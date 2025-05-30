from __future__ import annotations

from functools import partial
from typing import Tuple, Optional, Optional, List
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import equinox as eqx
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker

from einops import einsum, rearrange, repeat
import einx


def identity(x):
    return x


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def stop_grad(a):
    return jax.lax.stop_gradient(a)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def key_split_allowing_none(key, n=2, i=None):
    if key is not None:
        if i is not None:
            key = jr.fold_in(key, i)
        keys = jr.split(key, n)
    else: 
        keys = [None] * n
    return keys


class Upsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None, 
        *, 
        key: Key
    ):
        self.conv = eqx.nn.Conv2d(
            dim, 
            default(dim_out, dim), 
            kernel_size=3, 
            padding=1, 
            key=key
        )

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape
        x = jax.image.resize(
            x, shape=(c, h * 2, w * 2), method='bilinear'
        )
        x = self.conv(x)
        return x


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None, 
        *, 
        key: Key
    ):
        self.conv = eqx.nn.Conv2d(
            dim * 4, default(dim_out, dim), kernel_size=1, key=key
        )

    def __call__(self, x: Array) -> Array:
        x = rearrange(
            x, 'c (h p1) (w p2) -> (c p1 p2) h w', p1=2, p2=2
        )
        x = self.conv(x)
        return x


class RMSNorm(eqx.Module):
    scale: float # Array, learnable?
    gamma: Array

    def __init__(self, dim: int):
        self.scale = dim ** 0.5 # This is a default scaling
        self.gamma = jnp.zeros((dim, 1, 1))

    def __call__(self, x: Array) -> Array:
        # Normalise x, shift and scale 
        return (x / jnp.linalg.norm(x, ord=2, axis=0)) * (self.gamma + 1) * self.scale 


class SinusoidalPosEmb(eqx.Module):
    dim: int
    theta: int

    def __init__(self, dim: int, theta: int = 10_000):
        self.dim = dim
        self.theta = theta

    def __call__(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)
        half_dim = self.dim // 2
        emb = jnp.log(self.theta) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb) # emb = x * emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
        return emb


class RandomOrLearnedSinusoidalPosEmb(eqx.Module):
    random: bool
    weights: Array

    def __init__(self, dim: int, is_random: bool = False, *, key: Key):
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = jr.normal(key, (half_dim,))
        self.random = is_random

    def __call__(self, x: Array) -> Array:
        x = jnp.atleast_1d(x) 
        weights = jax.lax.stop_gradient(self.weights) if self.random else self.weights
        freqs = x * weights * 2. * jnp.pi 
        fouriered = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)])
        fouriered = jnp.concatenate([x, fouriered])
        return fouriered


class Block(eqx.Module):
    proj: eqx.nn.Conv2d
    norm: RMSNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        dropout: float = 0., 
        *, 
        key: Key
    ):
        self.proj = eqx.nn.Conv2d(
            dim, dim_out, kernel_size=3, padding=1, key=key
        )
        self.norm = RMSNorm(dim_out)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self, 
        x: Array, 
        scale_shift: Tuple[Array, Array] = None, 
        key: Optional[Key] = None
    ) -> Array:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1.) + shift

        return self.dropout(jax.nn.silu(x), key=key)


class ResnetBlock(eqx.Module):
    mlp: eqx.nn.Linear
    block1: Block
    block2: Block
    res_conv: eqx.nn.Conv2d

    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        *, 
        time_emb_dim: int = None, 
        dropout: float = 0., 
        key: Key
    ):
        keys = jr.split(key, 4)
        self.mlp = eqx.nn.Linear(
            time_emb_dim, dim_out * 2, key=keys[0]
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, dropout=dropout, key=keys[1])
        self.block2 = Block(dim_out, dim_out, key=keys[2])
        self.res_conv = eqx.nn.Conv2d(
            dim, dim_out, kernel_size=1, key=keys[3]
        ) if dim != dim_out else eqx.nn.Identity()

    def __call__(
        self, 
        x: Array, 
        time_emb: Optional[Array] = None, 
        *, 
        key: Key 
    ) -> Array:
        keys = key_split_allowing_none(key)

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(jax.nn.silu(time_emb))
            time_emb = rearrange(time_emb, 'c -> c 1 1')
            scale_shift = jnp.split(time_emb, 2)

        h = self.block1(x, scale_shift=scale_shift, key=keys[0])

        h = self.block2(h, key=keys[1])

        return h + self.res_conv(x)


class LinearAttention(eqx.Module):
    scale: float
    heads: int
    norm1: RMSNorm
    mem_kv: Array
    to_qkv: eqx.nn.Conv2d
    conv: eqx.nn.Conv2d
    norm2: RMSNorm

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        *,
        key: Key
    ):
        keys = jr.split(key, 3)

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm1 = RMSNorm(dim)

        self.mem_kv = jr.normal(keys[0], (2, heads, dim_head, num_mem_kv))
        self.to_qkv = eqx.nn.Conv2d(
            dim, hidden_dim * 3, kernel_size=1, use_bias=False, key=keys[1]
        )

        self.conv = eqx.nn.Conv2d(hidden_dim, dim, kernel_size=1, key=keys[2])
        self.norm2 = RMSNorm(dim)

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape

        x = self.norm1(x)

        qkv = jnp.split(self.to_qkv(x), 3)
        q, k, v = tuple(
            rearrange(t, '(h c) x y -> h c (x y)', h=self.heads) for t in qkv
        )

        mk, mv = tuple(repeat(t, 'h c n -> h c n') for t in self.mem_kv)
        k, v = map(partial(jnp.concatenate, axis=-1), ((mk, k), (mv, v)))

        q = jax.nn.softmax(q, axis=-2)
        k = jax.nn.softmax(k, axis=-1)

        q = q * self.scale

        context = einsum(k, v, 'h d n, h e n -> h d e')

        out = einsum(context, q, 'h d e, h d n -> h e n')
        out = rearrange(
            out, 'h c (x y) -> (h c) x y', h=self.heads, x=h, y=w
        )
        return self.norm2(self.conv(out))


class Attention(eqx.Module):
    scale: float
    heads: int
    norm: RMSNorm
    mem_kv: Array
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        flash: bool = False,
        *,
        key: Key
    ):
        keys = jr.split(key, 3)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = jr.normal(keys[0], (2, heads, num_mem_kv, dim_head))
        self.to_qkv = eqx.nn.Conv2d(
            dim, hidden_dim * 3, kernel_size=1, use_bias=False, key=keys[1]
        )
        self.to_out = eqx.nn.Conv2d(
            hidden_dim, dim, kernel_size=1, use_bias=False, key=keys[2]
        )

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape

        x = self.norm(x)

        qkv = jnp.split(self.to_qkv(x), 3)
        q, k, v = map(lambda t: rearrange(t, '(h c) x y -> h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> h n d'), self.mem_kv)
        k, v = map(partial(jnp.concatenate, axis=-2), ((mk, k), (mv, v)))

        q = q * self.scale
        sim = einsum(q, k, 'h i d, h j d -> h i j')

        attn = jax.nn.softmax(sim, axis=-1)
        out = einsum(attn, v, 'h i j, h j d -> h i d')

        out = rearrange(out, 'h (x y) d -> (h d) x y', x=h, y=w)
        return self.to_out(out)
 

class TimeMLP(eqx.Module):
    embed: RandomOrLearnedSinusoidalPosEmb | SinusoidalPosEmb
    layers: List[eqx.Module]

    def __init__(self, embed, fourier_dim, time_dim, a_dim, *, key):
        keys = jr.split(key)
        self.embed = embed 
        self.layers = [
            eqx.nn.Linear(
                fourier_dim + a_dim if a_dim is not None else fourier_dim, 
                time_dim, 
                key=keys[0]
            ),
            jax.nn.gelu,
            eqx.nn.Linear(
                time_dim + a_dim if a_dim is not None else time_dim, 
                time_dim, 
                key=keys[1]
            )
        ]

    def __call__(self, t, a):
        t = self.embed(t)
        t = self.layers[0](jnp.concatenate([t, a]) if a is not None else t)
        t = self.layers[1](t)
        t = self.layers[2](jnp.concatenate([t, a]) if a is not None else t)
        return t


class UNet(eqx.Module):
    channels: int
    init_conv: eqx.nn.Conv2d
    random_or_learned_sinusoidal_cond: bool
    time_mlp: List[eqx.Module]

    downs: List[eqx.Module]
    ups: List[eqx.Module]

    mid_block1: ResnetBlock
    mid_attn: Attention | LinearAttention
    mid_block2: ResnetBlock

    out_dim: int
    final_res_block: ResnetBlock
    final_conv: eqx.nn.Conv2d

    q_channels: int
    a_dim: int

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Sequence[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        q_channels: Optional[int] = None,
        a_dim: Optional[int] = None,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10_000,
        dropout: float = 0.,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        full_attn: bool = False , # Defaults to full attention only for inner most layer
        flash_attn: bool = False,
        *,
        key: Key[jnp.ndarray, "..."]
    ):
        key_modules, key_down, key_mid, key_up, key_final = jr.split(key, 5)
        keys = jr.split(key_modules, 3)

        # Determine dimensions
        self.channels = channels
        self.q_channels = q_channels
        self.a_dim = a_dim

        init_dim = default(init_dim, dim)
        self.init_conv = eqx.nn.Conv2d(
            channels + q_channels if q_channels is not None else channels, 
            init_dim, 
            kernel_size=7, 
            padding=3, 
            key=keys[0]
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features, key=keys[1]
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(
                dim, theta=sinusoidal_pos_emb_theta
            )
            fourier_dim = dim

        self.time_mlp = TimeMLP(
            sinu_pos_emb, fourier_dim, time_dim, a_dim, key=keys[3]
        )

        # Attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # Prepare blocks
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # Layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        # Downsampling layers 
        for ind, (
            (dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head
        ) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            keys = jr.split(jr.fold_in(key_down, ind), 4)

            is_last = ind >= (num_resolutions - 1)

            attn_class = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(
                [
                    resnet_block(dim_in, dim_in, key=keys[0]),
                    resnet_block(dim_in, dim_in, key=keys[1]),
                    attn_class(
                        dim_in, 
                        dim_head=layer_attn_dim_head, 
                        heads=layer_attn_heads, 
                        key=keys[2]
                    ),
                    Downsample(dim_in, dim_out, key=keys[3]) 
                    if not is_last else 
                    eqx.nn.Conv2d(
                        dim_in, dim_out, kernel_size=3, padding=1, key=keys[3]
                    )
                ]
            )

        # Middle layers + attention
        mid_dim = dims[-1]
        keys = jr.split(key_mid, 3)
        self.mid_block1 = resnet_block(mid_dim, mid_dim, key=keys[0])
        self.mid_attn = FullAttention(
            mid_dim, 
            heads=attn_heads[-1], 
            dim_head=attn_dim_head[-1],
            key=keys[1]
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim, key=keys[2])

        # Upsampling layers + skip connections
        for ind, (
            (dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            keys = jr.split(jr.fold_in(key_up, ind), 4)

            is_last = ind == (len(in_out) - 1)

            attn_class = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(
                [
                    resnet_block(dim_out + dim_in, dim_out, key=keys[0]),
                    resnet_block(dim_out + dim_in, dim_out, key=keys[1]),
                    attn_class(
                        dim_out, 
                        dim_head=layer_attn_dim_head, 
                        heads=layer_attn_heads, 
                        key=keys[2]
                    ),
                    Upsample(dim_out, dim_in, key=keys[3]) 
                    if not is_last else 
                    eqx.nn.Conv2d(
                        dim_out, dim_in, kernel_size=3, padding=1, key=keys[3]
                    )
                ]
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        keys = jr.split(key_final)
        self.final_res_block = resnet_block(init_dim * 2, init_dim, key=keys[0])
        self.final_conv = eqx.nn.Conv2d(
            init_dim + q_channels if q_channels is not None else init_dim, 
            self.out_dim, 
            kernel_size=1, 
            key=keys[1]
        )

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        t: Float[Array, ""], 
        x: Float[Array, "{self.channels} _ _"], 
        q: Optional[Float[Array, "{self.q_channels} _ _"]] = None, 
        a: Optional[Float[Array, "{self.a_dim}"]] = None, 
        key: Optional[Key[jnp.ndarray, "..."]] = None
    ) -> Float[Array, "{self.channels} _ _"]:
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), (
            f"Input dimensions {x.shape[-2:]} need to be divisible"
            f" by {self.downsample_factor}, given the UNet"
        )

        key_down, key_mid, key_up, key_final = key_split_allowing_none(key, n=4) 

        x = self.init_conv(jnp.concatenate([x, q]) if q is not None else x)
        r = x.copy()

        t = self.time_mlp(t, a)

        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            keys = key_split_allowing_none(key_down, i=i)

            x = block1(x, t, key=keys[0])
            h.append(x)

            x = block2(x, t, key=keys[1])
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        keys = key_split_allowing_none(key_mid)
        x = self.mid_block1(x, t, key=keys[0])
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t, key=keys[1])

        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            keys = key_split_allowing_none(key_up, i=i)

            x = jnp.concatenate([x, h.pop()])
            x = block1(x, t, key=keys[0])

            x = jnp.concatenate([x, h.pop()])
            x = block2(x, t, key=keys[1])
            x = attn(x) + x

            x = upsample(x)

        x = jnp.concatenate([x, r])
        x = self.final_res_block(x, t, key=key_final)

        return self.final_conv(jnp.concatenate([x, q]) if q is not None else x)