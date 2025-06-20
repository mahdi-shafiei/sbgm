from typing import Sequence, Optional, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import einops
import equinox as eqx
from jaxtyping import Key, Array, Float, jaxtyped
from beartype import beartype as typechecker


class AdaLayerNorm(eqx.Module):
    net: eqx.Module
    ln: eqx.nn.LayerNorm
    data_shape: Sequence[int]

    def __init__(
        self, 
        data_shape: Sequence[int], 
        condition_dim: int, 
        *, 
        key: Key
    ):
        """
            Adaptive layer norm; generate scale and shift parameters from conditioning context.
        """
        data_dim = jnp.prod(jnp.asarray(data_shape)).item() # Ensure this isn't a jax array
        self.net = eqx.nn.Linear(condition_dim, data_dim * 2, key=key)
        # Don't use bias or scale since these will be learnable through the conditioning context
        self.ln = eqx.nn.LayerNorm(data_shape, use_bias=False, use_weight=False)
        self.data_shape = data_shape

    def __call__(self, x: Array, conditioning: Array) -> Array:
        # Compute scale and shift parameters from conditioning context
        scale_and_shift = jax.nn.gelu(self.net(conditioning))
        scale, shift = jnp.split(scale_and_shift, 2)
        scale = scale.reshape(self.data_shape)
        shift = shift.reshape(self.data_shape)
        # Apply layer norm
        x = self.ln(x)
        # Apply scale and shift (same scale, shift to all elements)
        x = x * (1. + scale) + shift
        return x


class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    num_patches: int
    hidden_size: int
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    a_dim: int = None

    def __init__(
        self, 
        num_patches: int, 
        hidden_size: int, 
        mix_patch_size: int, 
        mix_hidden_size: int, 
        context_dim: Optional[int] = None,
        *, 
        key: Key
    ):
        tkey, ckey = jr.split(key)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, 
            num_patches, 
            mix_patch_size, 
            depth=1, 
            key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, 
            hidden_size, 
            mix_hidden_size, 
            depth=1, 
            key=ckey
        )
        if context_dim is not None:
            key1, key2 = jr.split(key)
            self.norm1 = AdaLayerNorm(
                (hidden_size, num_patches), context_dim, key=key1
            )
            self.norm2 = AdaLayerNorm(
                (num_patches, hidden_size), context_dim, key=key2
            )
        else:
            self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
            self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))
        self.hidden_size = hidden_size
        self.num_patches = num_patches 
        self.a_dim = context_dim

    def __call__(self, y: Array, a: Array = None) -> Array:
        if a is not None and self.a_dim is not None:
            y = y + jax.vmap(self.patch_mixer)(self.norm1(y, a))
        else:
            y = y + jax.vmap(self.norm1)(y)
        y = einops.rearrange(y, "c p -> p c")
        if a is not None and self.a_dim is not None:
            y = y + jax.vmap(self.hidden_mixer)(self.norm2(y, a))
        else:
            y = y + jax.vmap(self.norm2)(y)
        y = einops.rearrange(y, "p c -> c p")
        return y


def get_timestep_embedding(timesteps, embedding_dim):
    """Build sinusoidal embeddings (from Fairseq)."""
    # Convert scalar timesteps to an array
    assert embedding_dim % 2 == 0
    if jnp.isscalar(timesteps):
        timesteps = jnp.array(timesteps)
    timesteps *= 1000.
    half_dim = embedding_dim // 2
    emb = jnp.log(10_000.) / (half_dim - 1.)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=0)
    return emb


def get_activation_fn(fn):
    if isinstance(fn, str):
        return getattr(jax.nn, fn)
    else:
        return fn


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list[MixerBlock]
    norm: eqx.nn.LayerNorm
    t1: float
    embedding_dim: int
    final_activation: callable
    img_size: Sequence[int]
    q_dim: int
    a_dim: int

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: int,
        hidden_size: int,
        mix_patch_size: int,
        mix_hidden_size: int,
        num_blocks: int,
        t1: float,
        embedding_dim: int = 8,
        final_activation: Optional[Callable | str] = None,
        q_dim: Optional[int] = None,
        a_dim: Optional[int] = None,
        *,
        key: Key[jnp.ndarray, "..."]
    ):
        """
            A 2D MLP Mixer model.
            This model processes 2D images using a patch-based approach, where 
            each patch is processed through a series of mixer blocks. It also 
            supports optional conditioning through `q_dim` and `a_dim`.

            Parameters:
            -----------
            `img_size` : `Sequence[int]`
                Shape of the input image as a sequence (typically height and width).
            
            `patch_size` : `int`
                Size of the patches into which the input image is divided.
            
            `hidden_size` : `int`
                Size of the hidden layers within each Mixer block.
            
            `mix_patch_size` : `int`
                Size of the patches used for the patch mixing within each Mixer block.
            
            `mix_hidden_size` : `int`
                Size of the hidden layers used for mixing channels across patches.
            
            `num_blocks` : `int`
                Number of Mixer blocks in the model.
            
            `t1` : `float`
                Maximum time of diffusion process. Scales input times.
            
            `embedding_dim` : `int`, default: `4`
                Dimensionality of the time embedding. Defaults to `8`.

            `final_activation` : `Optional[Union[callable, str]]`, default: `None`
                Final activation function on output of model. Supply as a string
                of one of the attributes of `jax.nn` or a custom callable.
            
            `q_dim` : `Optional[int]`, default: `None`
                The number of channels in the conditioning map. Can be `None`.
                Must be same dimension as input `x` in `__call__`
            
            `a_dim` : `Optional[int]`, default: `None`
                The number of parameters in the conditioning. Can be `None`.
            
            `key` : `Key`
                JAX random key used for initialization.
        """

        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        _input_size = input_size + q_dim if q_dim is not None else input_size
        _context_dim = embedding_dim + a_dim if a_dim is not None else embedding_dim

        self.img_size = img_size
        self.q_dim = q_dim
        self.a_dim = a_dim

        self.conv_in = eqx.nn.Conv2d(
            _input_size, 
            hidden_size, 
            patch_size, 
            stride=patch_size, 
            key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, 
            input_size, 
            patch_size, 
            stride=patch_size, 
            key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, 
                hidden_size, 
                mix_patch_size, 
                mix_hidden_size, 
                context_dim=_context_dim,
                key=bkey
            ) 
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1
        self.embedding_dim = embedding_dim
        self.final_activation = get_activation_fn(final_activation)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        t: float | Float[Array, ""], 
        y: Float[Array, "..."], 
        q: Optional[Float[Array, "{self.q_dim} ..."]] = None, 
        a: Optional[Float[Array, "{self.a_dim}"]] = None, 
        *, 
        key: Optional[Key[jnp.ndarray, "..."]] = None
    ) -> Float[Array, "..."]:
        _, height, width = y.shape
        t = jnp.atleast_1d(t / self.t1)
        t = get_timestep_embedding(t, embedding_dim=self.embedding_dim)
        if q is not None:
            yq = jnp.concatenate([y, q])
            _input = yq
        else:
            _input = y
        y = self.conv_in(_input)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(y, "c h w -> c (h w)")
        if a is not None:
            a = jnp.atleast_1d(a)
            at = jnp.concatenate([a, t])
            _input = at
        else:
            _input = t
        for block in self.blocks:
            y = block(y, _input)
        y = self.norm(y)
        y = einops.rearrange(
            y, "c (h w) -> c h w", h=patch_height, w=patch_width
        )
        y = self.conv_out(y)
        if self.final_activation is not None:
            y = self.final_activation(y)
        return y