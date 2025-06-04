import jax
import jax.numpy as jnp 
import jax.random as jr
import optax
import matplotlib.pyplot as plt 
from einops import rearrange

import sbgm

"""
    Configuration
"""

n_devices = len(jax.devices())

key = jr.key(0)

data_key, model_key, train_key = jr.split(key, 3)

dataset = sbgm.data.quijote(data_key, n_pix=64)

(n_channels, n_pix, n_pix) = dataset.data_shape

# Data
dataset_name          = dataset.name 
img_shape             = (n_channels, n_pix, n_pix)

# Model (DiT)
n_heads               = 8  
embed_dim             = 512
patch_size            = 2 
depth                 = 4

# SDE
t1                    = 8.
t0                    = 0.
dt                    = 0.1
beta_integral         = lambda t: t 
weight_fn             = lambda t: 1. - jnp.exp(-beta_integral(t)) 

# Sampling
use_ema               = True
sample_size           = 5 # Squared for a grid
exact_log_prob        = False
ode_sample            = True # Sample the ODE during training
eu_sample             = True # Euler-Maruyama sample the SDE during training

# Optimisation hyperparameters
start_step            = 0
n_steps               = 200_000
batch_size            = 200 * len(jax.devices())
sample_and_save_every = 2_000
lr                    = 3e-4
opt                   = optax.adamw
opt_kwargs            = {} 

cmap                  = "gist_stern"

# Multiple GPU training if you are so inclined
sharding, replicated_sharding = sbgm.shard.get_shardings()

"""
    Score network and SDE
"""

model = sbgm.models.DiT(
    img_size=n_pix,
    channels=n_channels,
    embed_dim=embed_dim,
    patch_size=patch_size,
    depth=depth,
    n_heads=n_heads,
    a_dim=dataset.parameter_dim,
    key=model_key
)

sde = sbgm.sde.VPSDE(
    beta_integral_fn=beta_integral,
    dt=dt,
    t0=t0, 
    t1=t1,
    weight_fn=weight_fn
)

"""
    Train
"""

# Fit model to dataset
model = sbgm.train.train(
    train_key,
    model,
    sde,
    dataset,
    opt=opt(lr, **opt_kwargs),
    n_steps=n_steps,
    batch_size=batch_size,
    sample_size=sample_size,
    eu_sample=eu_sample,
    ode_sample=ode_sample,
    reload_opt_state=False,
    plot_train_data=True,
    sample_and_save_every=sample_and_save_every,
    sharding=sharding,
    replicated_sharding=replicated_sharding,
    cmap=cmap,
    save_dir="./"
)

"""
    Sample
"""

n_side = 10

key, key_Q, sample_key = jr.split(key, 3)
sample_keys = jr.split(sample_key, n_side ** 2)

# Sample random conditioning fields and P(k) parameters 
Q, A = dataset.label_fn(key_Q, n_side ** 2)

# EU sampling
sample_fn = sbgm.sample.get_eu_sample_fn(model, sde, dataset.data_shape)
eu_samples = jax.vmap(sample_fn)(sample_keys, Q, A)
eu_samples = dataset.process_fn.reverse(eu_samples)

# ODE sampling
sample_fn = sbgm.sample.get_ode_sample_fn(model, sde, dataset.data_shape)
ode_samples = jax.vmap(sample_fn)(sample_keys, Q, A)
ode_samples = dataset.process_fn.reverse(ode_samples)

fig, axs = plt.subplots(1, 2, dpi=150)
ax = axs[0]
ax.imshow(
    rearrange(eu_samples, "(h w) c x y -> (h x) (w y) c", h=n_side, w=n_side, x=n_pix, y=n_pix), 
)
ax = axs[1]
ax.imshow(
    rearrange(ode_samples, "(h w) c x y -> (h x) (w y) c", h=n_side, w=n_side, x=n_pix, y=n_pix)
)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()