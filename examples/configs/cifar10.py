import ml_collections
import jax.numpy as jnp


def cifar10_config():
    config = ml_collections.ConfigDict()

    config.seed                  = 0

    # Data
    config.dataset_name          = "cifar10" 

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.model_type             = "UNet"
    model.is_biggan              = False
    model.dim_mults              = [1, 1, 1]
    model.hidden_size            = 128
    model.heads                  = 4
    model.dim_head               = 64
    model.dropout_rate           = 0.3
    model.num_res_blocks         = 2
    model.attn_resolutions       = [8, 16, 32]
    model.final_activation       = None

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.sde                      = "VP"
    sde.t1                       = 8.
    sde.t0                       = 1e-5 
    sde.dt                       = 0.1
    sde.beta_integral            = lambda t: t 
    sde.weight_fn                = lambda t: 1. - jnp.exp(-sde.beta_integral(t))

    # Sampling
    config.use_ema               = False
    config.sample_size           = 5
    config.exact_log_prob            = False
    config.ode_sample            = True
    config.eu_sample             = True

    # Optimisation hyperparameters
    config.start_step            = 0
    config.n_steps               = 1_000_000
    config.batch_size            = 512 
    config.accumulate_gradients  = False
    config.n_minibatches         = 1
    config.sample_and_save_every = 1_000
    config.lr                    = 1e-4
    config.opt                   = "adabelief"
    config.opt_kwargs            = {} 
    config.num_workers           = 8

    # Other
    config.cmap                  = None

    return config