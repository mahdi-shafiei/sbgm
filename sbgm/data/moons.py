import jax.numpy as jnp
import jax.random as jr 
from sklearn.datasets import make_moons

from .utils import ScalerDataset, InMemoryDataLoader, Scaler, Normer


def key_to_seed(key):
    return int(jnp.asarray(jr.key_data(key)).sum())

def moons(key):
    key_train, key_valid = jr.split(key)

    data_shape = (2,)
    context_shape = None 
    parameter_dim = 1

    Xt, Yt = make_moons(
        40_000, noise=0.05, random_state=key_to_seed(key_train)
    )
    Xv, Yv = make_moons(
        40_000, noise=0.05, random_state=key_to_seed(key_valid)
    )
    Yt = Yt[:, jnp.newaxis].astype(jnp.float32)
    Yv = Yv[:, jnp.newaxis].astype(jnp.float32)

    process_fn = Normer(Xt.mean(), Xt.std())
    
    train_dataloader = InMemoryDataLoader(
        X=jnp.asarray(Xt), Q=jnp.asarray(Yt), A=None, process_fn=process_fn, key=key_train
    )
    valid_dataloader = InMemoryDataLoader(
        X=jnp.asarray(Xv), Q=jnp.asarray(Yv), A=None, process_fn=process_fn, key=key_valid
    )

    def label_fn(key, n):
        Q = jr.choice(key, jnp.array([0., 1.]), (n,))[:, jnp.newaxis]
        A = None
        return Q, A

    return ScalerDataset(
        name="moons",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        process_fn=process_fn, # Scaler(x_min=Xt.min(), x_max=Xt.max()),
        label_fn=label_fn
    )