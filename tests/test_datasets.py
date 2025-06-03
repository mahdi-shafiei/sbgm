import jax.random as jr
import jax.numpy as jnp

from sbgm.data.utils import Normer, Scaler, dataset_from_tensors


def test_dataset_from_tensors():

    key = jr.key(0)

    n_channels = 1
    n_pix = 8
    parameter_dim = 10

    X = jnp.ones((100, n_channels, n_pix, n_pix))  # Example of additional tensor
    Q = jnp.ones((100, n_channels, n_pix, n_pix))  # Example of additional tensor
    A = jnp.ones((100, parameter_dim))

    process_fn = Normer(jnp.mean(X, axis=0), jnp.std(X, axis=0))

    dataset = dataset_from_tensors(
        X=X, Q=Q, A=A, key=key, process_fn=process_fn, in_memory=True, name="dataset"
    )

    for i, (x, q, a) in zip(range(10), dataset.train_dataloader.loop(10)): 
        continue

    process_fn = Scaler(jnp.min(X, axis=0), jnp.max(X, axis=0))

    dataset = dataset_from_tensors(
        X=X, Q=Q, A=A, key=key, process_fn=process_fn, in_memory=False, name="dataset"
    )

    for i, (x, q, a) in zip(range(10), dataset.train_dataloader.loop(10)): 
        continue