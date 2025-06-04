import os
import jax.random as jr 
import jax.numpy as jnp
from jaxtyping import Key, Float, Array
from torch import Tensor
from torchvision import datasets, transforms

from .utils import Normer, Scaler, ScalerDataset, TorchDataLoader, InMemoryDataLoader


def tensor_to_array(tensor: Tensor) -> Array:
    return jnp.asarray(tensor.numpy())


def mnist(path:str, key: Key, *, in_memory: bool = True) -> ScalerDataset:

    key_train, key_valid = jr.split(key)

    n_pix = 28
    data_shape = (1, n_pix, n_pix)
    parameter_dim = 1 
    n_classes = 10

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor() 
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor()
        ]
    )

    # MNIST is small enough that the whole dataset can be placed in memory, so
    # we can actually use a faster method of data loading.
    train_dataset = datasets.MNIST(
        os.path.join(path, "datasets/mnist/"), 
        train=True, 
        download=True, 
        transform=train_transform,
        target_transform=transforms.Lambda(lambda x: x.float())
    )
    valid_dataset = datasets.MNIST(
        os.path.join(path, "datasets/mnist/"), 
        train=False, 
        download=True, 
        transform=valid_transform,
        target_transform=transforms.Lambda(lambda x: x.float())
    )

    if in_memory:
        train_data = tensor_to_array(train_dataset.data)[:, jnp.newaxis, ...] / 255.
        train_targets = tensor_to_array(train_dataset.targets)[:, jnp.newaxis]
        valid_data = tensor_to_array(valid_dataset.data)[:, jnp.newaxis, ...] / 255.
        valid_targets = tensor_to_array(valid_dataset.targets)[:, jnp.newaxis]

        train_targets = train_targets.astype(jnp.float32)
        valid_targets = valid_targets.astype(jnp.float32)

        mu, std = train_data.mean(), train_data.std()
        train_data = (train_data - mu) / std
        valid_data = (valid_data - mu) / std

        train_dataloader = InMemoryDataLoader(
            train_data, Q=None, A=train_targets, key=key_train
        )
        valid_dataloader = InMemoryDataLoader(
            valid_data, Q=None, A=valid_targets, key=key_valid
        )
    else:
        train_dataloader = TorchDataLoader(
            train_dataset, data_shape, context_shape=None, parameter_dim=parameter_dim, key=key_train
        )
        valid_dataloader = TorchDataLoader(
            valid_dataset, data_shape, context_shape=None, parameter_dim=parameter_dim, key=key_valid
        )


    def label_fn(key: Key, n: int) -> tuple[None, Float[Array, "n 1"]]:
        Q = None
        A = jr.choice(key, jnp.arange(n_classes), (n,))
        A = A[:, jnp.newaxis].astype(jnp.float32)
        return Q, A

    return ScalerDataset(
        name="mnist",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=None,
        parameter_dim=parameter_dim,
        process_fn=None,
        label_fn=label_fn
    )