import os
import jax.random as jr 
import jax.numpy as jnp
from jaxtyping import Key
from torchvision import transforms, datasets

from .utils import Scaler, ScalerDataset, TorchDataLoader, InMemoryDataLoader


def convert_torch_to_in_memory(dataset):
    # Convert torch cifar10 dataset to in-memory
    data = jnp.asarray(dataset.data)
    data = data.transpose(0, 3, 1, 2).astype(jnp.float32)
    data = data / data.max()
    targets = jnp.asarray(dataset.targets).astype(jnp.float32)
    targets = targets[:, jnp.newaxis]
    return data, targets


def cifar10(path: str, key: Key, *, in_memory: bool = True) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    n_pix = 32 # Native resolution for CIFAR10 
    data_shape = (3, n_pix, n_pix)
    parameter_dim = 1

    scaler = Scaler(x_min=0., x_max=1.)

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Lambda(scaler.forward) # [0,1] -> [-1,1]
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )
    train_dataset = datasets.CIFAR10(
        os.path.join(path, "datasets/cifar10/"),
        train=True, 
        download=True, 
        transform=train_transform
    )
    valid_dataset = datasets.CIFAR10(
        os.path.join(path, "datasets/cifar10/"),
        train=False, 
        download=True, 
        transform=valid_transform
    )

    if in_memory:
        Xt, At = convert_torch_to_in_memory(train_dataset) 
        Xv, Av = convert_torch_to_in_memory(valid_dataset) 
        train_dataloader = InMemoryDataLoader(X=Xt, A=At, key=key_train) 
        valid_dataloader = InMemoryDataLoader(X=Xv, A=Av, key=key_valid) 
    else:
        train_dataloader = TorchDataLoader(
            train_dataset, data_shape, context_shape=None, parameter_dim=parameter_dim, key=key_train
        )
        valid_dataloader = TorchDataLoader(
            valid_dataset, data_shape, context_shape=None, parameter_dim=parameter_dim, key=key_valid
        )

    def label_fn(key, n):
        Q = None
        A = jr.choice(key, jnp.arange(10), (n,))[:, jnp.newaxis]
        return Q, A

    return ScalerDataset(
        name="cifar10",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        parameter_dim=parameter_dim,
        context_shape=None,
        scaler=scaler,
        label_fn=label_fn
    )