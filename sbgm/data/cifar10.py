import os
import jax.random as jr 
import jax.numpy as jnp
from jaxtyping import Key
from torchvision import transforms, datasets

from .utils import Scaler, Normer, ScalerDataset, TorchDataLoader, InMemoryDataLoader


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
    context_shape = None
    parameter_dim = 1
    n_classes = 10

    scaler = Normer()

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Lambda(scaler.forward) 
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
        transform=train_transform,
        target_transform=transforms.Lambda(lambda x: x.float())
    )
    valid_dataset = datasets.CIFAR10(
        os.path.join(path, "datasets/cifar10/"),
        train=False, 
        download=True, 
        transform=valid_transform,
        target_transform=transforms.Lambda(lambda x: x.float())
    )

    if in_memory:
        Xt, At = convert_torch_to_in_memory(train_dataset) 
        Xv, Av = convert_torch_to_in_memory(valid_dataset) 

        At = At.astype(jnp.float32)
        Av = Av.astype(jnp.float32)

        process_fn = Normer(x_mean=Xt.mean(), x_std=Xt.std())

        train_dataloader = InMemoryDataLoader(
            X=Xt, A=At, process_fn=process_fn, key=key_train) 
        valid_dataloader = InMemoryDataLoader(
            X=Xv, A=Av, process_fn=process_fn, key=key_valid
        ) 
    else:
        process_fn = Scaler(x_min=0., x_max=1.)

        train_dataloader = TorchDataLoader(
            train_dataset, 
            data_shape=data_shape, 
            context_shape=context_shape,
            parameter_dim=parameter_dim, 
            key=key_train
        )
        valid_dataloader = TorchDataLoader(
            valid_dataset, 
            data_shape=data_shape, 
            context_shape=context_shape,
            parameter_dim=parameter_dim, 
            key=key_valid
        )

    def label_fn(key, n):
        Q = None
        A = jr.choice(key, jnp.arange(n_classes), (n,))
        A = A[:, jnp.newaxis].astype(jnp.float32)
        return Q, A

    return ScalerDataset(
        name="cifar10",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        parameter_dim=parameter_dim,
        context_shape=None,
        process_fn=process_fn, 
        label_fn=label_fn
    )