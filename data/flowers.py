import os
import jax.random as jr 
import jax.numpy as jnp
from jaxtyping import Key
from torchvision import transforms, datasets

from .utils import Scaler, ScalerDataset, TorchDataLoader


def flowers(path: str, key: Key, n_pix: int) -> ScalerDataset:

    key_train, key_valid = jr.split(key)

    data_shape = (3, n_pix, n_pix)
    parameter_dim = 1
    n_classes = 102

    scaler = Scaler()

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomCrop(n_pix, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomCrop(n_pix, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )

    train_dataset = datasets.Flowers102(
        os.path.join(path, "datasets/flowers/"), 
        split="train", 
        download=True, 
        transform=train_transform,
        target_transform=transforms.Lambda(lambda x: x.float())
    )
    valid_dataset = datasets.Flowers102(
        os.path.join(path, "datasets/flowers/"), 
        split="val", 
        download=True, 
        transform=valid_transform,
        target_transform=transforms.Lambda(lambda x: x.float())
    )

    train_dataloader = TorchDataLoader(
        train_dataset, 
        data_shape=data_shape, 
        context_shape=None, 
        parameter_dim=parameter_dim,
        key=key_train
    )
    valid_dataloader = TorchDataLoader(
        valid_dataset, 
        data_shape=data_shape, 
        context_shape=None, 
        parameter_dim=parameter_dim,
        key=key_valid
    )

    def label_fn(key, n):
        Q = None
        A = jr.choice(key, jnp.arange(n_classes), (n,))
        A = A[:, jnp.newaxis].astype(jnp.float32)
        return Q, A

    return ScalerDataset(
        name="flowers",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=None,
        parameter_dim=parameter_dim,
        process_fn=scaler,
        label_fn=label_fn
    )
