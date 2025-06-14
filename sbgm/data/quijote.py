import os 
from typing import Tuple 
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import Key, Array
import numpy as np
import torch
from torchvision import transforms

from .utils import Scaler, ScalerDataset, TorchDataLoader, InMemoryDataLoader

DATA_DIR = os.environ.get("DATA_DIR", "")


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        # Tuple of (images, contexts, targets), turn them into tensors
        self.tensors = tuple(
            torch.as_tensor(tensor) for tensor in tensors
        )
        self.transform = transform
        assert all(
            self.tensors[0].size(0) == tensor.size(0) 
            for tensor in self.tensors
        )

    def __getitem__(self, index):
        x = self.tensors[0][index] # Fields
        a = self.tensors[1][index] # Parameters

        if self.transform:
            x = self.transform(x)

        return x, a

    def __len__(self):
        return self.tensors[0].size(0)


def get_quijote_data(n_pix: int) -> Tuple[Array, Array]:

    assert (DATA_DIR is not None) and (DATA_DIR != ""), (
        "Note: you must supply a path with the Quijote hypercube parameters and simulations."
    )

    X = np.load(os.path.join(DATA_DIR, "quijote_fields.npy"))[:, np.newaxis, ...]
    A = np.load(os.path.join(DATA_DIR, "quijote_parameters.npy"))

    dx = int(256 / n_pix)
    X = X.reshape((-1, 1, n_pix, dx, n_pix, dx)).mean(axis=(3, 5))

    return X, A


def get_quijote_labels() -> Array:
    Q = np.load(os.path.join(DATA_DIR, "quijote_parameters.npy"))
    return Q


def quijote(key, n_pix, split=0.9):
    key_train, key_valid = jr.split(key)

    data_shape = (1, n_pix, n_pix)
    context_shape = None #(1, n_pix, n_pix)
    parameter_dim = 5

    X, A = get_quijote_data(n_pix) 

    print("Quijote data:", X.shape, A.shape)

    # min = X.min()
    # max = X.max()
    # X = (X - min) / (max - min) # ... -> [0, 1]
    # X = 2.0 * (X - min) / (max - min) - 1.0 # ... -> [-1, 1]
    X = (X - X.mean()) / X.std()

    A = (A - A.min()) / (A.max() - A.min()) # ... -> [0, 1]

    n_train = int(split * len(X))

    # scaler = Scaler() # [0,1] -> [-1,1]

    # Don't have many maps
    train_dataloader = InMemoryDataLoader(
        X=X[:n_train], A=A[:n_train], key=key_train
    )
    valid_dataloader = InMemoryDataLoader(
        X=X[n_train:], A=A[n_train:], key=key_valid
    )

    def label_fn(key, n):
        A = get_quijote_labels() 
        ix = jr.choice(key, jnp.arange(len(A)), (n,))
        Q = None
        A = A[ix]
        return Q, A

    return ScalerDataset(
        name="quijote",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        process_fn=None, #scaler,
        label_fn=label_fn
    )