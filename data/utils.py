import abc
from typing import Tuple, Callable, Optional, Generator, Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import Key, Array, jaxtyped
from beartype import beartype as typechecker
import numpy as np
import torch


def expand_if_scalar(x):
    return x[:, jnp.newaxis] if x.ndim == 1 else x


def default(v, d):
    return v if v is not None else d


class Scaler:
    forward: Callable 
    reverse: Callable
    def __init__(self, x_min=0., x_max=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


class Normer:
    forward: Callable 
    reverse: Callable
    def __init__(self, x_mean=0., x_std=1.):
        # [0, 1] -> [-1, 1]
        self.forward = lambda x: (x - x_mean) / x_std
        # [-1, 1] -> [0, 1]
        self.reverse = lambda y: y * x_std + x_mean


class Identity:
    forward: Callable 
    reverse: Callable
    def __init__(self):
        self.forward = lambda x: x
        self.reverse = lambda x: x


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        X: np.ndarray | Array, 
        Q: Optional[np.ndarray | Array] = None, 
        A: Optional[np.ndarray | Array] = None, 
        *, 
        process_fn: Optional[Scaler | Normer | Identity] = None,
        key: Key[jnp.ndarray, "..."]
    ):
        self.X = jnp.asarray(X)
        self.Q = jnp.asarray(Q) if Q is not None else Q
        self.A = jnp.asarray(A) if A is not None else A
        self.process_fn = default(process_fn, Identity()) 
        self.key = key

    def loop(
        self, batch_size: int
    ) -> Generator[Tuple[Array, Optional[Array], Optional[Array]], None, None]:
        dataset_size = self.X.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                # x, q, a
                yield (
                    self.process_fn.forward(self.X[batch_perm]), 
                    self.Q[batch_perm] if self.Q is not None else None, 
                    self.A[batch_perm] if self.A is not None else None 
                )
                start = end
                end = start + batch_size


class TorchDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        data_shape: Sequence[int],
        context_shape: Sequence[int],
        parameter_dim: int,
        *, 
        process_fn: Optional[Scaler | Normer | Identity] = None,
        num_workers: Optional[int] = None, 
        key: Key[jnp.ndarray, "..."]
    ):
        self.dataset = dataset
        self.context_shape = context_shape 
        self.parameter_dim = parameter_dim 
        self.seed = jr.randint(key, (), 0, 1_000_000).item() 
        self.process_fn = default(process_fn, Identity()) 
        self.num_workers = num_workers

    def loop(
        self, batch_size: int, num_workers: int = 2
    ) -> Generator[
        Tuple[Array, Optional[Array], Optional[Array]], None, None
    ]:
        generator = torch.Generator().manual_seed(self.seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=self.num_workers if self.num_workers is not None else num_workers,
            shuffle=True,
            drop_last=True,
            generator=generator
        )
        while True:
            for tensors in dataloader:

                x, *qa = tensors
                if self.context_shape and self.parameter_dim:
                    q, a = qa
                else:
                    if self.context_shape:
                        (q,) = qa
                    else:
                        q = None
                    if self.parameter_dim:
                        (a,) = qa
                    else:
                        a = None
                x = jnp.asarray(x)
                yield ( 
                    self.process_fn.forward(x),
                    expand_if_scalar(jnp.asarray(q)) if self.context_shape else None,
                    expand_if_scalar(jnp.asarray(a)) if self.parameter_dim else None
                )


@jaxtyped(typechecker=typechecker)
@dataclass
class ScalerDataset:
    name: str
    train_dataloader: TorchDataLoader | InMemoryDataLoader
    valid_dataloader: TorchDataLoader | InMemoryDataLoader
    data_shape: Tuple[int, ...]
    context_shape: Tuple[int, ...] | None
    parameter_dim: int | None
    process_fn: Scaler | Normer | Identity | None
    label_fn: Callable