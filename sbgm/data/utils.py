import abc
from functools import partial
from typing import Tuple, Callable, Optional, Generator, Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker
import numpy as np
import torch

typecheck = jaxtyped(typechecker=typechecker)

"""
    Various utilities for creating datasets. 
"""


def exists(v):
    return v is not None 


def default(v, d):
    return v if exists(v) else d


def stop_grad(x: Array) -> Array:
    return jax.lax.stop_gradient(x)


def maybe_convert(a: np.ndarray | Array) -> np.ndarray:
    return np.asarray(a) if isinstance(a, jnp.ndarray) else a


def expand_if_scalar(x: Array) -> Array:
    return x[:, jnp.newaxis] if x.ndim == 1 else x


class Scaler:
    """
        Scales input data to the range [-1, 1] and reverses it back to the original range.

        This is a linear scaling transformation defined by:
            forward(x) = 2 * (x - x_min) / (x_max - x_min) - 1
            reverse(y) = x_min + (y + 1) / 2 * (x_max - x_min)

        Attributes:
            forward (Callable[[Array], Array]): Function to scale input to [-1, 1].
            reverse (Callable[[Array], Array]): Function to reverse scaling back to original range.

        Args:
            x_min (Scalar | float, optional): Minimum value of the input range. Defaults to 0.
            x_max (Scalar | float, optional): Maximum value of the input range. Defaults to 1.
    """
    forward: Callable[[Array], Array] 
    reverse: Callable[[Array], Array]
    def __init__(self, x_min: Scalar | float = 0., x_max: Scalar | float = 1.):
        self.forward = lambda x: 2. * (x - x_min) / (x_max - x_min) - 1.
        self.reverse = lambda y: x_min + (y + 1.) / 2. * (x_max - x_min)


class Normer:
    forward: Callable[[Array], Array] 
    reverse: Callable[[Array], Array]
    """
        Normalizes input data to have zero mean and unit variance, and reverses the transformation.

        This is a standard z-score normalization defined by:
            forward(x) = (x - x_mean) / x_std
            reverse(y) = y * x_std + x_mean

        Attributes:
            forward (Callable[[Array], Array]): Function to normalize input.
            reverse (Callable[[Array], Array]): Function to denormalize back to original scale.

        Args:
            x_mean (Scalar | float, optional): Mean of the original data. Defaults to 0.
            x_std (Scalar | float, optional): Standard deviation of the original data. Defaults to 1.
    """
    def __init__(self, x_mean: Scalar | float = 0., x_std: Scalar | float = 1.):
        self.forward = lambda x: (x - x_mean) / x_std
        self.reverse = lambda y: y * x_std + x_mean


class Identity:
    """
        Identity transformation: does nothing to the input.

        Attributes:
            forward (Callable[[Array], Array]): Identity function, returns input unchanged.
            reverse (Callable[[Array], Array]): Identity function, returns input unchanged.
    """
    forward: Callable[[Array], Array] 
    reverse: Callable[[Array], Array]
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
    """
        A simple in-memory data loader that yields randomly permuted batches of data
        from arrays stored in memory. Optionally applies a preprocessing function
        to the input data.

        Attributes:
            X (Array): Input data array of shape (N, ...).
            Q (Optional[Array]): Optional conditioning data array.
            A (Optional[Array]): Optional conditioning data array.
            process_fn (Scaler | Normer | Identity): Optional preprocessing function to apply to X.
            key (PRNGKeyArray): JAX PRNG key for shuffling the data.

        Args:
            X (np.ndarray | Array): Input dataset of shape (N, ...).
            Q (Optional[np.ndarray | Array], optional): Optional auxiliary data. Defaults to None.
            A (Optional[np.ndarray | Array], optional): Optional additional data. Defaults to None.
            process_fn (Optional[Scaler | Normer | Identity], optional): A preprocessing object with
                a `.forward()` method to apply to each batch of `X`. Defaults to Identity().
            key (PRNGKeyArray): JAX PRNG key used for random shuffling.
    """
    def __init__(
        self, 
        X: np.ndarray | Array, 
        Q: Optional[np.ndarray | Array] = None, 
        A: Optional[np.ndarray | Array] = None, 
        *, 
        process_fn: Optional[Scaler | Normer | Identity] = None,
        key: PRNGKeyArray
    ):
        self.X = jnp.asarray(X)
        self.Q = jnp.asarray(Q) if exists(Q) else Q
        self.A = jnp.asarray(A) if exists(A) else A
        self.process_fn = default(process_fn, Identity()) 
        self.key = key

    def loop(
        self, batch_size: int
    ) -> Generator[Tuple[Array, Optional[Array], Optional[Array]], None, None]:
        """
            Infinite generator yielding batches of data randomly sampled without replacement
            from the in-memory dataset.

            Args:
                batch_size (int): Number of samples per batch.

            Yields:
                Tuple[Array, Optional[Array], Optional[Array]]: A tuple containing:
                    - A batch of input data (after applying `process_fn`, if provided),
                    - A batch of optional conditioning data Q (if present),
                    - A batch of optional conditioning data A (if present).

            Raises:
                ValueError: If the batch size is larger than the dataset size.
        """
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
                    self.process_fn.forward(self.X[batch_perm]) if exists(self.process_fn) else self.X[batch_perm], 
                    self.Q[batch_perm] if self.Q is not None else None, 
                    self.A[batch_perm] if self.A is not None else None 
                )
                start = end
                end = start + batch_size


class TorchDataLoader(_AbstractDataLoader):
    """
        A data loader that wraps a PyTorch Dataset and yields batches of data as JAX arrays.
        Optionally applies a preprocessing function to the input data and handles optional
        context and parameter targets.

        Attributes:
            dataset (torch.utils.data.Dataset): A PyTorch-compatible dataset, 
                returning tuples (x,) or (x, q), (x, a), or (x, q, a).
            context_shape (Optional[Sequence[int]]): Shape of context variable Q, if present.
            parameter_dim (Optional[int]): Dimensionality of parameter variable A, if present.
            seed (int): Seed used for deterministic shuffling via `torch.Generator`.
            process_fn (Scaler | Normer | Identity): Optional preprocessing object with a `.forward()` method.
            num_workers (Optional[int]): Number of worker processes for data loading. Overrides `num_workers` in `loop` if set.

        Args:
            dataset (torch.utils.data.Dataset): Dataset object that returns tuples (x,), (x, q), (x, a), or (x, q, a).
            data_shape (Sequence[int]): Shape of the input data `x`.
            context_shape (Optional[Sequence[int]]): Shape of the optional context data `q`. Use `None` if not applicable.
            parameter_dim (Optional[int]): Dimensionality of the optional parameter data `a`. Use `None` if not applicable.
            process_fn (Optional[Scaler | Normer | Identity], optional): Preprocessing function to apply to `x`. Defaults to Identity().
            num_workers (Optional[int], optional): Number of workers for data loading. Defaults to None.
            key (PRNGKeyArray): JAX PRNG key used to generate a reproducible torch seed.
    """
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        data_shape: Sequence[int],
        context_shape: Optional[Sequence[int]],
        parameter_dim: Optional[int],
        *, 
        process_fn: Optional[Scaler | Normer | Identity] = None,
        num_workers: Optional[int] = None, 
        key: PRNGKeyArray
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
        """
            Infinite generator that yields batches from the dataset using PyTorch's DataLoader.
            Shuffles the data each epoch and converts outputs to JAX arrays.

            Args:
                batch_size (int): Number of samples per batch.
                num_workers (int, optional): Number of worker processes to use for loading. Ignored if `self.num_workers` is set.

            Yields:
                Tuple[Array, Optional[Array], Optional[Array]]:
                    A tuple containing:
                    - A batch of input data (processed if `process_fn` is set),
                    - A batch of optional context data `q` (if `context_shape` is set),
                    - A batch of optional parameter data `a` (if `parameter_dim` is set).
        """
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
                    self.process_fn.forward(x) if exists(self.process_fn) else x,
                    expand_if_scalar(jnp.asarray(q)) if self.context_shape else None,
                    expand_if_scalar(jnp.asarray(a)) if self.parameter_dim else None
                )


class TensorDataset(torch.utils.data.Dataset):
    """
        A PyTorch-style dataset that holds in-memory tensors for inputs (x),
        optional context (q), and optional parameters (a), with optional per-variable transforms.

        This dataset supports variable combinations: (x,), (x, q), (x, a), or (x, q, a).

        Attributes:
            names (List[str]): The names of the data slots: "x", "q", and "a".
            data (Dict[str, Optional[torch.Tensor]]): Dictionary mapping names to tensors, or None.
            transforms (Dict[str, Optional[Callable]]): Optional transforms for each tensor.

        Args:
            tensors (Tuple[Union[np.ndarray, torch.Tensor, None], ...]):
                A tuple of up to three tensors corresponding to (x, q, a).
                Each can be a NumPy array, Torch tensor, or None.
            x_transform (Callable, optional): Optional transform to apply to x at retrieval time.
            q_transform (Callable, optional): Optional transform to apply to q at retrieval time.
            a_transform (Callable, optional): Optional transform to apply to a at retrieval time.

        Raises:
            AssertionError: If any non-None tensor has a mismatched first dimension.
    """
    def __init__(self, tensors, x_transform=None, q_transform=None, a_transform=None):
        self.names = ["x", "q", "a"]
        self.data = {
            name: torch.as_tensor(np.copy(maybe_convert(t))) if exists(t) else None
            for name, t in zip(self.names, tensors)
        }

        self.transforms = {
            name: transform if exists(transform) else None
            for name, transform in zip(self.names, [x_transform, q_transform, a_transform])
        }

        # Sanity check: all non-None tensors must have same first dimension
        lengths = [v.shape[0] for v in self.data.values() if v is not None]
        assert len(set(lengths)) == 1, "All input tensors must have the same length."

    def __getitem__(self, index):
        output = []
        for key in self.names:
            tensor = self.data.get(key)
            if exists(tensor):
                val = tensor[index]
                if self.transforms[key]:
                    val = self.transforms[key](val)
                output.append(val)
        return tuple(output)

    def __len__(self):
        return next(v.shape[0] for v in self.data.values() if v is not None)


@typecheck
@dataclass
class ScalerDataset:
    """
        Dataset container that wraps dataloaders and preprocessing utilities.

        Parameters
        ----------
        name : str
            A name for the dataset.
        
        train_dataloader : TorchDataLoader | InMemoryDataLoader
            Dataloader instance for training data. Can be a streaming or in-memory loader.
        
        valid_dataloader : TorchDataLoader | InMemoryDataLoader
            Dataloader instance for validation data. Same format as `train_dataloader`.
        
        data_shape : Tuple[int, ...]
            Shape of each data sample (excluding batch dimension), e.g., input shape of training data.
        
        context_shape : Tuple[int, ...] or None
            Shape of each conditioning variable sample (if any). 
        
        parameter_dim : int or None
            Dimensionality of the conditioning parameter space.
        
        process_fn : Scaler | Normer | Identity | None
            Optional preprocessing function applied to the input data (e.g., normalization or standardization).
        
        label_fn : Callable
            Function that generates a batch of context/parameter pairs (Q, A) given a random key and batch size.
            Typically used during training to retrieve target values.

        Notes
        -----
        This class is designed to interface cleanly with JAX training loops and SBI pipelines.
    """
    name: str
    train_dataloader: TorchDataLoader | InMemoryDataLoader
    valid_dataloader: TorchDataLoader | InMemoryDataLoader
    data_shape: Tuple[int, ...]
    context_shape: Optional[Tuple[int, ...]]
    parameter_dim: Optional[int]
    process_fn: Optional[Scaler | Normer | Identity] 
    label_fn: Callable[
        [PRNGKeyArray, int], 
        Tuple[
            Optional[Float[Array, "n ..."]], 
            Optional[Float[Array, "n _"]]
        ]
    ]


@typecheck
def dataset_from_tensors(
    X: Float[Array, "n ..."],
    Q: Optional[Float[Array, "n ..."]] = None,
    A: Optional[Float[Array, "n _"]] = None,
    *,
    key: PRNGKeyArray,
    process_fn: Optional[Scaler | Normer | Identity] = None,
    split: float = 0.8,
    in_memory: bool = False,
    name: Optional[str] = None
):
    """
        Creates a ScalerDataset object from in-memory tensors with optional conditioning and parameter targets.

        Splits the data into training and validation subsets, applies optional preprocessing, 
        and returns a dataset with a label function for choosing conditioning variables for sampling.

        Parameters
        ----------
        X : Float[Array, "n ..."]
            Input data array of shape (n, ...) where n is the number of data points.
        Q : Optional[Float[Array, "n ..."]]
            Conditioning variables (e.g. features or context), optional. Must align with `X` if provided.
        A : Optional[Float[Array, "n _"]]
            Parameters or labels (e.g. target variables) associated with each sample, optional.
        key : PRNGKeyArray
            PRNG key for random operations such as sampling indices or splitting.
        process_fn : Optional[Scaler | Normer | Identity], default=None
            Optional processing function to normalize or transform the inputs.
        split : float, default=0.8
            Fraction of data to use for training. The remainder is used for validation.
        in_memory : bool, default=False
            Whether to load the entire dataset into memory or use an iterable-style loader.
        name : Optional[str], default=None
            Name identifier for the dataset.

        Returns
        -------
        ScalerDataset
            A dataset wrapper object containing training/validation dataloaders, 
            shape metadata, optional preprocessing, and a label generation function.

        Notes
        -----
        - If `Q` and `A` are both provided, `label_fn` will randomly sample a subset of them.
        - If `in_memory=True`, `InMemoryDataLoader` will be used instead of `TorchDataLoader`.
    """
    key_train, key_valid = jr.split(key)

    process_fn = default(process_fn, Identity())

    n_train = int(split * X.shape[0])

    data_shape = X.shape[1:]  # Exclude the first dimension (n)
    context_shape = Q.shape[1:] if exists(Q) else None
    parameter_dim = A.shape[1] if exists(A) else None

    train_set = (
        X[:n_train], 
        Q[:n_train] if exists(Q) else None, 
        A[:n_train] if exists(A) else None,
    )
    valid_set = (
        X[n_train:], 
        Q[n_train:] if exists(Q) else None, 
        A[n_train:] if exists(A) else None,
    )

    def label_fn(
        key: PRNGKeyArray, 
        n: int,
        Q: Optional[Float[Array, "n ..."]],
        A: Optional[Float[Array, "n _"]]
    ) -> Tuple[
        Optional[Float[Array, "n ..."]], 
        Optional[Float[Array, "n _"]]
    ]:
        if exists(Q):
            length = len(Q)
        elif exists(A):
            length = len(A)
        else:
            return None, None
        ix = jr.choice(key, jnp.arange(length), (n,))
        _Q = Q[ix] if exists(Q) else None
        _A = A[ix] if exists(A) else None
        return _Q, _A

    if in_memory:
        train_dataloader = InMemoryDataLoader(
            *train_set, process_fn=process_fn, key=key_train
        )
        valid_dataloader = InMemoryDataLoader(
            *valid_set, process_fn=process_fn, key=key_valid
        )
    else:
        train_dataloader = TorchDataLoader(
            TensorDataset(train_set), 
            data_shape=data_shape,
            context_shape=context_shape, 
            parameter_dim=parameter_dim,
            process_fn=process_fn, 
            key=key_train
        )
        valid_dataloader = TorchDataLoader(
            TensorDataset(valid_set), 
            data_shape=data_shape,
            context_shape=context_shape, 
            parameter_dim=parameter_dim,
            process_fn=process_fn, 
            key=key_valid
        )

    return ScalerDataset(
        name=default(name, "dataset"),
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        parameter_dim=parameter_dim,
        label_fn=partial(label_fn, Q=Q, A=A),
        process_fn=process_fn
    )