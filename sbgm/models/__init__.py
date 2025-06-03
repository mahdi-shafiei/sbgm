from typing import Sequence, Optional
import equinox as eqx
from jaxtyping import Key
import numpy as np
from ml_collections import ConfigDict

from ._mixer import Mixer2d
from ._mlp import ResidualNetwork 
from ._unet import UNet
from ._dit import DiT


def get_model(
    model_key: Key, 
    model_type: str, 
    config: ConfigDict, 
    data_shape: Sequence[int], 
    context_shape: Optional[Sequence[int]] = None, 
    parameter_dim: Optional[int] = None
) -> eqx.Module:
    """
        Get the model based on the specified type and configuration.

        Args:
            model_key: JAX random key for model initialization.
            model_type: Type of the model to create (e.g., "Mixer", "UNet", "mlp", "DiT").
            config: Configuration dictionary containing model parameters.
            data_shape: Shape of the input data (e.g. image dimensions, channels first).
            context_shape: Shape of the context map, if applicable.
            parameter_dim: Dimension of the additional conditioning.
        Returns:
            An initialized instance of the specified model type.

        Raises:
            ValueError: If the model type is not recognized.
    """

    # Grab channel assuming 'q' is a map like x
    if context_shape is not None:
        context_channels, *_ = context_shape.shape 
    else:
        context_channels = None

    if model_type not in ["Mixer", "UNet", "mlp", "DiT"]:
        raise ValueError(
            f"Model type {model_type} is not recognized. "
            "Choose from 'Mixer', 'UNet', 'mlp', or 'DiT'."
        )

    if model_type == "Mixer":
        model = Mixer2d(
            data_shape,
            patch_size=config.model.patch_size,
            hidden_size=config.model.hidden_size,
            mix_patch_size=config.model.mix_patch_size,
            mix_hidden_size=config.model.mix_hidden_size,
            num_blocks=config.model.num_blocks,
            t1=config.sde.t1,
            final_activation=config.model.final_activation,
            q_dim=context_channels,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "UNet":
        n_channels, *_ = data_shape
        model = UNet(
            dim=config.model.hidden_size,
            channels=n_channels,
            dim_mults=config.model.dim_mults,
            attn_heads=config.model.heads,
            attn_dim_head=config.model.dim_head,
            dropout=config.model.dropout_rate,
            learned_sinusoidal_cond=True,
            random_fourier_features=True,
            q_channels=context_channels,
            a_dim=parameter_dim,
            key=model_key
        )
    if model_type == "mlp":
        model = ResidualNetwork(
            in_size=np.prod(data_shape),
            width_size=config.model.width_size,
            depth=config.model.depth,
            activation=config.model.activation,
            dropout_p=config.model.dropout_p,
            q_dim=context_channels,
            a_dim=parameter_dim,
            t1=config.sde.t1,
            key=model_key
        )
    if model_type == "DiT":
        model = DiT(
            img_size=config.img_size,
            channels=config.n_channels,
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            depth=config.depth,
            n_heads=config.n_heads,
            q_dim=context_channels, # Number of channels in conditioning map
            a_dim=parameter_dim,    # Number of parameters in power spectrum model
            key=model_key
        )

    return model