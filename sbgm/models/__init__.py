from typing import Sequence, Optional
import equinox as eqx
from jaxtyping import Key
import numpy as np
import ml_collections

from ._mixer import Mixer2d
from ._mlp import ResidualNetwork 
from ._unet import UNet


def get_model(
    model_key: Key, 
    model_type: str, 
    config: ml_collections.ConfigDict, 
    data_shape: Sequence[int], 
    context_shape: Optional[Sequence[int]] = None, 
    parameter_dim: Optional[int] = None
) -> eqx.Module:
    # Grab channel assuming 'q' is a map like x
    if context_shape is not None:
        context_channels, *_ = context_shape.shape 
    else:
        context_channels = None

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
    if model_type == "CCT":
        raise NotImplementedError
    if model_type == "DiT":
        raise NotImplementedError
    return model