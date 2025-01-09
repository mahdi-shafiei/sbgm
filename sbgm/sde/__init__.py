from ml_collections import ConfigDict

from ._sde import SDE, default_weight_fn
from ._ve import VESDE
from ._vp import VPSDE
from ._subvp import SubVPSDE


def get_sde(config: ConfigDict) -> SDE:
    assert config.sde in ["VE", "VP", "SubVP"]

    name = config.sde + "SDE"

    sdes = [VESDE, VPSDE, SubVPSDE]
    sde = sdes[[sde.__name__ for sde in sdes].index(name)]

    if hasattr(config, "weight_fn"):
        weight_fn = config.weight_fn
    else:
        weight_fn = default_weight_fn

    if config.sde in ["VP", "SubVP"]:
        _sde = sde(
            beta_integral_fn=config.beta_integral,
            dt=config.dt,
            t0=config.t0, 
            t1=config.t1,
            weight_fn=weight_fn
        )
    if config.sde in ["VE"]:
        _sde = sde(
            sigma_fn=config.sigma_fn,
            dt=config.dt,
            t0=config.t0, 
            t1=config.t1,
            weight_fn=weight_fn
        )
    return _sde