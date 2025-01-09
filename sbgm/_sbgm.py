from __future__ import annotations

from typing import Tuple, Optional, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import diffrax as dfx
from jaxtyping import PRNGKeyArray, Array 

from .sde import SDE
from ._sample import single_ode_sample_fn, single_eu_sample_fn
from ._ode import log_likelihood

"""
    Net needs to be precondition-wrapped before being passed to SBGM.
    This is because the net is passed to the sampling/likelihood methods
"""


def default(v, d):
    return v if v is not None else d


class SBGM(eqx.Module):
    net: eqx.Module
    sde: SDE
    x_shape: Sequence[int]
    solver: dfx.AbstractSolver
    precondition: bool

    def __init__(
        self, 
        net: eqx.Module, 
        sde: SDE, 
        x_shape: Sequence[int],
        solver: dfx.AbstractSolver = dfx.Heun(),
        *,
        precondition: bool = False
    ):
        self.net = PreconditionedNet(net, sde) if precondition else net
        self.sde = sde
        self.x_shape = x_shape
        self.solver = solver
        self.precondition = precondition

    def score(
        self, 
        t: float | Array,
        x: Array,
        q: Array,
        a: Array,
        key: PRNGKeyArray,
    ) -> Array:
        t = jnp.atleast_1d(t)
        return self.net(t, x, q, a, key=key)

    @eqx.filter_jit
    def likelihood(
        self, 
        x: Array,
        q: Array,
        a: Array,
        key: PRNGKeyArray,
        *,
        solver: Optional[dfx.AbstractSolver] = None,
        exact_logp: bool = False,
        n_eps: int = 1,
        return_latents: bool = False
    ) -> Tuple[Array, Array] | Array:
        solver = default(solver, self.solver)
        L, z = log_likelihood(
            key, 
            self.net, 
            self.sde, 
            self.x_shape, 
            x, 
            q, 
            a, 
            exact_logp=exact_logp, 
            n_eps=n_eps, 
            solver=solver
        )
        return (L, z) if return_latents else L

    @eqx.filter_jit
    def sample_ode(
        self,
        q: Array, 
        a: Array, 
        key: PRNGKeyArray, 
        *,
        solver: Optional[dfx.AbstractSolver] = None
    ) -> Array:
        solver = default(solver, self.solver)
        return single_ode_sample_fn(
            self.net, 
            self.sde, 
            self.x_shape, 
            key, 
            q, 
            a, 
            solver=solver
        )

    @eqx.filter_jit
    def sample_eu(
        self,
        q: Array, 
        a: Array, 
        key: PRNGKeyArray, 
        *,
        T_sample: int = 1000
    ) -> Array:
        return single_eu_sample_fn(
            self.net, 
            self.sde, 
            self.x_shape, 
            key, 
            q, 
            a, 
            T_sample=T_sample
        )


class PreconditionedNet(eqx.Module):
    net: eqx.Module
    sde: SDE

    def __init__(self, net: eqx.Module, sde: SDE):
        self.net = net
        self.sde = sde

    @property
    def name(self):
        return f"{self.__class__.__name__}{self.net.__name__}"

    def expectation(
        self,
        t: float | Array,
        x: Array,
        q: Array,
        a: Array,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        _, sigma_t = self.sde.marginal_prob(x, t)
        log_sigma_t = jnp.log(sigma_t)
        x_ = x / jnp.sqrt(sigma_t ** 2. + 1.)
        h = self.net(log_sigma_t, x_, q, a, key=key)
        d = x / (sigma_t ** 2. + 1.) + sigma_t / jnp.sqrt(sigma_t ** 2. + 1.) * h
        return d

    def score(
        self,
        t: float | Array,
        x: Array,
        q: Array,
        a: Array,
        key: Optional[PRNGKeyArray] = None,    
    ) -> Array:
        d = self.expectation(t, x, q, a, key)
        return (x - d) / jnp.maximum(t, 1e-8) # This also needs scaling/shifting like d above?

    def __call__(
        self,
        t: float | Array,
        x: Array,
        q: Array,
        a: Array,
        key: Optional[PRNGKeyArray] = None,    
    ) -> Array:
        return self.score(t, x, q, a, key)