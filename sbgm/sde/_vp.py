from typing import Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array, Float, Scalar, jaxtyped
from beartype import beartype as typechecker

from ._sde import SDE, _get_log_prob_fn, Time, TimeFn


def get_beta_fn(beta_integral_fn: TimeFn | eqx.Module) -> TimeFn:
    """ Obtain beta function from a beta integral. """
    def _beta_fn(t: Time) -> Scalar:
        _, beta = jax.jvp(
            beta_integral_fn, 
            primals=(t,), 
            tangents=(jnp.ones_like(t),),
            has_aux=False
        )
        return beta
    return _beta_fn


class VPSDE(SDE):
    beta_integral_fn: TimeFn | eqx.Module 
    beta_fn: TimeFn 
    weight_fn: TimeFn

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, 
        beta_integral_fn: TimeFn, 
        weight_fn: Optional[TimeFn] = None, 
        dt: float = 0.1, 
        t0: float = 0., 
        t1: float = 1.
    ):
        """
            Construct a Variance Preserving SDE.
        """
        super().__init__(dt=dt, t0=t0, t1=t1)
        self.beta_integral_fn = beta_integral_fn 
        self.beta_fn = get_beta_fn(beta_integral_fn)
        self.weight_fn = weight_fn

    @jaxtyped(typechecker=typechecker)
    def sde(self, x: Float[Array, "..."], t: Time) -> Tuple[Float[Array, "..."], Scalar]:
        """ 
            dx = f(x, t) * dt + g(t) * dw 
            dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
        """
        beta_t = self.beta_fn(t)
        drift = -0.5 * beta_t * x 
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    @jaxtyped(typechecker=typechecker)
    def marginal_prob(self, x: Float[Array, "..."], t: Time) -> Tuple[Float[Array, "..."], Scalar]:
        """ 
            VP SDE p_t(x(t)|x(0)) is
                x(t) ~ G[x(t)|mu(x(0), t), sigma^2(t)] 
            where
                mu(x(0), t) = x(0) * exp(-0.5 * int[beta(s)])
                sigma^2(t) = I * (1 - exp(-int[beta(s)]))
        """
        beta_integral = self.beta_integral_fn(t)
        mean = jnp.exp(-0.5 * beta_integral) * x 
        std = jnp.sqrt(-jnp.expm1(-beta_integral)) 
        return mean, std

    @jaxtyped(typechecker=typechecker)
    def weight(self, t: Time, likelihood_weight: bool = False) -> Scalar:
        # likelihood weighting: above Eq 8 https://arxiv.org/pdf/2101.09258.pdf
        if self.weight_fn is not None and not likelihood_weight:
            weight = self.weight_fn(t)
        else:
            if likelihood_weight:
                weight = self.beta_fn(t) # beta(t)
            else:
                weight = -jnp.expm1(-self.beta_integral_fn(t))
        return weight

    def prior_sample(self, key: PRNGKeyArray, shape: Sequence[int]) -> Float[Array, "..."]:
        return jr.normal(key, shape)

    def prior_log_prob(self, z: Float[Array, "..."]) -> Scalar:
        return _get_log_prob_fn(scale=1.)(z)