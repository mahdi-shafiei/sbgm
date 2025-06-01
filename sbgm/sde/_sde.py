from typing import Sequence, Tuple, Self, Callable, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array, Float, Scalar

Time = Scalar | float
TimeFn = Callable[[Time], Scalar]


def default_weight_fn(t, *, beta_integral=None, sigma_fn=None): 
    assert not ((beta_integral is not None) and (sigma_fn is not None))
    return 1. - jnp.exp(-beta_integral(t))


class SDE(eqx.Module):
    """
        Abstract base class for Stochastic Differential Equations (SDEs) used in 
        score-based generative modeling and related diffusion models.

        This class defines the required interface and provides base functionality for 
        forward and reverse-time SDEs, prior sampling, and log-probability computation. 
        The user should subclass `SDE` and implement the following methods:
            - `sde()`: returns drift and diffusion coefficients
            - `marginal_prob()`: returns the parameters of the marginal distribution at time t
            - `prior_sample()`: returns samples from the terminal distribution p_T(x)
            - `prior_log_prob()`: returns log-probabilities under p_T(x)
            - `weight()`: returns weighting for loss functions

        Attributes:
            dt (float): Time discretization step size.
            t0 (float): Start time of the diffusion process.
            t1 (float): End time of the diffusion process.
    """
    dt: float
    t0: float
    t1: float

    def __init__(self, dt: float = 0.01, t0: float = 0., t1: float = 1.):
        """
            Initialize the base SDE with time parameters.

            Args:
                dt (float): Time step for the SDE solver.
                t0 (float): Initial time of the process.
                t1 (float): Terminal time of the process.
        """
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.dt = dt

    def sde(self, x: Array, t: Time) -> Tuple[Array, Array]:
        """
            Return the drift and diffusion coefficients f(x, t), g(t) of the SDE.

            Must be implemented by subclass.
        """
        ...

    def marginal_prob(self, x: Array, t: Time) -> Tuple[Array, Array]:
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$. """
        ...

    def prior_sample(self, key: Key, shape: Sequence[int]) -> Array:
        """ 
            Generate one sample from the prior distribution, $p_T(x)$. 
        """
        ...

    def weight(self, t: Time, likelihood_weight: bool = False) -> Array:
        """
            Return the training loss weight at time t.

            Args:
                t (float or Array): Time value(s).
                likelihood_weight (bool): Whether to use likelihood weighting (optional).

            Returns:
                Array: Scalar or array of weights.
        """
        ...

    def prior_log_prob(self, z: Array) -> Array:
        """
            Compute log-density of the prior distribution.

            Useful for computing the log-likelihood via probability flow ODE.

            Args:
                z: latent code

            Returns:
                log probability density
        """
        ...

    def reverse(self, score_fn: eqx.Module, probability_flow: bool = False) -> Self:
        """
            Create the reverse-time SDE/ODE.

            Args:
                score_fn: A time-dependent score-based model that takes x and t and returns the score.
                probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.

            Returns:
                SDE: A subclass implementing the reverse-time SDE.
        """

        sde_fn = self.sde

        if hasattr(self, "beta_integral_fn"):
            _sde_fn = self.beta_integral_fn 
        if hasattr(self, "sigma_fn"):
            _sde_fn = self.sigma_fn

        _dt = self.dt
        _t0 = self.t0
        _t1 = self.t1

        # Build the class for the reverse-time SDE.
        class RSDE(self.__class__, SDE):
            probability_flow: bool

            def __init__(self):
                self.probability_flow = probability_flow
                super().__init__(_sde_fn, dt=_dt, t0=_t0, t1=_t1)

            def sde(
                self, 
                x: Float[Array, "..."], 
                t: Time, 
                q: Optional[Float[Array, "..."]] = None,
                a: Optional[Float[Array, "..."]] = None
            ) -> Tuple[Float[Array, "..."], Scalar]:
                """ 
                    Create the drift and diffusion functions for the reverse SDE/ODE. 
                    - forward time SDE:
                        dx = f(x, t) * dt + g(t) * dw
                    - reverse time SDE:
                        dx = [f(x, t) - g^2(t) * score(x, t)] * dt + g(t) * dw
                    - ode of SDE:
                        dx = [f(x, t) - 0.5 * g^2(t) * score(x, t)] * dt (ODE => No dw)
                """
                t = jnp.asarray(t)
                c = 0.5 if self.probability_flow else 1.
                drift, diffusion = sde_fn(x, t)
                score = score_fn(t, x, q, a)
                # Drift coefficient of reverse SDE and probability flow only different by a factor
                drift = drift - jnp.square(diffusion) * score * c
                # Set the diffusion function to zero for ODEs (dw=0)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

        return RSDE()


def _get_log_prob_fn(scale: float = 1.) -> Callable:
    def _log_prob_fn(z: Array) -> Array:
        return jax.scipy.stats.norm.logpdf(z, loc=0., scale=scale).sum()
    return _log_prob_fn