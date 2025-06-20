import jax
import jax.numpy as jnp
import jax.random as jr

from sbgm.sde import VESDE, VPSDE, SubVPSDE


def test_sdes():

    key = jr.key(0)

    data_dim = 2

    x0 = jnp.ones((data_dim,))
    eps = jr.normal(key, (data_dim,))
    t = jnp.array(0.5)

    def diffuse(sde, x, t, eps):
        mu, std = sde.marginal_prob(x, t)
        return mu + std * eps

    sde = VESDE(sigma_fn=lambda t: t) 

    xt = diffuse(sde, x0, t, eps)

    f, g = sde.sde(x0, t) 

    print(f.shape, g.shape, xt.shape)

    assert f.shape == (data_dim,)
    assert g.shape == ()
    assert xt.shape == x0.shape
    assert jnp.all(jnp.isfinite(xt))
    assert jnp.all(jnp.isfinite(f))
    assert jnp.all(jnp.isfinite(g))

    sde = VPSDE(beta_integral_fn=lambda t: t) 

    xt = diffuse(sde, x0, t, eps)

    f, g = sde.sde(x0, t) 

    assert f.shape == (data_dim,)
    assert g.shape == ()
    assert xt.shape == x0.shape
    assert jnp.all(jnp.isfinite(xt))
    assert jnp.all(jnp.isfinite(f))
    assert jnp.all(jnp.isfinite(g))

    sde = SubVPSDE(beta_integral_fn=lambda t: t) 

    xt = diffuse(sde, x0, t, eps)

    f, g = sde.sde(x0, t) 

    assert f.shape == (data_dim,)
    assert g.shape == ()
    assert xt.shape == x0.shape
    assert jnp.all(jnp.isfinite(xt))
    assert jnp.all(jnp.isfinite(f))
    assert jnp.all(jnp.isfinite(g))