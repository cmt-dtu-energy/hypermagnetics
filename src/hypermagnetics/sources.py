import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics import plots


def _potential(m, r0, r):
    """Dipole potential in two dimensions."""
    core = 1.0
    d = r - r0
    d_norm = jnp.linalg.norm(d)
    m_dot_r = jnp.dot(m, d)
    close_to_source = d_norm <= core
    interior = m_dot_r / core / (2 * jnp.pi * core)
    exterior = m_dot_r / d_norm / (2 * jnp.pi * d_norm)
    return jnp.where(close_to_source, interior, exterior)


def _field(m, r0, r):
    """Dipole field in two dimensions."""
    return -jax.grad(_potential, argnums=2)(m, r0, r)


def _total(fun, m, r0, r):
    """Aggregate the field or potential of all sources."""
    points = jax.vmap(fun, in_axes=(None, None, 0))
    batch = jax.vmap(points, in_axes=(0, 0, None))
    components = jax.vmap(batch, in_axes=(1, 1, None))(m, r0, r)
    return jnp.sum(components, axis=0)


def configure(n_samples, n_sources, lim=3, res=32, seed=0):
    """
    Configures samples of sources.

    Args:
        n_samples (int): Number of samples to generate.
        n_sources (int): Number of sources in each sample.
        lim (int, optional): Domain range, in units of source radius. Defaults to 3.
        res (int, optional): Resolution of the field grid. Defaults to 32.
        key (jr.PRNGKey): Random number generator key.
    """

    key = jr.PRNGKey(seed)
    r0key, mkey, rkey = jr.split(key, 3)
    r0 = (lim / 3) * jr.normal(shape=(n_samples, n_sources, 2), key=r0key)
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, 2))

    range = jnp.linspace(-lim, lim, res)
    x, y = jnp.meshgrid(range, range)
    grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
    r = sample_grid(rkey, lim, res)
    return {
        "sources": jnp.concatenate([m, r0], axis=-1),
        "r": r,
        "potential": _total(_potential, m, r0, r),
        "field": _total(_field, m, r0, r),
        "grid": grid,
        "potential_grid": _total(_potential, m, r0, grid),
        "field_grid": _total(_field, m, r0, grid),
    }


def sample_grid(key, lim, res, n=None):
    if n is None:
        n = res * res
    return jr.uniform(minval=-lim, maxval=lim, shape=(n, 2), key=key)


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 2,
        "seed": 40,
        "lim": 3,
        "res": 128,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)
    plots(train_data, model=None, idx=0)
