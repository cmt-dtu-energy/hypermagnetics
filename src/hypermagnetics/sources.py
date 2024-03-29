from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics import plots


def _F(x, y, z):
    r = jnp.array([x, y, z])
    d = jnp.linalg.norm(r)
    terms = [jnp.arctan(y * z / (x * d)), -jnp.log(z + d), -jnp.log(y + d)]
    return jnp.array(terms) @ r


def _faces(x, y, z, a, b, c):
    return (
        _F(x + a, y + b, z + c)
        - _F(x + a, y + b, z - c)
        - _F(x + a, y - b, z + c)
        + _F(x + a, y - b, z - c)
        - _F(x - a, y + b, z + c)
        + _F(x - a, y + b, z - c)
        + _F(x - a, y - b, z + c)
        - _F(x - a, y - b, z - c)
    )


@jax.jit
def _prism(m: jax.Array, r0: jax.Array, r: jax.Array, size: jax.Array):
    x, y, z = r - r0
    a, b, c = size

    fx = _faces(x, y, z, a, b, c)
    fy = _faces(y, z, x, b, c, a)
    fz = _faces(z, x, y, c, a, b)

    return -(1 / 4 * jnp.pi) * m @ jnp.array([fx, fy, fz])


@jax.jit
def _sphere(m: jax.Array, r0: jax.Array, r: jax.Array, size=1.0, dim=2):
    """Finite sphere potential in two or three dimensions."""
    d = r - r0
    d_norm = jnp.linalg.norm(d)
    m_dot_r = jnp.dot(m, d)
    close_to_source = d_norm <= size
    interior = m_dot_r / size / (2 * (dim - 1) * jnp.pi * size ** (dim - 1))
    exterior = m_dot_r / d_norm / (2 * (dim - 1) * jnp.pi * d_norm ** (dim - 1))
    return jnp.where(close_to_source, interior, exterior)


def _potential(sources, r, shape):
    """Dispatcher for source potential calculation."""
    dim = sources.shape[-1] // 2
    m, r0 = jnp.split(sources, 2, axis=-1)
    if shape == "sphere":
        size = 1.0
        return _sphere(m, r0, r, size, dim)
    elif shape == "prism":
        size = jnp.array([1.0, 1.0, 1.0])
        return _prism(m, r0, r, size)
    else:
        raise ValueError(f"Unknown source shape: {shape}")


def _field(sources, r, shape):
    """Finite sphere field in two or three dimensions."""
    _potential_with_shape = partial(_potential, shape=shape)
    return -jax.grad(_potential_with_shape, argnums=1)(sources, r)


def _total(fun, sources, r, shape):
    """Aggregate the field or potential of all sources."""
    fun_with_shape = partial(fun, shape=shape)
    points = jax.vmap(fun_with_shape, in_axes=(None, 0))
    batch = jax.vmap(points, in_axes=(0, None))
    components = jax.vmap(batch, in_axes=(1, None))(sources, r)
    return jnp.sum(components, axis=0)


def configure(n_samples, n_sources, dim=2, lim=3, res=32, seed=0, shape="sphere"):
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
    r0 = (lim / 3) * jr.normal(shape=(n_samples, n_sources, dim), key=r0key)
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, dim))

    range = jnp.linspace(-lim, lim, res)
    grids = jnp.meshgrid(*[range] * dim)
    grid = jnp.concatenate([g.ravel()[:, None] for g in grids], axis=-1)
    r = sample_grid(rkey, lim, res, dim)
    sources = jnp.concatenate([m, r0], axis=-1)
    return {
        "sources": sources,
        "r": r,
        "potential": _total(_potential, sources, r, shape),
        "field": _total(_field, sources, r, shape),
        "grid": grid,
        "potential_grid": _total(_potential, sources, grid, shape),
        "field_grid": _total(_field, sources, grid, shape),
    }


def sample_grid(key, lim, res, dim=2, n=None):
    if n is None:
        n = res**dim
    return jr.uniform(minval=-lim, maxval=lim, shape=(n, dim), key=key)


if __name__ == "__main__":
    # Two dimensions
    config = {
        "shape": "sphere",
        "n_samples": 10,
        "n_sources": 2,
        "seed": 40,
        "lim": 3,
        "res": 128,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)
    plots(train_data, model=None, idx=0)

    # Three dimensions
    config = {
        "shape": "sphere",
        "n_samples": 1,
        "n_sources": 2,
        "seed": 40,
        "lim": 3,
        "res": 16,
        "dim": 3,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)

    # Prism
    config = {
        "shape": "prism",
        "n_samples": 1,
        "n_sources": 1,
        "seed": 40,
        "lim": 3,
        "res": 16,
        "dim": 3,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)
