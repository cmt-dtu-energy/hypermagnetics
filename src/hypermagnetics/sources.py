from functools import partial
from pathlib import Path
import time

import os
import sys
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
from magtense import magstatics

from hypermagnetics import plots


def replace_inf_nan(x):
    x = jax.lax.select(jnp.isinf(x), 0.0, x)
    x = jax.lax.select(jnp.isnan(x), 0.0, x)
    return x


def _F(x, y, z):
    r = jnp.array([x, y, z])
    d = jnp.linalg.norm(r)
    terms = [jnp.arctan(y * z / (x * d)), -jnp.log(z + d), -jnp.log(y + d)]
    return jnp.array(terms) @ r


def _F2(X, Y):
    d = jnp.array([X, Y])
    return Y + jnp.linalg.norm(d)


def _faces(x, y, z, a, b, c):
    return (
        +_F(x + a, y + b, z + c)
        - _F(x + a, y + b, z - c)
        - _F(x + a, y - b, z + c)
        + _F(x + a, y - b, z - c)
        - _F(x - a, y + b, z + c)
        + _F(x - a, y + b, z - c)
        + _F(x - a, y - b, z + c)
        - _F(x - a, y - b, z - c)
    )


def _edges(x, y, a, b):
    e = replace_inf_nan(jnp.log(_F2(x - a, y + b)))
    e += replace_inf_nan(jnp.log(_F2(x + a, y - b)))
    e -= replace_inf_nan(jnp.log(_F2(x - a, y - b)))
    e -= replace_inf_nan(jnp.log(_F2(x + a, y + b)))
    return -e


@jax.jit
def _prism(m: jax.Array, r0: jax.Array, r: jax.Array, size: jax.Array):
    x, y, z = r - r0
    a, b, c = size

    fx = _faces(x, y, z, a, b, c)
    fy = _faces(y, z, x, b, c, a)
    fz = _faces(z, x, y, c, a, b)

    value = -(1 / 4 * jnp.pi) * m @ jnp.array([fx, fy, fz])
    value = jax.lax.select(jnp.isinf(value), 0.0, value)
    value = jax.lax.select(jnp.isnan(value), 0.0, value)
    return -value


@jax.jit
def _prism2(m: jax.Array, r0: jax.Array, r: jax.Array, size: jax.Array):
    x, y = r - r0
    a, b = size[:2]

    ex = _edges(x, y, a, b)
    ey = _edges(y, x, b, a)

    value = -(1 / 2 * jnp.pi) * m @ jnp.array([ex, ey])
    # value = jax.lax.select(jnp.isinf(value), 0.0, value)
    # value = jax.lax.select(jnp.isnan(value), 0.0, value)
    return value


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
    m, r0, size = jnp.split(sources, 3, axis=-1)
    dim = m.shape[-1]
    if shape == "sphere":
        size = 1.0
        return _sphere(m, r0, r, size, dim)
    elif shape == "prism":
        if dim == 2:
            phi = _prism2(m, r0, r, size)
        else:
            phi = _prism(m, r0, r, size)
        return phi
    else:
        raise ValueError(f"Unknown source shape: {shape}")


def _field(sources, r, shape):
    """Finite sphere field in two or three dimensions."""
    _potential_with_shape = partial(_potential, shape=shape)
    return -jax.grad(_potential_with_shape, argnums=1)(sources, r)


def _field_mt(sources, r, shape):
    """Finite field in two or three dimensions with MagTense."""
    mu0 = 4 * jnp.pi * 1e-7
    # Shapes: n_samples, n_sources, dim
    m, r0, size = jnp.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape

    if shape == "sphere":
        tile_type = 7
    elif shape == "prism":
        tile_type = 2
    else:
        raise ValueError(f"Unknown source shape: {shape}")

    size = size * 2
    if dim == 2:
        r0 = jnp.concatenate([r0, jnp.zeros((n_samples, n_sources, 1))], axis=-1)
        size = jnp.concatenate([size, jnp.ones((n_samples, n_sources, 1))], axis=-1)
        m = jnp.concatenate([m, jnp.zeros((n_samples, n_sources, 1))], axis=-1)
        r = jnp.concatenate([r, jnp.zeros((r.shape[0], 1))], axis=-1)

    m_norm = jnp.linalg.norm(m, axis=-1, keepdims=True)
    mag_angles = jnp.concatenate(
        [
            jnp.arccos(m[..., 2] / m_norm[..., 0]).reshape(n_samples, n_sources, 1),
            jnp.arctan2(m[..., 1], m[..., 0]).reshape(n_samples, n_sources, 1),
        ],
        axis=-1,
    )

    field = jnp.zeros((n_samples, r.shape[0], dim))
    for i in range(n_samples):
        tiles = magstatics.Tiles(
            n=n_sources,
            M_rem=m_norm[i] / mu0,
            mag_angle=mag_angles[i],
            tile_type=tile_type,
            size=size[i],
            offset=r0[i],
        )
        # it_tiles = magstatics.iterate_magnetization(tiles)
        # demag_tensor = magstatics.get_demag_tensor(it_tiles, r)
        # H_out = magstatics.get_H_field(it_tiles, r, demag_tensor)
        devnull = open("/dev/null", "w")
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        start_time = time.time()
        _, H_out = magstatics.run_simulation(tiles, r)
        duration = time.time() - start_time
        os.dup2(oldstdout_fno, 1)
        field = field.at[i].set(jnp.array(H_out[:, :dim]) * mu0)

    return field, duration


def _total(fun, sources, r, shape):
    """Aggregate the field or potential of all sources."""
    fun_with_shape = partial(fun, shape=shape)
    points = jax.vmap(fun_with_shape, in_axes=(None, 0))
    batch = jax.vmap(points, in_axes=(0, None))
    components = jax.vmap(batch, in_axes=(1, None))(sources, r)
    return jnp.sum(components, axis=0)


def configure(
    n_samples,
    n_sources,
    dim=2,
    lim=3,
    res=32,
    seed=0,
    min_size=0.12,
    max_size=0.48,
    shape="sphere",
    save_data=False,
):
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
    r0key, mkey, rkey, skey = jr.split(key, 4)
    # r0 = (lim / 3) * jr.normal(key=r0key, shape=(n_samples, n_sources, dim))
    r0 = jr.uniform(
        key=r0key,
        shape=(n_samples, n_sources, dim),
        minval=-lim + min_size,
        maxval=lim - min_size,
    )
    if dim == 3:
        r0 = r0.at[:, :, 2].set(0.0)
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, dim))
    if dim == 3:
        m = m.at[:, :, 2].set(0.0)
    size = jr.uniform(
        key=skey, shape=(n_samples, n_sources, 1), minval=min_size, maxval=max_size
    )
    size = jnp.concatenate([size, size], axis=-1)
    if dim == 3:
        size = jnp.concatenate([size, size[:, :, 0:1]], axis=-1)
        size = size.at[:, :, 2].set(1.0)

    lim_range = jnp.linspace(-lim, lim, res)
    if dim == 3:
        grids = jnp.meshgrid(lim_range, lim_range, jnp.linspace(0, 0, 1))
    else:
        grids = jnp.meshgrid(*[lim_range] * dim)
    grid = jnp.concatenate([g.ravel()[:, None] for g in grids], axis=-1)
    r = sample_grid(rkey, lim, res, r0, size, dim, masking=False)
    sources = jnp.concatenate([m, r0, size], axis=-1)

    if save_data:
        datapath = Path(__file__).parent / ".." / ".." / "data"
        datapath.mkdir(parents=True, exist_ok=True)

        db = h5py.File(datapath / f"{seed}_{n_samples}.h5", "w")
        db.create_dataset("m", shape=(n_samples, n_sources, dim), dtype="float32")
        db.create_dataset("r0", shape=(n_samples, n_sources, dim), dtype="float32")
        db.create_dataset("size", shape=(n_samples, n_sources, dim), dtype="float32")
        db.create_dataset("r", shape=(res**2, dim), dtype="float32")
        db.create_dataset("potential", shape=(n_samples, res**2), dtype="float32")
        db.create_dataset("field", shape=(n_samples, res**2, dim), dtype="float32")
        db.create_dataset("grid", shape=(res**2, dim), dtype="float32")
        db.create_dataset("potential_grid", shape=(n_samples, res**2), dtype="float32")
        db.create_dataset("field_grid", shape=(n_samples, res**2, dim), dtype="float32")

        db["r"][:] = r
        db["grid"][:] = grid

        step = 1000
        for i in range(n_samples // step):
            db["m"][i * step : (i + 1) * step] = m[i * step : (i + 1) * step]
            db["r0"][i * step : (i + 1) * step] = r0[i * step : (i + 1) * step]
            db["size"][i * step : (i + 1) * step] = size[i * step : (i + 1) * step]
            db["potential"][i * step : (i + 1) * step] = _total(
                _potential, sources[i * step : (i + 1) * step], r, shape
            )
            db["field"][i * step : (i + 1) * step] = _total(
                _field, sources[i * step : (i + 1) * step], r, shape
            )
            db["potential_grid"][i * step : (i + 1) * step] = _total(
                _potential, sources[i * step : (i + 1) * step], grid, shape
            )
            db["field_grid"][i * step : (i + 1) * step] = _total(
                _field, sources[i * step : (i + 1) * step], grid, shape
            )

        # Remove fields with nan values
        nan_idx = jnp.where(jnp.isnan(db["field"][:, :, 0]))[0]
        for i, idx in enumerate(nan_idx):
            db["m"][idx] = db["m"][n_samples - 1000 + i]
            db["r0"][idx] = db["r0"][n_samples - 1000 + i]
            db["size"][idx] = db["size"][n_samples - 1000 + i]
            db["potential"][idx] = db["potential"][n_samples - 1000 + i]
            db["field"][idx] = db["field"][n_samples - 1000 + i]
            db["potential_grid"][idx] = db["potential_grid"][n_samples - 1000 + i]
            db["field_grid"][idx] = db["field_grid"][n_samples - 1000 + i]

        db.close()
        return None

    else:
        return {
            "sources": sources,
            "r": r,
            "potential": _total(_potential, sources, r, shape),
            "field": _total(_field, sources, r, shape),
            # "field_mt": _field_mt(sources, r, shape),
            "grid": grid,
            "potential_grid": _total(_potential, sources, grid, shape),
            "field_grid": _total(_field, sources, grid, shape),
        }


def configure_eval(
    n_samples,
    n_sources,
    dim=2,
    lim=3,
    seed=0,
    min_size=0.12,
    max_size=0.48,
    shape="sphere",
):
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
    r0key, mkey, skey = jr.split(key, 3)
    r0 = jr.uniform(
        key=r0key,
        shape=(n_samples, n_sources, dim),
        minval=-lim + min_size,
        maxval=lim - min_size,
    )
    if dim == 3:
        r0 = r0.at[:, :, 2].set(0.0)
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, dim))
    if dim == 3:
        m = m.at[:, :, 2].set(0.0)
    size = jr.uniform(
        key=skey, shape=(n_samples, n_sources, 1), minval=min_size, maxval=max_size
    )
    size = jnp.concatenate([size, size], axis=-1)
    if dim == 3:
        size = jnp.concatenate([size, size[:, :, 0:1]], axis=-1)
        size = size.at[:, :, 2].set(1.0)

    sources = jnp.concatenate([m, r0, size], axis=-1)
    r = r0[0]

    return {
        "sources": sources,
        "r": r,
        "potential": _total(_potential, sources, r, shape),
        "field": _total(_field, sources, r, shape),
    }


def sample_grid(key, lim, res, r0, size, dim=2, n=None, masking=False):
    if n is None:
        n = res**2
    r = jr.uniform(minval=-lim, maxval=lim, shape=(n, dim), key=key)
    if dim == 3:
        r = r.at[:, 2].set(0.0)

    if masking:
        idx_sample = 0
        ## Remove points inside magnetic sources
        for i in range(r0.shape[1]):
            # r = _remove_sources(r, r0[:, i], size[:, i])
            mask = jnp.logical_or(
                r[:, 0] < r0[idx_sample, i, 0] - 1.1 * size[idx_sample, i, 0],
                r[:, 0] > r0[idx_sample, i, 0] + 1.1 * size[idx_sample, i, 0],
            )
            mask = jnp.logical_or(
                mask,
                jnp.logical_or(
                    r[:, 1] < r0[idx_sample, i, 1] - size[idx_sample, i, 1],
                    r[:, 1] > r0[idx_sample, i, 1] + size[idx_sample, i, 1],
                ),
            )
            r = r[mask]
    return r


def fourier_decomposition(
    n_samples, n_sources, dim=2, lim=3, res=32, seed=0, shape="sphere"
):
    # Create a 2D signal for demonstration
    x = jnp.linspace(-lim, lim, res)
    y = jnp.linspace(-lim, lim, res)
    X, Y = jnp.meshgrid(x, y)

    key = jr.PRNGKey(seed)
    r0key, mkey, rkey, skey = jr.split(key, 4)
    r0 = (lim / 3) * jr.normal(key=r0key, shape=(n_samples, n_sources, dim))
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, dim))
    size = jr.uniform(
        key=skey, shape=(n_samples, n_sources, dim), minval=1.0, maxval=1.0
    )
    sources = jnp.concatenate([m, r0, size], axis=-1)
    r = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    potential = _total(_potential, sources, r, shape)

    # Perform the 2D Fourier Transform
    F = jnp.fft.fft2(potential)

    # Shift the zero-frequency component to the center
    F_shifted = jnp.fft.fftshift(F)

    # Compute the magnitudes (absolute values) of the complex numbers
    magnitudes = jnp.abs(F_shifted)

    # Reconstructing signal
    # Z_reconstructed = jnp.fft.ifft2(potential)

    # # The reconstructed data is complex, take only the real part
    # Z_reconstructed = jnp.real(Z_reconstructed)

    # # Display the reconstructed data
    # plt.imshow(Z_reconstructed, extent=(-3, 3, -3, 3))
    # plt.colorbar()
    # plt.show()

    return magnitudes


def read_db(filename: str):
    datapath = Path("/home/spol/Documents/repos/hypermagnetics/data")
    db = h5py.File(datapath / filename, "r")
    data = {
        "sources": jnp.concatenate([db["m"][:], db["r0"][:], db["size"][:]], axis=-1),
        "r": jnp.array(db["r"][:]),
        "potential": jnp.array(db["potential"][:]),
        "field": jnp.array(db["field"][:]),
        "grid": jnp.array(db["grid"][:]),
        "potential_grid": jnp.array(db["potential_grid"][:]),
        "field_grid": jnp.array(db["field_grid"][:]),
    }
    db.close()

    return data


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
