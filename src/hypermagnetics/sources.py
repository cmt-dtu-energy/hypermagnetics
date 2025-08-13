from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from hypermagnetics import plots
from hypermagnetics.quadtree import random_quadtree


def replace_inf_nan(x):
    x = jax.lax.select(jnp.isinf(x), 0.0, x)
    x = jax.lax.select(jnp.isnan(x), 0.0, x)
    return x


def _F(x, y, z):
    r = jnp.array([x, y, z])
    d = jnp.linalg.norm(r)
    terms = [jnp.arctan(y * z / (x * d)), -jnp.log(z + d), -jnp.log(y + d)]
    return jnp.array(terms) @ r


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


@jax.jit
def _prism(m: jax.Array, r0: jax.Array, r: jax.Array, size: jax.Array):
    x, y, z = r - r0
    a, b, c = size

    fx = _faces(x, y, z, a, b, c)
    fy = _faces(y, z, x, b, c, a)
    fz = _faces(z, x, y, c, a, b)

    # To be consistent with dipolar potential implementation
    # Convert magnetic moment to magnetization
    M = m / (2 * a * 2 * b)
    value = M @ jnp.array([fx, fy, fz]) / (4 * jnp.pi)
    return replace_inf_nan(value)


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
    if shape == "sphere":
        return _sphere(m, r0, r, size[..., 0], dim=r0.shape[-1])
    elif shape == "prism":
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


def configure(
    n_samples: int,
    n_sources: int,
    dim: int = 2,
    lim: int = 3,
    res: int = 32,
    min_size: float = 0.12,
    max_size: float = 0.48,
    shape: str = "sphere",
    save_data: bool = False,
    line: bool = False,
    eps: float = 1e-5,
    quadtree: bool = False,
    source_val: bool = False,
    field_eval: bool = True,
    grid_eval: bool = True,
    seed: int = 0,
):
    """
    Configures samples of sources.

    Parameters:
        n_samples (int): Number of samples to generate.
        n_sources (int): Number of sources in each sample.
        dim (int): Dimension of the sources.
        lim (int): Domain range, in units of source radius.
        res (int): Resolution of the field grid.
        min_size (float): Minimum side length / radius of the sources.
        max_size (float): Maximum side length / radius of the sources.
        shape (str): Shape of the sources. Can be "sphere" or "prism".
        save_data (bool): Whether to save the generated data in a database.
        line (bool): Whether to evaluate along a line. Otherwise a uniform grid is used.
        eps (float): Small value to avoid singularities.
        quadtree (bool): Whether to use a quadtree for source placement.
        source_val (bool): Whether to evaluate in the center point of the sources.
        field_eval (bool): Whether to evaluate the field.
        grid_eval (bool): Whether to evaluate the grid.
        seed (int): Random seed for reproducibility.
    """

    key = jr.PRNGKey(seed)
    r0key, mkey, rkey, skey = jr.split(key, 4)

    m = jr.normal(key=mkey, shape=(n_samples, n_sources, 2))

    if quadtree:
        r0 = np.zeros((n_samples, n_sources, 2))
        size = np.zeros((n_samples, n_sources, 1))

        for i in range(n_samples):
            # Generate random quadtree
            cells, r0key = random_quadtree(*(-lim, -lim, lim, lim), n_sources, r0key)
            # Collect centers
            r0[i] = jnp.array([c.center() for c in cells][:n_sources])
            size[i, :, 0] = jnp.array([c.width / 2 for c in cells][:n_sources])
    else:
        r0 = jr.uniform(
            key=r0key,
            shape=(n_samples, n_sources, 2),
            minval=-lim + min_size,
            maxval=lim - min_size,
        )
        size = jr.uniform(
            key=skey, shape=(n_samples, n_sources, 1), minval=min_size, maxval=max_size
        )

    if shape == "sphere":
        if dim == 2:
            size = jnp.concatenate([size, size], axis=-1)
        elif dim == 3:
            size = jnp.concatenate([size, size, size], axis=-1)
            r0 = jnp.concatenate([r0, jnp.zeros((n_samples, n_sources, 1))], axis=-1)
            m = jnp.concatenate([m, jnp.zeros((n_samples, n_sources, 1))], axis=-1)

    elif shape == "prism":
        if dim == 2:
            size = jnp.concatenate(
                [size, size, jnp.ones((n_samples, n_sources, 1)) * 10], axis=-1
            )
            r0 = jnp.concatenate([r0, jnp.zeros((n_samples, n_sources, 1))], axis=-1)
            m = jnp.concatenate([m, jnp.zeros((n_samples, n_sources, 1))], axis=-1)
        elif dim == 3:
            size = jnp.concatenate([size, size, size], axis=-1)

    sources = jnp.concatenate([m, r0, size], axis=-1)

    if dim == 2 and shape == "sphere":
        if line:
            grids = jnp.meshgrid(
                jnp.linspace(0, lim, res),
                jnp.linspace(r0[0, 0, 1], r0[0, 0, 1], 1) + eps,
                indexing="xy",
            )
        else:
            grids = jnp.meshgrid(*[jnp.linspace(-lim, lim, res)] * dim, indexing="xy")
    else:
        dim = 3
        if line:
            grids = jnp.meshgrid(
                jnp.linspace(0, lim, res),
                jnp.linspace(r0[0, 0, 1], r0[0, 0, 1], 1) + eps,
                jnp.linspace(r0[0, 0, 2], r0[0, 0, 2], 1),
                indexing="xy",
            )
        else:
            grids = jnp.meshgrid(
                jnp.linspace(-lim, lim, res),
                jnp.linspace(-lim, lim, res),
                jnp.linspace(0, 0, 1),
                indexing="xy",
            )
    grid = jnp.concatenate([g.ravel()[:, None] for g in grids], axis=-1)

    if source_val:
        # Add a small value to r0 to avoid singularities for gradient evaluation
        r = r0 + eps
    else:
        r = jr.uniform(minval=-lim, maxval=lim, shape=(res**2, 2), key=rkey)
        if dim == 3:
            r = jnp.concatenate([r, jnp.zeros((res**2, 1))], axis=-1)

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
        # Potential calculation
        if source_val:
            msp = np.zeros((n_samples, r.shape[1]))
            field = np.zeros((n_samples, r.shape[1], dim))
            for i, r_sample in enumerate(r):
                # Memory constraints above 10k sources
                # N_sources ** 2 has to be below 100M
                if sources[i].shape[0] > 1e4:
                    n_sources_per_sim = int(1e8 // sources[i].shape[0])
                    for j in range(0, sources[i].shape[0], n_sources_per_sim):
                        msp[i : i + 1] += _total(
                            _potential,
                            sources[
                                i : i + 1,
                                j : j + n_sources_per_sim,
                            ],
                            r_sample,
                            shape,
                        )
                        if field_eval:
                            field[i : i + 1] += _total(
                                _field,
                                sources[
                                    i : i + 1,
                                    j : j + n_sources_per_sim,
                                ],
                                r_sample,
                                shape,
                            )
                else:
                    msp[i] = _total(_potential, sources[i : i + 1], r_sample, shape)[0]
                    if field_eval:
                        field[i] = _total(_field, sources[i : i + 1], r_sample, shape)[
                            0
                        ][:, :dim]

        else:
            msp = _total(_potential, sources, r, shape)
            if field_eval:
                field = _total(_field, sources, r, shape)
        return {
            "sources": sources,
            "r": r,
            "potential": msp,
            "field": field,
            "grid": grid,
            "potential_grid": _total(_potential, sources, grid, shape)
            if grid_eval
            else None,
            "field_grid": _total(_field, sources, grid, shape) if grid_eval else None,
            "shape": shape,
        }


def read_db(filename: str, read_grid=False):
    datapath = Path("/home/spol/Documents/repos/hypermagnetics/data")
    db = h5py.File(datapath / filename, "r")
    if read_grid:
        data = {
            "sources": jnp.concatenate(
                [db["m"][:], db["r0"][:], db["size"][:]], axis=-1
            ),
            "r": jnp.array(db["r"][:]),
            "potential": jnp.array(db["potential"][:]),
            "field": jnp.array(db["field"][:]),
            "grid": jnp.array(db["grid"][:]),
            "potential_grid": jnp.array(db["potential_grid"][:]),
            "field_grid": jnp.array(db["field_grid"][:]),
        }
    else:
        data = {
            "sources": jnp.concatenate(
                [db["m"][:], db["r0"][:], db["size"][:]], axis=-1
            ),
            "r": jnp.array(db["r"][:]),
            "potential": jnp.array(db["potential"][:]),
            "field": jnp.array(db["field"][:]),
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
        "dim": 2,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)
    plots(train_data, edge=True, idx=0, prefix="test", output="save")
    plots(train_data, edge=True, idx=1, prefix="test1", output="save")

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
    plots(train_data, edge=True, idx=0, prefix="test3d", output="save")

    # Prism
    config = {
        "shape": "prism",
        "n_samples": 1,
        "n_sources": 1,
        "seed": 40,
        "lim": 3,
        "res": 16,
        "dim": 2,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)
    plots(train_data, edge=True, idx=0, prefix="test_prism2d", output="save")
