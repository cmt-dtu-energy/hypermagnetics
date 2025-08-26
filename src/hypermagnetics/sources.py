from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from hypermagnetics import plots
from hypermagnetics.quadtree import random_quadtree

jax.config.update("jax_enable_x64", True)


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
    value = jnp.where(close_to_source, interior, exterior)
    return replace_inf_nan(value)


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
    min_size: float = 0.1,
    max_size: float = 1,
    shape: str = "sphere",
    save_data: bool = False,
    line: bool = False,
    eps: float = 1e-5,
    quadtree: bool = False,
    target_source: bool = False,
    field_eval: bool = True,
    grid_eval: bool = True,
    batch_size: int = 1000,
    db_prefix: str = "",
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
        target_source (bool): Whether to evaluate in the center point of the sources.
        field_eval (bool): Whether to evaluate the field.
        grid_eval (bool): Whether to evaluate the grid.
        batch_size (int): Number of samples per batch.
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
            dim = 3
        elif dim == 3:
            size = jnp.concatenate([size, size, size], axis=-1)

    sources = jnp.concatenate([m, r0, size], axis=-1)

    if save_data:
        datapath = Path(__file__).parent / ".." / ".." / "data"
        datapath.mkdir(parents=True, exist_ok=True)

        db = h5py.File(datapath / f"{db_prefix}_{seed}_{n_samples}_{n_sources}.h5", "w")
        db.attrs["shape"] = shape
        db.attrs["field_eval"] = field_eval
        db.attrs["grid_eval"] = grid_eval
        db.attrs["target_source"] = target_source
        db.attrs["lim"] = lim
        db.attrs["eps"] = eps
        db.attrs["quadtree"] = quadtree
        db.attrs["seed"] = seed
        if not quadtree:
            db.attrs["min_size"] = min_size
            db.attrs["max_size"] = max_size
        db.create_dataset("m", shape=(n_samples, n_sources, dim), dtype="float32")
        db.create_dataset("r0", shape=(n_samples, n_sources, dim), dtype="float32")
        db.create_dataset("size", shape=(n_samples, n_sources, dim), dtype="float32")
        db.create_dataset(
            "r",
            shape=(n_samples, n_sources, dim) if target_source else (res**2, dim),
            dtype="float32",
        )
        db.create_dataset(
            "msp",
            shape=(n_samples, n_sources) if target_source else (n_samples, res**2),
            dtype="float32",
        )
        db.create_dataset(
            "field",
            shape=(n_samples, n_sources, dim)
            if target_source
            else (n_samples, res**2, dim),
            dtype="float32",
        )
        db.create_dataset("grid", shape=(res**2, dim), dtype="float32")
        db.create_dataset("msp_grid", shape=(n_samples, res**2), dtype="float32")
        db.create_dataset("field_grid", shape=(n_samples, res**2, dim), dtype="float32")

    if grid_eval:
        if dim == 2 and shape == "sphere":
            if line:
                grids = jnp.meshgrid(
                    jnp.linspace(0, lim, res),
                    jnp.linspace(r0[0, 0, 1], r0[0, 0, 1], 1) + eps,
                )
            else:
                grids = jnp.meshgrid(*[jnp.linspace(-lim, lim, res)] * dim)
        else:
            if line:
                grids = jnp.meshgrid(
                    jnp.linspace(0, lim, res),
                    jnp.linspace(r0[0, 0, 1], r0[0, 0, 1], 1) + eps,
                    jnp.linspace(r0[0, 0, 2], r0[0, 0, 2], 1),
                )
            else:
                grids = jnp.meshgrid(
                    jnp.linspace(-lim, lim, res),
                    jnp.linspace(-lim, lim, res),
                    jnp.linspace(0, 0, 1),
                )
        grid = jnp.concatenate([g.ravel()[:, None] for g in grids], axis=-1)

        ### Calculation for grid
        # If n_samples > batch_size, process in batches
        for i in range(max(1, n_samples // batch_size + 1)):
            batch = min(batch_size, n_samples - i * batch_size)
            b_sources = sources[i * batch : (i + 1) * batch]
            msp_grid = _total(_potential, b_sources, grid, shape)
            if field_eval:
                field_grid = _total(_field, b_sources, grid, shape)
            else:
                field_grid = None

            if save_data:
                db["msp_grid"][i * batch : (i + 1) * batch] = msp_grid
                db["field_grid"][i * batch : (i + 1) * batch] = field_grid

    else:
        grid = None
        msp_grid = None
        field_grid = None

    if target_source:
        # Add a small value to r0 to avoid singularities for gradient evaluation
        r = r0 + eps
    else:
        r = jr.uniform(minval=-lim, maxval=lim, shape=(res**2, 2), key=rkey)
        if dim == 3:
            r = jnp.concatenate([r, jnp.zeros((res**2, 1))], axis=-1)

    if save_data:
        db["r"][:] = r
        db["grid"][:] = grid

    ### Calculation for r
    for k in range(max(1, n_samples // batch_size + 1)):
        batch = min(batch_size, n_samples - k * batch_size)
        b_sources = sources[k * batch : (k + 1) * batch]

        if save_data:
            db["m"][k * batch : (k + 1) * batch] = m[k * batch : (k + 1) * batch]
            db["r0"][k * batch : (k + 1) * batch] = r0[k * batch : (k + 1) * batch]
            db["size"][k * batch : (k + 1) * batch] = size[k * batch : (k + 1) * batch]

        if target_source:
            msp = np.zeros((batch, r.shape[1]))
            if field_eval:
                field = np.zeros((batch, r.shape[1], dim))
            else:
                field = None

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
            msp = _total(_potential, b_sources, r, shape)
            if field_eval:
                field = _total(_field, b_sources, r, shape)
            else:
                field = None

        if save_data:
            db["msp"][k * batch : (k + 1) * batch] = msp
            db["field"][k * batch : (k + 1) * batch] = field

    # Postprocessing - Remove fields with nan values
    if save_data:
        print(f"Database '{db_prefix}_{seed}_{n_samples}_{n_sources}.h5' created!")
        if n_samples > 1000:
            nan_idx = jnp.where(jnp.isnan(db["field"][:, :, 0]))[0]
            for i, idx in enumerate(nan_idx):
                db["m"][idx] = db["m"][n_samples - 1000 + i]
                db["r0"][idx] = db["r0"][n_samples - 1000 + i]
                db["size"][idx] = db["size"][n_samples - 1000 + i]
                db["msp"][idx] = db["msp"][n_samples - 1000 + i]
                db["field"][idx] = db["field"][n_samples - 1000 + i]
                db["msp_grid"][idx] = db["msp_grid"][n_samples - 1000 + i]
                db["field_grid"][idx] = db["field_grid"][n_samples - 1000 + i]

        db.close()

    return {
        "sources": sources,
        "r": r,
        "msp": msp,
        "field": field,
        "grid": grid,
        "msp_grid": msp_grid,
        "field_grid": field_grid,
        "shape": shape,
    }


def read_db(filename: str, read_grid=False):
    datapath = Path(__file__).parent / ".." / ".." / "data"
    db = h5py.File(datapath / filename, "r")
    if read_grid:
        data = {
            "sources": np.concatenate(
                [db["m"][:], db["r0"][:], db["size"][:]], axis=-1
            ),
            "r": np.array(db["r"][:]),
            "msp": np.array(db["msp"][:]),
            "field": np.array(db["field"][:]),
            "grid": np.array(db["grid"][:]),
            "msp_grid": np.array(db["msp_grid"][:]),
            "field_grid": np.array(db["field_grid"][:]),
            "field_eval": db.attrs["field_eval"],
            "target_source": db.attrs["target_source"],
            "shape": db.attrs["shape"],
        }
    else:
        data = {
            "sources": np.concatenate(
                [db["m"][:], db["r0"][:], db["size"][:]], axis=-1
            ),
            "r": np.array(db["r"][:]),
            "msp": np.array(db["msp"][:]),
            "field": np.array(db["field"][:]),
            "field_eval": db.attrs["field_eval"],
            "target_source": db.attrs["target_source"],
            "shape": db.attrs["shape"],
        }
    db.close()

    _, _, size = np.split(data["sources"], 3, axis=-1)
    data["max_size"] = np.max(size[..., :2])
    data["min_size"] = np.min(size[..., :2])

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
    print(train_data["msp"].shape, train_data["field"].shape)
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
    print(train_data["msp"].shape, train_data["field"].shape)
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
    print(train_data["msp"].shape, train_data["field"].shape)
    plots(train_data, edge=True, idx=0, prefix="test_prism2d", output="save")
