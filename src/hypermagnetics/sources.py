from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from hypermagnetics import plots
from hypermagnetics.quadtree import random_quadtree
# from hypermagnetics.fmm_sources import potential2D

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
    # M = m / (2 * a * 2 * b)
    value = m @ jnp.array([fx, fy, fz]) / (4 * jnp.pi)
    return replace_inf_nan(value)


@jax.jit
def _sphere(m: jax.Array, r0: jax.Array, r: jax.Array, size=1.0, dim=2):
    """Finite sphere potential in two or three dimensions."""
    # Convert magnetization to magnetic moment
    m = m * (jnp.pi * size**2)
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


def _total_batch(fun, sources, r, shape):
    """Aggregate the field or potential of all sources."""
    fun_with_shape = partial(fun, shape=shape)
    points = jax.vmap(fun_with_shape, in_axes=(None, 0))
    batch = jax.vmap(points, in_axes=(0, 0))
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
    p_target_source: float = 1.0,
    dipole_correction: bool = False,
    field_eval: bool = True,
    grid_eval: bool = True,
    size_log: bool = True,
    r0_gap: bool = True,
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
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, 2)) * 10

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
            minval=-lim + min_size if r0_gap else -lim,
            maxval=lim - min_size if r0_gap else lim,
        )
        if size_log:
            size = jnp.exp(
                jr.uniform(
                    key=skey,
                    shape=(n_samples, n_sources, 1),
                    minval=jnp.log(min_size),
                    maxval=jnp.log(max_size),
                )
            )
        else:
            size = jr.uniform(
                key=skey,
                shape=(n_samples, n_sources, 1),
                minval=min_size,
                maxval=max_size,
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
        db.attrs["p_target_source"] = p_target_source
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
        # if dipole_correction:
        #     db.create_dataset(
        #         "r",
        #         shape=(n_samples + 1, res**2, 3),
        #         dtype="float32",
        #     )
        #     db.create_dataset(
        #         "msp",
        #         shape=(n_samples, 2 * res**2),
        #         dtype="float32",
        #     )
        #     db.create_dataset(
        #         "field",
        #         (n_samples, 2 * res**2, dim),
        #         dtype="float32",
        #     )
        # else:
        if p_target_source < 1.0:
            shape_r = (n_samples, res**2, dim)
        else:
            if target_source:
                shape_r = (n_samples, n_sources, dim)
            else:
                shape_r = (res**2, dim)
        db.create_dataset(
            "r",
            shape=shape_r,
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
            msp_grid = np.array(_total(_potential, b_sources, grid, shape))
            if field_eval:
                field_grid = np.array(_total(_field, b_sources, grid, shape))
            else:
                field_grid = None

            # if dipole_correction:
            #     msp_fmm, field_fmm, _ = potential2D(b_sources, "sphere", grid)
            #     msp_grid -= msp_fmm

            #     if field_eval:
            #         field_grid[..., :2] -= field_fmm
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
        if p_target_source < 1.0:
            r_all = jnp.zeros((n_samples, res**2, 2))
            akey, bkey, ckey, dkey, rkey = jr.split(rkey, num=5)
            r_avail = jr.uniform(
                key=akey, minval=-lim, maxval=lim, shape=(n_samples, 4 * res**2, 2)
            )
            for i in range(n_samples):
                for n in range(n_sources):
                    if shape == "sphere":
                        idx_in = np.where(
                            np.linalg.norm(r_avail[i] - r0[i][n], axis=1)
                            <= size[i, n, 0]
                        )[0]
                    elif shape == "prism":
                        idx_in = np.where(
                            (np.abs(r_avail[i][:, 0] - r0[i][n, 0]) <= size[i, n, 0])
                            & (np.abs(r_avail[i][:, 1] - r0[i][n, 1]) <= size[i, n, 1])
                        )[0]
                    else:
                        raise ValueError("Unknown shape")

                    # Accumulate indices of candidate points that lie inside any source for this sample
                    if n == 0:
                        idx_union = set(idx_in.tolist())
                    else:
                        idx_union.update(idx_in.tolist())

                    # After processing all sources for the first sample, build r_all so that
                    # approximately p_target_source fraction of the res**2 points are inside sources
                    if n == n_sources - 1:
                        total_points = res**2
                        p_in = int(round(p_target_source * total_points))

                        avail_indices = jnp.arange(r_avail.shape[1])
                        idx_in_arr = jnp.array(sorted(idx_union))

                        outside_indices = jnp.setdiff1d(
                            avail_indices, idx_in_arr, assume_unique=False
                        )

                        # Sample points inside sources (up to n_in) and fill the rest from outside
                        chosen_in = np.array([], dtype=int)
                        if idx_in_arr.size > 0 and p_in > 0:
                            chosen_in = np.random.choice(
                                idx_in_arr,
                                size=min(p_in, idx_in_arr.size),
                                replace=False,
                            )

                        n_remaining = total_points - chosen_in.size
                        chosen_out = (
                            jr.choice(
                                bkey,
                                outside_indices,
                                shape=(n_remaining,),
                                replace=False,
                            )
                            if n_remaining > 0
                            else jnp.array([], dtype=int)
                        )

                        selected = jnp.concatenate([chosen_in, chosen_out]).astype(int)

                        # In case of any shortfall (very unlikely), pad with random available indices
                        if selected.size < total_points:
                            more = jnp.setdiff1d(
                                avail_indices, selected, assume_unique=False
                            )
                            need = total_points - selected.size
                            extra = jr.choice(ckey, more, shape=(need,), replace=False)
                            selected = jnp.concatenate([selected, extra])

                        r_all = r_all.at[i].set(
                            jnp.array(r_avail[i])[jr.permutation(dkey, selected)]
                        )

            if dim == 3:
                r_all = jnp.concatenate(
                    [r_all, jnp.zeros((n_samples, res**2, 1))], axis=-1
                )
        else:
            r_all = jr.uniform(key=rkey, minval=-lim, maxval=lim, shape=(res**2, 2))

            if dim == 3:
                r_all = jnp.concatenate(
                    [r_all, jnp.zeros((n_samples, res**2, 1))], axis=-1
                )

        # if dipole_correction:
        #     r_dipole = (
        #         jr.normal(key=rkey, shape=(n_samples, res**2, 2)) * size[:, 0:1, 0:2]
        #         + r0[:, 0:1, 0:2]
        #     )
        #     if dim == 3:
        #         r_dipole = jnp.concatenate(
        #             [r_dipole, jnp.zeros((n_samples, res**2, 1))], axis=-1
        #         )

        #     r = jnp.concatenate([r_dipole, r_all[None]], axis=0)
        # else:
        r = r_all

    if save_data:
        db["r"][:] = r
        db["grid"][:] = grid

    ### Calculation for r
    for k in range(max(1, n_samples // batch_size + 1)):
        batch = min(batch_size, n_samples - k * batch_size)
        b_sources = sources[k * batch : (k + 1) * batch]
        # if dipole_correction:
        #     b_r = jnp.concatenate(
        #         [
        #             r[k * batch : (k + 1) * batch],
        #             jnp.repeat(r[-1:], batch, axis=0),
        #         ],
        #         axis=1,
        #     )
        if p_target_source < 1.0:
            b_r = r[k * batch : (k + 1) * batch]
        else:
            if target_source:
                b_r = r[k * batch : (k + 1) * batch]
            else:
                b_r = r

        if save_data:
            db["m"][k * batch : (k + 1) * batch] = m[k * batch : (k + 1) * batch]
            db["r0"][k * batch : (k + 1) * batch] = r0[k * batch : (k + 1) * batch]
            db["size"][k * batch : (k + 1) * batch] = size[k * batch : (k + 1) * batch]

        if target_source or p_target_source < 1.0:
            msp = np.zeros((batch, b_r.shape[1]))
            if field_eval:
                field = np.zeros((batch, b_r.shape[1], dim))
            else:
                field = None

            for o, r_sample in enumerate(b_r):
                # Memory constraints above 10k sources
                # N_sources ** 2 has to be below 100M
                if sources[o].shape[0] > 1e4:
                    n_sources_per_sim = int(1e8 // sources[o].shape[0])
                    for j in range(0, sources[o].shape[0], n_sources_per_sim):
                        msp[o : o + 1] += _total(
                            _potential,
                            sources[
                                o : o + 1,
                                j : j + n_sources_per_sim,
                            ],
                            r_sample,
                            shape,
                        )
                        if field_eval:
                            field[o : o + 1] += _total(
                                _field,
                                sources[
                                    o : o + 1,
                                    j : j + n_sources_per_sim,
                                ],
                                r_sample,
                                shape,
                            )
                else:
                    msp[o] = _total(_potential, b_sources, r_sample, shape)[0]
                    if field_eval:
                        field[o] = _total(_field, b_sources, r_sample, shape)[0][
                            :, :dim
                        ]

        else:
            if dipole_correction:
                pass
                # msp = np.array(_total(_potential, b_sources, b_r, shape))
                # if field_eval:
                #     field = np.array(_total(_field, b_sources, b_r, shape))
                # else:
                #     field = None

                # msp_fmm, field_fmm, _ = potential2D(
                #     b_sources, shape, b_r
                # )  # , batch_r=True)
                # msp -= msp_fmm

                # if field_eval:
                #     field[..., :2] -= field_fmm

            else:
                msp = np.array(_total(_potential, b_sources, b_r, shape))
                if field_eval:
                    field = np.array(_total(_field, b_sources, b_r, shape))
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


def read_db(filename: str):
    datapath = Path(__file__).parent / ".." / ".." / "data"
    db = h5py.File(datapath / filename, "r")
    data = {
        "sources": np.concatenate(
            [db["m"][:], db["r0"][:], db["size"][:]],
            axis=-1,
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
    db.close()

    _, _, size = np.split(data["sources"], 3, axis=-1)
    data["max_size"] = np.max(size[..., :2])
    data["min_size"] = np.min(size[..., :2])

    return data


def read_db_old(filename: str, max_samples: int = -1):
    datapath = Path(__file__).parent / ".." / ".." / "data"
    db = h5py.File(datapath / filename, "r")
    data = {
        "sources": np.concatenate(
            [db["m"][:max_samples], db["r0"][:max_samples], db["size"][:max_samples]],
            axis=-1,
        ),
        "r": np.array(db["r"][:]),
        "msp": np.array(db["potential"][:max_samples]),
        "field": np.array(db["field"][:max_samples]),
        "grid": np.array(db["grid"][:]),
        "msp_grid": np.array(db["potential_grid"][:max_samples]),
        "field_grid": np.array(db["field_grid"][:max_samples]),
        "field_eval": True,
        "shape": "prism",
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
