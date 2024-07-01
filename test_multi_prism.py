from pathlib import Path
import h5py
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics.sources import sample_grid, _potential, _field, _total


path = Path("/home/spol/Documents/repos/hypermagnetics/")
folder = path / "data" / "Meshes_complete_2D"

datapath = Path(__file__).parent / "data"
datapath.mkdir(parents=True, exist_ok=True)
n_samples = 1
res = 100
dim = 3
lim = 1.2
shape = "prism"

for file_idx in range(10):
    seed = 10 + file_idx

    with open(list(folder.iterdir())[file_idx], "r") as file:
        file_len = len(file.readlines()) - 1

    skip_cnt = 0
    with open(list(folder.iterdir())[file_idx], "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            columns = line.split()
            r0_file = columns[:3]
            size_file = columns[3:]

            if float(size_file[0]) < 1.2e-8:
                skip_cnt += 1

    n_sources = file_len - skip_cnt

    db = h5py.File(datapath / f"squares_{n_samples}_{file_idx}.h5", "w")
    db.create_dataset("m", shape=(n_samples, n_sources, dim), dtype="float32")
    db.create_dataset("r0", shape=(n_samples, n_sources, dim), dtype="float32")
    db.create_dataset("size", shape=(n_samples, n_sources, dim), dtype="float32")
    db.create_dataset("r", shape=(res**2, dim), dtype="float32")
    db.create_dataset("potential", shape=(n_samples, res**2), dtype="float32")
    db.create_dataset("field", shape=(n_samples, res**2, dim), dtype="float32")
    db.create_dataset("grid", shape=(res**2, dim), dtype="float32")
    db.create_dataset("potential_grid", shape=(n_samples, res**2), dtype="float32")
    db.create_dataset("field_grid", shape=(n_samples, res**2, dim), dtype="float32")

    r0 = np.zeros((n_samples, n_sources, dim))
    size = np.zeros((n_samples, n_sources, dim))
    idx = 0

    with open(list(folder.iterdir())[file_idx], "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            columns = line.split()
            r0_file = columns[:3]
            size_file = columns[3:]

            if float(size_file[0]) < 1.2e-8:
                continue

            # Convert the coordinates and size to integers
            r0[0, idx, :] = [float(coord) * 1e7 - lim for coord in r0_file]
            size[0, idx, :] = [float(a) * 1e7 for a in size_file]
            idx += 1

    if dim == 3:
        r0[:, :, 2] = 0.0

    if dim == 3:
        size[:, :, 2] = 1.0

    key = jr.PRNGKey(seed)
    mkey, rkey = jr.split(key, 2)
    m = jr.normal(key=mkey, shape=(n_samples, n_sources, dim))
    if dim == 3:
        m = m.at[:, :, 2].set(0.0)

    lim_range = jnp.linspace(-lim, lim, res)
    if dim == 3:
        grids = jnp.meshgrid(lim_range, lim_range, jnp.linspace(0, 0, 1))
    else:
        grids = jnp.meshgrid(*[lim_range] * dim)
    grid = jnp.concatenate([g.ravel()[:, None] for g in grids], axis=-1)
    r = sample_grid(rkey, lim, res, r0, size, dim, masking=False)

    sources = jnp.concatenate(
        [
            m,
            jnp.array(r0),
            jnp.array(size),
        ],
        axis=-1,
    )
    db["r"][:] = r
    db["grid"][:] = grid

    db["m"][:] = m
    db["r0"][:] = r0
    db["size"][:] = size
    db["potential"][:] = _total(_potential, sources, r, shape)
    db["field"][:] = _total(_field, sources, r, shape)
    db["potential_grid"][:] = _total(_potential, sources, grid, shape)
    db["field_grid"][:] = _total(_field, sources, grid, shape)

    # Remove fields with nan values
    # nan_idx = jnp.where(jnp.isnan(db["field"][:, :, 0]))[0]
    # for i, idx in enumerate(nan_idx):
    #     db["m"][idx] = db["m"][n_samples - 1000 + i]
    #     db["r0"][idx] = db["r0"][n_samples - 1000 + i]
    #     db["size"][idx] = db["size"][n_samples - 1000 + i]
    #     db["potential"][idx] = db["potential"][n_samples - 1000 + i]
    #     db["field"][idx] = db["field"][n_samples - 1000 + i]
    #     db["potential_grid"][idx] = db["potential_grid"][n_samples - 1000 + i]
    #     db["field_grid"][idx] = db["field_grid"][n_samples - 1000 + i]

    db.close()
