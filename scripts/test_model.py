from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from hypermagnetics.sources import read_db
from hypermagnetics.mt_eval import field_cylinder_exact, field_mt
from hypermagnetics.models.hyper_mlp import HyperLayer

plot_single_err = False
model_cfg = HyperLayer(
    width=400,
    depth=3,
    hwidth=2,
    hdepth=3,
    seed=42,
)
model_path = Path(__file__).parent / ".." / "models"
figs_path = Path(__file__).parent / ".." / "figs"
model_name = "fcilr_400_200k_quadtree.eqx"
model = eqx.tree_deserialise_leaves(model_path / model_name, model_cfg)

data = read_db("val_qt_40_1020_1.h5")

out = []

if plot_single_err:
    n_samples = 10
else:
    n_samples = data["sources"].shape[0] // 4

for i in range(n_samples):
    # Run model for field
    fcilr_field = jax.vmap(model.field, in_axes=(0, None))(
        data["sources"][i : i + 1], data["grid"]
    )[0]

    # Run MagTense
    if data["shape"] == "sphere":
        mt_h, mt_dur = field_cylinder_exact(
            data["sources"][i : i + 1], data["grid"], length=25
        )
    else:
        mt_h, mt_dur = field_mt(
            data["sources"][i : i + 1],
            data["grid"],
            data["shape"],
        )

    diff_fcilr = jnp.array(mt_h)[..., :2] - jnp.array(fcilr_field)
    rel_err_fcilr = jnp.linalg.norm(diff_fcilr, axis=-1) / jnp.linalg.norm(
        jnp.array(mt_h)[..., :2], axis=-1
    )
    m_fcilr = jnp.nanmean(rel_err_fcilr, axis=-1)

    out.append([data["sources"][i, 0, 7], m_fcilr[0]])

    if plot_single_err:
        x_grid = np.array(data["grid"][:, 0].reshape((32, 32)))
        y_grid = np.array(data["grid"][:, 1].reshape((32, 32)))
        diff = np.linalg.norm(
            np.abs(np.array(fcilr_field) - mt_h[0, ..., :2]), axis=-1
        ).reshape((32, 32))
        norm_f = np.linalg.norm(mt_h[0, ..., :2], axis=-1).reshape((32, 32))
        plt.contourf(x_grid, y_grid, np.clip((diff / norm_f) * 100, 0, 150))
        plt.title(f"Relative Error of field [%] - Size {data['sources'][i, 0, 7]:.3f}")
        plt.colorbar()
        plt.savefig(
            figs_path
            / f"size_err_fcilr_{data['shape']}_{data['sources'][i, 0, 7]:.3f}.svg"
        )
        plt.clf()

if not plot_single_err:
    plt.scatter(jnp.array(out)[:, 0], jnp.array(out)[:, 1] * 100)
    plt.xlabel("Size")
    plt.xscale("log")
    plt.ylim(0, 200)
    plt.ylabel("Mean Relative Error (%)")
    plt.grid()
    plt.savefig(figs_path / f"size_err_fcilr_{data['shape']}.svg")
    plt.clf()
