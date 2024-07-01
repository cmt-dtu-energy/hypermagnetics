import jax
import jax.numpy as jnp
import equinox as eqx
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from hypermagnetics.sources import read_db, _field_mt, configure
from hypermagnetics.models.hyper_mlp import HyperLayer
from hypermagnetics.measures import replace_inf_nan

from_db = False
n_eval = 800
config = {"width": 400, "depth": 3, "hwidth": 1.5, "hdepth": 3, "seed": 42}

model_orig = HyperLayer(
    width=config["width"],
    depth=config["depth"],
    hwidth=config["hwidth"],
    hdepth=config["hdepth"],
    seed=config["seed"],
)

filename = "/home/spol/Documents/repos/hypermagnetics/models/ic_inr_400_50k_fcinr_lim_uniform.eqx"
model = eqx.tree_deserialise_leaves(filename, model_orig)

mt_time = []
mt_acc = []
model_time = []
model_acc = []
x_axis_ticks = []

for test_idx in range(0, n_eval, max(1, 100)):
    if from_db:
        test = read_db(f"squares_1_{test_idx}.h5")
    else:
        source_config = {
            "shape": "prism",
            "n_samples": 1,
            "lim": 1.2,
            "res": 100,
            "dim": 3,
            "save_data": False,
        }
        test = configure(**source_config, n_sources=max(1, test_idx), seed=101)

    mr = test["sources"][0:1]
    m, r0, size = jnp.split(mr, 3, axis=-1)
    grid = test["grid"]

    res = int(jnp.sqrt(len(grid)))
    N = len(mr)

    start_time = time.time()
    field_model = jax.vmap(model.field, in_axes=(0, None))(mr, test["grid"])
    model_time.append(time.time() - start_time)

    field_mt, mt_dur = _field_mt(mr, test["grid"], "prism")
    mt_time.append(mt_dur)

    sources, r, target = test["sources"], test["r"], test["field_grid"]
    diff_mt = target[..., :2] - field_mt[..., :2] * jnp.pi**2
    mt_acc.append(
        jnp.median(
            replace_inf_nan(
                jnp.linalg.norm(diff_mt, axis=-1)
                / jnp.linalg.norm(target[..., :2], axis=-1)
            )
        )
        * 100
    )
    diff_model = target[..., :2] - field_model[..., :2]
    model_acc.append(
        jnp.median(
            replace_inf_nan(
                jnp.linalg.norm(diff_model, axis=-1)
                / jnp.linalg.norm(target[..., :2], axis=-1)
            )
        )
        * 100
    )

    x_axis_ticks.append(test["sources"][:].shape[1])
    print(f"Number of sources: {test['sources'][:].shape[1]}")
    print(f"Model time: {model_time[-1]:.3f}s, MagTense time: {mt_time[-1]:.4f}s")
    print(f"Model acc: {model_acc[-1]:.2f}%, MagTense acc: {mt_acc[-1]:.2f}%")

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Number of sources")
ax1.set_ylabel("Relative median field error (%)", color=color)
ax1.plot(mt_acc, color=color, linestyle="--")
ax1.plot(model_acc, color=color)
ax1.tick_params(axis="y", labelcolor=color)
# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel("Runtime (s)", color=color)  # we already handled the x-label with ax1
ax2.plot(mt_time, color=color, linestyle="--")
ax2.plot(model_time, color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.xticks(range(len(x_axis_ticks)), x_axis_ticks)
# Custom legend
legend_elements = [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="MagTense"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="Model"),
]
plt.legend(handles=legend_elements, loc="upper left")
fig.tight_layout()  # To ensure there's no overlap

# Save the plot to the 'figs' directory
plt.savefig("/home/spol/Documents/repos/hypermagnetics/figs/metrics.png")

# Clear the current figure after saving to avoid conflicts with future plots
plt.clf()
