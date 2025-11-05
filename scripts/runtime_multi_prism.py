import jax
import jax.numpy as jnp
import equinox as eqx
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import prettytable

from hypermagnetics.sources import configure
from hypermagnetics.mt_eval import field_mt
from hypermagnetics.models.hyper_mlp import HyperLayer

n_eval = 500
n_ensemble = 1
min_sources = 10
config = {"width": 400, "depth": 3, "hwidth": 2, "hdepth": 3, "seed": 42}

model_orig = HyperLayer(
    width=config["width"],
    depth=config["depth"],
    hwidth=config["hwidth"],
    hdepth=config["hdepth"],
    seed=config["seed"],
)

filename = "/home/spol/Documents/repos/hypermagnetics/models/ic_inr_400_hwidth_2_50k_fcinr_lim_uniform.eqx"
model = eqx.tree_deserialise_leaves(filename, model_orig)

mt_acc = []
mt_acc_std = []
mt_t_avg = []
field_acc = []
field_acc_std = []
field_t_avg = []
pot_acc = []
pot_acc_std = []
pot_t_avg = []
x_axis_ticks = []

for test_sources in range(0, n_eval, max(min_sources, 250)):
    n_sources = max(min_sources, test_sources)
    t_mt = []
    mt_out = []
    t_field = []
    field_out = []
    t_pot = []
    pot_out = []
    x_axis_ticks.append(n_sources)

    source_config = {
        "shape": "prism",
        "n_samples": n_ensemble,
        "lim": 1.2,
        "res": 16,
        "dim": 2,
        "source_val": True,
        # "calc_2d": True,
    }
    test = configure(**source_config, n_sources=n_sources, seed=0)

    for i in range(n_ensemble):
        mr = test["sources"][i : i + 1]
        m, r0, size = jnp.split(mr, 3, axis=-1)

        # Run model for field
        start_time = time.time()
        field_out.append(jax.vmap(model.field, in_axes=(0, None))(mr, test["r"][i])[0])
        t_field.append(time.time() - start_time)

        # Run model for potential
        start_time_pot = time.time()
        pot_out.append(jax.vmap(model, in_axes=(0, None))(mr, test["r"][i])[0])
        t_pot.append(time.time() - start_time_pot)

        # Run MagTense
        mt_h, mt_dur = field_mt(mr, test["r"][i], "prism")
        mt_out.append(mt_h[0])
        t_mt.append(mt_dur)

    mt_t_avg.append(sum(t_mt) / len(t_mt))
    field_t_avg.append(sum(t_field) / len(t_field))
    pot_t_avg.append(sum(t_pot) / len(t_pot))

    sources, r, target, target_pot = (
        test["sources"],
        test["r"],
        test["field"],
        test["potential"],
    )
    # Eval MagTense
    diff_mt = target[..., :2] - jnp.array(mt_out)[..., :2] * jnp.pi**2
    rel_err_mt = jnp.linalg.norm(diff_mt, axis=-1) / jnp.linalg.norm(
        target[..., :2], axis=-1
    )
    median_mt = jnp.nanmedian(rel_err_mt, axis=-1)
    mt_acc.append(jnp.mean(median_mt) * 100)
    mt_acc_std.append(jnp.std(median_mt) * 100)

    # Field
    diff_model = target[..., :2] - jnp.array(field_out)[..., :2]
    rel_err_field = jnp.linalg.norm(diff_model, axis=-1) / jnp.linalg.norm(
        target[..., :2], axis=-1
    )
    median_field = jnp.nanmedian(rel_err_field, axis=-1)
    field_acc.append(jnp.mean(median_field) * 100)
    field_acc_std.append(jnp.std(median_field) * 100)

    # Potential
    diff_model_pot = target_pot - jnp.array(pot_out)
    rel_err_pot = jnp.abs(diff_model_pot / target_pot)
    median_pot = jnp.nanmedian(rel_err_pot, axis=-1)
    pot_acc.append(jnp.mean(median_pot) * 100)
    pot_acc_std.append(jnp.std(median_pot) * 100)

    # Plotting takes long time for a large number of sources
    # plots(test, idx=0, model=model, prefix=f"{n_sources}", output="save")

    res_table = prettytable.PrettyTable()
    res_table.field_names = [
        "#Sources",
        "MagTense time [s]",
        "Model time [s]",
        "Potential time [s]",
        "MagTense error [%]",
        "Field error [%]",
        "Potential error [%]",
    ]
    res_table.add_row(
        [
            int(n_sources),
            float(mt_t_avg[-1]),
            float(field_t_avg[-1]),
            float(pot_t_avg[-1]),
            float(mt_acc[-1]),
            float(field_acc[-1]),
            float(pot_acc[-1]),
        ]
    )
    res_table.float_format = "5.3"
    print(res_table)

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Number of sources")
ax1.set_ylabel("Relative median error (%)", color=color)
# Plot mean and standard deviation for errors
# ax1.errorbar(
#     range(len(x_axis_ticks)),
#     mt_acc,
#     yerr=mt_acc_std,
#     fmt="o",
#     color=color,
#     linestyle="--",
# )
# ax1.errorbar(
#     range(len(x_axis_ticks)),
#     field_acc,
#     yerr=field_acc_std,
#     fmt="o",
#     color=color,
#     linestyle="dotted",
# )
ax1.errorbar(
    range(len(x_axis_ticks)),
    pot_acc,
    yerr=pot_acc_std,
    fmt="o",
    color=color,
)
# ax1.plot(mt_acc, color=color, linestyle="--")
# ax1.plot(field_acc, color=color, linestyle="dotted")
ax1.plot(pot_acc, color=color)
ax1.set_ylim(0, 11)
ax1.tick_params(axis="y", labelcolor=color)
# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel("Runtime (s)", color=color)  # we already handled the x-label with ax1
ax2.plot(mt_t_avg, color=color, linestyle="--")
# ax2.plot(field_t_avg, color=color, linestyle="dotted")
ax2.plot(pot_t_avg, color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.xticks(range(len(x_axis_ticks)), x_axis_ticks)
# Custom legend
legend_elements = [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Exact"),
    # Line2D([0], [0], color="black", lw=2, linestyle="dotted", label="Field - Model"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="FC + ILR"),
]
plt.legend(handles=legend_elements, loc="upper left")
fig.tight_layout()  # To ensure there's no overlap

# Save the plot to the 'figs' directory
plt.savefig("/home/spol/Documents/repos/hypermagnetics/figs/metrics_prism2d_check.svg")

# Clear the current figure after saving to avoid conflicts with future plots
plt.clf()
