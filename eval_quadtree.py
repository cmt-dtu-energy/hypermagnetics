import h5py
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import prettytable

from hypermagnetics.fmm_sources import potential2D_sources
from hypermagnetics.mt_eval import field_mt

n_eval = 8
n_ensemble = 10
min_sources = 10
step_sources = 250

mt_acc = []
mt_acc_std = []
mt_t_avg = []
field_acc = []
field_acc_std = []
pot_acc = []
pot_acc_std = []
pot_t_avg = []
x_axis_ticks = []

for n in range(n_eval + 1):
    n_sources = max(min_sources, step_sources * n)
    x_axis_ticks.append(n_sources)
    mt_out = []
    field_out = []

    db_test = h5py.File(
        Path(__file__).parent / ".." / ".." / "data" / f"eval_{n_sources}.h5", "r"
    )
    sources = np.concatenate([db_test["m"], db_test["ro"], db_test["size"]], axis=-1)

    pot_out = np.zeros((n_ensemble, n_sources))
    t_pot = np.zeros(n_ensemble)
    t_mt = np.zeros(n_ensemble)
    for i in range(n_ensemble):
        # Function once to eliminate any overhead for first call
        if i == 0:
            msp_fmm, field_fmm = potential2D_sources(sources[i : i + 1], "prism")
        # Run model for potential
        start_time_pot = time.time()
        msp_fmm, field_fmm = potential2D_sources(sources[i : i + 1], "prism")
        pot_out[i] = msp_fmm
        t_pot[i] = time.time() - start_time_pot
        if db_test.attrs["field_eval"]:
            field_out.append(field_fmm)

            # Run MagTense
            mt_h, mt_dur = field_mt(
                sources[i : i + 1],
                db_test["r"][i] if db_test.attrs["t_source"] else db_test["r"],
                "prism",
            )
            mt_out.append(mt_h[0])
            t_mt[i] = mt_dur

    pot_t_avg.append(np.mean(t_pot))

    # Potential
    diff_model_pot = db_test["msp"] - jnp.array(pot_out)
    rel_err_pot = jnp.abs(diff_model_pot / db_test["msp"])
    median_pot = jnp.nanmedian(rel_err_pot, axis=-1)
    pot_acc.append(jnp.mean(median_pot) * 100)
    pot_acc_std.append(jnp.std(median_pot) * 100)

    if db_test.attrs["field_eval"]:
        mt_t_avg.append(np.mean(t_mt))

        # Eval MagTense
        diff_mt = db_test["field"][..., :2] - jnp.array(mt_out)[..., :2]  # * jnp.pi**2
        rel_err_mt = jnp.linalg.norm(diff_mt, axis=-1) / jnp.linalg.norm(
            db_test["field"][..., :2], axis=-1
        )
        median_mt = jnp.nanmedian(rel_err_mt, axis=-1)
        mt_acc.append(jnp.mean(median_mt) * 100)
        mt_acc_std.append(jnp.std(median_mt) * 100)

        # Field
        diff_model = db_test["field"][..., :2] - jnp.array(field_out)[..., :2]
        rel_err_field = jnp.linalg.norm(diff_model, axis=-1) / jnp.linalg.norm(
            db_test["field"][..., :2], axis=-1
        )
        median_field = jnp.nanmedian(rel_err_field, axis=-1)
        field_acc.append(jnp.mean(median_field) * 100)
        field_acc_std.append(jnp.std(median_field) * 100)

    res_table = prettytable.PrettyTable()
    if db_test.attrs["field_eval"]:
        res_table.field_names = [
            "#Sources",
            "MagTense time [s]",
            "Potential time [s]",
            "MagTense error [%]",
            "Field error [%]",
            "Potential error [%]",
        ]

        res_table.add_row(
            [
                int(n_sources),
                float(mt_t_avg[-1]),
                float(pot_t_avg[-1]),
                float(mt_acc[-1]),
                float(field_acc[-1]),
                float(pot_acc[-1]),
            ]
        )
    else:
        res_table.field_names = [
            "#Sources",
            "Potential time [s]",
            "Potential error [%]",
        ]

        res_table.add_row(
            [
                int(n_sources),
                float(pot_t_avg[-1]),
                float(pot_acc[-1]),
            ]
        )
    res_table.float_format = "5.4"
    print(res_table)
    db_test.close()


fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Number of sources")
ax1.set_ylabel("Relative median error (%)", color=color)
# Plot mean and standard deviation for errors
ax1.errorbar(
    range(len(x_axis_ticks)),
    pot_acc,
    yerr=pot_acc_std,
    fmt="o",
    color=color,
)
ax1.plot(pot_acc, color=color)

if len(mt_acc_std) > 0:
    ax1.errorbar(
        range(len(x_axis_ticks)),
        mt_acc,
        yerr=mt_acc_std,
        fmt="o",
        color=color,
        linestyle="--",
    )

    ax1.errorbar(
        range(len(x_axis_ticks)),
        field_acc,
        yerr=field_acc_std,
        fmt="o",
        color=color,
        linestyle="dotted",
    )

    ax1.plot(mt_acc, color=color, linestyle="--")
    ax1.plot(field_acc, color=color, linestyle="dotted")

ax1.tick_params(axis="y", labelcolor=color)

# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel("Runtime (s)", color=color)
if len(mt_t_avg) > 0:
    ax2.plot(mt_t_avg, color=color, linestyle="--")

ax2.plot(pot_t_avg, color=color)
ax2.tick_params(axis="y", labelcolor=color)

# Only display every second x tick
xtick_indices = list(range(0, len(x_axis_ticks), 2))
plt.xticks(xtick_indices, [x_axis_ticks[i] for i in xtick_indices])
# Custom legend
legend_elements = [
    # Line2D([0], [0], color="black", lw=2, linestyle="--", label="MagTense"),
    # Line2D([0], [0], color="black", lw=2, linestyle="dotted", label="Field - Model"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="FMM"),
]
plt.legend(handles=legend_elements, loc="upper left")
fig.tight_layout()  # To ensure there's no overlap

# Save the plot to the 'figs' directory
plt.savefig("/home/spol/Documents/repos/hypermagnetics/figs/metrics_fmm_quadtree.svg")

# Clear the current figure after saving to avoid conflicts with future plots
plt.clf()
