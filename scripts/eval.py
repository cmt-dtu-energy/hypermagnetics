from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import prettytable

from hypermagnetics.fmm_sources import potential2D
from hypermagnetics.mt_eval import field_cylinder_exact, field_mt
from hypermagnetics.sources import read_db
from hypermagnetics.models.hyper_mlp import HyperLayer

n_eval = 4
n_ensemble = 5
min_sources = 10
step_sources = 250
db_name = "eval_qt_exact_42"
plot_t_fmm = True
plot_err_mt = False
plot_pot_fmm = False
grid_eval = True
mean_eval = True
model_eval = True
model_path = Path(__file__).parent / ".." / "models"
figs_path = Path(__file__).parent / ".." / "figs"

if model_eval:
    model_cfg = HyperLayer(
        width=400,
        depth=3,
        hwidth=2,
        hdepth=3,
        seed=42,
    )
    model_name = "fcilr_400_200k_quadtree.eqx"
    model = eqx.tree_deserialise_leaves(model_path / model_name, model_cfg)
    fcilr_t_avg = []
    fcilr_field_acc = []
    fcilr_field_acc_std = []
    fcilr_pot_acc = []
    fcilr_pot_acc_std = []

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
    fcilr_field_out = []
    fcilr_pot_out = []

    data = read_db(f"{db_name}_{n_ensemble}_{n_sources}.h5")

    if grid_eval:
        pot_out = np.zeros((n_ensemble, data["grid"].shape[0]))
    else:
        pot_out = np.zeros((n_ensemble, n_sources))
    t_pot = np.zeros(n_ensemble)
    t_fcilr = np.zeros(n_ensemble)
    t_mt = np.zeros(n_ensemble)

    for i in range(n_ensemble):
        if grid_eval:
            eval_loc = data["grid"]
            cor_source = False
        else:
            eval_loc = data["r"][i]
            cor_source = True
        # Function once to eliminate any overhead for first call
        if i == 0:
            potential2D(data["sources"][i : i + 1], data["shape"], eval_loc)
            if model_eval:
                jax.vmap(model, in_axes=(0, None))(
                    data["sources"][i : i + 1], eval_loc
                )[0]
        # Run model for potential
        start_time_pot = time.time()

        msp_fmm, field_fmm = potential2D(
            data["sources"][i : i + 1],
            data["shape"],
            eval_loc,
            correction_source=cor_source,
        )

        pot_out[i] = msp_fmm
        t_pot[i] = time.time() - start_time_pot
        field_out.append(field_fmm[0])

        # Run MagTense
        if data["shape"] == "sphere":
            mt_h, mt_dur = field_cylinder_exact(
                data["sources"][i : i + 1], eval_loc, length=25
            )
        else:
            mt_h, mt_dur = field_mt(
                data["sources"][i : i + 1],
                eval_loc,
                data["shape"],
            )
        mt_out.append(mt_h[0])
        t_mt[i] = mt_dur

        if model_eval:
            # Run model for field
            t_start = time.time()
            fcilr_field_out.append(
                jax.vmap(model.field, in_axes=(0, None))(
                    data["sources"][i : i + 1], eval_loc
                )[0]
            )
            t_fcilr[i] = time.time() - t_start

            # Run model for potential
            fcilr_pot_out.append(
                jax.vmap(model, in_axes=(0, None))(
                    data["sources"][i : i + 1], eval_loc
                )[0]
            )

    pot_t_avg.append(np.mean(t_pot))
    if model_eval:
        fcilr_t_avg.append(np.mean(t_fcilr))

    # Potential
    if grid_eval:
        diff_model_pot = data["msp_grid"] - np.array(pot_out)
        rel_err_pot = np.abs(diff_model_pot / data["msp_grid"])
    else:
        diff_model_pot = data["msp"] - np.array(pot_out)
        rel_err_pot = np.abs(diff_model_pot / data["msp"])

    if mean_eval:
        m_pot = np.nanmean(rel_err_pot, axis=-1)
    else:
        m_pot = np.nanmedian(rel_err_pot, axis=-1)
    pot_acc.append(np.mean(m_pot) * 100)
    pot_acc_std.append(np.std(m_pot) * 100)

    # Field
    mt_t_avg.append(np.mean(t_mt))
    diff_model = np.array(mt_out)[..., :2] - np.array(field_out)[..., :2]
    rel_err_field = np.linalg.norm(diff_model, axis=-1) / np.linalg.norm(
        np.array(mt_out)[..., :2], axis=-1
    )
    if mean_eval:
        m_field = np.nanmean(rel_err_field, axis=-1)
    else:
        m_field = np.nanmedian(rel_err_field, axis=-1)
    field_acc.append(np.mean(m_field) * 100)
    field_acc_std.append(np.std(m_field) * 100)

    if data["field_eval"]:
        # Eval MagTense
        diff_mt = data["field"][..., :2] - np.array(mt_out)[..., :2]
        rel_err_mt = np.linalg.norm(diff_mt, axis=-1) / np.linalg.norm(
            data["field"][..., :2], axis=-1
        )
        if mean_eval:
            m_mt = np.nanmean(rel_err_mt, axis=-1)
        else:
            m_mt = np.nanmedian(rel_err_mt, axis=-1)
        mt_acc.append(np.mean(m_mt) * 100)
        mt_acc_std.append(np.std(m_mt) * 100)

    if model_eval:
        # Potential
        if grid_eval:
            diff_model_fcilr_pot = data["msp_grid"] - np.array(fcilr_pot_out)
            rel_err_fcilr_pot = np.abs(diff_model_fcilr_pot / data["msp_grid"])
        else:
            diff_model_fcilr_pot = data["msp"] - np.array(fcilr_pot_out)
            rel_err_fcilr_pot = np.abs(diff_model_fcilr_pot / data["msp"])

        if mean_eval:
            m_fcilr_pot = np.nanmean(rel_err_fcilr_pot, axis=-1)
        else:
            m_fcilr_pot = np.nanmedian(rel_err_fcilr_pot, axis=-1)
        fcilr_pot_acc.append(np.mean(m_fcilr_pot) * 100)
        fcilr_pot_acc_std.append(np.std(m_fcilr_pot) * 100)

        # Field
        diff_fcilr = np.array(mt_out)[..., :2] - np.array(fcilr_field_out)
        rel_err_fcilr = np.linalg.norm(diff_fcilr, axis=-1) / np.linalg.norm(
            np.array(mt_out)[..., :2], axis=-1
        )
        if mean_eval:
            m_fcilr = np.nanmean(rel_err_fcilr, axis=-1)
        else:
            m_fcilr = np.nanmedian(rel_err_fcilr, axis=-1)
        fcilr_field_acc.append(np.mean(m_fcilr) * 100)
        fcilr_field_acc_std.append(np.std(m_fcilr) * 100)

    res_table = prettytable.PrettyTable()
    res_table.field_names = [
        "#Sources",
        "MagTense time [s]",
        "FMM2D time [s]",
        "FCILR time [s]",
        "MSP error [%]",
        "FCILR MSP error [%]",
        "Field error [%]",
        "FCILR Field error [%]",
        "MagTense error [%]",
    ]

    res_table.add_row(
        [
            int(n_sources),
            float(mt_t_avg[-1]),
            float(pot_t_avg[-1]),
            float(fcilr_t_avg[-1]) if model_eval else "-",
            float(pot_acc[-1]),
            float(fcilr_pot_acc[-1]) if model_eval else "-",
            float(field_acc[-1]),
            float(fcilr_field_acc[-1]) if model_eval else "-",
            float(mt_acc[-1]) if data["field_eval"] else "-",
        ]
    )
    res_table.float_format = "5.4"
    print(res_table)


fig, ax1 = plt.subplots()
fig.set_size_inches(12, 6)

color = "tab:green"
ax1.set_xlabel("Number of sources")

if plot_pot_fmm:
    if mean_eval:
        ax1.set_ylabel("Relative mean error (%)", color=color)
    else:
        ax1.set_ylabel("Relative median error (%)", color=color)
    # Plot mean and standard deviation for errors
    ax1.errorbar(
        range(len(x_axis_ticks)),
        pot_acc,
        yerr=pot_acc_std,
        fmt="o",
        color=color,
        linestyle="-.",
    )
    ax1.plot(pot_acc, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

color = "tab:red"
if len(mt_acc_std) > 0 and plot_err_mt:
    ax1.errorbar(
        range(len(x_axis_ticks)),
        mt_acc,
        yerr=mt_acc_std,
        fmt="o",
        color=color,
        linestyle="--",
    )

# Instantiate a third y-axis that shares the same x-axis
ax4 = ax1.twinx()
if plot_pot_fmm:
    ax4.spines["left"].set_position(("axes", -0.1))
else:
    ax1.set_yticklabels([])
    ax1.set_ylabel("")
    ax1.set_yticks([])

ax4.spines["left"].set_visible(True)
ax4.yaxis.set_label_position("left")
ax4.yaxis.set_ticks_position("left")
ax4.tick_params(axis="y", labelcolor=color)
if mean_eval:
    ax4.set_ylabel("Relative mean error (%) - Field", color=color)
else:
    ax4.set_ylabel("Relative median error (%) - Field", color=color)

ax4.errorbar(
    np.array(range(len(x_axis_ticks))) + 0.05,
    field_acc,
    yerr=field_acc_std,
    fmt="o",
    color=color,
    linestyle=":",
)

if model_eval:
    ax4.errorbar(
        np.array(range(len(x_axis_ticks))) + 0.1,
        fcilr_field_acc,
        yerr=fcilr_field_acc_std,
        fmt="o",
        color=color,
        linestyle="-",
    )

# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel("Runtime (s)", color=color)
if len(mt_t_avg) > 0:
    ax2.plot(mt_t_avg, color=color, linestyle="--")
ax2.tick_params(axis="y", labelcolor=color)

if model_eval:
    if len(fcilr_t_avg) > 0:
        ax2.plot(fcilr_t_avg, color=color, linestyle="-")

# Instantiate a third y-axis that shares the same x-axis
if plot_t_fmm:
    if grid_eval:
        ax2.set_ylabel("Runtime (s)", color=color)
        ax2.plot([val_t for val_t in pot_t_avg], color=color, linestyle=":")
    else:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        ax3.spines["right"].set_visible(True)
        ax3.set_ylabel("Runtime - FMM (ms)", color=color)
        ax3.plot([val_t * 1e3 for val_t in pot_t_avg], color=color, linestyle=":")
        ax3.tick_params(axis="y", labelcolor=color)

# Only display every second x tick
xtick_indices = list(range(0, len(x_axis_ticks), 2))
plt.xticks(xtick_indices, [x_axis_ticks[i] for i in xtick_indices])
# Custom legend
legend_elements = [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="MagTense"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="FCILR"),
    Line2D([0], [0], color="black", lw=2, linestyle=":", label="FMM"),
]
if plot_pot_fmm:
    legend_elements.append(
        Line2D([0], [0], color="black", lw=2, linestyle="-.", label="FMM - Potential")
    )
plt.legend(handles=legend_elements, loc="upper left")
fig.tight_layout()  # To ensure there's no overlap

# Save the plot to the 'figs' directory
plt.savefig(figs_path / f"metrics_fcilr_{data['shape']}.svg")
plt.clf()
