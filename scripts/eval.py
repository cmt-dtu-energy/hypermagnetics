import warnings
import psutil
import os
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

warnings.filterwarnings("ignore")


def run_on_one_cpu(func, cpu_id=0, *args, **kwargs):
    p = psutil.Process(os.getpid())
    p.cpu_affinity([cpu_id])  # works on Linux and Windows
    result = func(*args, **kwargs)
    return result


n_eval = 6
n_ensemble = 10
min_sources = 10
step_sources = 250
res = 512
db_name = "eval_large_m_42"
plot_t_fmm = True
plot_err_mt = False
plot_pot_fmm = True
grid_eval = True
mean_eval = False
model_eval = True
ax1_log = False
normalized_time = False
fcilr_overhead_time = True  # If true, check jax.vmap in model
plain_fmm_time = True
fmm_nooverhead_time = True
percentage_test = False
nocorrsource = True
prism_mt = True
norm_n = 10
model_path = Path(__file__).parent / ".." / "models"
figs_path = Path(__file__).parent / ".." / "figs"
tmp_path = Path(__file__).parent / ".." / "tmp"
fig_name = "tlmr_corr_tile_mt"

if model_eval:
    model_cfg = HyperLayer(
        width=400,
        depth=3,
        hwidth=2,
        hdepth=3,
        seed=42,
    )
    model_name = "fc_ilr_400_200k_res32_large_m.eqx"
    model = eqx.tree_deserialise_leaves(model_path / model_name, model_cfg)
    fcilr_t_avg = []
    fcilr_field_acc = []
    fcilr_field_acc_std = []
    fcilr_pot_acc = []
    fcilr_pot_acc_std = []

if normalized_time:
    data_n1 = read_db(f"{db_name}_{n_ensemble}_{norm_n}.h5")
    t_n1_model_list = []
    t_n1_fmm_list = []
    t_n1_mt_list = []

    for i in range(n_ensemble):
        # Function once to eliminate any overhead
        if plain_fmm_time:
            run_on_one_cpu(
                potential2D,
                cpu_id=0,
                sources=data_n1["sources"][i : i + 1],
                shape=data_n1["shape"],
                grid=data_n1["r"][i],
                correction_source=True,
                prism_mt=prism_mt,
            )
        else:
            potential2D(
                data_n1["sources"][i : i + 1],
                data_n1["shape"],
                data_n1["r"][i],
                correction_source=(not nocorrsource),
                prism_mt=prism_mt,
            )
        if model_eval:
            model(data_n1["sources"][i], data_n1["r"][i])
        field_mt(data_n1["sources"][i : i + 1], data_n1["r"][i], data_n1["shape"])

        # Run MagTense
        if data_n1["shape"] == "sphere":
            _, mt_dur = field_cylinder_exact(
                data_n1["sources"][i : i + 1], data_n1["r"][i], length=25
            )
        else:
            _, mt_dur = field_mt(
                data_n1["sources"][i : i + 1],
                data_n1["r"][i],
                data_n1["shape"],
            )
        t_n1_mt_list.append(mt_dur)

        if plain_fmm_time:
            _, _, dur = run_on_one_cpu(
                potential2D,
                cpu_id=0,
                sources=data_n1["sources"][i : i + 1],
                shape=data_n1["shape"],
                grid=data_n1["r"][i],
                correction_source=(not nocorrsource),
                prism_mt=prism_mt,
            )
            t_n1_fmm_list.append(dur)
        else:
            start_time_pot = time.time()
            _, _, dur = potential2D(
                data_n1["sources"][i : i + 1],
                data_n1["shape"],
                data_n1["r"][i],
                correction_source=True,
                prism_mt=prism_mt,
            )
            if fmm_nooverhead_time:
                t_n1_fmm_list.append(dur)
            else:
                t_n1_fmm_list.append(time.time() - start_time_pot)
        # print(f"t(FMM): {t_n1_fmm_list[-1]:.6f}")
        if model_eval:
            t_start_n1 = time.time()
            jax.block_until_ready(model(data_n1["sources"][i], data_n1["r"][i]))
            t_n1_model_list.append(time.time() - t_start_n1)

    t_n1_mt = np.mean(np.array(t_n1_mt_list))
    t_n1_fmm = np.mean(np.array(t_n1_fmm_list))
    if model_eval:
        t_n1_model = np.mean(np.array(t_n1_model_list))

else:
    t_n1_mt = 1.0
    t_n1_model = 1.0
    t_n1_fmm = 1.0

mt_acc = []
mt_acc_std = []
mt_t_avg = []
field_acc = []
field_acc_std = []
pot_acc = []
pot_acc_std = []
pot_t_avg = []
x_axis_ticks = []

if percentage_test:
    eval_list = [0, 0.1, 0.25, 0.5, 0.75, 0.99]
else:
    eval_list = [1]  # 10, 50, 250, 1000]  # range(n_eval + 1)

for n, p in enumerate(eval_list):
    if percentage_test:
        n_sources = 5
        x_axis_ticks.append(p)
    else:
        n_sources = p  # max(min_sources, step_sources * n)
        x_axis_ticks.append(n_sources)

    pot_out = []
    field_out = []
    mt_out = []
    fcilr_field_out = []
    fcilr_pot_out = []

    if percentage_test:
        db_filename = f"eval_large_m_p{int(p * 100)}_100_42_5_5.h5"
    else:
        db_filename = (
            "val_eval_large_m_92_1000_1.h5"  # f"{db_name}_{n_ensemble}_{n_sources}.h5"
        )
    data = read_db(db_filename)

    t_pot = np.zeros(n_ensemble)
    t_fcilr = np.zeros(n_ensemble)
    t_mt = np.zeros(n_ensemble)

    for i in range(n_ensemble):
        sources = data["sources"][:n_ensemble][i : i + 1]
        eval_loc = data["grid"] if grid_eval else data["r"][i]
        if percentage_test or nocorrsource or grid_eval:
            cor_source = False
        else:
            cor_source = True

        # Function once to eliminate any overhead
        if plain_fmm_time:
            run_on_one_cpu(
                potential2D,
                cpu_id=0,
                sources=sources,
                shape=data["shape"],
                grid=eval_loc,
                correction_source=cor_source,
                prism_mt=prism_mt,
            )
        else:
            potential2D(
                sources,
                data["shape"],
                eval_loc,
                correction_source=cor_source,
                prism_mt=prism_mt,
            )
        if model_eval:
            jax.vmap(model, in_axes=(0, None))(sources, eval_loc)[0]

        # Run model for potential
        if plain_fmm_time:
            msp_fmm, field_fmm, dur = run_on_one_cpu(
                potential2D,
                cpu_id=0,
                sources=sources,
                shape=data["shape"],
                grid=eval_loc,
                correction_source=cor_source,
                prism_mt=prism_mt,
            )
            t_pot[i] = dur / t_n1_fmm
        else:
            start_time_pot = time.time()
            msp_fmm, field_fmm, dur = potential2D(
                sources,
                data["shape"],
                eval_loc,
                correction_source=cor_source,
                prism_mt=prism_mt,
            )
            if fmm_nooverhead_time:
                t_pot[i] = dur / t_n1_fmm
            else:
                t_pot[i] = (time.time() - start_time_pot) / t_n1_fmm
        pot_out.append(msp_fmm[0])
        field_out.append(field_fmm[0])
        # (f"t(FMM): {t_pot[i]:.6f}")

        # Run MagTense
        if data["shape"] == "sphere":
            mt_h, mt_dur = field_cylinder_exact(sources, eval_loc, length=25)
        else:
            mt_h, mt_dur = field_mt(
                sources,
                eval_loc,
                data["shape"],
            )
        mt_out.append(mt_h[0])
        t_mt[i] = mt_dur / t_n1_mt

        if model_eval:
            # Run model for field
            fcilr_field_out.append(
                jax.vmap(model.field, in_axes=(0, None))(sources, eval_loc)[0]
            )

            # Run model for potential
            t_start = time.time()
            fcilr_pot_out.append(
                jax.block_until_ready(
                    jax.vmap(model, in_axes=(0, None))(sources, eval_loc)[0]
                )
            )
            t_fcilr[i] = (time.time() - t_start) / t_n1_model
            # print(f"FCILR: {t_fcilr[i]:.6f}")

    pot_t_avg.append(np.mean(t_pot))
    mt_t_avg.append(np.mean(t_mt))
    if model_eval:
        fcilr_t_avg.append(np.mean(t_fcilr))

    # Potential
    if grid_eval:
        diff_model_pot = data["msp_grid"][:n_ensemble] - np.array(pot_out)
        rel_err_pot = np.abs(diff_model_pot / (data["msp_grid"][:n_ensemble]))
    else:
        diff_model_pot = data["msp"] - np.array(pot_out)
        rel_err_pot = np.abs(diff_model_pot / (data["msp"]))

    if mean_eval:
        m_pot = np.nanmean(rel_err_pot, axis=-1)
    else:
        m_pot = np.nanmedian(rel_err_pot, axis=-1)
    # print(m_pot)
    pot_acc.append(np.mean(m_pot) * 100)
    pot_acc_std.append(np.std(m_pot) * 100)

    # Field
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
        if grid_eval:
            diff_mt = (
                data["field_grid"][:n_ensemble][..., :2] - np.array(mt_out)[..., :2]
            )
            rel_err_mt = np.linalg.norm(diff_mt, axis=-1) / np.linalg.norm(
                data["field_grid"][:n_ensemble][..., :2], axis=-1
            )
        else:
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
            diff_model_fcilr_pot = data["msp_grid"][:n_ensemble] - np.array(
                fcilr_pot_out
            )
            rel_err_fcilr_pot = np.abs(
                diff_model_fcilr_pot / (data["msp_grid"][:n_ensemble])
            )
        else:
            diff_model_fcilr_pot = data["msp"] - np.array(fcilr_pot_out)
            rel_err_fcilr_pot = np.abs(diff_model_fcilr_pot / (data["msp"]))

        if mean_eval:
            m_fcilr_pot = np.nanmean(rel_err_fcilr_pot, axis=-1)
        else:
            m_fcilr_pot = np.nanmedian(rel_err_fcilr_pot, axis=-1)
        # print(m_fcilr_pot)
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
    print("FMM:", pot_acc_std[-1])
    print("FCILR:", fcilr_pot_acc_std[-1])

### Plotting ###
fig, ax1 = plt.subplots()
fig.set_size_inches(12, 6)

# color = "tab:green"
if percentage_test:
    ax1.set_xlabel("Percentage of points inside sources (%)")
    x_ticks = np.linspace(0, 1, 11)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"{int(x * 100)}" for x in x_ticks])
    ax1.set_xlim(x_ticks[0] - 0.025, x_ticks[-1] + 0.025)
else:
    ax1.set_xlabel("Number of sources")

color = "tab:red"

if mean_eval:
    ax1.set_ylabel("Relative mean error (%)", color=color)
else:
    ax1.set_ylabel("Relative median error (%)", color=color)

if plot_pot_fmm:
    # Plot mean and standard deviation for errors
    if ax1_log:
        ax1.plot(
            x_axis_ticks,
            pot_acc,
            color=color,
            linestyle=":",
        )
    else:
        ax1.errorbar(
            x_axis_ticks,
            pot_acc,
            yerr=pot_acc_std,
            fmt="o",
            color=color,
            linestyle=":",
        )
    ax1.tick_params(axis="y", labelcolor=color)

if len(mt_acc_std) > 0 and plot_err_mt:
    ax1.errorbar(
        x_axis_ticks,
        mt_acc,
        yerr=mt_acc_std,
        fmt="o",
        color=color,
        linestyle="--",
    )

# Instantiate a third y-axis that shares the same x-axis
# ax4 = ax1.twinx()
# if plot_pot_fmm:
#     ax4.spines["left"].set_position(("axes", -0.1))
# else:
#     ax1.set_yticklabels([])
#     ax1.set_ylabel("")
#     ax1.set_yticks([])

# ax4.spines["left"].set_visible(True)
# ax4.yaxis.set_label_position("left")
# ax4.yaxis.set_ticks_position("left")
if ax1_log:
    ax1.set_yscale("log")
    ax1.set_ylim(1, 300)
else:
    if percentage_test:
        ax1.set_ylim(0, 5.5)
    else:
        ax1.set_ylim(0, 10)

ax1.tick_params(axis="y", labelcolor=color)
# if mean_eval:
#     ax4.set_ylabel("Relative mean error (%) - Field", color=color)
# else:
#     ax4.set_ylabel("Relative median error (%) - Potential", color=color)

# ax4.errorbar(
#     np.array(range(len(x_axis_ticks))) + 0.05,
#     field_acc,
#     yerr=field_acc_std,
#     fmt="o",
#     color=color,
#     linestyle=":",
# )

if model_eval:
    if ax1_log:
        ax1.plot(
            x_axis_ticks,
            fcilr_pot_acc,
            color=color,
            linestyle="-",
        )
    else:
        if percentage_test:
            added_offset = 0.02
        else:
            added_offset = 5
        ax1.errorbar(
            [x_ax_t + added_offset for x_ax_t in x_axis_ticks],
            fcilr_pot_acc,
            yerr=fcilr_pot_acc_std,
            fmt="o",
            color=color,
            linestyle="-",
        )

# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

color = "tab:blue"
if len(mt_t_avg) > 0:
    ax2.plot(
        x_axis_ticks,
        mt_t_avg,
        color=color,
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_yscale("log")
if normalized_time:
    ax2.set_ylabel("Normalized runtime", color=color)
    ax2.set_ylim(1, 1e4)
else:
    ax2.set_ylabel("Runtime (s)", color=color)
    ax2.set_ylim(0.5e-3, 20)

if model_eval:
    len_t = len(fcilr_t_avg)
    if fcilr_overhead_time:
        pass
        # fcilr_t_avg = [
        #     36e-3 + 9e-3,
        #     39e-3 + 9.5e-3,
        #     39e-3 + 10e-3,
        #     39.5e-3 + 10e-3,
        #     42e-3 + 9.5e-3,
        #     38.5e-3 + 10e-3,
        #     40e-3 + 9.3e-3,
        # ][:len_t]
    else:
        if normalized_time:
            if norm_n == 1:
                fcilr_t_avg = [
                    3.63,
                    34.27,
                    63.48,
                    95.07,
                    127.08,
                    158.52,
                    189.42,
                ][:len_t]
                # fcilr_t_avg = [
                #     1,
                #     25**2,
                #     50**2,
                #     75**2,
                #     100**2,
                #     125**2,
                #     150**2,
                # ][:len_t]
            elif norm_n == 10:
                fcilr_t_avg = [1.0, 9.44, 17.49, 26.19, 35.01, 43.67, 52.18][:len_t]
            else:
                raise ValueError("norm_n must be 1 or 10")
        else:
            fcilr_t_avg = [
                1.52e-3,
                14.31e-3,
                26.51e-3,
                39.7e-3,
                53.07e-3,
                66.2e-3,
                79.1e-3,
            ][:len_t]

    if len(fcilr_t_avg) > 0:
        # ax3 = ax1.twinx()
        # ax3.spines["right"].set_position(("axes", 1.12))
        # ax3.spines["right"].set_visible(True)
        # ax3.set_ylabel("Runtime - FCILR (ms)", color=color)
        # ax3.tick_params(axis="y", labelcolor=color)
        ax2.plot(
            x_axis_ticks,
            fcilr_t_avg,
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )

# Instantiate a third y-axis that shares the same x-axis
if plot_t_fmm:
    if grid_eval:
        ax2.set_ylabel("Runtime (s)", color=color)
        ax2.plot(x_axis_ticks, pot_t_avg, color=color, linestyle=":")
    else:
        # ax3 = ax1.twinx()
        # ax3.spines["right"].set_position(("axes", 1.1))
        # ax3.spines["right"].set_visible(True)
        # ax3.set_ylabel("Runtime - FMM (ms)", color=color)
        # ax3.tick_params(axis="y", labelcolor=color)
        # pot_t_avg = [
        #     1,
        #     25*2,
        #     50*2,
        #     75*2,
        #     100*2,
        #     125*2,
        #     150*2,
        # ][:len_t]

        ax2.plot(
            x_axis_ticks, pot_t_avg, color=color, linestyle=":", linewidth=2, alpha=0.7
        )

# Only display every second x tick
if not percentage_test:
    plt.xticks(x_axis_ticks[::2], x_axis_ticks[::2])

# Custom legend
legend_elements = [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="MagTense"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="FCILR")
    if model_eval
    else None,
    Line2D([0], [0], color="black", lw=2, linestyle=":", label="FMM")
    if plot_t_fmm
    else None,
]

# Remove None entries from legend_elements
legend_elements = [elem for elem in legend_elements if elem is not None]

plt.legend(handles=legend_elements, loc="upper left", fontsize=16)
fig.tight_layout()  # To ensure there's no overlap

plt.rcParams.update({"font.size": 16})
for ax in fig.get_axes():
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
    ax.title.set_fontsize(18)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
plt.tight_layout()

# Save the plot to the 'figs' directory
plt.savefig(figs_path / f"{fig_name}_{data['shape']}.svg")
plt.clf()
