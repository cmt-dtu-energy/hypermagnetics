import numpy as np
import matplotlib.pyplot as plt
import prettytable

from hypermagnetics.fmm_sources import potential2D
from hypermagnetics.mt_eval import field_cylinder_exact, field_mt
from hypermagnetics.sources import _field, _total, _potential, read_db

n_ensemble = 5
n_sources = 10
data = read_db(f"eval_qt_exact_42_{n_ensemble}_{n_sources}.h5")

pot_out = np.zeros((n_ensemble, n_sources))
field_out = np.zeros((n_ensemble, n_sources, 2))
field_mt_out = np.zeros((n_ensemble, n_sources, 2))
pot_single_out = np.zeros((n_ensemble, n_sources, n_sources))
target_single_out = np.zeros((n_ensemble, n_sources, n_sources))
field_single_out = np.zeros((n_ensemble, n_sources, n_sources, 2))
target_field_single_out = np.zeros((n_ensemble, n_sources, n_sources, 2))

target_field_mt = True
plot_single_corr = False

for i in range(n_ensemble):
    # Run model for potential
    msp_fmm, field_fmm = potential2D(
        data["sources"][i : i + 1],
        data["shape"],
        data["r"][i],
        correction=False,
        correction_source=True,
    )
    pot_out[i] = msp_fmm
    field_out[i] = field_fmm

    # Run MagTense
    if data["shape"] == "sphere":
        mt_h, mt_dur = field_cylinder_exact(
            data["sources"][i : i + 1], data["r"][i], length=25
        )
    else:
        mt_h, mt_dur = field_mt(
            data["sources"][i : i + 1],
            data["r"][i] if data["target_source"] else data["r"],
            data["shape"],
        )
    field_mt_out[i] = mt_h[..., :2]

    for n in range(n_sources):
        msp_fmm, field_fmm = potential2D(
            data["sources"][i : i + 1, n : n + 1],
            data["shape"],
            data["r"][i],
            correction=False,
            correction_source=True,
            idx_single=n,
        )
        pot_single_out[i, n] = msp_fmm
        field_single_out[i, n] = field_fmm

        target_single_out[i, n] = _total(
            _potential,
            data["sources"][
                i : i + 1,
                n : n + 1,
            ].astype(np.float64),
            data["r"][i].astype(np.float64),
            data["shape"],
        )

        if target_field_mt:
            # Run MagTense
            if data["shape"] == "sphere":
                mt_h, mt_dur = field_cylinder_exact(
                    data["sources"][i : i + 1, n : n + 1], data["r"][i], length=25
                )
            else:
                mt_h, mt_dur = field_mt(
                    data["sources"][i : i + 1, n : n + 1],
                    data["r"][i] if data["target_source"] else data["r"],
                    data["shape"],
                )

            target_field_single_out[i, n] = mt_h[..., :2]
        else:
            target_field_single_out[i, n] = _total(
                _field,
                data["sources"][
                    i : i + 1,
                    n : n + 1,
                ].astype(np.float64),
                data["r"][i].astype(np.float64),
                data["shape"],
            )[..., :2]

assert np.allclose(pot_out, np.sum(pot_single_out, axis=1), rtol=1e-8, atol=1e-5)

# Potential
rel_err_pot = np.abs((data["msp"] - pot_out) / data["msp"])
median_pot = np.nanmedian(rel_err_pot, axis=-1)

# Field
diff_model = field_mt_out - np.array(field_out)[..., :2]
rel_err_field = np.linalg.norm(diff_model, axis=-1) / np.linalg.norm(
    field_mt_out, axis=-1
)
median_field = np.nanmedian(rel_err_field, axis=-1)

# Potential from individual sources
rel_err_pot_single = np.abs((target_single_out - pot_single_out) / target_single_out)
median_pot_single = np.nanmedian(rel_err_pot_single, axis=-1)

# Field from individual sources
rel_err_field_single = np.abs(
    (target_field_single_out - field_single_out) / target_field_single_out
)
median_field_single = np.nanmedian(rel_err_field_single, axis=-1)


if plot_single_corr:
    corr_field_single = np.mean(rel_err_field_single, axis=-1)[0] * 100
    corr_pot_single = rel_err_pot_single[0] * 100
    plt.imshow(np.clip(corr_field_single, 0, 40), cmap="viridis")
    # plt.imshow(np.clip(corr_pot_single, 0, 10), cmap="viridis")
    plt.colorbar(label="Relative Error")
    plt.title("Relative Error of Field from Single Sources")
    plt.xlabel("Source Index")
    plt.ylabel("Target Index")
    plt.xticks(np.arange(n_sources), np.arange(n_sources))
    plt.yticks(np.arange(n_sources), np.arange(n_sources))
    # Save the plot to the 'figs' directory
    plt.savefig("/home/spol/Documents/repos/hypermagnetics/figs/corr_fmm_single.svg")


res_table = prettytable.PrettyTable()
res_table.field_names = [
    "#Sources",
    "Potential error - Mean [%]",
    "Potential error - Std [%]",
    "Field error - Mean [%]",
    "Single error - Mean [%]",
    "Single error - Std [%]",
    "Single field error - Mean [%]",
]

res_table.add_row(
    [
        int(n_sources),
        float(np.mean(median_pot) * 100),
        float(np.std(median_pot) * 100),
        float(np.mean(median_field) * 100),
        float(np.mean(median_pot_single) * 100),
        float(np.std(median_pot_single) * 100),
        float(np.mean(median_field_single) * 100),
    ]
)
res_table.float_format = "5.4"
print(res_table)
