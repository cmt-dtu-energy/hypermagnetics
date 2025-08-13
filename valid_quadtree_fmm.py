import numpy as np
import prettytable

from hypermagnetics.fmm_sources import potential2D, potential2D_sources
from hypermagnetics.sources import _total, _potential, read_db

n_ensemble = 5
n_sources = 500
data = read_db(f"eval_42_5_{n_sources}.h5")
m, r0, size = np.split(data["sources"], 3, axis=-1)

pot_out = np.zeros((n_ensemble, n_sources))
pot_single_out = np.zeros((n_ensemble, n_sources, n_sources))
target_single_out = np.zeros((n_ensemble, n_sources, n_sources))

for i in range(n_ensemble):
    # Run model for potential
    msp_fmm, _ = potential2D_sources(data["sources"][i : i + 1], "prism")
    pot_out[i] = msp_fmm

    for n in range(n_sources):
        msp_fmm, _ = potential2D(
            data["sources"][i : i + 1, n : n + 1],
            r0[i, ..., :2],
            "prism",
            correction=False,
        )
        pot_single_out[i, n] = msp_fmm
        target_single_out[i, n] = _total(
            _potential,
            data["sources"][
                i : i + 1,
                n : n + 1,
            ],
            r0[i],
            "prism",
        )

assert np.allclose(pot_out, np.sum(pot_single_out, axis=1), rtol=1e-8, atol=1e-5)

# Potential
rel_err_pot = np.abs((data["msp"] - pot_out) / data["msp"])
median_pot = np.nanmedian(rel_err_pot, axis=-1)


# Potential from individual sources
rel_err_pot_single = np.abs((target_single_out - pot_single_out) / target_single_out)
median_pot_single = np.nanmedian(rel_err_pot_single, axis=-1)

res_table = prettytable.PrettyTable()
res_table.field_names = [
    "#Sources",
    "Potential error - Mean [%]",
    "Potential error - Std [%]",
    "Single error - Mean [%]",
    "Single error - Std [%]",
]

res_table.add_row(
    [
        int(n_sources),
        float(np.mean(median_pot) * 100),
        float(np.std(median_pot) * 100),
        float(np.mean(median_pot_single) * 100),
        float(np.std(median_pot_single) * 100),
    ]
)
res_table.float_format = "5.4"
print(res_table)
