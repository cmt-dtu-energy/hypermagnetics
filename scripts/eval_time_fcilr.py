import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import equinox as eqx
import jax

from hypermagnetics.sources import read_db
from hypermagnetics.models.hyper_mlp import HyperLayer

warnings.filterwarnings("ignore")

"""
Profiling with jax (jax.vmap - hyper_mlp.py:74 44.1 ms)
"""


def profile_model(
    n_ensemble: int = 5,
    n_eval: int = 10,
    db_name: str = "eval_large_m_42",
    model_path: Path = Path(__file__).parent / ".." / "models",
    tmp_path: Path = Path(__file__).parent / ".." / "tmp",
    model_name: str = "fc_ilr_400_200k_res32_large_m.eqx",
):
    model_cfg = HyperLayer(
        width=400,
        depth=3,
        hwidth=2,
        hdepth=3,
        seed=42,
    )
    model = eqx.tree_deserialise_leaves(model_path / model_name, model_cfg)

    for n in [5]:  # [1, 10, 250, 500, 750, 1000, 1250, 1500]:
        data_n = read_db(
            f"eval_large_m_p{int(0.1 * 100)}_100_42_5_5.h5"
        )  # read_db(f"{db_name}_{n_ensemble}_{n}.h5")
        for i in range(n_eval):
            # Call function once to eliminate any overhead)
            model(data_n["sources"][i], data_n["r"][i])
            options = jax.profiler.ProfileOptions()
            options.advanced_configuration = {
                "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"
            }
            with jax.profiler.trace(tmp_path / f"jax_trace_{n}"):
                jax.block_until_ready(model(data_n["sources"][i], data_n["r"][i]))


def make_res_df(norm_n: int = 1):
    # profile_model()
    results_df = pd.DataFrame(columns=[1, 10, 250, 500, 750, 1000, 1250, 1500])

    # Hypernetwork - [ms]
    results_df.at["g", 1] = np.array(
        [0.375, 0.364, 0.4, 0.4, 0.359, 0.51, 0.41, 0.358, 0.5, 0.5]
    ).mean()
    results_df.at["g", 10] = np.array(
        [1.15, 1.12, 1.07, 1.18, 1.12, 1.14, 1.39, 1.21, 1.18, 1.29]
    ).mean()
    results_df.at["g", 250] = np.array(
        [7.0, 6.78, 7.64, 6.9, 6.8, 7.0, 7.0, 8.1, 7.4, 6.9]
    ).mean()
    results_df.at["g", 500] = np.array(
        [13.4, 13.5, 13.3, 13.8, 13.5, 12.3, 13.6, 13.8, 13.9, 13.8]
    ).mean()
    results_df.at["g", 750] = np.array(
        [20.7, 20.7, 20.8, 20.8, 20.6, 18.6, 21.1, 20.8, 20.6, 21]
    ).mean()
    results_df.at["g", 1000] = np.array(
        [27.7, 27.7, 28.9, 28.5, 28.1, 27.9, 27.5, 27.9, 27.9, 27.7]
    ).mean()
    results_df.at["g", 1250] = np.array(
        [34.7, 35.1, 34.7, 34, 34.1, 33.9, 33.6, 30.9, 34.1, 34.4]
    ).mean()
    results_df.at["g", 1500] = np.array(
        [41.5, 36.9, 42, 41.8, 42.6, 41.9, 41.9, 41.5, 41.1, 41.8]
    ).mean()

    # FC-ILR - [ms]
    results_df.at["f", 1] = np.array([0]).mean()
    results_df.at["f", 10] = np.array(
        [0.22, 0.36, 0.35, 0.22, 0.35, 0.35, 0.36, 0.38, 0.35, 0.37]
    ).mean()
    results_df.at["f", 250] = np.array(
        [6.5, 6.6, 6.4, 6.3, 6.7, 6.8, 6.3, 7.4, 12.4, 6.2]
    ).mean()
    results_df.at["f", 500] = np.array(
        [12.7, 12.3, 12.9, 13.7, 13.1, 12.9, 13.1, 13.0, 12.8, 13.7]
    ).mean()
    results_df.at["f", 750] = np.array(
        [19, 19.5, 19.4, 18.8, 19.1, 18.7, 19.2, 19.0, 19.3, 19.3]
    ).mean()
    results_df.at["f", 1000] = np.array(
        [25.3, 25.1, 25.1, 25.1, 25, 25.1, 24.8, 25, 25.2, 25.2]
    ).mean()
    results_df.at["f", 1250] = np.array(
        [32.1, 32.1, 31.6, 31, 31, 30.5, 31.2, 38.9, 32.2, 31.9]
    ).mean()
    results_df.at["f", 1500] = np.array(
        [38.2, 35.2, 37.6, 38, 38.7, 38.6, 37.9, 37.9, 37.9, 38]
    ).mean()

    results_df.loc["sum"] = [
        float(g) + float(f) for g, f in zip(results_df.loc["g"], results_df.loc["f"])
    ]
    results_df.loc["normed"] = [
        float(sum) / float(results_df.at["sum", norm_n])
        for sum in results_df.loc["sum"]
    ]

    return results_df


if __name__ == "__main__":
    # df = make_res_df(norm_n=10)
    # print(df)
    profile_model(n_eval=3, tmp_path=Path(__file__).parent / ".." / "tmp_tile")
    print("Profiling complete.")
