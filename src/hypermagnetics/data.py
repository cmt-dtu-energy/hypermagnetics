import numpy as np

from hypermagnetics.sources import configure


def fno_input_converter(
    sources: np.ndarray,
    shape: str = "sphere",
    dim: int = 2,
    lim: int = 3,
    res: int = 32,
):
    """Convert source parameters to FNO input format."""
    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, _ = r0.shape

    grid_x = np.linspace(-lim, lim, res)
    grid_y = np.linspace(-lim, lim, res)
    if dim > 2:
        grid_z = np.linspace(-lim, lim, res)
        grids = np.meshgrid(grid_x, grid_y, grid_z, indexing="xy")
    else:
        grids = np.meshgrid(grid_x, grid_y, indexing="xy")

    grid = np.concatenate([g.ravel()[:, None] for g in grids], axis=-1)
    input_data = np.zeros((n_samples, res**dim, dim))

    for i in range(n_samples):
        for n in range(n_sources):
            if shape == "sphere":
                idx_in = np.where(
                    np.linalg.norm(grid - r0[i][n], axis=1) <= size[i, n, 0]
                )[0]
            elif shape == "prism":
                idx_in = np.where(
                    (np.abs(grid[:, 0] - r0[i, n, 0]) <= size[i, n, 0])
                    & (np.abs(grid[:, 1] - r0[i, n, 1]) <= size[i, n, 1])
                )[0]
            else:
                raise ValueError("Unknown shape")

            input_data[i][idx_in] += m[i, n, :dim]

    return input_data


def create_data(
    n_eval: int = 8,
    n_ensemble: int = 5,
    min_sources: int = 10,
    step_sources: int = 250,
    field_eval: bool = False,
    grid_eval: bool = False,
    seed: int = 42,
    shape: str = "prism",
    quadtree: bool = True,
    name: str = "eval",
    res: int = 32,
    lim: float = 1.0,
    eps: float = 0.0,
    max_size: float = 0.5,
    min_size: float = 0.05,
    batch_size: int = 1000,
    dp_correction: bool = True,
    size_log: bool = True,
    r0_gap: bool = False,
    target_source: bool = False,
    p_target_source: float = 1.0,
    start_idx: int = 0,
):
    for n in range(start_idx, n_eval + 1):
        source_config = {
            "shape": shape,
            "n_samples": n_ensemble,
            "n_sources": max(min_sources, step_sources * n),
            "lim": lim,
            "res": res,
            "target_source": target_source,
            "p_target_source": p_target_source,
            "eps": eps,
            "grid_eval": grid_eval,
            "field_eval": field_eval,
            "quadtree": quadtree,
            "save_data": True,
            "db_prefix": name,
            "seed": seed,
            "min_size": min_size,
            "max_size": max_size,
            "size_log": size_log,
            "r0_gap": r0_gap,
            "batch_size": batch_size,
            "dipole_correction": dp_correction,
        }
        configure(**source_config)


if __name__ == "__main__":
    res = 32
    lim = 1.2
    # for p in [0, 0.1, 0.25, 0.5, 0.75, 0.99]:
    name = "eval_large_m_qt"
    create_data(
        n_eval=0,
        n_ensemble=10,
        min_sources=50,
        step_sources=250,
        field_eval=True,
        name=f"{name}",
        shape="prism",
        quadtree=True,
        grid_eval=True,
        res=res,
        lim=lim,
        eps=0.0,
        max_size=0.48,
        min_size=0.12,
        batch_size=1,
        seed=42,
        dp_correction=False,
        size_log=False,
        r0_gap=True,
        target_source=True,
        start_idx=0,
    )

    # create_data(
    #     n_eval=0,
    #     n_ensemble=1000,
    #     min_sources=1,
    #     step_sources=0,
    #     field_eval=True,
    #     name=f"val_{name}",
    #     shape="sphere",
    #     quadtree=False,
    #     grid_eval=True,
    #     res=res,
    #     lim=lim,
    #     eps=0.0,
    #     max_size=0.48,
    #     min_size=0.12,
    #     batch_size=30,
    #     seed=92,
    #     dp_correction=False,
    #     size_log=False,
    #     r0_gap=True,
    #     target_source=True,
    # )

    # create_data(
    #     n_eval=0,
    #     n_ensemble=1000,
    #     min_sources=10,
    #     step_sources=0,
    #     field_eval=True,
    #     name=f"val_{name}",
    #     shape="prism",
    #     quadtree=True,
    #     grid_eval=True,
    #     res=res,
    #     lim=lim,
    #     eps=0.0,
    #     max_size=0.5,
    #     min_size=0.1,
    #     batch_size=30,
    #     seed=41,
    #     dp_correction=False,
    # )

    # create_data(
    #     n_eval=0,
    #     n_ensemble=200000,
    #     min_sources=1,
    #     step_sources=0,
    #     field_eval=True,
    #     name=f"train_{name}",
    #     shape="prism",
    #     quadtree=False,
    #     grid_eval=True,
    #     res=res,
    #     lim=lim,
    #     eps=0.0,
    #     max_size=0.5,
    #     min_size=0.05,
    #     batch_size=40,
    #     seed=42,
    #     dp_correction=False,
    #     r0_gap=True,
    #     size_log=False,
    # )

    # import h5py
    # from pathlib import Path

    # db_name = "train_res32_large_m_42_200000_1"
    # datapath = Path(__file__).parent / ".." / ".." / "data"

    # db_orig = h5py.File(datapath / f"{db_name}.h5", "r")
    # db = h5py.File(datapath / f"{db_name}_fno.h5", "w")
    # data = {
    #     "sources": np.concatenate(
    #         [db_orig["m"][:], db_orig["r0"][:], db_orig["size"][:]],
    #         axis=-1,
    #     ),
    #     "msp_grid": np.array(db_orig["msp_grid"][:]),
    # }
    # input = fno_input_converter(data["sources"], shape="prism", res=32, lim=1.2)
    # db.create_dataset("input", shape=input.shape, dtype="float32")
    # db.create_dataset("output", shape=data["msp_grid"].shape, dtype="float32")
    # print(input.shape, data["msp_grid"].shape)
    # db["input"][:] = input.astype("float32")
    # db["output"][:] = data["msp_grid"].astype("float32")
    # db.close()
    # print("FNO data created.")
