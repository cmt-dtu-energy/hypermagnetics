from hypermagnetics.sources import configure


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
    name: str = "eval_",
    lim: float = 1.0,
):
    for n in range(5, n_eval + 1):
        source_config = {
            "shape": shape,
            "n_samples": n_ensemble,
            "n_sources": max(min_sources, step_sources * n),
            "lim": lim,
            "res": 32,
            "t_source": True,
            "eps": 1e-6,
            "grid_eval": grid_eval,
            "field_eval": field_eval,
            "quadtree": quadtree,
            "save_data": True,
            "db_prefix": name,
            "seed": seed,
            "min_size": 0.05,
            "max_size": 0.5,
        }
        configure(**source_config)


if __name__ == "__main__":
    create_data(
        n_eval=8,
        field_eval=True,
        name="eval_qt_",
        shape="prism",
        quadtree=True,
        grid_eval=False,
        lim=1.0,
        seed=42,
    )
