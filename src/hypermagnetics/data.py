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
    name: str = "eval",
    lim: float = 1.0,
    eps: float = 0.0,
    max_size: float = 0.5,
    min_size: float = 0.05,
    batch_size: int = 1000,
):
    for n in range(n_eval + 1):
        source_config = {
            "shape": shape,
            "n_samples": n_ensemble,
            "n_sources": max(min_sources, step_sources * n),
            "lim": lim,
            "res": 32,
            "target_source": False,
            "eps": eps,
            "grid_eval": grid_eval,
            "field_eval": field_eval,
            "quadtree": quadtree,
            "save_data": True,
            "db_prefix": name,
            "seed": seed,
            "min_size": min_size,
            "max_size": max_size,
            "batch_size": batch_size,
            "dipole_correction": True,
        }
        configure(**source_config)


if __name__ == "__main__":
    create_data(
        n_eval=0,
        n_ensemble=50050,
        min_sources=1,
        step_sources=0,
        field_eval=True,
        name="train_qt_dipole",
        shape="prism",
        quadtree=False,
        grid_eval=True,
        lim=1.0,
        eps=0.0,
        max_size=0.5,
        min_size=0.005,
        batch_size=10,
        seed=42,
    )
