from hypermagnetics.sources import configure


def create_data(
    n_eval: int = 8,
    n_ensemble: int = 10,
    min_sources: int = 10,
    step_sources: int = 250,
    field_eval: bool = False,
    seed: int = 42,
    shape: str = "prism",
    quadtree: bool = True,
    name: str = "eval_",
):
    for n in range(n_eval + 1):
        source_config = {
            "shape": shape,
            "n_samples": n_ensemble,
            "n_sources": max(min_sources, step_sources * n),
            "lim": 1,
            "res": 32,
            "t_source": True,
            "eps": 0,
            "grid_eval": False,
            "field_eval": field_eval,
            "quadtree": quadtree,
            "save_data": True,
            "db_prefix": name,
            "seed": seed,
        }
        configure(**source_config)


if __name__ == "__main__":
    create_data(field_eval=True)
