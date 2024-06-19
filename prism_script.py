import optax
from hypermagnetics.sources import configure
from hypermagnetics.models.hyper_fourier import FourierModel
from hypermagnetics.runner import fit

if __name__ == "__main__":
    source_config = {
        "shape": "prism",
        "n_samples": 10000,
        "lim": 3,
        "res": 32,
        "dim": 3,
    }
    train = configure(**source_config, n_sources=1, seed=100)
    val = configure(**source_config, n_sources=4, seed=101)

    model = FourierModel(32, hwidth=0.25, hdepth=3, seed=42)

    schedule = [
        {"optim": optax.adam, "epochs": 5000, "params": {"learning_rate": 1e-2}},
        {"optim": optax.adam, "epochs": 5000, "params": {"learning_rate": 1e-3}},
        {"optim": optax.adam, "epochs": 5000, "params": {"learning_rate": 1e-4}},
        {"optim": optax.adam, "epochs": 5000, "params": {"learning_rate": 1e-5}},
    ]

    for trainer_config in schedule:
        optim = trainer_config["optim"](**trainer_config["params"])
        model = fit(trainer_config, optim, model, train, val, every=1000)
        # print(model.k)
