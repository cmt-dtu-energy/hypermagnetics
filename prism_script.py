import optax
import equinox as eqx
from pathlib import Path
import h5py
import jax.numpy as jnp
from hypermagnetics.sources import configure
from hypermagnetics.models.hyper_fourier import FourierModel
from hypermagnetics.models.hyper_mlp import HyperLayer
from hypermagnetics.measures import loss, accuracy
from hypermagnetics.runner import fit


def read_db(filename: str):
    datapath = Path("/home/spol/Documents/repos/hypermagnetics/data")
    db = h5py.File(datapath / filename, "r")
    data = {
        "sources": jnp.concatenate([db["m"][:], db["r0"][:], db["size"][:]], axis=-1),
        "r": jnp.array(db["r"][:]),
        "potential": jnp.array(db["potential"][:]),
        "field": jnp.array(db["field"][:]),
        "grid": jnp.array(db["grid"][:]),
        "potential_grid": jnp.array(db["potential_grid"][:]),
        "field_grid": jnp.array(db["field_grid"][:]),
    }
    db.close()

    return data


if __name__ == "__main__":
    # source_config = {
    #     "shape": "prism",
    #     "n_samples": 1000,
    #     "lim": 3,
    #     "res": 32,
    #     "dim": 3,
    # }
    epochs = 2000
    filepath = Path("/home/spol/Documents/repos/hypermagnetics/")
    # train = configure(**source_config, n_sources=1, seed=100)
    # val = configure(**source_config, n_sources=4, seed=101)

    train = read_db("100_10000_train.h5")
    val = read_db("101_1000_val.h5")

    # model = FourierModel(32, hwidth=0.25, hdepth=3, seed=42)
    model = HyperLayer(width=400, depth=3, hwidth=1, hdepth=2, seed=42)

    schedule = [
        {"optim": optax.adam, "epochs": epochs, "params": {"learning_rate": 1e-2}},
        {"optim": optax.adam, "epochs": epochs, "params": {"learning_rate": 1e-3}},
        {"optim": optax.adam, "epochs": epochs, "params": {"learning_rate": 1e-4}},
        {"optim": optax.adam, "epochs": epochs, "params": {"learning_rate": 1e-5}},
    ]

    for trainer_config in schedule:
        optim = trainer_config["optim"](**trainer_config["params"])
        model = fit(trainer_config, optim, model, train, val, every=100)

    eqx.tree_serialise_leaves(filepath / "models" / "ic_inr_400_10k_samples.eqx", model)
