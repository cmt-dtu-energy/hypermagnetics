import wandb
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
    config = {
        "shape": "prism",
        "n_samples": 101000,
        "lim": 3,
        "res": 32,
        "dim": 3,
        "epochs": 100,
        "width": 400,
        "depth": 3,
        "hwidth": 1.5,
        "hdepth": 3,
        "seed": 42,
    }

    # train = configure(**source_config, n_sources=3, seed=110)
    # val = configure(**source_config, n_sources=4, seed=101)
    # val_single = configure(**source_config, n_sources=1, seed=102)

    train = read_db(f"100_{config['n_samples']}_train.h5")
    val = read_db("101_1000_val.h5")
    val_single = read_db("102_1000_val_single.h5")

    # model = FourierModel(32, hwidth=0.25, hdepth=3, seed=42)
    model = HyperLayer(width=400, depth=3, hwidth=1.5, hdepth=3, seed=42)

    wandb.init(
        entity="dl4mag",
        project="hypermagnetics",
        config=config,
    )
    wandb.log({"nparams": model.nparams})

    schedule = [
        {
            "optim": optax.adam,
            "epochs": config["epochs"],
            "params": {"learning_rate": 1e-2},
        },
        {
            "optim": optax.adam,
            "epochs": config["epochs"],
            "params": {"learning_rate": 1e-3},
        },
        {
            "optim": optax.adam,
            "epochs": config["epochs"],
            "params": {"learning_rate": 1e-4},
        },
        {
            "optim": optax.adam,
            "epochs": config["epochs"],
            "params": {"learning_rate": 1e-5},
        },
    ]

    for trainer_config in schedule:
        optim = trainer_config["optim"](**trainer_config["params"])
        model = fit(trainer_config, optim, model, train, val, log=wandb.log, every=10)

    train_err = accuracy(model, train)
    val_single_err = accuracy(model, val_single)
    val_multi_err = accuracy(model, val)
    wandb.log(
        {
            "train_err": train_err.item(),
            "val_single_err": val_single_err.item(),
            "val_multi_err": val_multi_err.item(),
        }
    )

    wandb.finish()

    filepath = Path("/home/spol/Documents/repos/hypermagnetics/")
    eqx.tree_serialise_leaves(filepath / "models" / "ic_inr_400_101k_fcinr.eqx", model)
