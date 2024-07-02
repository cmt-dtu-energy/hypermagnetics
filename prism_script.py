import wandb
import optax
import equinox as eqx
from pathlib import Path
import jax.numpy as jnp
from hypermagnetics.sources import configure, read_db
from hypermagnetics.models.hyper_fourier import FourierModel
from hypermagnetics.models.hyper_mlp import HyperLayer
from hypermagnetics.measures import loss, accuracy
from hypermagnetics.runner import fit


if __name__ == "__main__":
    config = {
        "shape": "prism",
        "n_samples": 200200,
        "lim": 2.4,
        "res": 32,
        "dim": 3,
        "epochs": 75,
        "width": 400,
        "depth": 3,
        "hwidth": 2,
        "hdepth": 3,
        "seed": 42,
        "lambda_field": 0.25,
        "batch_size": 800,
    }

    # train = configure(**source_config, n_sources=3, seed=110)
    # val = configure(**source_config, n_sources=4, seed=101)
    # val_single = configure(**source_config, n_sources=1, seed=102)

    train = read_db(f"100_{config['n_samples']}_train.h5")
    val = read_db("101_1010_val_lim.h5")
    val_single = read_db("102_1010_val_lim_single.h5")

    # model = FourierModel(32, hwidth=0.25, hdepth=3, seed=42)
    model = HyperLayer(
        width=config["width"],
        depth=config["depth"],
        hwidth=config["hwidth"],
        hdepth=config["hdepth"],
        seed=config["seed"],
    )

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
            "epochs": config["epochs"] * 2,
            "params": {"learning_rate": 1e-3},
        },
        {
            "optim": optax.adam,
            "epochs": config["epochs"] * 2,
            "params": {"learning_rate": 1e-4},
        },
        # {
        #     "optim": optax.adam,
        #     "epochs": config["epochs"],
        #     "params": {"learning_rate": 1e-5},
        # },
    ]

    for trainer_config in schedule:
        optim = trainer_config["optim"](**trainer_config["params"])
        model = fit(
            trainer_config,
            optim,
            model,
            train,
            val,
            log=wandb.log,
            every=10,
            batch_size=config["batch_size"],
            lambda_field=config["lambda_field"],
        )

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
    eqx.tree_serialise_leaves(
        filepath / "models" / "ic_inr_400_200k_fcinr_lim_uniform.eqx", model
    )
