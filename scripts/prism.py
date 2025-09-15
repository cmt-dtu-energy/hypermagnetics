import wandb
import optax
import equinox as eqx
from pathlib import Path
from hypermagnetics.sources import read_db
from hypermagnetics.models.hyper_mlp import HyperLayer
from hypermagnetics.measures import accuracy
from hypermagnetics.runner import fit


if __name__ == "__main__":
    config = {
        "shape": "prism",
        "n_samples": 50050,
        "lim": 1,
        "res": 32,
        "dim": 2,
        "epochs": 50,
        "width": 400,
        "depth": 3,
        "hwidth": 2,
        "hdepth": 3,
        "seed": 42,
        "lambda_field": 1,
        "batch_size": 200,
    }

    train = read_db("train_qt_dipole_res128_42_50050_1.h5")
    val = read_db("val_qt_dipole_res128_41_1020_10.h5")
    val_single = read_db("val_qt_dipole_res128_40_1020_1.h5")

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

    filepath = Path(__file__).parent / ".." / "models"
    eqx.tree_serialise_leaves(filepath / "fc_ilr_400_200k_dipole_correction_res128.eqx", model)
