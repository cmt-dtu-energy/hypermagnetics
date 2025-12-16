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
        "n_samples": 200000,
        "lim": 1.25,
        "res": 32,
        "dim": 3,
        "epochs": 20,
        "width": 400,
        "depth": 3,
        "hwidth": 2,
        "hdepth": 3,
        "seed": 42,
        "lambda_field": 0.25,
        "batch_size": 15,
    }

    run_name = "res32_3D"
    train_name = f"train_{run_name}_42_{config['n_samples']}_1.h5"
    val = read_db(f"val_{run_name}_41_1000_10.h5")
    val_single = read_db(f"val_{run_name}_40_1000_1.h5")

    # model = FourierModel(32, hwidth=0.25, hdepth=3, seed=42)
    model = HyperLayer(
        width=config["width"],
        depth=config["depth"],
        hwidth=config["hwidth"],
        hdepth=config["hdepth"],
        seed=config["seed"],
        dim=config["dim"],
        in_size=config["dim"],
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
            "epochs": 2 * config["epochs"],
            "params": {"learning_rate": 1e-3},
        },
        {
            "optim": optax.adam,
            "epochs": config["epochs"] // 2,
            "params": {"learning_rate": 1e-4},
        },
        {
            "optim": optax.adam,
            "epochs": config["epochs"] // 2,
            "params": {"learning_rate": 1e-5},
        },
    ]

    for trainer_config in schedule:
        optim = trainer_config["optim"](**trainer_config["params"])
        model = fit(
            trainer_config,
            optim,
            model,
            train_name,
            val,
            log=wandb.log,
            every=10,
            batch_size=config["batch_size"],
            lambda_field=config["lambda_field"],
            n_samples=config["n_samples"],
        )

    train_err = accuracy(model, read_db(train_name, config["batch_size"]))
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
    eqx.tree_serialise_leaves(filepath / f"fc_ilr_400_200k_{run_name}.eqx", model)
