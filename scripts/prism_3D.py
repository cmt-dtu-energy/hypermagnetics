import wandb
import optax
import equinox as eqx
import jax.random as jr
from pathlib import Path
from hypermagnetics.sources import read_db_batch
from hypermagnetics.models.hyper_mlp import HyperLayer
from hypermagnetics.measures import accuracy
from hypermagnetics.runner import fit


if __name__ == "__main__":
    config = {
        "shape": "prism",
        "n_samples": 120000,
        "lim": 1.25,
        "res": 32,
        "dim": 3,
        "epochs": 50,
        "width": 800,
        "depth": 3,
        "hwidth": 3,
        "hdepth": 3,
        "seed": 42,
        "lambda_field": 0.25,
        "batch_size": 400,
    }

    run_name = f"res{config['res']}_{config['dim']}D"
    train_name = f"train_{run_name}_42_{config['n_samples']}_1.h5"
    val, _ = read_db_batch(
        jr.PRNGKey(41),
        f"val_{run_name}_41_1000_10.h5",
        0,
        config["batch_size"],
        config["res"],
    )
    val_single, _ = read_db_batch(
        jr.PRNGKey(40),
        f"val_{run_name}_40_1000_1.h5",
        0,
        config["batch_size"],
        config["res"],
    )

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
            val_single,
            log=wandb.log,
            every=10,
            batch_size=config["batch_size"],
            lambda_field=config["lambda_field"],
            n_samples=config["n_samples"],
            res=config["res"],
            seed=config["seed"],
        )

    filepath = Path(__file__).parent / ".." / "models"
    eqx.tree_serialise_leaves(
        filepath
        / f"fc_ilr_{config['width']}_{config['n_samples'] // 1000}k_{run_name}.eqx",
        model,
    )

    train_err = accuracy(
        model,
        read_db_batch(
            jr.PRNGKey(0),
            train_name,
            0,
            config["batch_size"],
            config["res"],
        )[0],
    )
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
