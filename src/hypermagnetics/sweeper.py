import matplotlib.pyplot as plt
import optax
import scienceplots  # noqa
import yaml

import hypermagnetics.sources as sources
import wandb
from hypermagnetics import plots
from hypermagnetics.measures import accuracy
from hypermagnetics.models.hyper_fourier import FourierModel  # noqa
from hypermagnetics.models.hyper_mlp import HyperLayer, HyperMLP  # noqa
from hypermagnetics.runner import fit

plt.style.use(["science", "ieee"])

with open("config/sweep-configuration.yaml", "r") as f:
    sweep_configuration = yaml.safe_load(f)

sweep_id = wandb.sweep(
    sweep=sweep_configuration, entity="dl4mag", project="hypermagnetics"
)


def main():
    wandb.init()

    source_config = sweep_configuration["sources"]
    train = sources.configure(**source_config["train"])
    test = sources.configure(**source_config["test"])
    val_single = sources.configure(**source_config["val-single"])
    val_multi = sources.configure(**source_config["val-multi"])

    model_config = sweep_configuration["model"]
    # model = FourierModel(**model_config["fourier"])
    model = HyperLayer(**model_config["hyperlayer"])
    # model = HyperMLP(**model_config["hypernetwork"])
    wandb.log({"nparams": model.nparams})

    schedule = sweep_configuration["schedule"]
    for trainer_config in schedule:
        optim = optax.adam(**trainer_config["params"])
        model = fit(trainer_config, optim, model, train, test, log=wandb.log, every=10)

    train_err = accuracy(model, train)
    val_single_err = accuracy(model, val_single)
    val_multi_err = accuracy(model, val_multi)

    wandb.log(
        {
            "train_err": train_err.item(),
            "val_single_err": val_single_err.item(),
            "val_multi_err": val_multi_err.item(),
        }
    )

    plots(train, model, idx=0, output="wandb")
    plots(val_single, model, idx=0, output="wandb")
    plots(val_multi, model, idx=0, output="wandb")
    wandb.finish()


# Start sweep job.
wandb.agent(sweep_id, function=main)
