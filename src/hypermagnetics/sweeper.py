import matplotlib.pyplot as plt
import optax
import scienceplots  # noqa
import yaml
import jax.numpy as jnp

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

    train = sources.configure(n_samples=1000, n_sources=1, lim=3, res=32, seed=100)
    test = sources.configure(n_samples=1000, n_sources=4, lim=3, res=32, seed=101)
    val_single = sources.configure(n_samples=100, n_sources=1, lim=3, res=50, seed=102)
    val_multi = sources.configure(n_samples=1, n_sources=4, lim=3, res=50, seed=102)

    model = FourierModel(
        order=wandb.config.order, hwidth=wandb.config.hwidth, hdepth=wandb.config.hdepth, seed=41
    )
    # model = HyperLayer(
    #     width=wandb.config.width, depth=wandb.config.depth, hwidth=2, hdepth=3, seed=41
    # )
    # model = HyperMLP(**model_config["hypernetwork"])
    wandb.log({"nparams": model.nparams})

    lr = 10 ** (-jnp.log10(model.nparams))
    trainer_config = {"epochs": 25000, "params": {"learning_rate": lr}}
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
