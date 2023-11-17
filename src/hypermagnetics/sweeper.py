import jax.random as jr
import optax
import yaml

import hypermagnetics.sources as sources
import wandb
from hypermagnetics.models import HyperMLP
from hypermagnetics.trainer import fit

with open("config/sweep-configuration.yaml", "r") as f:
    sweep_configuration = yaml.safe_load(f)

sweep_id = wandb.sweep(
    sweep=sweep_configuration, entity="dl4mag", project="hypermagnetics"
)


def main():
    wandb.init()

    source_config = wandb.config.source
    train = sources.configure(**source_config, key=jr.PRNGKey(40))
    val = sources.configure(**source_config, key=jr.PRNGKey(41))

    hyperkey, mainkey = jr.split(jr.PRNGKey(42), 2)
    model_config = wandb.config.model
    model = HyperMLP(**model_config, hyperkey=hyperkey, mainkey=mainkey)
    wandb.log({"nparams": model.nparams})

    trainer_config = wandb.config.trainer
    optim = optax.adam(learning_rate=trainer_config["learning_rate"])
    fit(trainer_config, optim, model, train, val, log=wandb.log)
    wandb.finish()


# Start sweep job.
wandb.agent(sweep_id, function=main)
