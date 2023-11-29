import equinox as eqx
import jax.random as jr

import wandb
from hypermagnetics.models.hyper_fourier import FourierModel  # noqa: F401
from hypermagnetics.models.hyper_mlp import (
    AdditiveMLP,  # noqa: F401
    HyperMLP,  # noqa: F401
)


def save(model, id):
    # Save model to file and prepare artifact
    model_path = f"models/{id}.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_path)

    # Add hyperparameters to artifact metadata
    hyperparameters = {
        "model_class": model.__class__.__name__,
        "hyperparameters": model.get_hyperparameters(),
    }
    artifact.metadata = hyperparameters

    wandb.log_artifact(artifact)


def load(id):
    # Download artifact from wandb
    artifact = wandb.use_artifact(f"model:{id}", type="model")
    hyperparameters = artifact.metadata["hyperparameters"]
    model_path = artifact.file()

    # Instantiate new model with saved hyperparameters
    hyperkey, mainkey = jr.split(jr.PRNGKey(42), 2)
    model_class = globals()[artifact.metadata["model_class"]]
    model = model_class(**hyperparameters, hyperkey=hyperkey, mainkey=mainkey)

    # Load parameters into model
    return eqx.tree_deserialise_leaves(model_path, like=model)
