import equinox as eqx

import wandb
from hypermagnetics.models.hyper_fourier import FourierModel  # noqa: F401
from hypermagnetics.models.hyper_mlp import (
    AdditiveMLP,  # noqa: F401
    HyperMLP,  # noqa: F401
)


def save(model, id):
    model_path = f"models/{id}.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def load(id):
    model_path = f"models/{id}.eqx"
    return eqx.tree_deserialise_leaves(model_path)
