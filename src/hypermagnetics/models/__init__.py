import equinox as eqx
import jax
import jax.random as jr

import wandb


def count_mlp_params(in_features, out_features, width, depth):
    return (
        (in_features + 1) * width
        + (width + 1) * width * (depth - 1)
        + (width + 1) * out_features
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


class HyperModel(eqx.Module):
    """A hypermodel is a model whose parameters are themselves parameterised."""

    def prepare_weights(self, sources):
        """Compute inference model weights."""
        raise NotImplementedError

    def prepare_model(self, weights, bias):
        """Construct inference model for evaluation."""
        raise NotImplementedError

    def field(self, sources, r):
        """Evaluate the field given sources (sources) and evaluation points (r)."""
        weights, bias = self.prepare_weights(sources)
        model = self.prepare_model(weights, bias)
        return -jax.vmap(jax.grad(model))(r)

    def __call__(self, sources, r):
        """Evaluate the potential given sources (sources) and evaluation points (r)."""
        weights, bias = self.prepare_weights(sources)
        model = self.prepare_model(weights, bias)
        return jax.vmap(model)(r)
