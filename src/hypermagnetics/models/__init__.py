import equinox as eqx
import jax

import wandb


def count_params(model):
    param_tree = eqx.filter(model, eqx.is_array)
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(lambda x: x.size, param_tree), 0
    )


def count_mlp_params(in_features: int, out_features: int, width: int, depth: int):
    return (
        (in_features + 1) * width
        + (width + 1) * width * (depth - 1)
        + (width + 1) * out_features
    )


def upload(model, id):
    # Serialise model weights
    model_path = f"models/{id}.eqx"
    eqx.tree_serialise_leaves(model_path, model)

    # Upload model artifact to wandb
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_path)
    artifact.metadata = {
        "model_class": model.__class__.__name__,
        "hyperparameters": model.hparams,
    }
    wandb.log_artifact(artifact)


def download(id):
    # Download artifact from wandb
    artifact = wandb.use_artifact(f"model:{id}", type="model")
    hyperparameters = artifact.metadata["hyperparameters"]
    model_path = artifact.file()

    # Instantiate new model with saved hyperparameters
    model_class = globals()[artifact.metadata["model_class"]]
    model = model_class(**hyperparameters)
    return eqx.tree_deserialise_leaves(model_path, like=model)


class HyperModel(eqx.Module):
    """A hypermodel is a model whose parameters are themselves parameterised."""

    @property
    def nparams(self):
        """Number of parameters in the model."""
        return count_params(self)

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


class MLPHyperModel(HyperModel):
    """A hypernetwork using two fully-connected network architectures."""

    width: int
    depth: int
    hwidth: int
    hdepth: int
    seed: int

    @property
    def hparams(self):
        """Hyperparameters of the model."""
        return {
            "width": self.width,
            "depth": self.depth,
            "hwidth": self.hwidth,
            "hdepth": self.hdepth,
            "seed": self.seed,
        }
