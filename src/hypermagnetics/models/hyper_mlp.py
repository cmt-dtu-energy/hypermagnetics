import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics import plots
from hypermagnetics.models import MLPHyperModel, count_params
from hypermagnetics.sources import configure


def is_linear(x):
    """Identify the linear layer component of the model,
    to update their weights."""
    return isinstance(x, eqx.nn.Linear)


def get_weights(model):
    """Get the weights of all linear layers in the MLP."""
    leaves = jax.tree_util.tree_leaves(model, is_leaf=is_linear)
    return [x.weight for x in leaves if is_linear(x)]


def get_biases(model):
    """Get the biases of all linear layers in the MLP."""
    leaves = jax.tree_util.tree_leaves(model, is_leaf=is_linear)
    return [x.bias for x in leaves if is_linear(x)]


def reshape_params(old_params, flat_params):
    """Reshape a flat parameter vector into a list of weight and bias matrices."""
    new_params = ()
    idx = 0
    for w in old_params:
        new_params += (jnp.reshape(flat_params[idx : idx + w.size], w.shape),)
        idx += w.size
    return new_params


class HyperLayer(MLPHyperModel):
    hypermodel: eqx.nn.MLP = eqx.field(init=False)
    model: eqx.nn.MLP = eqx.field(init=False)
    final_layer: eqx.nn.MLP = eqx.field(static=True, init=False)

    def __post_init__(self):
        key = jr.PRNGKey(self.seed)
        hyperkey, modelkey, finalkey = jr.split(key, 3)
        self.model = eqx.nn.MLP(
            self.in_size,
            self.width,
            self.width,
            self.depth,
            activation=jax.nn.gelu,
            final_activation=jax.nn.gelu,
            use_bias=True,
            key=modelkey,
        )
        self.final_layer = eqx.nn.MLP(self.width, "scalar", self.width, 0, key=finalkey)

        p = self.width + 1
        q = int(self.hwidth * p)
        self.hypermodel = eqx.nn.MLP(
            2 * self.in_size, p, q, self.hdepth, jax.nn.gelu, key=hyperkey
        )

    @property
    def nparams(self):
        # Avoid double counting params in the final layer
        return count_params(self.model) + count_params(self.hypermodel)

    def prepare_weights(self, sources):
        wb = jnp.sum(jax.vmap(self.hypermodel)(sources), axis=0)
        weights, bias = wb[:-1], wb[-1:]
        return weights, bias

    def prepare_model(self, weights, bias):
        final_layer = eqx.tree_at(
            get_weights,
            self.final_layer,
            reshape_params(get_weights(self.final_layer), weights),
        )
        final_layer = eqx.tree_at(
            get_biases, final_layer, reshape_params(get_biases(final_layer), bias)
        )
        return lambda r: final_layer(self.model(r))


class HyperMLP(MLPHyperModel):
    """A hypernetwork that generates weights and biases for a given MLP architecture,
    applies them to a template MLP model, and evaluates the resulting model on a given input."""

    hypermodel: eqx.nn.MLP = eqx.field(init=False)
    nbiases: int = eqx.field(init=False)
    nweights: int = eqx.field(init=False)
    model: eqx.nn.MLP = eqx.field(static=True, init=False)

    def __post_init__(self):
        self.in_size = 2 if self.in_size is None else self.in_size
        key = jr.PRNGKey(self.seed)
        hyperkey, mainkey = jr.split(key, 2)
        self.model = eqx.nn.MLP(
            self.in_size,
            "scalar",
            self.width,
            self.depth,
            activation=jax.nn.gelu,
            use_bias=True,
            key=mainkey,
        )
        self.nweights = sum(w.size for w in get_weights(self.model))
        self.nbiases = sum(b.size for b in get_biases(self.model))
        p = self.nweights + self.nbiases
        q = int(self.hwidth * p)
        self.hypermodel = eqx.nn.MLP(
            2 * self.in_size, p, q, self.hdepth, jax.nn.gelu, key=hyperkey
        )

    def prepare_weights(self, sources):
        wb = jnp.sum(jax.vmap(self.hypermodel)(sources), axis=0)
        weights, biases = wb[: self.nweights], wb[self.nweights :]
        return weights, biases

    def prepare_model(self, weights, biases):
        model = eqx.tree_at(
            get_weights, self.model, reshape_params(get_weights(self.model), weights)
        )
        model = eqx.tree_at(
            get_biases, model, reshape_params(get_biases(model), biases)
        )
        return model


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 2,
        "seed": 40,
        "lim": 3,
        "res": 32,
        "shape": "sphere",
    }
    train_data = configure(**config)
    sources, r = train_data["sources"], train_data["grid"]

    # Show output from evaluating HyperMLP model on source configuration
    seed = 39
    model = HyperMLP(width=4, depth=3, hwidth=2, hdepth=2, seed=seed)
    print(model.hparams)
    print(jax.vmap(model, in_axes=(0, None))(sources, r))
    plots(train_data, model, idx=0)

    additive_model = HyperLayer(width=4, depth=3, hwidth=1, hdepth=2, seed=seed)
    print(additive_model.hparams)
    print(jax.vmap(additive_model, in_axes=(0, None))(sources, r))
    plots(train_data, additive_model, idx=0)
