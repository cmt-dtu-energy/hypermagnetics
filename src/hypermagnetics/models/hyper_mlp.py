import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics import plots
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


def count_mlp_params(in_features, out_features, width, depth):
    return (
        (in_features + 1) * width
        + (width + 1) * width * (depth - 1)
        + (width + 1) * out_features
    )


class AdditiveMLP(eqx.Module):
    hypermodel: eqx.nn.MLP
    nparams: int
    model: eqx.nn.MLP
    final_layer: eqx.nn.MLP = eqx.field(static=True)

    def __init__(self, width, depth, hwidth, hdepth, hyperkey, mainkey):
        modelkey, finalkey = jr.split(mainkey, 2)
        self.model = eqx.nn.MLP(
            2,
            width,
            width,
            depth,
            activation=jax.nn.gelu,
            final_activation=jax.nn.gelu,
            use_bias=True,
            key=modelkey,
        )
        self.final_layer = eqx.nn.MLP(width, "scalar", width, 0, key=finalkey)

        p = width + 1
        self.hypermodel = eqx.nn.MLP(
            4, p, hwidth * p, hdepth, jax.nn.gelu, key=hyperkey
        )
        self.nparams = count_mlp_params(4, p, hwidth * p, hdepth) + count_mlp_params(
            2, width, width, depth
        )

    def prepare_weights(self, sources):
        wb = jnp.sum(jax.vmap(self.hypermodel)(sources), axis=0)
        weights, bias = wb[:-1], wb[-1:]
        return weights, bias

    def prepare_final_layer(self, weights, bias):
        final_layer = eqx.tree_at(
            get_weights,
            self.final_layer,
            reshape_params(get_weights(self.final_layer), weights),
        )
        final_layer = eqx.tree_at(
            get_biases, final_layer, reshape_params(get_biases(final_layer), bias)
        )
        return final_layer

    def __call__(self, sources, r, field=False):
        """Evaluate the hypernetwork given sources (sources) and field evaluation points (r).
        Returns the potential by default, or the magnetic field if field=True."""
        weights, bias = self.prepare_weights(sources)
        final_layer = self.prepare_final_layer(weights, bias)
        return jax.vmap(lambda r: final_layer(self.model(r)))(r)


class HyperMLP(eqx.Module):
    """A hypernetwork that generates weights and biases for a given MLP architecture,
    applies them to a template MLP model, and evaluates the resulting model on a given input."""

    hypermodel: eqx.nn.MLP
    nbiases: int
    nweights: int
    nparams: int
    model: eqx.nn.MLP = eqx.field(static=True)

    def __init__(self, width, depth, hwidth, hdepth, hyperkey, mainkey):
        self.model = eqx.nn.MLP(
            2, "scalar", width, depth, jax.nn.gelu, use_bias=True, key=mainkey
        )
        self.nweights = sum(w.size for w in get_weights(self.model))
        self.nbiases = sum(b.size for b in get_biases(self.model))
        p = self.nweights + self.nbiases
        self.hypermodel = eqx.nn.MLP(
            4, p, (hwidth * p), hdepth, jax.nn.gelu, key=hyperkey
        )
        self.nparams = count_mlp_params(4, p, hwidth * p, hdepth)

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

    def field(self, sources, r):
        """Evaluate the hypernetwork given sources (sources) and field evaluation points (r).
        Returns the potential by default, or the magnetic field if field=True."""
        weights, biases = self.prepare_weights(sources)
        model = self.prepare_model(weights, biases)
        return -jax.vmap(jax.grad(model))(r)

    def __call__(self, sources, r):
        """Evaluate the hypernetwork given sources (sources) and field evaluation points (r).
        Returns the potential by default, or the magnetic field if field=True."""
        weights, biases = self.prepare_weights(sources)
        model = self.prepare_model(weights, biases)
        return jax.vmap(model)(r)


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 2,
        "key": jr.PRNGKey(40),
        "lim": 3,
        "res": 32,
    }
    train_data = configure(**config)
    sources, r = train_data["sources"], train_data["grid"]

    # Show output from evaluating HyperMLP model on source configuration
    hyperkey, mainkey = jr.split(jr.PRNGKey(41), 2)
    model = HyperMLP(4, 3, 2, 2, hyperkey, mainkey)
    print(jax.vmap(model, in_axes=(0, None))(sources, r))

    additive_model = AdditiveMLP(4, 3, 1, 2, hyperkey, mainkey)
    print(jax.vmap(additive_model, in_axes=(0, None))(sources, r))

    plots(train_data, idx=0, model=model, show_field=True)
