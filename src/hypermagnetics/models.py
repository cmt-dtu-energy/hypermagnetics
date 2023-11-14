import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

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


class HyperMLP(eqx.Module):
    """A hypernetwork that generates weights and biases for a given MLP architecture,
    applies them to a template MLP model, and evaluates the resulting model on a given input."""

    rho: eqx.nn.MLP
    nbiases: int
    nweights: int
    model: eqx.nn.MLP = eqx.field(static=True)

    def __init__(self, width, depth, hdepth, hyperkey, mainkey):
        self.model = eqx.nn.MLP(
            2, "scalar", width, depth, jax.nn.gelu, use_bias=True, key=mainkey
        )
        self.nweights = sum(w.size for w in get_weights(self.model))
        self.nbiases = sum(b.size for b in get_biases(self.model))
        nparams = self.nweights + self.nbiases
        self.rho = eqx.nn.MLP(4, nparams, nparams, hdepth, jax.nn.gelu, key=hyperkey)

    def prepare_weights(self, sources):
        wb = jnp.sum(jax.vmap(self.rho)(sources), axis=0)
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

    def __call__(self, sources, r, field=False):
        """Evaluate the hypernetwork given sources (sources) and field evaluation points (r).
        Returns the potential by default, or the magnetic field if field=True."""
        weights, biases = self.prepare_weights(sources)
        model = self.prepare_model(weights, biases)
        return jax.vmap(model)(r) if not field else -jax.vmap(jax.grad(model))(r)


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 1,
        "key": jr.PRNGKey(40),
        "lim": 3,
        "res": 32,
    }
    train_data = configure(**config)
    key, hyperkey, mainkey = jr.split(jr.PRNGKey(41), 3)
    model = HyperMLP(4, 3, 2, hyperkey, mainkey)

    # Show output from evaluating model on source configuration
    print(jax.vmap(model, in_axes=(0, None))(train_data["sources"], train_data["grid"]))
