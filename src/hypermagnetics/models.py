import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr


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


@jax.jit
def reshape_params(old_params, params):
    """Reshape a flat parameter vector into a list of weight and bias matrices."""
    new_params = []
    for w in old_params:
        new_params.append(jnp.reshape(params[: w.size], w.shape))
        params = params[w.size :]
    return new_params


class MLP(nn.Module):
    """A scalar output flax MLP model with a given width and depth."""

    width: int
    depth: int
    n_out: int = 1

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = nn.gelu(x)
        return nn.Dense(self.n_out)(x)


class HyperMLP(nn.Module):
    """A hypernetwork that generates weights and biases for a given MLP architecture,
    applies them to a template MLP model, and evaluates the resulting model on a given input."""

    width: int
    depth: int
    hdepth: int
    hyperkey: jr.PRNGKey
    mainkey: jr.PRNGKey

    def setup(self):
        self.inference = MLP(width=self.width, depth=self.depth)

    def __init__(self, width, depth, hdepth, hyperkey, mainkey):
        self.model = eqx.nn.MLP(
            2, "scalar", width, depth, jax.nn.gelu, use_bias=True, key=mainkey
        )
        self.nweights = sum(w.size for w in get_weights(self.model))
        self.nbiases = sum(b.size for b in get_biases(self.model))
        nparams = self.nweights + self.nbiases
        self.rho = eqx.nn.MLP(4, nparams, nparams, hdepth, jax.nn.gelu, key=hyperkey)

    def prepare_weights(self, m_r):
        wb = jnp.sum(jax.vmap(self.rho)(m_r), axis=0)
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

    def __call__(self, m_r, r, field=False):
        """Evaluate the hypernetwork given sources (m_r) and field evaluation points (r).
        Returns the potential by default, or the magnetic field if field=True."""
        weights, biases = self.prepare_weights(m_r)
        model = self.prepare_model(weights, biases)
        return jax.vmap(model)(r) if not field else -jax.vmap(jax.grad(model))(r)


if __name__ == "__main__":
    key, hyperkey, mainkey = jr.split(jr.PRNGKey(41), 3)
    model = HyperMLP(16, 3, 2, hyperkey, mainkey)
    print(model)
