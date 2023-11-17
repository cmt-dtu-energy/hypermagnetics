import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics.sources import configure

# Helper functions and class definition for Fourier hypernetwork


def basis_term(fun1, fun2, omega, r):
    """Compute basis terms for the Fourier series expansion."""
    return fun1(omega[:, None, None] * r[None, :, 0]) * fun2(
        omega[None, :, None] * r[None, :, 1]
    )


class FourierHyperModel(eqx.Module):
    w: eqx.nn.MLP
    b: eqx.nn.Linear

    def __init__(self, out, width, depth, weightkey, biaskey):
        self.w = eqx.nn.MLP(4, out, width, depth, jax.nn.gelu, key=weightkey)
        self.b = eqx.nn.Linear(4, "scalar", key=biaskey)

    def __call__(self, sources):
        return self.w(sources), self.b(sources)


class FourierModel(eqx.Module):
    hypermodel: FourierHyperModel
    order: int
    omega: jnp.ndarray = eqx.field(static=True)
    basis_terms: jnp.ndarray = eqx.field(static=True)

    def __init__(self, order, r, wkey, bkey):
        self.order = order
        self.hypermodel = FourierHyperModel(4 * order**2, order**2, 3, wkey, bkey)
        self.omega = 2 * jnp.pi * jnp.arange(1, order + 1) / 10
        self.basis_terms = jnp.stack(
            [
                basis_term(jnp.cos, jnp.cos, self.omega, r),
                basis_term(jnp.sin, jnp.sin, self.omega, r),
                basis_term(jnp.cos, jnp.sin, self.omega, r),
                basis_term(jnp.sin, jnp.cos, self.omega, r),
            ],
            axis=0,
        )

    def fourier_expansion(self, weights, bias):
        weights = jnp.reshape(weights, (4, self.order, self.order))
        elementwise_product = weights[..., None] * self.basis_terms
        summed_product = jnp.sum(elementwise_product, axis=(0, 1, 2))
        return bias + summed_product

    def prepare_weights(self, sources):
        w, b = jax.vmap(self.hypermodel)(sources)
        return jnp.sum(w, axis=0), jnp.sum(b, axis=0)

    def __call__(self, sources):
        return self.fourier_expansion(*self.prepare_weights(sources))


# Helper functions and class definition for MLP hypernetwork


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
    nparams: int
    model: eqx.nn.MLP = eqx.field(static=True)

    def __init__(self, width, depth, hwidth, hdepth, hyperkey, mainkey):
        self.model = eqx.nn.MLP(
            2, "scalar", width, depth, jax.nn.gelu, use_bias=True, key=mainkey
        )
        self.nweights = sum(w.size for w in get_weights(self.model))
        self.nbiases = sum(b.size for b in get_biases(self.model))
        p = self.nweights + self.nbiases
        self.rho = eqx.nn.MLP(4, p, (hwidth * p), hdepth, jax.nn.gelu, key=hyperkey)

        total_weights = (
            4 * (hwidth * p)
            + (hwidth * p) * (hwidth * p) * (hdepth - 1)
            + (hwidth * p) * p
        )
        total_biases = hdepth * (hwidth * p) + p
        self.nparams = total_weights + total_biases

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

    # Show output from evaluating FourierModel model on source configuration
    wkey, bkey = jr.split(jr.PRNGKey(41), 2)
    model = FourierModel(4, r, wkey, bkey)
    print(jax.vmap(model)(sources))
