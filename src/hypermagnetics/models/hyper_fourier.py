import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics import plots
from hypermagnetics.sources import configure


def basis_term(fun1, fun2, omega, r):
    """Compute basis terms for the Fourier series expansion."""
    return fun1(omega[:, None] * r[0]) * fun2(omega[None, :] * r[1])


@jax.jit
def evaluate_basis(omega, r):
    return jnp.stack(
        [
            basis_term(jnp.cos, jnp.cos, omega, r),
            basis_term(jnp.sin, jnp.sin, omega, r),
            basis_term(jnp.cos, jnp.sin, omega, r),
            basis_term(jnp.sin, jnp.cos, omega, r),
        ],
        axis=0,
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
    # basis_terms: jnp.ndarray = eqx.field(static=True)

    def __init__(self, order, r, wkey, bkey):
        self.order = order
        self.hypermodel = FourierHyperModel(4 * order**2, order**2, 3, wkey, bkey)
        self.omega = 2 * jnp.pi * jnp.arange(1, order + 1) / 10
        # self.basis_terms = jax.vmap(lambda r: evaluate_basis(self.omega, r))(r)

    def fourier_expansion(self, weights, bias, r=None):
        weights = jnp.reshape(weights, (4, self.order, self.order))
        basis_terms = evaluate_basis(self.omega, r)
        # basis_terms = self.basis_terms if r is None else evaluate_basis(self.omega, r)
        elementwise_product = weights * basis_terms
        summed_product = jnp.sum(elementwise_product, axis=(0, 1, 2))
        return bias + summed_product

    def prepare_weights(self, sources):
        w, b = jax.vmap(self.hypermodel)(sources)
        return jnp.sum(w, axis=0), jnp.sum(b, axis=0)

    def prepare_model(self, weights, bias):
        return lambda r: self.fourier_expansion(weights, bias, r)

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

    # Show output from evaluating FourierModel model on source configuration
    wkey, bkey = jr.split(jr.PRNGKey(41), 2)
    model = FourierModel(4, r, wkey, bkey)
    print(jax.vmap(model, in_axes=(0, None))(sources, r))

    plots(train_data, model, idx=0)
