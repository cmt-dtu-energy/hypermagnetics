import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import hypermagnetics.measures as measures
from hypermagnetics import plots
from hypermagnetics.models import HyperModel
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


class FourierModel(HyperModel):
    hypermodel: FourierHyperModel
    order: int
    omega_scale: jax.Array
    basis_terms: jax.Array = eqx.field(static=True)

    def __init__(self, order, r, wkey, bkey):
        self.order = order
        self.hypermodel = FourierHyperModel(4 * order**2, order**2, 3, wkey, bkey)
        self.omega_scale = jnp.ones(1) * 12
        omega = 2 * jnp.pi * jnp.arange(1, self.order + 1) / self.omega_scale
        self.basis_terms = jax.vmap(evaluate_basis, in_axes=(None, 0))(
            omega, r
        ).transpose(1, 2, 3, 0)

    def cached_evaluation(self, weights, bias):
        weights = jnp.reshape(weights, (4, self.order, self.order))
        elementwise_product = weights[..., None] * self.basis_terms
        summed_product = jnp.sum(elementwise_product, axis=(0, 1, 2))
        return bias + summed_product

    def fourier_expansion(self, weights, bias, r=None):
        weights = jnp.reshape(weights, (4, self.order, self.order))
        omega = 2 * jnp.pi * jnp.arange(1, self.order + 1) / self.omega_scale
        basis_terms = evaluate_basis(omega, r)
        elementwise_product = weights * basis_terms
        summed_product = jnp.sum(elementwise_product, axis=(0, 1, 2))
        return bias + summed_product

    def prepare_weights(self, sources):
        w, b = jax.vmap(self.hypermodel)(sources)
        return jnp.sum(w, axis=0), jnp.sum(b, axis=0)

    def prepare_model(self, weights, bias):
        return lambda r: self.fourier_expansion(weights, bias, r)

    def loss(self, model, data):
        return measures.cached_loss(model, data)


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
    print(jax.vmap(model.cached_evaluation)(*jax.vmap(model.prepare_weights)(sources)))
    print(model.loss(model, train_data))

    plots(train_data, model, idx=0)
