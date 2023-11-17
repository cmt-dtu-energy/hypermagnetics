import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics.sources import configure


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
    print(jax.vmap(model)(sources))
