import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics.sources import configure


@jax.jit
def basis_terms(omega, r):
    """Compute basis terms for the Fourier series expansion."""
    return jnp.stack(
        [
            jnp.cos(omega[:, None, None] * r[None, :, 0])
            * jnp.cos(omega[None, :, None] * r[None, :, 1]),
            jnp.sin(omega[:, None, None] * r[None, :, 0])
            * jnp.sin(omega[None, :, None] * r[None, :, 1]),
            jnp.cos(omega[:, None, None] * r[None, :, 0])
            * jnp.sin(omega[None, :, None] * r[None, :, 1]),
            jnp.sin(omega[:, None, None] * r[None, :, 0])
            * jnp.cos(omega[None, :, None] * r[None, :, 1]),
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

    def __init__(self, order, wkey, bkey):
        self.order = order
        self.hypermodel = FourierHyperModel(4 * order**2, order**2, 3, wkey, bkey)

    def compute_basis_terms(self, r):
        omega = 2 * jnp.pi * jnp.arange(1, self.order + 1) / 10
        return basis_terms(omega, r)

    def fourier_expansion(self, weights, bias, r):
        weights = jnp.reshape(weights, (4, self.order, self.order))
        elementwise_product = weights[..., None] * self.compute_basis_terms(r)
        summed_product = jnp.sum(elementwise_product, axis=(0, 1, 2))
        return bias + summed_product

    def prepare_weights(self, sources):
        w, b = jax.vmap(jax.vmap(self.hypermodel))(sources)
        return jnp.sum(w, axis=1), jnp.sum(b, axis=1)

    def __call__(self, sources, r):
        w, b = self.prepare_weights(sources)
        return jax.vmap(self.fourier_expansion, in_axes=(0, 0, None))(w, b, r)


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 5,
        "key": jr.PRNGKey(40),
        "lim": 3,
        "res": 32,
    }
    train_data = configure(**config)
    sources, r = train_data["sources"], train_data["grid"]

    # Show output from evaluating FourierModel model on source configuration
    wkey, bkey = jr.split(jr.PRNGKey(41), 2)
    model = FourierModel(4, wkey, bkey)
    output = model(sources, r)
    print(output.shape)
    print(model(sources, r))
