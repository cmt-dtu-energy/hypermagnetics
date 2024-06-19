import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from hypermagnetics import plots
from hypermagnetics.models import HyperModel
from hypermagnetics.sources import configure


def basis_term(fun1, fun2, k, r):
    """Compute basis terms for the Fourier series expansion."""
    return fun1(k[:, None] * r[0]) * fun2(k[None, :] * r[1])


@jax.jit
def evaluate_basis(k, r):
    return jnp.stack(
        [
            basis_term(jnp.cos, jnp.cos, k, r),
            basis_term(jnp.sin, jnp.cos, k, r),
            basis_term(jnp.cos, jnp.sin, k, r),
            basis_term(jnp.sin, jnp.sin, k, r),
        ],
        axis=0,
    )


class FourierHyperModel(eqx.Module):
    w: eqx.nn.MLP

    def __init__(self, out: int, width: int, depth: int, key: jax.Array):
        self.w = eqx.nn.MLP(5, out, width, depth, jax.nn.gelu, key=key)

    def __call__(self, sources):
        # mx, my, rx, ry, a
        inputs = jnp.concatenate(
            [sources[..., :2], sources[..., 3:5], sources[..., 6:7]], axis=-1
        )
        return self.w(inputs)


class FourierModel(HyperModel):
    order: int
    hwidth: int = 1
    hdepth: int = 2
    seed: int = 1
    hypermodel: FourierHyperModel = eqx.field(init=False)
    kl: jax.Array = eqx.field(init=False)

    def __post_init__(self):
        key = jr.PRNGKey(self.seed)
        self.kl = jnp.array([-2.5, 1.1])
        # self.kl = jnp.array([-2.9, 0.75])

        out = 4 * self.order * self.order
        hwidth = int(self.hwidth * out)
        self.hypermodel = FourierHyperModel(out, hwidth, self.hdepth, key)
        # self.hypermodel = jnp.zeros((1, out))

    @property
    def hparams(self):
        return {
            "order": self.order,
            "hwidth": self.hwidth,
            "hdepth": self.hdepth,
            "seed": self.seed,
        }

    @property
    def k(self):
        logmodes = jnp.squeeze(jnp.linspace(self.kl[0], self.kl[1], self.order - 1))
        return jnp.concatenate((jnp.zeros(1), 10**logmodes))  # Append zero mode

    @eqx.filter_jit
    def fourier_expansion(self, weights, r):
        weights = jnp.reshape(weights, (4, self.order, self.order))
        basis_terms = evaluate_basis(self.k, r)
        elementwise_product = weights * basis_terms
        summed_product = jnp.sum(elementwise_product, axis=(0, 1, 2))
        return summed_product

    def prepare_weights(self, sources):
        w = jax.vmap(self.hypermodel)(sources)
        # w = self.hypermodel
        return jnp.sum(w, axis=0), None

    def prepare_model(self, weights, bias):
        return lambda r: self.fourier_expansion(weights, r)


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 2,
        "seed": 40,
        "lim": 3,
        "res": 32,
    }
    train_data = configure(**config)
    sources, r = train_data["sources"], train_data["grid"]

    model = FourierModel(order=32, seed=41)
    print(model.hparams)
    print(jax.vmap(model, in_axes=(0, None))(sources, r))
    plots(train_data, model, idx=0)
