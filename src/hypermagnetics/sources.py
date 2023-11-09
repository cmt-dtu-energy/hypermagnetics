from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass
class Sources:
    N: int  # Number of samples to generate
    M: int  # Number of sources in each sample
    key: jr.PRNGKey  # Random number generator key
    core: float = field(default=1.0, init=True)  # Radius of sources
    lim: int = field(default=3, init=True)  # Domain range, in units of source radius
    res: int = field(default=32, init=True)  # Resolution of the field grid

    def __post_init__(self):
        self.key, subkey = jr.split(self.key, 2)
        self.m, self.r0 = jnp.split(
            jr.normal(key=subkey, shape=(self.N, self.M, 4)), 2, axis=-1
        )
        self.m_r = jnp.concatenate([self.m, self.r0], axis=-1)

        range = jnp.linspace(-self.lim, self.lim, self.res)
        self.x, self.y = jnp.meshgrid(range, range)
        self.grid = jnp.stack([self.x.flatten(), self.y.flatten()], axis=1)

        self.potential = self._total(self._potential, self.m, self.r0, self.grid)
        self.field = self._total(self._field, self.m, self.r0, self.grid)

    def sample_grid(self, key, n=None):
        if n is None:
            n = self.res * self.res
        return jr.uniform(minval=-self.lim, maxval=self.lim, shape=(n, 2), key=key)

    def _potential(self, m, r0, r):
        core = 1.0
        d = r - r0
        d_norm = jnp.linalg.norm(d)
        m_dot_r = jnp.dot(m, d)
        close_to_source = d_norm <= core
        interior = m_dot_r / core / (2 * jnp.pi * core)
        exterior = m_dot_r / d_norm / (2 * jnp.pi * d_norm)
        return jnp.where(close_to_source, interior, exterior)

    def _field(self, m, r0, r):
        return -jax.grad(self._potential, argnums=2)(m, r0, r)

    def _total(self, fun, m, r0, r):
        points = jax.vmap(fun, in_axes=(None, None, 0))
        batch = jax.vmap(points, in_axes=(0, 0, None))
        components = jax.vmap(batch, in_axes=(1, 1, None))(m, r0, r)
        return jnp.sum(components, axis=0)


if __name__ == "__main__":
    sources = Sources(N=10, M=1, key=jr.PRNGKey(40), lim=3, res=32)
    print(sources.potential.shape, sources.field.shape)
