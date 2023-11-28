import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np


def _potential(m, r0, r):
    """Dipole potential in two dimensions."""
    core = 1.0
    d = r - r0
    d_norm = jnp.linalg.norm(d)
    m_dot_r = jnp.dot(m, d)
    close_to_source = d_norm <= core
    interior = m_dot_r / core / (2 * jnp.pi * core)
    exterior = m_dot_r / d_norm / (2 * jnp.pi * d_norm)
    return jnp.where(close_to_source, interior, exterior)


def _field(m, r0, r):
    """Dipole field in two dimensions."""
    return -jax.grad(_potential, argnums=2)(m, r0, r)


def _total(fun, m, r0, r):
    """Aggregate the field or potential of all sources."""
    points = jax.vmap(fun, in_axes=(None, None, 0))
    batch = jax.vmap(points, in_axes=(0, 0, None))
    components = jax.vmap(batch, in_axes=(1, 1, None))(m, r0, r)
    return jnp.sum(components, axis=0)


def configure(n_samples, n_sources, lim=3, res=32, key=jr.PRNGKey(0)):
    """
    Configures samples of sources.

    Args:
        n_samples (int): Number of samples to generate.
        n_sources (int): Number of sources in each sample.
        lim (int, optional): Domain range, in units of source radius. Defaults to 3.
        res (int, optional): Resolution of the field grid. Defaults to 32.
        key (jr.PRNGKey): Random number generator key.
    """

    key, subkey = jr.split(key, 2)
    m, r0 = jnp.split(
        jr.normal(key=subkey, shape=(n_samples, n_sources, 4)), 2, axis=-1
    )

    range = jnp.linspace(-lim, lim, res)
    x, y = jnp.meshgrid(range, range)
    grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
    return {
        "sources": jnp.concatenate([m, r0], axis=-1),
        "grid": grid,
        "potential": _total(_potential, m, r0, grid),
        "field": _total(_field, m, r0, grid),
    }


def plots(sources, idx=0, model=None, show_field=False):
    """Plots the sources and field/potential of a single sample."""
    mr = sources["sources"]
    m, r0 = mr[:, :, 0:1], mr[:, :, 2:3]
    grid = sources["grid"]

    res = int(jnp.sqrt(len(grid)))
    N = len(mr)

    x_grid = np.array(grid[:, 0].reshape((res, res)))
    y_grid = np.array(grid[:, 1].reshape((res, res)))
    potential = sources["potential"].reshape((N, res, res))
    field = sources["field"].reshape((N, res, res, 2))

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: Magnetic Scalar Potential
    _ = axes[0].contourf(x_grid, y_grid, potential[idx])
    # plt.colorbar(cp, ax=axes[0])
    axes[0].scatter(r0[idx, :, 0], r0[idx, :, 1], color="red")
    axes[0].quiver(
        r0[idx, :, 0],
        r0[idx, :, 1],
        m[idx, :, 0],
        m[idx, :, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
    )
    axes[0].set_title("Magnetic Scalar Potential")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Subplot 2: Magnetic Field
    if show_field is not False:
        axes[1].streamplot(
            x_grid,
            y_grid,
            field[idx, :, :, 0],
            field[idx, :, :, 1],
            density=1.5,
            linewidth=1,
            arrowsize=1.5,
            arrowstyle="->",
        )
        axes[1].scatter(r0[idx, :, 0], r0[idx, :, 1], color="red")
        axes[1].quiver(
            r0[idx, :, 0],
            r0[idx, :, 1],
            m[idx, :, 0],
            m[idx, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
        )
        axes[1].set_title("Magnetic Field")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()


def sample_grid(key, lim, res, n=None):
    if n is None:
        n = res * res
    return jr.uniform(minval=-lim, maxval=lim, shape=(n, 2), key=key)


if __name__ == "__main__":
    config = {
        "n_samples": 10,
        "n_sources": 2,
        "key": jr.PRNGKey(40),
        "lim": 3,
        "res": 32,
    }
    train_data = configure(**config)
    print(train_data["potential"].shape, train_data["field"].shape)
    plots(train_data, idx=0, show_field=True)
