import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plots(sources, idx=0, model=None, show_field=False):
    """Plots the sources and field/potential of a single sample."""
    mr = sources["sources"]
    m, r0 = mr[:, :, 0:1], mr[:, :, 2:3]
    grid = sources["grid"]

    res = int(jnp.sqrt(len(grid)))
    N = len(mr)

    x_grid = np.array(grid[:, 0].reshape((res, res)))
    y_grid = np.array(grid[:, 1].reshape((res, res)))

    if model is None:
        potential = sources["potential"].reshape((N, res, res))
        field = sources["field"].reshape((N, res, res, 2))
    else:
        potential = jax.vmap(model, in_axes=(0, None))(mr, grid).reshape((N, res, res))
        # field = jax.vmap(model, in_axes=(0, None, None))(mr, grid, field=True)

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
