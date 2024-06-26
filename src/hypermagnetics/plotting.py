import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import wandb


def _plot(axes, x_grid, y_grid, potential, field, m, r0, idx, prefix):
    xlims = (x_grid.min(), x_grid.max())
    ylims = (y_grid.min(), y_grid.max())

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
    axes[0].set_title(prefix + " " + "Magnetic Scalar Potential")
    units_str = ", in units of source radius"
    axes[0].set_xlabel("$x$" + units_str)
    axes[0].set_ylabel("$y$" + units_str)
    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)

    # Subplot 2: Magnetic Field
    axes[1].streamplot(
        x_grid,
        y_grid,
        field[idx, :, :, 0],
        field[idx, :, :, 1],
        density=1.5,
        linewidth=0.5,
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
    axes[1].set_title(prefix + " " + "Magnetic Field")
    axes[1].set_xlabel("x" + units_str)
    axes[1].set_ylabel("y" + units_str)
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)


def plots(sources, model, idx=0, prefix="", output="show"):
    """Plots the sources and field/potential of a single sample."""
    mr = sources["sources"][idx : idx + 1]
    m, r0 = jnp.split(mr, 2, axis=-1)
    grid = sources["grid"]

    res = int(jnp.sqrt(len(grid)))
    N = len(mr)

    x_grid = np.array(grid[:, 0].reshape((res, res)))
    y_grid = np.array(grid[:, 1].reshape((res, res)))

    target_potential = sources["potential_grid"][idx : idx + 1].reshape((N, res, res))
    target_field = sources["field_grid"][idx : idx + 1].reshape((N, res, res, 2))

    if model is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        _plot(axes, x_grid, y_grid, target_potential, target_field, m, r0, idx, prefix)
    else:
        model_potential = jax.vmap(model, in_axes=(0, None))(mr, grid).reshape(
            (N, res, res)
        )
        model_field = jax.vmap(model.field, in_axes=(0, None))(mr, grid).reshape(
            (N, res, res, 2)
        )

        _, axes = plt.subplots(2, 2, figsize=(8, 8))
        _plot(
            axes[0], x_grid, y_grid, target_potential, target_field, m, r0, idx, prefix
        )
        _plot(axes[1], x_grid, y_grid, model_potential, model_field, m, r0, idx, prefix)

    plt.tight_layout()
    if output == "show":
        plt.show()
    elif output == "wandb":
        wandb.log({"chart": wandb.Image(plt)})
