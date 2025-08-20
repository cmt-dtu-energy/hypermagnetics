from pathlib import Path

import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.axes import Axes
import numpy as np
import wandb


def _plot_shape(ax, r0, m, size, shape, idx, loc, edge):
    # Dot at source location
    if loc:
        ax.scatter(r0[idx, :, 0], r0[idx, :, 1], color="red")
        ax.quiver(
            r0[idx, :, 0],
            r0[idx, :, 1],
            m[idx, :, 0],
            m[idx, :, 1],
            angles="xy",
            scale_units="xy",
            scale=5,
            color="red",
        )
    # Edge of magnetized bodies
    if edge:
        for i in range(r0.shape[1]):
            if shape == "sphere":
                edge = Circle(
                    (r0[idx, i, 0], r0[idx, i, 1]),
                    size[idx, i, 0],
                    fill=False,
                    edgecolor="red",
                )
            elif shape == "prism":
                edge = Rectangle(
                    (
                        r0[idx, i, 0] - size[idx, i, 0],
                        r0[idx, i, 1] - size[idx, i, 1],
                    ),
                    size[idx, i, 0] * 2,
                    size[idx, i, 1] * 2,
                    fill=False,
                    edgecolor="red",
                )
            else:
                raise ValueError(f"Unknown shape: {shape}")

            ax.add_patch(edge)


def _plot(
    axes: Axes,
    data: dict,
    idx: int,
    prefix: str,
    loc: bool,
    edge: bool,
    number: bool,
    model=None,
):
    mr = data["sources"][idx : idx + 1]
    m, r0, size = np.split(mr, 3, axis=-1)

    res = int(np.sqrt(len(data["grid"])))
    x_grid = np.array(data["grid"][:, 0].reshape((res, res)))
    y_grid = np.array(data["grid"][:, 1].reshape((res, res)))

    if model is None:
        msp = data["msp_grid"][idx]
        field = data["field_grid"][idx][..., :2]
    else:
        msp = jax.vmap(model, in_axes=(0, None))(mr, data["grid"])[idx]
        field = jax.vmap(model.field, in_axes=(0, None))(mr, data["grid"])[idx][..., :2]

    xlims = (x_grid.min(), x_grid.max())
    ylims = (y_grid.min(), y_grid.max())

    # Subplot 1: Magnetic Scalar Potential
    axes[0].contourf(x_grid, y_grid, msp.reshape((res, res)))
    _plot_shape(axes[0], r0, m, size, data["shape"], idx, loc, edge)
    if number:
        for i in range(r0.shape[1]):
            axes[0].text(
                r0[idx, i, 0] + 0.05,
                r0[idx, i, 1] + 0.05,
                str(i),
                color="black",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
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
        field[..., 0].reshape((res, res)),
        field[..., 1].reshape((res, res)),
        density=1.5,
        linewidth=0.5,
        arrowsize=1.5,
        arrowstyle="->",
    )
    _plot_shape(axes[1], r0, m, size, data["shape"], idx, loc, edge)

    axes[1].set_title(prefix + " " + "Magnetic Field")
    axes[1].set_xlabel("x" + units_str)
    axes[1].set_ylabel("y" + units_str)
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)


def plots(
    data: dict,
    loc: bool = True,
    edge: bool = False,
    number: bool = False,
    idx: int = 0,
    prefix: str = "",
    output: str = "show",
    model=None,
):
    """
    Plots the sources and field/potential of a single sample.

    Parameters:
        sources (dict): The source data containing positions, magnetizations, and sizes.
        loc (bool): Whether to plot the source positions.
        edge (bool): Whether to plot the shapes of the sources.
        numbers (bool): Whether to plot the source numbers.
        idx (int): The index of the sample to plot.
        prefix (str): A prefix for the plot titles.
        output (str): The output mode for the plot (e.g., "show", "save").
        model (HyperLayer): The trained model to use for predictions.
    """
    if model is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        _plot(axes, data, idx, prefix, loc, edge, number)
    else:
        _, axes = plt.subplots(2, 2, figsize=(8, 8))
        _plot(axes[0], data, idx, prefix, loc, edge, number)
        _plot(axes[1], data, idx, prefix, loc, edge, number, model=model)

    plt.tight_layout()

    if output == "show":
        plt.show()
    elif output == "save":
        plt.savefig(
            Path(__file__).parent / ".." / ".." / "figs" / f"{prefix}_plot_{idx}.svg"
        )
    elif output == "wandb":
        wandb.log({"chart": wandb.Image(plt)})
    else:
        raise ValueError(f"Unknown output mode: {output}")
