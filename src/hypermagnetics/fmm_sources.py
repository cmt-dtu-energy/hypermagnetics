import time
import numpy as np
import fmm2dpy as fmm

from hypermagnetics.mt_eval import field_mt
from hypermagnetics.sources import _potential, _total


def potential2D(
    sources: np.ndarray,
    shape: str = "sphere",
    grid: np.ndarray | None = None,
    correction_source: bool = False,
    prism_mt: bool = False,
    idx_single: int | None = None,
    batch_r: bool = False,
):
    """
    Compute the potential at the grid points due to the sources.

    Args:
        sources: Array of source points and their properties.
        shape: Shape of the source (e.g., "sphere" or "prism").
        grid: Optional grid points to evaluate the potential.
        correction_source: Whether to apply correction to the source.
        idx_single: Index of a single source to evaluate.
        batch_r: Whether the grid is provided in batches.

    Returns:
        msp: The magnetic scalar potential at the grid points.
        field: The magnetic field at the grid points.

    The package fmm2dpy calculates the dipolar potential from a magnetic moment.

    Pecularities of fmm2d:
    - Convention in FMM requires negative magnetic moment
    - Adapt to syntax of fmm2dpy [1,1,2] to [2,1]
    - Factor 1 / (2 * pi) needs to manually added to match the ground-truth
    - Dipole vector is written in complex form and potential as directional derivative of monopole
    - Code snippet (https://github.com/flatironinstitute/fmm2d/blob/main/src/laplace/rfmm2d.f#L144):
        if (ifdipole .eq. 1) then
            do i = 1,ns
                do j = 1,nd
                    ztmp = -(dipvec(j,1,i)+eye*dipvec(j,2,i))
                    dipstr1(j,i) = dipstr(j,i)*ztmp
                enddo
            enddo
        endif


    Further notes:
    - In our setup, m is used as magnetic moment (sphere) as well as magnetization (prism).
    - Accordingly, this has to be taken into account when computing the potential.
    """

    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape
    if dim == 3:
        m = m[..., :2]
        r0 = r0[..., :2]
        if grid is not None:
            grid = grid[..., :2]

    if grid is None:
        n_points = n_sources
        targets = None
        source_eval = 2
        target_eval = 0
    else:
        if batch_r:
            n_points = grid.shape[1]
            targets = grid.swapaxes(1, 2)
        else:
            n_points = grid.shape[0]
            targets = grid.swapaxes(0, 1)
        source_eval = 0
        target_eval = 2

    # Convert magnetization to magnetic moment
    if shape == "sphere":
        m = m * (np.pi * size[..., 0:1] ** 2)
    elif shape == "prism":
        m = m * (4 * size[..., 0:1] * size[..., 1:2])

    msp = np.zeros((n_samples, n_points))
    field = np.zeros((n_samples, n_points, 2))
    dur = np.zeros((n_samples,))

    for i in range(n_samples):
        if batch_r:
            targets_i = targets[i]
            grid_i = grid[i]
        else:
            targets_i = targets
            grid_i = grid

        t_start = time.time()
        out = fmm.rfmm2d(
            eps=10 ** (-5),
            sources=r0[i].swapaxes(0, 1),
            charges=None,
            dipstr=np.ones(shape=(n_sources,)),
            dipvec=-m[i].swapaxes(0, 1),
            targets=targets_i,
            nd=1,
            pg=source_eval,
            pgt=target_eval,
        )
        dur[i] = time.time() - t_start

        if grid is None:
            msp[i] = out.pot / (2 * np.pi)
            field[i] = (-1) * out.grad.swapaxes(0, 1) / (2 * np.pi)
        else:
            msp[i] = out.pottarg / (2 * np.pi)
            field[i] = (-1) * out.gradtarg.swapaxes(0, 1) / (2 * np.pi)

        if grid is None or correction_source:
            if type(idx_single) is int:
                slice_single = slice(idx_single, idx_single + 1)
            else:
                slice_single = slice(0, n_sources)

            if shape == "sphere":
                area_n = np.pi * size[i, :, 0:1] ** 2
            elif shape == "prism":
                area_n = 2 * size[i, :, 0:1] * 2 * size[i, :, 1:2]
            else:
                raise ValueError("Unknown shape")

            field[i, slice_single] -= m[i] / 2 / area_n
        else:
            # Correction for physical dipole - Adds an M in the complexity
            for n in range(n_sources):
                if shape == "sphere":
                    idx_in = np.where(
                        np.linalg.norm(grid_i - r0[i][n], axis=1) <= size[i, n, 0]
                    )[0]
                    area_n = np.pi * size[i, n, 0] ** 2
                elif shape == "prism":
                    idx_in = np.where(
                        (np.abs(grid_i[:, 0] - r0[i][n, 0]) <= size[i, n, 0])
                        & (np.abs(grid_i[:, 1] - r0[i][n, 1]) <= size[i, n, 1])
                    )[0]
                    area_n = 2 * size[i, n, 0] * 2 * size[i, n, 1]
                else:
                    raise ValueError("Unknown shape")

                d_in = grid_i[idx_in] - r0[i][n]
                d_in_norm = np.linalg.norm(d_in, axis=1)
                d_in_norm2 = np.where(d_in_norm == 0, np.inf, d_in_norm**2)
                mdotd = np.dot(m[i][n], d_in.T)

                # Correction for point-like dipole
                msp[i, idx_in] -= mdotd / (2 * np.pi * d_in_norm2)
                field[i, idx_in] += (
                    m[i][n] / d_in_norm2[:, None]
                    - 2 * mdotd[:, None] * (d_in) / d_in_norm2[:, None] ** 2
                ) / (2 * np.pi)

                if shape == "sphere":
                    # Correction for physical dipole (elongated cylinder)
                    msp[i, idx_in] += mdotd / 2 / area_n
                    field[i, idx_in] -= m[i][n] / 2 / area_n

                elif shape == "prism":
                    # Correction for physical dipole (elongated prism)
                    if prism_mt:
                        pts = np.concatenate(
                            [
                                grid_i[idx_in],
                                np.zeros((grid_i[idx_in].shape[0], 1)),
                            ],
                            axis=-1,
                        )
                        mt_msp = np.array(
                            _total(
                                _potential, sources[i : i + 1, n : n + 1], pts, "prism"
                            )
                        )
                        msp[i, idx_in] += mt_msp[0]

                        mt_sim, _ = field_mt(
                            sources[i : i + 1, n : n + 1], pts, "prism"
                        )
                        field[i, idx_in] += mt_sim[0, ..., :2]
                    else:
                        msp[i, idx_in] += mdotd / 2 / area_n
                        field[i, idx_in] -= m[i][n] / 2 / area_n
                else:
                    raise ValueError("Unknown shape")

    return msp, field, dur.mean()


def potential2D_loop(
    sources: np.ndarray,
    shape: str = "sphere",
    grid: np.ndarray | None = None,
):
    """
    Compute the potential at the grid points due to the sources.
    The package fmm2dpy calculates the dipolar potential from a magnetic moment.

    Pecularities of fmm2d:
    - Convention in FMM requires negative magnetic moment
    - Adapt to syntax of fmm2dpy [1,1,2] to [2,1]
    - Factor 1 / (2 * pi) needs to manually added to match the ground-truth
    - Dipole vector is written in complex form and potential as directional derivative of monopole
    - Code snippet (https://github.com/flatironinstitute/fmm2d/blob/main/src/laplace/rfmm2d.f#L144):
        if (ifdipole .eq. 1) then
            do i = 1,ns
                do j = 1,nd
                    ztmp = -(dipvec(j,1,i)+eye*dipvec(j,2,i))
                    dipstr1(j,i) = dipstr(j,i)*ztmp
                enddo
            enddo
        endif


    Further notes:
    - In our setup, m is used as magnetic moment (sphere) as well as magnetization (prism).
    - Accordingly, this has to be taken into account when computing the potential.
    """

    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape
    if dim == 3 and shape == "prism":
        m = m[..., :2]
        r0 = r0[..., :2]

    if grid is None:
        msp = np.zeros((n_samples, n_sources))
        field = np.zeros((n_samples, n_sources, 2))
        targets = r0[0].swapaxes(0, 1)
    else:
        if dim == 3 and shape == "prism":
            grid = grid[..., :2]
        msp = np.zeros((n_samples, grid.shape[0]))
        field = np.zeros((n_samples, grid.shape[0], 2))
        targets = grid.swapaxes(0, 1)

    for i in range(n_samples):
        for n in range(n_sources):
            out = fmm.rfmm2d(
                eps=10 ** (-5),
                sources=r0[i, n : n + 1].swapaxes(0, 1),
                charges=None,
                dipstr=np.ones(shape=(1,)),
                dipvec=-m[i, n : n + 1].swapaxes(0, 1),
                targets=targets,
                nd=1,
                pg=0,
                pgt=2,
            )

            if grid is None:
                if shape == "sphere":
                    area_n = np.pi * size[i, n, 0:1] ** 2
                elif shape == "prism":
                    area_n = 2 * size[i, n, 0:1] * 2 * size[i, n, 1:2]
                else:
                    raise ValueError("Unknown shape")
                # Prefactor is missing in FMM
                msp[i] += out.pottarg / (2 * np.pi)
                field[i] -= out.gradtarg.swapaxes(0, 1) / (2 * np.pi)
                field[i][n] -= m[i][n] / area_n / 2
            else:
                # Correction for physical dipole - Adds an M in the complexity
                if shape == "sphere":
                    idx_in = np.where(
                        np.linalg.norm(grid - r0[i][n], axis=1) <= size[i, n, 0]
                    )[0]
                    area_n = np.pi * size[i, n, 0] ** 2
                elif shape == "prism":
                    idx_in = np.where(
                        (np.abs(grid[:, 0] - r0[i][n, 0]) <= size[i, n, 0])
                        & (np.abs(grid[:, 1] - r0[i][n, 1]) <= size[i, n, 1])
                    )[0]
                    area_n = 2 * size[i, n, 0] * 2 * size[i, n, 1]
                else:
                    raise ValueError("Unknown shape")

                idx_out = np.setdiff1d(np.arange(grid.shape[0]), idx_in)

                msp[i, idx_in] += (
                    np.dot(m[i][n], (grid[idx_in] - r0[i][n]).T) / 2 / area_n
                )
                msp[i, idx_out] += out.pottarg[idx_out] / (2 * np.pi)
                field[i, idx_in] -= m[i][n] / area_n / 2
                field[i, idx_out] -= out.gradtarg.swapaxes(0, 1)[idx_out] / (2 * np.pi)

    return msp, field


def field2d_grid(config, msp):
    """
    Requires potential evaluation to be as a meshgrid.
    Take care of numpy's convention for gradient direction.
    """
    msp_grid = msp.reshape((msp.shape[0], config["res"], config["res"]))
    field = np.zeros((*msp_grid.shape, 2))
    for n in range(msp_grid.shape[0]):
        field[n, ..., 0] = np.gradient(
            msp_grid[n], 2 * config["lim"] / config["res"], axis=1
        )
        field[n, ..., 1] = np.gradient(
            msp_grid[n], 2 * config["lim"] / config["res"], axis=0
        )
    return (-1) * field
