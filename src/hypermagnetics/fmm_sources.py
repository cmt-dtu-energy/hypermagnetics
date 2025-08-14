import numpy as np
import fmm2dpy as fmm


def replace_inf_nan(x):
    x = np.where(np.isinf(x), 0.0, x)
    x = np.where(np.isnan(x), 0.0, x)
    return x


def potential2D(
    sources: np.ndarray,
    shape: str = "sphere",
    grid: np.ndarray | None = None,
    correction: bool = True,
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
        source_eval = 2
        target_eval = 0
        targets = None
        msp = np.zeros((n_samples, n_sources))
        field = np.zeros((n_samples, n_sources, 2))
    else:
        if dim == 3 and shape == "prism":
            grid = grid[..., :2]
        source_eval = 0
        target_eval = 2
        targets = grid.swapaxes(0, 1)
        msp = np.zeros((n_samples, grid.shape[0]))
        field = np.zeros((n_samples, grid.shape[0], 2))

    for i in range(n_samples):
        out = fmm.rfmm2d(
            eps=10 ** (-5),
            sources=r0[i].swapaxes(0, 1),
            charges=None,
            dipstr=np.ones(shape=(n_sources,)),
            dipvec=-m[i].swapaxes(0, 1),
            targets=targets,
            nd=1,
            pg=source_eval,
            pgt=target_eval,
        )
        if grid is None:
            msp[i] = out.pot / (2 * np.pi)
            field[i] = (-1) * out.grad.swapaxes(0, 1) / (2 * np.pi)

            if correction:
                field[i] = -m[i] / (np.pi * size[i, :, 0:1] ** 2) / 2
        else:
            # Prefactor is missing in FMM
            msp[i] = out.pottarg / (2 * np.pi)
            field[i] = (-1) * out.gradtarg.swapaxes(0, 1) / (2 * np.pi)

            # Correction for physical dipole - Adds an M in the complexity
            # This works only if sources do not overlap
            if correction:
                for n in range(n_sources):
                    if shape == "sphere":
                        inside_idx = np.where(
                            np.linalg.norm(grid - r0[i][n], axis=1) <= size[i, n, 0]
                        )[0]
                    elif shape == "prism":
                        inside_idx = np.where(
                            (np.abs(grid[:, 0] - r0[i][n, 0]) <= size[i, n, 0])
                            & (np.abs(grid[:, 1] - r0[i][n, 1]) <= size[i, n, 1])
                        )[0]
                    else:
                        raise ValueError("Unknown shape")

                    msp[i, inside_idx] += (
                        np.dot(m[i][n], (grid[inside_idx] - r0[i][n]).T)
                        / size[i, n, 0]
                        / (2 * np.pi * size[i, n, 0])
                    ) - replace_inf_nan(
                        np.dot(m[i][n], (grid[inside_idx] - r0[i][n]).T)
                        / np.linalg.norm(grid[inside_idx] - r0[i][n], axis=1)
                        / (
                            2
                            * np.pi
                            * np.linalg.norm(grid[inside_idx] - r0[i][n], axis=1)
                        )
                    )

                    field[i, inside_idx] = -m[i][n] / (np.pi * size[i, n, 0] ** 2) / 2

    return msp, field


def potential2D_loop(
    sources: np.ndarray,
    shape: str = "sphere",
    grid: np.ndarray | None = None,
    correction: bool = True,
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
        targets = r0[0]
        msp = np.zeros((n_samples, n_sources))
        field = np.zeros((n_samples, n_sources, 2))
    else:
        if dim == 3 and shape == "prism":
            grid = grid[..., :2]
        targets = grid
        msp = np.zeros((n_samples, grid.shape[0]))
        field = np.zeros((n_samples, grid.shape[0], 2))

    for i in range(n_samples):
        for n in range(n_sources):
            out = fmm.rfmm2d(
                eps=10 ** (-5),
                sources=r0[i, n : n + 1].swapaxes(0, 1),
                charges=None,
                dipstr=np.ones(shape=(1,)),
                dipvec=-m[i, n : n + 1].swapaxes(0, 1),
                targets=targets.swapaxes(0, 1),
                nd=1,
                pg=0,
                pgt=2,
            )

            if grid is None:
                # Prefactor is missing in FMM
                msp[i] += out.pottarg / (2 * np.pi)
                field[i] -= out.gradtarg.swapaxes(0, 1) / (2 * np.pi)
                if correction:
                    field[i][n] -= m[i][n] / (np.pi * size[i, n, 0:1] ** 2) / 2
            else:
                # Correction for physical dipole - Adds an M in the complexity
                # This works only if sources do not overlap
                if correction:
                    if shape == "sphere":
                        inside_idx = np.where(
                            np.linalg.norm(grid - r0[i][n], axis=1) <= size[i, n, 0]
                        )[0]
                    elif shape == "prism":
                        inside_idx = np.where(
                            (np.abs(grid[:, 0] - r0[i][n, 0]) <= size[i, n, 0])
                            & (np.abs(grid[:, 1] - r0[i][n, 1]) <= size[i, n, 1])
                        )[0]
                    else:
                        raise ValueError("Unknown shape")

                    outside_idx = np.setdiff1d(np.arange(grid.shape[0]), inside_idx)

                    msp[i, inside_idx] += (
                        np.dot(m[i][n], (grid[inside_idx] - r0[i][n]).T)
                        / size[i, n, 0]
                        / (2 * np.pi * size[i, n, 0])
                    )
                    msp[i, outside_idx] += out.pottarg[outside_idx] / (2 * np.pi)

                    field[i, inside_idx] -= m[i][n] / (np.pi * size[i, n, 0] ** 2) / 2
                    field[i, outside_idx] -= out.gradtarg.swapaxes(0, 1)[
                        outside_idx
                    ] / (2 * np.pi)
                else:
                    msp[i] += out.pottarg / (2 * np.pi)
                    field[i] -= out.gradtarg.swapaxes(0, 1) / (2 * np.pi)

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
