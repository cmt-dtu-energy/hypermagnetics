import numpy as np
import fmm2dpy as fmm


def replace_inf_nan(x):
    x = np.where(np.isinf(x), 0.0, x)
    x = np.where(np.isnan(x), 0.0, x)
    return x


def potential2D(sources, grid, shape, correction=True):
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
        grid = grid[..., :2]

    msp = np.zeros((n_samples, grid.shape[0]))
    field = np.zeros((n_samples, grid.shape[0], 2))
    for n in range(n_samples):
        out = fmm.rfmm2d(
            eps=10 ** (-5),
            sources=r0[n].swapaxes(0, 1),
            charges=None,
            dipstr=np.ones(shape=(n_sources,)),
            dipvec=-m[n].swapaxes(0, 1),
            targets=grid.swapaxes(0, 1),
            nd=1,
            pg=0,
            pgt=2,
        )
        # Prefactor is missing in FMM
        msp[n] = out.pottarg / (2 * np.pi)
        field[n] = (-1) * out.gradtarg.swapaxes(0, 1) / (2 * np.pi)

        # Correction for physical dipole - Adds an M in the complexity
        # This works only if sources do not overlap
        if correction:
            for i in range(n_sources):
                if shape == "sphere":
                    inside_idx = np.where(
                        np.linalg.norm(grid - r0[n][i], axis=1) <= size[n, i, 0]
                    )[0]
                elif shape == "prism":
                    inside_idx = np.where(
                        (np.abs(grid[:, 0] - r0[n][i, 0]) <= size[n, i, 0])
                        & (np.abs(grid[:, 1] - r0[n][i, 1]) <= size[n, i, 1])
                    )[0]

                msp[n, inside_idx] += (
                    np.dot(m[n][i], (grid[inside_idx] - r0[n][i]).T)
                    / size[n, i, 0]
                    / (2 * np.pi * size[n, i, 0])
                ) - replace_inf_nan(
                    np.dot(m[n][i], (grid[inside_idx] - r0[n][i]).T)
                    / np.linalg.norm(grid[inside_idx] - r0[n][i], axis=1)
                    / (2 * np.pi * np.linalg.norm(grid[inside_idx] - r0[n][i], axis=1))
                )

    return msp, field


def potential2D_sources(sources, shape):
    """
    Compute the potential at center point r0 of the sources.

    Pecularities of fmm2d:
    - Convention in FMM requires negative magnetic moment
    - Adapt to syntax of fmm2dpy [n,1,2] to [2,1]
    - Factor 1 / (2 * pi) needs to manually added to match the ground-truth
    """

    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape
    if dim == 3 and shape == "prism":
        m = m[..., :2]
        r0 = r0[..., :2]

    msp = np.zeros((n_samples, n_sources))
    field = np.zeros((n_samples, n_sources, 2))
    for n in range(n_samples):
        out = fmm.rfmm2d(
            eps=10 ** (-5),
            sources=r0[n].swapaxes(0, 1),
            charges=None,
            dipstr=np.ones(shape=(n_sources,)),
            dipvec=-m[n].swapaxes(0, 1),
            nd=1,
            pg=2,
            pgt=0,
        )
        # Prefactor is missing in FMM
        msp[n] = out.pot / (2 * np.pi)
        field[n] = (-1) * out.grad.swapaxes(0, 1) / (2 * np.pi)

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
