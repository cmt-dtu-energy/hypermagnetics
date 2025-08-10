import numpy as np
import fmm2dpy as fmm


def potential2D(sources, grid, shape):
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
            pgt=1,
        )
        # Prefactor is missing in FMM
        msp[n] = out.pottarg / (2 * np.pi)

        # Correction for physical dipole
        # This works only if sources do not overlap
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
            ) - (
                np.dot(m[n][i], (grid[inside_idx] - r0[n][i]).T)
                / np.linalg.norm(grid[inside_idx] - r0[n][i], axis=1)
                / (2 * np.pi * np.linalg.norm(grid[inside_idx] - r0[n][i], axis=1))
            )

    return msp


def field2d(config, msp):
    """
    Requires potential evaluation to be a meshgrid
    """
    msp_grid = msp.reshape((msp.shape[0], config["res"], config["res"]))
    field = np.gradient(msp_grid, config["lim"] / config["res"], axis=1)
    return field
