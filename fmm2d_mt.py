import os
import sys
import time
import numpy as np
import fmm2dpy as fmm

from scipy.special import elliprf, elliprj, ellipk, ellipe

from magtense import magstatics


# Elliptic integrals
def ellippi(n, m):
    """
    Complete elliptic integral of the third kind in terms of symmetric elliptic integrals.
    The relation is given in this stackoverflow post:
    https://stackoverflow.com/questions/77488171/complete-integral-of-the-third-kind-using-scipy-python-and-mathematica
    and on this wiki page:
    https://en.wikipedia.org/wiki/Carlson_symmetric_form

    The wiki page also explains how the 'Carlson symmetric forms of elliptic integrals' is a set of
    5 integrals which form a basis of all elliptic integrals. An older basis is the 3 Legendre forms,
    i.e. the complete elliptic integral of first, second and third kind (K, E and Π respectively).
    Supposedly the Carlson forms are computationally cheaper, so scipy computes the Legendre forms in
    terms of the Carlson forms, except Π which is not implemented at all.
    """
    return elliprf(0, 1 - m, 1) + (n / 3) * elliprj(0, 1 - m, 1, 1 - n)


def K_func(ksq):
    """The function K(1-k^2) where ksq = k^2"""
    return ellipk(1 - ksq)


def E_func(ksq):
    """The function E(1-k^2) where ksq = k^2"""
    return ellipe(1 - ksq)


def P_func(gamma, ksq):
    """The function Π(1-γ, 1-k^2) where ksq = k^2"""
    return ellippi(1 - gamma**2, 1 - ksq)


# Minor functions
def alpha(z, rho, R, L, p=1):
    """
    z, rho are cylindrical coordinates. R, L are the radius and length of the cylinder.
    There is a plus and minus version of alpha, specified by p = +- 1.

    Note that Cagliaci et al. defines L as half the cylinders length (see fig. 1 of their paper).
    I prefer to define L as the full length, hence the factor of 1/2 wherever L appears in the formulas.
    """
    xi = z + p * L / 2
    return 1 / np.sqrt(xi**2 + (rho + R) ** 2)


def ksq_func(z, rho, R, L, p=1):
    """See documentation for alpha"""
    xi = z + p * L / 2
    return (xi**2 + (rho - R) ** 2) / (xi**2 + (rho + R) ** 2)


# Combinations of elliptic integrals
def P1_func(z, rho, R, L, p=1):
    """See documentation for alpha"""
    ksq = ksq_func(z, rho, R, L, p=p)
    K = K_func(ksq)
    E = E_func(ksq)
    return K - 2 / (1 - ksq) * (K - E)


def P2_func(z, rho, R, L, p=1):
    """See documentation for alpha

    When rho -> R, I found that P diverges and gamma*P is discontinuous,
    but you get the right B-field outside the cylinder (i.e. for |z| > L/2),
    when setting P = 0.
    The P2 function is discontinuous, but a couple terms cancel in the expressions
    for the full B-field, removing the discontinuity for |z| > L/2.
    The same holds for P3 and P4.
    """
    gamma = (rho - R) / (rho + R)
    ksq = ksq_func(z, rho, R, L, p=p)
    K = K_func(ksq)
    P = P_func(gamma, ksq)
    P[np.logical_and(rho == R, np.abs(z) > L / 2)] = (
        0  # Handle special case where rho = R
    )
    return -gamma / (1 - gamma**2) * (P - K) - 1 / (1 - gamma**2) * (gamma**2 * P - K)


def P3_func(z, rho, R, L, p=1):
    """See documentation for P2_func"""
    gamma = (rho - R) / (rho + R)
    ksq = ksq_func(z, rho, R, L, p=p)
    K = K_func(ksq)
    E = E_func(ksq)
    P = P_func(gamma, ksq)
    P[np.logical_and(rho == R, np.abs(z) > L / 2)] = (
        0  # Handle special case where rho = R
    )
    return 1 / (1 - ksq) * (K - E) - gamma**2 / (1 - gamma**2) * (P - K)


def P4_func(z, rho, R, L, p=1, P1=None):
    """See documentation for P2_func"""
    gamma = (rho - R) / (rho + R)
    ksq = ksq_func(z, rho, R, L, p=p)
    K = K_func(ksq)
    P = P_func(gamma, ksq)
    P[np.logical_and(rho == R, np.abs(z) > L / 2)] = (
        0  # Handle special case where rho = R
    )
    if P1 is None:
        P1 = P1_func(z, rho, R, L, p=p)
    return (
        gamma / (1 - gamma**2) * (P - K)
        + gamma / (1 - gamma**2) * (gamma**2 * P - K)
        - P1
    )


def cylindrical_magnet_field(m, R, length, r):
    """
    Total magnetic field from a uniformly magnetised cylinder.
    Implemented as a single function to avoid repeating intermediate steps.
    Based on the paper 'Exact expression for the magnetic field of a finite
    cylinder with arbitrary uniform magnetization' by Cacliagi et al.
    (https://www.sciencedirect.com/science/article/pii/S0304885317334662)

    Parameters:
        m: Magnetisation
        R: Cylinder radius
        length: Cylinder length
        r: Evaluation points from np.meshgrid

    Returns:
        Bx, By, Bz: Arrays with magnetic field in Cartesian coordinates.

    Note: we are still repeating the calculations of some elliptic integrals, because
    the same elliptic integral appears in several of the Pn functions. In other words,
    the functions K_func, E_func and P_func are called several times with the same input.
    """
    mu0 = 4 * np.pi * 10 ** (-7)
    Mx, My, Mz = m

    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Auxiliary functions
    alphaPlus = alpha(z, rho, R, length, p=1)
    betaPlus = (z + length / 2) * alphaPlus
    alphaMinus = alpha(z, rho, R, length, p=-1)
    betaMinus = (z - length / 2) * alphaMinus
    P1_plus = P1_func(z, rho, R, length, p=1)
    P1_minus = P1_func(z, rho, R, length, p=-1)
    P2_plus = P2_func(z, rho, R, length, p=1)
    P2_minus = P2_func(z, rho, R, length, p=-1)
    P3_plus = P3_func(z, rho, R, length, p=1)
    P3_minus = P3_func(z, rho, R, length, p=-1)
    P4_plus = P4_func(z, rho, R, length, p=1, P1=P1_plus)
    P4_minus = P4_func(z, rho, R, length, p=-1, P1=P1_minus)

    # Map to local coordinate system where the transverse magnetisation is along the x-axis
    Mtrans = np.sqrt(Mx**2 + My**2)
    phiM = np.arctan2(My, Mx)  # Azimuthal angle of magnetisation
    phi = phi - phiM

    ### Magnetic field components ###
    # z component
    Bz_ax = (
        mu0 * Mz * R / (np.pi * (rho + R)) * (betaPlus * P2_plus - betaMinus * P2_minus)
    )  # z-component from axial magnetisation
    Bz_trans = (
        mu0
        * Mtrans
        * R
        * np.cos(phi)
        / np.pi
        * (alphaPlus * P1_plus - alphaMinus * P1_minus)
    )  # z-component from transverse magnetisation
    Bz = Bz_ax + Bz_trans  # z-component of total field

    # rho component
    Brho_ax = (
        mu0 * Mz * R / np.pi * (alphaPlus * P1_plus - alphaMinus * P1_minus)
    )  # rho-component from axial magnetisation
    Hrho_trans = (
        Mtrans
        * R
        * np.cos(phi)
        / (2 * np.pi * rho)
        * (betaPlus * P4_plus - betaMinus * P4_minus)
    )  # rho-component of H-field from transverse magnetisation

    # phi component
    Bphi = (
        mu0
        * Mtrans
        * R
        * np.sin(phi)
        / (np.pi * rho)
        * (betaPlus * P3_plus - betaMinus * P3_minus)
    )

    # x component
    if type(Hrho_trans) is np.ndarray:  # Check if array or scalar
        Bx = (mu0 * Hrho_trans + Brho_ax) * np.cos(phi) - Bphi * np.sin(phi)
        selector = np.logical_and(
            rho < R, np.abs(z) < length / 2
        )  # Select points inside the cylinder
        Bx[selector] = Bx[selector] + mu0 * Mtrans  # Add magnetisation inside cylinder
    else:
        if rho < R and np.abs(z) < length / 2:
            Bx = (mu0 * Hrho_trans + Brho_ax) * np.cos(phi) - Bphi * np.sin(phi)
        else:
            Bx = (
                (mu0 * Hrho_trans + Brho_ax) * np.cos(phi)
                + mu0 * Mtrans
                - Bphi * np.sin(phi)
            )

    # y component
    By = (mu0 * Hrho_trans + Brho_ax) * np.sin(phi) + Bphi * np.cos(phi)

    # At rho = 0, there are division-by-zero errors in the numerical implementation, but
    # the rho -> 0 limit can be taken analytically.
    # Use the analytical solutions for this special case
    if type(Bz) is np.ndarray:
        # Auxiliary variables
        selectorOnAxis = np.logical_and(
            rho == 0, np.abs(z) < length / 2
        )  # Select points on the central axis inside the cylinder
        xiPlus = z[rho == 0] + length / 2
        xiMinus = z[rho == 0] - length / 2

        # For the on-axis z-component, only the axial magnetisation contributes
        # See Eq. 14 of "The magnetic field of a finite solenoid" 1960, by E. Callaghan and S. Maslan
        Bz[rho == 0] = (
            mu0
            * Mz
            / 2
            * (
                xiPlus / np.sqrt(xiPlus**2 + R**2)
                - xiMinus / np.sqrt(xiMinus**2 + R**2)
            )
        )

        # The on-axis x-component equals the on-axis rho-component with phi = 0.
        # For the on-axis rho-component, only the transverse magnetisation contributes.
        # See Eq. 26 of Caciagli et al.
        Hrho_trans[rho == 0] = (
            -Mtrans
            / 4
            * (
                xiPlus / np.sqrt(xiPlus**2 + R**2)
                - xiMinus / np.sqrt(xiMinus**2 + R**2)
            )
        )
        Bx[rho == 0] = mu0 * Hrho_trans[rho == 0]
        Bx[selectorOnAxis] = Bx[selectorOnAxis] + mu0 * Mtrans

        # There is no azimuthal B-field on the rho=0 axis, nor a y-component
        Bphi[rho == 0] = 0
        By[rho == 0] = 0

    # Map B-field back to global coordinates
    Bx, By = (
        np.cos(phiM) * Bx - np.sin(phiM) * By,
        np.sin(phiM) * Bx + np.cos(phiM) * By,
    )

    return Bx, By, Bz


def field_cylinder_exact(sources, r, length=100):
    """
    Finite field in two or three dimensions with excact implementation for cylindrical tiles
    """
    mu0 = 4 * np.pi * 1e-7
    # Shapes: n_samples, n_sources, dim
    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape

    # Magnetization is used in MagTense
    # Magnetic moment is used for dipole formula
    # m = m / (np.pi * size[..., 0:1] ** 2)

    if dim == 2:
        r0 = np.concatenate([r0, np.zeros((n_samples, n_sources, 1))], axis=-1)
        m = np.concatenate([m, np.zeros((n_samples, n_sources, 1))], axis=-1)
        r = np.concatenate([r, np.zeros((r.shape[0], 1))], axis=-1)
        size = np.concatenate([size, np.ones((n_samples, n_sources, 1))], axis=-1)

    field = np.zeros((n_samples, r.shape[0], dim))
    start_time = time.time()
    for i in range(n_samples):
        for n in range(n_sources):
            Bx, By, Bz = cylindrical_magnet_field(
                m[i, n], size[i, n, 0], length, r - r0[i, n]
            )
            field[i, :, 0] += Bx / mu0
            field[i, :, 1] += By / mu0
            if dim == 3:
                field[i, :, 2] += Bz / mu0

            # Convert to H-field
            idx_in = np.where(np.linalg.norm(r - r0[i][n], axis=1) <= size[i, n, 0])[0]
            field[i, idx_in, :2] -= m[i, n, :2]
            if dim == 3:
                field[i, idx_in, 2] -= m[i, n, 2]

    duration = time.time() - start_time

    return field, duration


def field_mt(sources, r, shape, length=100):
    """Finite field in two or three dimensions with MagTense."""
    mu0 = 4 * np.pi * 1e-7
    # Shapes: n_samples, n_sources, dim
    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape
    if n_samples == 0:
        raise ValueError("No sources provided for MagTense field calculation.")

    center_pos = np.zeros(shape=(n_samples, n_sources, 3))
    dev_center = np.zeros(shape=(n_samples, n_sources, 3))

    if shape == "sphere":
        # Magnetization is used in MagTense
        # Magnetic moment is used for dipole formula
        # m = m / (np.pi * size[..., 0:1] ** 2)

        # 2D is simulated with an elongated cylinder
        if dim == 2:
            tile_type = 1
            center_pos = np.concatenate(
                [
                    size[..., 0].reshape((n_samples, n_sources, 1)) / 2,
                    np.zeros((n_samples, n_sources, 2)),
                ],
                axis=-1,
            )
            dev_center = np.concatenate(
                [
                    size[..., 0].reshape((n_samples, n_sources, 1)),
                    np.ones((n_samples, n_sources, 1)) * 1.9999 * np.pi,
                    np.ones((n_samples, n_sources, 1)) * length,
                ],
                axis=-1,
            )

        else:
            tile_type = 7

    elif shape == "prism":
        tile_type = 2
        # Prism with side lengths [2a, 2b] defined in this repo
        size = size * 2
        # Magnetization is used in MagTense
        # Magnetic moment is used for dipole formula
        # m = m / (size[..., 0:1] * size[..., 1:2])
    else:
        raise ValueError(f"Unknown source shape: {shape}")

    if dim == 2:
        r0 = np.concatenate([r0, np.zeros((n_samples, n_sources, 1))], axis=-1)
        m = np.concatenate([m, np.zeros((n_samples, n_sources, 1))], axis=-1)
        r = np.concatenate([r, np.zeros((r.shape[0], 1))], axis=-1)
        size = np.concatenate(
            [size, np.ones((n_samples, n_sources, 1)) * length], axis=-1
        )

    m_norm = np.linalg.norm(m, axis=-1, keepdims=True)
    mag_angles = np.concatenate(
        [
            np.arccos(m[..., 2] / m_norm[..., 0]).reshape(n_samples, n_sources, 1),
            np.arctan2(m[..., 1], m[..., 0]).reshape(n_samples, n_sources, 1),
        ],
        axis=-1,
    )

    field = np.zeros((n_samples, r.shape[0], dim))
    for i in range(n_samples):
        tiles = magstatics.Tiles(
            n=n_sources,
            M_rem=m_norm[i] / mu0,
            mag_angle=mag_angles[i],
            tile_type=tile_type,
            size=size[i],
            offset=r0[i],
            center_pos=center_pos[i],
            dev_center=dev_center[i],
        )
        devnull = open("/dev/null", "w")
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        start_time = time.time()
        _, H_out = magstatics.run_simulation(tiles, r)
        duration = time.time() - start_time
        os.dup2(oldstdout_fno, 1)
        field[i] = np.array(H_out[:, :dim]) * mu0

    return field, duration


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

    for i in range(n_samples):
        if batch_r:
            targets_i = targets[i]
            grid_i = grid[i]
        else:
            targets_i = targets
            grid_i = grid

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
                    msp[i, idx_in] += mdotd / 2 / area_n
                    if prism_mt:
                        mt_sim, _ = field_mt(
                            sources[i : i + 1, n : n + 1],
                            np.concatenate(
                                [
                                    grid_i[idx_in],
                                    np.zeros((grid_i[idx_in].shape[0], 1)),
                                ],
                                axis=-1,
                            ),
                            "prism",
                        )
                        field[i, idx_in] += mt_sim[0, ..., :2]
                    else:
                        field[i, idx_in] -= m[i][n] / 2 / area_n
                else:
                    raise ValueError("Unknown shape")

    return msp, field
