import os
import sys
import time

import numpy as np

from magtense import magstatics


def field_mt(sources, r, shape):
    """Finite field in two or three dimensions with MagTense."""
    mu0 = 4 * np.pi * 1e-7
    # Shapes: n_samples, n_sources, dim
    m, r0, size = np.split(sources, 3, axis=-1)
    n_samples, n_sources, dim = r0.shape

    center_pos = np.zeros(shape=(n_samples, n_sources, 3))
    dev_center = np.zeros(shape=(n_samples, n_sources, 3))

    if shape == "sphere":
        # Magnetization is used in MagTense
        # Magnetic moment is used for dipole formula
        m = m / (np.pi * size[..., 0:1] ** 2)

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
                    np.ones((n_samples, n_sources, 1)) * 10,
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
        m = m / (size[..., 0:1] * size[..., 1:2])
    else:
        raise ValueError(f"Unknown source shape: {shape}")

    if dim == 2:
        r0 = np.concatenate([r0, np.zeros((n_samples, n_sources, 1))], axis=-1)
        m = np.concatenate([m, np.zeros((n_samples, n_sources, 1))], axis=-1)
        r = np.concatenate([r, np.zeros((r.shape[0], 1))], axis=-1)
        size = np.concatenate([size, np.ones((n_samples, n_sources, 1))], axis=-1)

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
        field = field.at[i].set(np.array(H_out[:, :dim]) * mu0)

    return field, duration
