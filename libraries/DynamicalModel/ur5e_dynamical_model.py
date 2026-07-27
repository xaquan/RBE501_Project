"""Numerical inverse dynamics for the UR5e."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from .ur5e_M import ur5e_M
    from .ur5e_C import ur5e_C
    from .ur5e_G import ur5e_G
except ImportError:  # Allow this file to be run directly.
    from ur5e_M import ur5e_M
    from ur5e_C import ur5e_C
    from ur5e_G import ur5e_G


def ur5e_torque(
    q: ArrayLike,
    q_dot: ArrayLike,
    q_ddot: ArrayLike,
) -> NDArray[np.float64]:
    """Return ``M(q) q_ddot + C(q, q_dot) q_dot + G(q)``."""
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    qd_arr = np.asarray(q_dot, dtype=float).reshape(-1)
    qdd_arr = np.asarray(q_ddot, dtype=float).reshape(-1)

    for name, vector in (
        ("q", q_arr),
        ("q_dot", qd_arr),
        ("q_ddot", qdd_arr),
    ):
        if vector.size != 6 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain six finite values.")

    return (
        ur5e_M(q_arr) @ qdd_arr
        + ur5e_C(q_arr, qd_arr) @ qd_arr
        + ur5e_G(q_arr)
    )
