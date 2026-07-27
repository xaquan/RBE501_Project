<<<<<<< HEAD
"""Numerical inverse dynamics for the UR5e."""
=======
"""UR5e inverse dynamics including a variable payload."""
>>>>>>> 476d01e9a5e5b54cdfb1cb10b65786f65462e704

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

<<<<<<< HEAD
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
=======
from ur5e_M import ur5e_M
from ur5e_C import ur5e_C
from ur5e_G import ur5e_G
from ur5e_payload_M import ur5e_payload_M
from ur5e_payload_C import ur5e_payload_C
from ur5e_payload_G import ur5e_payload_G


def ur5e_torque(q, q_dot, q_ddot, payload_mass):
    #Compute UR5e joint torque including the attached payload.
    #When the robot is not lifting anything, payload mass can be entered as 0.
    #Otherwise, payload mass is the mass of the box(the dynamics are designed for a box of 0.1m x 0.1m x 0.1m size
    #The inputs are joint positions(q_arr), joint velocities(qd_arr), joint accelerations(qdd_arr), and the payload

    q_arr = np.asarray(q, dtype=float).reshape(-1)
    qd_arr = np.asarray(q_dot, dtype=float).reshape(-1)
    qdd_arr = np.asarray(q_ddot, dtype=float).reshape(-1)
    payload_mass = float(payload_mass)

    M = ur5e_M(q_arr) + ur5e_payload_M(q_arr, payload_mass)
    C = ur5e_C(q_arr, qd_arr) + ur5e_payload_C(
        q_arr,
        qd_arr,
        payload_mass,
    )
    G = ur5e_G(q_arr) + ur5e_payload_G(q_arr, payload_mass)

    tau = M @ qdd_arr + C@qd_arr + G
>>>>>>> 476d01e9a5e5b54cdfb1cb10b65786f65462e704

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
