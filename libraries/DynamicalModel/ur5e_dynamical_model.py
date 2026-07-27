"""UR5e inverse dynamics including a variable payload."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

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

    return tau
