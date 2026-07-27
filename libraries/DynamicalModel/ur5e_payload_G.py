#UR5e gravity matrix

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ur5e_payload_G(
    q: ArrayLike,
    box_mass: float,
) -> NDArray[np.float64]:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size != 6:
        raise ValueError(f"q must contain exactly 6 values; received {q_arr.size}.")
    box_mass = float(box_mass)

    theta2 = q_arr[1]
    theta3 = q_arr[2]
    theta4 = q_arr[3]
    theta5 = q_arr[4]

    t2 = np.sin(theta5)
    t3 = theta2+theta3
    t4 = np.cos(t3)
    t5 = t3+theta4
    t6 = np.cos(t5)
    t7 = np.sin(t5)
    t8 = t4*3.922e-1
    t9 = t7*-9.97e-2
    t10 = t7*9.97e-2
    t11 = t2*t6*4.96e-2
    return np.asarray([0.0, box_mass*(t8+t9+t11+np.cos(theta2)*4.25e-1)*-9.81, box_mass*(t8+t9+t11)*-9.81, box_mass*(t10-t11)*9.81, box_mass*t7*np.cos(theta5)*-4.86576e-1, 0.0], dtype=float)
