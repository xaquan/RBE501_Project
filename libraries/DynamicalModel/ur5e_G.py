# This script translates the Gravity matrix, as derived in matlab, to python for later 
# use in the controller

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ur5e_G(q: ArrayLike) -> NDArray[np.float64]:
    #Return the UR5e gravity torque vector G(q)
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size != 6:
        raise ValueError(f"q must contain exactly 6 values; received {q_arr.size}.")
    
    theta1 = q_arr[0]
    theta2 = q_arr[1]
    theta3 = q_arr[2]
    theta4 = q_arr[3]
    theta5 = q_arr[4]
    theta6 = q_arr[5]

    t2 = np.sin(theta5)
    t3 = theta2+theta3
    t4 = np.cos(t3)
    t5 = t3+theta4
    t6 = np.cos(t5)
    t7 = np.sin(t5)
    t8 = t4*-1.8439152642e+1
    t10 = t7*1.825114203
    t11 = t2*t6*-5.6086678665e-1
    return np.array([0.0, t8+t10+t11-np.cos(theta2)*4.13172675e+1, t8+t10+t11, t10+t11, t7*np.cos(theta5)*-5.6086678665e-1, 0.0], dtype=float)
