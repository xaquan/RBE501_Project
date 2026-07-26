# This script combines the M, C, and G matrices, as derived in matlab, to python for later
# use in the controller

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ur5e_M import ur5e_M
from ur5e_C import ur5e_C
from ur5e_G import ur5e_G


def ur5e_torque(q: ArrayLike, q_dot: ArrayLike, q_ddot: ArrayLike,) -> NDArray[np.float64]:
    #Compute Torque: tau = M(q) q_ddot + C(q, q_dot) q_dot + G(q)
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    qd_arr = np.asarray(q_dot, dtype=float).reshape(-1)
    qdd_arr = np.asarray(q_ddot, dtype=float).reshape(-1)
        
    tau = (ur5e_M(q_arr) @ qdd_arr + ur5e_C(q_arr, qd_arr) @ qd_arr + ur5e_G(q_arr))

    return tau
