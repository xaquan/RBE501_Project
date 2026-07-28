"""Numerical UR5e inverse dynamics with an optional box payload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from .ur5e_C import ur5e_C
    from .ur5e_G import ur5e_G
    from .ur5e_M import ur5e_M
    from .ur5e_payload_C import ur5e_payload_C
    from .ur5e_payload_G import ur5e_payload_G
    from .ur5e_payload_M import ur5e_payload_M
except ImportError:  # Allow direct execution from this directory.
    from ur5e_C import ur5e_C
    from ur5e_G import ur5e_G
    from ur5e_M import ur5e_M
    from ur5e_payload_C import ur5e_payload_C
    from ur5e_payload_G import ur5e_payload_G
    from ur5e_payload_M import ur5e_payload_M


def _joint_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != 6 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain six finite values.")
    return vector


def ur5e_torque(
    q: ArrayLike,
    q_dot: ArrayLike,
    q_ddot: ArrayLike,
    payload_mass: float = 0.0,
) -> NDArray[np.float64]:
    """Return UR5e inverse-dynamics torque for an optional payload.

    ``payload_mass`` is the mass, in kilograms, of a 0.1 m cubic box held by
    the tool. Set it to zero for an unloaded arm.
    """
    q_arr = _joint_vector(q, "q")
    qd_arr = _joint_vector(q_dot, "q_dot")
    qdd_arr = _joint_vector(q_ddot, "q_ddot")
    payload_mass = float(payload_mass)
    if not np.isfinite(payload_mass) or payload_mass < 0.0:
        raise ValueError("payload_mass must be finite and non-negative.")

    mass = ur5e_M(q_arr) + ur5e_payload_M(q_arr, payload_mass)
    coriolis = ur5e_C(q_arr, qd_arr) + ur5e_payload_C(
        q_arr, qd_arr, payload_mass
    )
    gravity = ur5e_G(q_arr) + ur5e_payload_G(q_arr, payload_mass)

    return mass @ qdd_arr + coriolis @ qd_arr + gravity


def torque_history(
    time: ArrayLike,
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    payload_mass: float = 0.0,
) -> NDArray[np.float64]:
    """Calculate one six-joint torque vector for each time sample."""
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    position_arr = np.asarray(position, dtype=float)
    velocity_arr = np.asarray(velocity, dtype=float)
    acceleration_arr = np.asarray(acceleration, dtype=float)
    expected_shape = (time_arr.size, 6)

    if time_arr.size == 0 or not np.all(np.isfinite(time_arr)):
        raise ValueError("time must contain at least one finite sample.")
    for name, values in (
        ("position", position_arr),
        ("velocity", velocity_arr),
        ("acceleration", acceleration_arr),
    ):
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must have shape {expected_shape}.")

    return np.vstack(
        [
            ur5e_torque(q, q_dot, q_ddot, payload_mass)
            for q, q_dot, q_ddot in zip(
                position_arr, velocity_arr, acceleration_arr
            )
        ]
    )


def save_torque_csv(
    path: str | Path,
    time: ArrayLike,
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    payload_mass: float = 0.0,
) -> NDArray[np.float64]:
    """Write ``time,tau1,...,tau6`` to a CSV file and return the torques."""
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    torques = np.abs(torque_history(
        time_arr, position, velocity, acceleration, payload_mass
    ))
    output = np.column_stack((time_arr, torques))
    np.savetxt(
        path,
        output,
        delimiter=",",
        header="time,tau1,tau2,tau3,tau4,tau5,tau6",
        comments="",
    )
    return torques
