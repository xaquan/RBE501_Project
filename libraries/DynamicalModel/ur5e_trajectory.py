"""UR5e quintic trajectories with real-valued Lagrangian dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from ..lagrangianDynamicReal import LagrangeDynamicReal
    from ..manipulatorKinematics import FK_Exponential
except ImportError:  # Allow this file to be run directly.
    import sys

    LIBRARIES_DIR = Path(__file__).resolve().parents[1]
    if str(LIBRARIES_DIR) not in sys.path:
        sys.path.insert(0, str(LIBRARIES_DIR))
    from lagrangianDynamicReal import LagrangeDynamicReal
    from manipulatorKinematics import FK_Exponential


FloatArray = NDArray[np.float64]
JOINT_COUNT = 6


@dataclass(frozen=True)
class Trajectory:
    """Samples of a UR5e trajectory and its required feed-forward torque."""

    time: FloatArray
    position: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    torque: FloatArray

    def save_csv(self, path: str | Path) -> None:
        """Save time, position, velocity, acceleration, and torque samples."""
        output = np.column_stack(
            (
                self.time,
                self.position,
                self.velocity,
                self.acceleration,
                self.torque,
            )
        )
        joint_headers = [f"q{i}" for i in range(1, JOINT_COUNT + 1)]
        velocity_headers = [f"q{i}_dot" for i in range(1, JOINT_COUNT + 1)]
        acceleration_headers = [
            f"q{i}_ddot" for i in range(1, JOINT_COUNT + 1)
        ]
        torque_headers = [f"tau{i}" for i in range(1, JOINT_COUNT + 1)]
        header = ",".join(
            ["time"]
            + joint_headers
            + velocity_headers
            + acceleration_headers
            + torque_headers
        )
        np.savetxt(path, output, delimiter=",", header=header, comments="")


def _joint_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != JOINT_COUNT:
        raise ValueError(
            f"{name} must contain exactly {JOINT_COUNT} values; "
            f"received {vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


@lru_cache(maxsize=1)
def create_ur5e_real_dynamics() -> LagrangeDynamicReal:
    """Create the numerical UR5e Lagrangian model once per process."""
    d1 = 0.163
    d2 = 0.138
    a2 = 0.425
    d3 = 0.131
    a3 = 0.392
    d4 = 0.127
    d5 = 0.100

    # Center-of-mass offsets used by the project's numerical UR5e model.
    c1y = -0.02561
    c1z = 0.00193
    c2z = 0.2125
    c3z = 0.150
    c4y = 0.127
    c5z = 0.100
    wrist_3_mass = 0.365
    wrist_3_c6y = 0.071

    # The project world attaches this fixed VacuumGripper to toolSlot.
    payload_mass = 0.100
    payload_c6y = 0.125
    combined_wrist_3_mass = wrist_3_mass + payload_mass
    c6y = (
        wrist_3_mass * wrist_3_c6y
        + payload_mass * payload_c6y
    ) / combined_wrist_3_mass

    link_transforms = [
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d1], [0, 0, 0, 1]],
        [[0, 0, 1, 0], [0, 1, 0, d2], [-1, 0, 0, d1], [0, 0, 0, 1]],
        [
            [0, 0, 1, a2],
            [0, 1, 0, d2 - d3],
            [-1, 0, 0, d1],
            [0, 0, 0, 1],
        ],
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3],
            [0, 0, -1, d1],
            [0, 0, 0, 1],
        ],
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4],
            [0, 0, -1, d1],
            [0, 0, 0, 1],
        ],
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4],
            [0, 0, -1, d1 - d5],
            [0, 0, 0, 1],
        ],
    ]

    center_transforms = [
        [
            [1, 0, 0, 0],
            [0, 1, 0, c1y],
            [0, 0, 1, d1 + c1z],
            [0, 0, 0, 1],
        ],
        [
            [0, 0, 1, c2z],
            [0, 1, 0, d2],
            [-1, 0, 0, d1],
            [0, 0, 0, 1],
        ],
        [
            [0, 0, 1, a2 + c3z],
            [0, 1, 0, d2 - d3],
            [-1, 0, 0, d1],
            [0, 0, 0, 1],
        ],
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + c4y],
            [0, 0, -1, d1],
            [0, 0, 0, 1],
        ],
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4],
            [0, 0, -1, d1 - c5z],
            [0, 0, 0, 1],
        ],
        [
            [-1, 0, 0, a2 + a3],
            [0, 1, 0, d2 - d3 + d4 + c6y],
            [0, 0, -1, d1 - d5],
            [0, 0, 0, 1],
        ],
    ]

    joint_axes = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
    joint_points = [
        [0, 0, d1],
        [0, d2, d1],
        [a2, d2 - d3, d1],
        [a2 + a3, d2 - d3, d1],
        [a2 + a3, d2 - d3 + d4, d1],
        [a2 + a3, d2 - d3 + d4, d1 - d5],
    ]

    masses = np.array(
        [
            3.761,
            8.058,
            2.846,
            1.370,
            1.300,
            combined_wrist_3_mass,
        ]
    )
    # UR5e link inertia tensors in kg m². The previous implementation set
    # five tensors to zero, which severely underestimated D(q) and produced
    # unusably small model-based PD gains.
    inertias = np.array(
        [
            [
                [0.00700210, 0.00000073, -0.00001053],
                [0.00000073, 0.00648091, 0.00049994],
                [-0.00001053, 0.00049994, 0.00657286],
            ],
            [
                [0.01505885, -0.00005400, 0.00000563],
                [-0.00005400, 0.33388086, -0.00000181],
                [0.00000563, -0.00000181, 0.33247207],
            ],
            [
                [0.00399632, -0.00001365, 0.00137272],
                [-0.00001365, 0.07879254, -0.00000660],
                [0.00137272, -0.00000660, 0.07848510],
            ],
            [
                [0.00165491, -0.00000282, -0.00000438],
                [-0.00000282, 0.00135962, 0.00010157],
                [-0.00000438, 0.00010157, 0.00126279],
            ],
            [
                [0.00135617, -0.00000274, 0.00000444],
                [-0.00000274, 0.00127827, -0.00005048],
                [0.00000444, -0.00005048, 0.00096614],
            ],
            [
                [1.00041563, 0.00000006, -0.00000017],
                [0.00000006, 1.00018908, -0.00000092],
                [-0.00000017, -0.00000092, 1.00048625],
            ],
        ],
        dtype=float,
    )
    initial_q = np.zeros(JOINT_COUNT)

    link_fk = FK_Exponential(
        link_transforms, joint_axes, joint_points, initial_q
    )
    center_fk = FK_Exponential(
        center_transforms, joint_axes, joint_points, initial_q
    )
    return LagrangeDynamicReal(
        link_fk,
        center_fk,
        masses,
        gravity=9.81,
        inertia_array=inertias,
    )


def inverse_dynamics(
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
) -> FloatArray:
    """Return torque from the real-valued Lagrangian dynamics model."""
    q_value = _joint_vector(position, "position")
    qdot_value = _joint_vector(velocity, "velocity")
    qddot_value = _joint_vector(acceleration, "acceleration")
    return create_ur5e_real_dynamics().get_torque(
        q_value, qdot_value, qddot_value
    )


def quintic_trajectory(
    start: ArrayLike,
    goal: ArrayLike,
    duration: float,
    sample_period: float,
) -> Trajectory:
    """Generate a rest-to-rest quintic joint trajectory.

    The endpoint is always included, even when ``duration`` is not an exact
    multiple of ``sample_period``.
    """
    q_start = _joint_vector(start, "start")
    q_goal = _joint_vector(goal, "goal")
    duration = float(duration)
    sample_period = float(sample_period)

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be a positive finite number.")
    if not np.isfinite(sample_period) or sample_period <= 0.0:
        raise ValueError("sample_period must be a positive finite number.")

    time = np.arange(0.0, duration, sample_period, dtype=float)
    time = np.append(time, duration)
    normalized_time = time / duration
    displacement = q_goal - q_start

    blend = (
        10.0 * normalized_time**3
        - 15.0 * normalized_time**4
        + 6.0 * normalized_time**5
    )
    blend_dot = (
        30.0 * normalized_time**2
        - 60.0 * normalized_time**3
        + 30.0 * normalized_time**4
    ) / duration
    blend_ddot = (
        60.0 * normalized_time
        - 180.0 * normalized_time**2
        + 120.0 * normalized_time**3
    ) / duration**2

    position = q_start + np.outer(blend, displacement)
    velocity = np.outer(blend_dot, displacement)
    acceleration = np.outer(blend_ddot, displacement)
    torque = np.vstack(
        [
            inverse_dynamics(q_value, qdot_value, qddot_value)
            for q_value, qdot_value, qddot_value in zip(
                position, velocity, acceleration
            )
        ]
    )

    return Trajectory(time, position, velocity, acceleration, torque)


if __name__ == "__main__":
    # Example motion using the first two poses from the robot controller.
    trajectory = quintic_trajectory(
        start=np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0]),
        goal=np.deg2rad([45.0, -60.0, 90.0, -120.0, -90.0, 30.0]),
        duration=5.0,
        sample_period=0.032,
    )
    output_path = Path(__file__).with_name("ur5e_trajectory.csv")
    trajectory.save_csv(output_path)
    print(f"Saved {trajectory.time.size} trajectory samples to {output_path}")
