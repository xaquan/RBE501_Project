"""Real-valued Lagrangian dynamics calculated with NumPy.

This module performs no symbolic differentiation. Given numerical forward-
kinematics models, it evaluates the joint-space mass matrix, Coriolis matrix,
gravity vector, and inverse-dynamics torque using floating-point operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class LagrangeDynamicReal:
    """Numerical rigid-body dynamics for a serial revolute manipulator.

    Args:
        fk: Forward-kinematics model for the link frames.
        fkc: Forward-kinematics model whose frame origins are link centers of
            mass. If omitted, the origins from ``fk`` are used.
        masses: One mass per joint/link, in kilograms.
        gravity: Positive gravitational acceleration, in m/s².
        inertia_array: One 3x3 body-frame inertia matrix per link.
        difference_step: Joint perturbation used by central differences.
    """

    def __init__(
        self,
        fk: Any,
        fkc: Any,
        masses: ArrayLike,
        gravity: float,
        inertia_array: ArrayLike | None,
        difference_step: float = 1e-5,
    ) -> None:
        self.fk = fk
        self.fkc = fkc
        self.masses = np.asarray(masses, dtype=float).reshape(-1)
        self.num_joints = self.masses.size
        self.g = float(gravity)
        self.difference_step = float(difference_step)

        if self.num_joints == 0:
            raise ValueError("masses must contain at least one value.")
        if not np.all(np.isfinite(self.masses)) or np.any(self.masses < 0.0):
            raise ValueError("masses must contain finite non-negative values.")
        if not np.isfinite(self.g):
            raise ValueError("gravity must be finite.")
        if (
            not np.isfinite(self.difference_step)
            or self.difference_step <= 0.0
        ):
            raise ValueError("difference_step must be positive and finite.")

        if inertia_array is None:
            self.inertia_array = np.zeros(
                (self.num_joints, 3, 3), dtype=float
            )
        else:
            self.inertia_array = np.asarray(
                inertia_array, dtype=float
            )
            expected_shape = (self.num_joints, 3, 3)
            if self.inertia_array.shape != expected_shape:
                raise ValueError(
                    f"inertia_array must have shape {expected_shape}; "
                    f"received {self.inertia_array.shape}."
                )
            if not np.all(np.isfinite(self.inertia_array)):
                raise ValueError(
                    "inertia_array must contain only finite values."
                )

        self.q = np.zeros(self.num_joints)
        self.qdot = np.zeros(self.num_joints)
        self.qddot = np.zeros(self.num_joints)
        self.DCG_matrices: list[FloatArray] = []
        self._precompute_kinematic_constants()

    def _precompute_kinematic_constants(self) -> None:
        """Extract constant FK data once for pure-NumPy evaluations."""
        try:
            self._joint_axes = np.asarray(
                self.fk.home_omegas, dtype=float
            ).reshape(self.num_joints, 3)
            self._joint_points = np.asarray(
                self.fk.home_positions, dtype=float
            ).reshape(self.num_joints, 3)
            self._link_home_transforms = np.asarray(
                self.fk.home_trans_mtx, dtype=float
            ).reshape(self.num_joints, 4, 4)

            center_model = self.fkc if self.fkc is not None else self.fk
            self._center_home_transforms = np.asarray(
                center_model.home_trans_mtx, dtype=float
            ).reshape(self.num_joints, 4, 4)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "FK models must expose numerical home_omegas, "
                "home_positions, and home_trans_mtx values."
            ) from error

        self._joint_linear_twists = -np.cross(
            self._joint_axes, self._joint_points
        )
        self._axis_skew = np.array(
            [self._skew(axis) for axis in self._joint_axes],
            dtype=float,
        )
        self._axis_skew_squared = np.matmul(
            self._axis_skew, self._axis_skew
        )

    def get_torque(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        qddot: ArrayLike,
    ) -> FloatArray:
        """Return ``D(q) qddot + C(q, qdot) qdot + G(q)``."""
        qddot_array = self._joint_vector(qddot, "qddot")
        mass, coriolis, gravity = self.get_DCG_matrices(
            q, qdot, qddot_array
        )
        return (
            mass @ qddot_array
            + coriolis @ self.qdot
            + gravity[:, 0]
        )

    def __pd_params_calculate(self, b, xi, ts, j, disturbance=0.0):
        wn = 5.3/(ts*xi)
        kp = wn**2 * j + disturbance
        kd = 2*xi*wn*j - b
        return kp, kd
    
    def calculate_pd_params_for_ur5e(self, D):
        """Calculate PD parameters for each joint of the UR5e robot."""
        HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
        
        pd_params = []
        for i in range(len(self.masses) - 1, -1, -1):
            print(f"Mass {i}: {self.masses[i]} kg")
            j = D[i, i]
            kp, kd = self.__pd_params_calculate(b=1, xi=1, ts=0.5, j=j)
            pd_params.append((kp, kd))
            
        return pd_params

    def get_DCG_matrices(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        qddot: ArrayLike,
    ) -> list[FloatArray]:
        """Return numerical ``[D(q), C(q, qdot), G(q)]`` arrays.

        ``G`` is returned as an ``(n, 1)`` column to preserve the shape used by
        the earlier symbolic implementation.
        """
        self.q = self._joint_vector(q, "q")
        self.qdot = self._joint_vector(qdot, "qdot")
        self.qddot = self._joint_vector(qddot, "qddot")

        mass, potential = self._mass_and_potential(self.q)
        mass_derivatives = np.empty(
            (
                self.num_joints,
                self.num_joints,
                self.num_joints,
            ),
            dtype=float,
        )
        gravity_vector = np.empty(self.num_joints, dtype=float)

        for coordinate in range(self.num_joints):
            q_plus = self.q.copy()
            q_minus = self.q.copy()
            q_plus[coordinate] += self.difference_step
            q_minus[coordinate] -= self.difference_step

            mass_plus, potential_plus = self._mass_and_potential(q_plus)
            mass_minus, potential_minus = self._mass_and_potential(q_minus)
            denominator = 2.0 * self.difference_step
            mass_derivatives[coordinate] = (
                mass_plus - mass_minus
            ) / denominator
            gravity_vector[coordinate] = (
                potential_plus - potential_minus
            ) / denominator

        coriolis = self._coriolis_matrix(mass_derivatives, self.qdot)
        gravity_column = gravity_vector.reshape(self.num_joints, 1)

        self.potential_energy = potential
        self.DCG_matrices = [mass, coriolis, gravity_column]
        return self.DCG_matrices

    def _joint_vector(self, values: ArrayLike, name: str) -> FloatArray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != self.num_joints:
            raise ValueError(
                f"{name} must contain {self.num_joints} values; "
                f"received {vector.size}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector

    def _mass_and_potential(
        self, q: FloatArray
    ) -> tuple[FloatArray, float]:
        """Evaluate the mass matrix and potential energy at one real q."""
        (
            link_rotations,
            center_positions,
            angular_axes,
            spatial_linear,
        ) = self._numerical_kinematics(q)

        mass_matrix = np.zeros(
            (self.num_joints, self.num_joints), dtype=float
        )
        potential_energy = 0.0

        for link in range(self.num_joints):
            center_position = center_positions[link]

            linear_jacobian = np.zeros(
                (3, self.num_joints), dtype=float
            )
            angular_jacobian = np.zeros(
                (3, self.num_joints), dtype=float
            )
            for joint in range(link + 1):
                omega = angular_axes[:, joint]
                linear_jacobian[:, joint] = (
                    np.cross(omega, center_position)
                    + spatial_linear[:, joint]
                )
                angular_jacobian[:, joint] = omega

            rotation = link_rotations[link]
            world_inertia = (
                rotation
                @ self.inertia_array[link]
                @ rotation.T
            )
            mass_matrix += (
                self.masses[link]
                * linear_jacobian.T
                @ linear_jacobian
                + angular_jacobian.T
                @ world_inertia
                @ angular_jacobian
            )
            potential_energy += (
                self.masses[link] * self.g * center_position[2]
            )

        # Remove insignificant numerical asymmetry.
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        return mass_matrix, float(potential_energy)

    def _numerical_kinematics(
        self, q: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate poses and spatial twists with NumPy product of exponentials."""
        cumulative = np.eye(4, dtype=float)
        link_rotations = np.empty(
            (self.num_joints, 3, 3), dtype=float
        )
        center_positions = np.empty(
            (self.num_joints, 3), dtype=float
        )
        angular_axes = np.empty(
            (3, self.num_joints), dtype=float
        )
        spatial_linear = np.empty(
            (3, self.num_joints), dtype=float
        )

        for joint in range(self.num_joints):
            rotation = cumulative[:3, :3]
            translation = cumulative[:3, 3]
            omega = rotation @ self._joint_axes[joint]
            linear = (
                np.cross(translation, omega)
                + rotation @ self._joint_linear_twists[joint]
            )
            angular_axes[:, joint] = omega
            spatial_linear[:, joint] = linear

            cumulative = cumulative @ self._twist_exponential(
                joint, q[joint]
            )
            link_pose = (
                cumulative @ self._link_home_transforms[joint]
            )
            center_pose = (
                cumulative @ self._center_home_transforms[joint]
            )
            link_rotations[joint] = link_pose[:3, :3]
            center_positions[joint] = center_pose[:3, 3]

        return (
            link_rotations,
            center_positions,
            angular_axes,
            spatial_linear,
        )

    def _twist_exponential(
        self, joint: int, angle: float
    ) -> FloatArray:
        """Return exp([S] angle) for one precomputed revolute screw axis."""
        skew = self._axis_skew[joint]
        skew_squared = self._axis_skew_squared[joint]
        sine = np.sin(angle)
        cosine = np.cos(angle)

        rotation = (
            np.eye(3)
            + sine * skew
            + (1.0 - cosine) * skew_squared
        )
        translation_operator = (
            angle * np.eye(3)
            + (1.0 - cosine) * skew
            + (angle - sine) * skew_squared
        )

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = (
            translation_operator @ self._joint_linear_twists[joint]
        )
        return transform

    @staticmethod
    def _skew(vector: FloatArray) -> FloatArray:
        x, y, z = vector
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=float,
        )

    def _coriolis_matrix(
        self,
        mass_derivatives: FloatArray,
        qdot: FloatArray,
    ) -> FloatArray:
        """Calculate C from numerical Christoffel symbols."""
        coriolis = np.zeros(
            (self.num_joints, self.num_joints), dtype=float
        )

        # mass_derivatives[k, i, j] = d D[i, j] / d q[k]
        for row in range(self.num_joints):
            for column in range(self.num_joints):
                value = 0.0
                for coordinate in range(self.num_joints):
                    christoffel = 0.5 * (
                        mass_derivatives[coordinate, row, column]
                        + mass_derivatives[column, row, coordinate]
                        - mass_derivatives[row, column, coordinate]
                    )
                    value += christoffel * qdot[coordinate]
                coriolis[row, column] = value

        return coriolis
