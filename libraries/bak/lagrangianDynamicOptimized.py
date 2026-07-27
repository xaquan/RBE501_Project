"""Efficient symbolic Lagrangian dynamics for serial manipulators.

This module preserves the public ``LagrangeDynamic`` class and
``DCG_matrices`` interface from ``lagrangianDynamic.py`` while avoiding the
largest sources of redundant symbolic work.
"""

from __future__ import annotations

from typing import Optional

import sympy as sp

from libraries.bak.manipulatorKinematicsOptimized import FK_Exponential


class LagrangeDynamic:
    """Build the symbolic mass, Coriolis, and gravity matrices.

    Args:
        thetas: Generalized joint coordinates.
        fk: Forward kinematics evaluated at each link frame.
        fkc: Forward kinematics evaluated at each center of mass. If omitted,
            every link is treated as a point mass at its link-frame origin.
        masses: One mass per joint/link.
        gravity: Gravitational acceleration magnitude.
        simplify_results: Apply SymPy's general simplifier to D, C, and G.
            Disabling this can substantially reduce construction time for
            large manipulators; callers can simplify selected entries later.
    """

    def __init__(
        self,
        thetas,
        fk: FK_Exponential,
        fkc: Optional[FK_Exponential],
        masses,
        gravity,
        inertia_array: Optional[list[sp.Matrix]] = None,
        *,
        simplify_results: bool = True,
    ) -> None:
        self.thetas = sp.Matrix(thetas)
        self.fk = fk
        self.fkc = fkc
        self.masses = sp.Matrix(masses)
        self.num_joints = len(self.thetas)
        self.g = sp.sympify(gravity)
        self.simplify_results = simplify_results

        if len(self.masses) != self.num_joints:
            raise ValueError("masses must contain one value per joint")

        # Match the original module's public symbol names exactly.
        self.qdot = tuple(sp.symarray("qdot", self.num_joints + 1)[1:])
        self.qddot = tuple(sp.symarray("qddot", self.num_joints + 1)[1:])
        self._qdot_vector = sp.Matrix(self.qdot)
        self._qddot_vector = sp.Matrix(self.qddot)

        self._inertia_array: Optional[list[sp.Matrix]] = inertia_array
        self._velocities: Optional[list[sp.Matrix]] = None
        self._mass_point_v_squared: Optional[list[sp.Expr]] = None

        if self.fkc is None:
            self.Jv_mtx = [
                sp.Matrix(jacobian)
                for jacobian in self.fk.get_jacobian_linear_velocities()
            ]
            self.Jw_mtx: list[sp.Matrix] = []
        else:
            self.Jv_mtx, self.Jw_mtx = self._compute_center_mass_jacobians()

        self.potential_energies = self._compute_potential_energies()
        self.Potential_energy_P = sp.Add(*self.potential_energies)

        D = self._compute_mass_matrix()
        C = self._compute_coriolis_matrix(D)
        G = self._compute_gravity_vector()
        self.DCG_matrices = (D, C, G)

        # Compatibility values retained from the original implementation.
        self.total_K = (self._qdot_vector.T * D * self._qdot_vector)[0] / 2
        self.Lagrange = self.total_K - self.Potential_energy_P
        self.torques = D * self._qddot_vector + C * self._qdot_vector + G

    @property
    def inertia_array(self) -> list[sp.Matrix]:
        """Return lazily-created symmetric link inertia tensors."""
        if self._inertia_array is None:
            tensors: list[sp.Matrix] = []
            for index in range(1, self.num_joints + 1):
                ixx, ixy, ixz, iyy, iyz, izz = sp.symbols(
                    f"Ixx{index} Ixy{index} Ixz{index} "
                    f"Iyy{index} Iyz{index} Izz{index}"
                )
                tensors.append(
                    sp.Matrix(
                        [
                            [ixx, ixy, ixz],
                            [ixy, iyy, iyz],
                            [ixz, iyz, izz],
                        ]
                    )
                )
            self._inertia_array = tensors
        return self._inertia_array

    @property
    def velocities(self) -> list[sp.Matrix]:
        """Return link linear velocities, computed only if requested."""
        if self._velocities is None:
            self._velocities = [
                jacobian * self._qdot_vector for jacobian in self.Jv_mtx
            ]
        return self._velocities

    @property
    def MassPoint_v_squared_all_joints(self) -> list[sp.Expr]:
        """Retain the original public squared-velocity attribute lazily."""
        if self._mass_point_v_squared is None:
            self._mass_point_v_squared = [
                sp.sympify((velocity.T * velocity)[0]) for velocity in self.velocities
            ]
        return self._mass_point_v_squared

    def _maybe_simplify(self, expression):
        if not self.simplify_results:
            return expression
        return sp.simplify(expression)

    def _compute_center_mass_jacobians(
        self,
    ) -> tuple[list[sp.Matrix], list[sp.Matrix]]:
        """Compute geometric Jacobians at all centers of mass."""
        if self.fkc is None:
            raise RuntimeError("center-of-mass kinematics are unavailable")

        link_transforms = [
            sp.Matrix(transform) for transform in self.fk.get_transformation_matrices()
        ]
        com_transforms = [
            sp.Matrix(transform) for transform in self.fkc.get_transformation_matrices()
        ]

        linear_jacobians: list[sp.Matrix] = []
        angular_jacobians: list[sp.Matrix] = []

        for link_index in range(self.num_joints):
            linear = sp.zeros(3, self.num_joints)
            angular = sp.zeros(3, self.num_joints)
            center = com_transforms[link_index + 1][:3, 3]

            for joint_index in range(link_index + 1):
                transform = link_transforms[joint_index]
                axis = transform[:3, 2]
                origin = transform[:3, 3]
                linear[:, joint_index] = axis.cross(center - origin)
                angular[:, joint_index] = axis

            linear_jacobians.append(linear)
            angular_jacobians.append(angular)

        return linear_jacobians, angular_jacobians

    def _compute_mass_matrix(self) -> sp.Matrix:
        """Compute D(q) directly from translational and rotational Jacobians."""
        mass_matrix = sp.zeros(self.num_joints)
        rotations = (
            [sp.Matrix(rotation) for rotation in self.fk.get_rotation_matrices()]
            if self.fkc is not None
            else []
        )

        for index in range(self.num_joints):
            linear = self.Jv_mtx[index]
            mass_matrix += self.masses[index] * linear.T * linear

            if self.fkc is not None:
                angular = self.Jw_mtx[index]
                rotation = rotations[index]
                world_inertia = rotation * self.inertia_array[index] * rotation.T
                mass_matrix += angular.T * world_inertia * angular

        # Enforce exact symmetry before simplification. This also avoids doing
        # duplicate simplification work for mirrored entries.
        for row in range(self.num_joints):
            for column in range(row, self.num_joints):
                value = self._maybe_simplify(mass_matrix[row, column])
                mass_matrix[row, column] = value
                mass_matrix[column, row] = value

        return mass_matrix

    def _compute_coriolis_matrix(self, mass_matrix: sp.Matrix) -> sp.Matrix:
        """Compute C(q, qdot) directly from Christoffel coefficients."""
        coriolis = sp.zeros(self.num_joints)

        for row in range(self.num_joints):
            for column in range(self.num_joints):
                value = sp.S.Zero
                for index in range(self.num_joints):
                    coefficient = (
                        sp.diff(mass_matrix[row, column], self.thetas[index])
                        + sp.diff(mass_matrix[row, index], self.thetas[column])
                        - sp.diff(mass_matrix[column, index], self.thetas[row])
                    ) / 2
                    value += coefficient * self.qdot[index]
                coriolis[row, column] = self._maybe_simplify(value)

        return coriolis

    def _compute_potential_energies(self) -> list[sp.Expr]:
        transforms = (
            self.fkc.get_transformation_matrices()
            if self.fkc is not None
            else self.fk.get_transformation_matrices()
        )

        return [
            self.masses[index] * self.g * sp.Matrix(transforms[index + 1])[2, 3]
            for index in range(self.num_joints)
        ]

    def _compute_gravity_vector(self) -> sp.Matrix:
        gravity = sp.Matrix(
            [sp.diff(self.Potential_energy_P, theta) for theta in self.thetas]
        )
        if not self.simplify_results:
            return gravity
        return gravity.applyfunc(sp.simplify)
