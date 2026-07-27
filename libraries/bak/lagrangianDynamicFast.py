"""Fast symbolic Lagrangian dynamics for serial manipulators.

Drop-in replacement for ``lagrangianDynamic.LagrangeDynamic`` that makes the
6-DOF (UR5) case tractable.

Why the original hangs
----------------------
For a 6-DOF arm the mass matrix ``D(q)``, Coriolis matrix ``C(q, qdot)`` and
gravity vector ``G(q)`` are gigantic expressions built from nested
``sin(t_i)`` / ``cos(t_i)`` terms.  The original code calls ``sympy.simplify``
and ``sympy.trigsimp`` on those matrices (and the example calls ``trigsimp``
again for printing *and* LaTeX).  Both routines are super-linear and, on
6-DOF trig expressions, effectively never finish.

The fix used here
-----------------
* Substitute ``cos(t_i) -> c_i`` and ``sin(t_i) -> s_i`` up front, so every
  quantity becomes a **polynomial** in the ``c_i`` / ``s_i`` symbols.  No trig
  simplification engine is ever invoked.
* Differentiate with respect to joint angles analytically in that polynomial
  space via the chain rule (``d/dt_k = -s_k d/dc_k + c_k d/ds_k``), which is a
  cheap polynomial operation.
* Only ``expand`` (polynomial, fast) is used for normalisation.  Full
  ``simplify`` is opt-in and off by default.

The public interface matches the original: construct with the same arguments
and read ``self.DCG_matrices``.  Results are returned back in ``sin``/``cos``
form by default so downstream code and LaTeX output are unchanged.
"""

from __future__ import annotations

from typing import Optional

import sympy as sp

try:  # Prefer the optimized kinematics if present, else the standard one.
    from libraries.bak.manipulatorKinematicsOptimized import FK_Exponential  # type: ignore
except Exception:  # pragma: no cover - fallback for the default module name
    from manipulatorKinematics import FK_Exponential  # type: ignore


class LagrangeDynamic:
    """Build D(q), C(q, qdot), G(q) for a serial manipulator, fast.

    Args:
        thetas: Generalized joint coordinates (symbols).
        fk: Forward kinematics evaluated at each link frame.
        fkc: Forward kinematics evaluated at each center of mass.  If ``None``,
            each link is treated as a point mass at its link-frame origin.
        masses: One mass per joint/link.
        gravity: Gravitational acceleration magnitude.
        inertia_array: Optional list of 3x3 link inertia tensors (COM frame).
        simplify_results: If ``True`` run ``sympy.simplify`` on the final
            matrices.  Off by default because it is exactly what makes the
            6-DOF case hang; the polynomial results are already compact.
        return_trig: If ``True`` (default) convert results back to
            ``sin``/``cos`` of ``thetas`` for display/LaTeX compatibility.
    """

    def __init__(
        self,
        thetas,
        fk: FK_Exponential,
        fkc: Optional[FK_Exponential],
        masses,
        gravity,
        inertia_array: Optional[list] = None,
        *,
        simplify_results: bool = False,
        return_trig: bool = True,
    ) -> None:
        self.thetas = sp.Matrix(thetas)
        self.fk = fk
        self.fkc = fkc
        self.masses = sp.Matrix(masses)
        self.num_joints = len(self.thetas)
        self.g = sp.sympify(gravity)
        self.simplify_results = simplify_results
        self.return_trig = return_trig

        if len(self.masses) != self.num_joints:
            raise ValueError("masses must contain one value per joint")

        # Public velocity/acceleration symbols, matching the original names.
        self.qdot = tuple(sp.symarray("qdot", self.num_joints + 1)[1:])
        self.qddot = tuple(sp.symarray("qddot", self.num_joints + 1)[1:])
        self._qdot_vec = sp.Matrix(self.qdot)
        self._qddot_vec = sp.Matrix(self.qddot)

        self._inertia_array = inertia_array

        # ---- trig <-> polynomial bookkeeping -----------------------------
        # c_i = cos(t_i), s_i = sin(t_i).  Working in these symbols keeps every
        # expression polynomial and avoids the trig simplification blow-up.
        self._c = sp.symbols(f"__c1:{self.num_joints + 1}", real=True)
        self._s = sp.symbols(f"__s1:{self.num_joints + 1}", real=True)
        self._to_poly = {}
        self._to_trig = {}
        for i, th in enumerate(self.thetas):
            self._to_poly[sp.cos(th)] = self._c[i]
            self._to_poly[sp.sin(th)] = self._s[i]
            self._to_trig[self._c[i]] = sp.cos(th)
            self._to_trig[self._s[i]] = sp.sin(th)

        # ---- kinematic Jacobians (in polynomial space) -------------------
        if self.fkc is None:
            self._Jv = [
                self._polyize(sp.Matrix(J))
                for J in self.fk.get_jacobian_linear_velocities()
            ]
            self._Jw: list = []
        else:
            self._Jv, self._Jw = self._center_mass_jacobians()

        # ---- energies & matrices -----------------------------------------
        self._potential = self._potential_energy()  # scalar, polynomial
        D = self._mass_matrix()
        C = self._coriolis_matrix(D)
        G = self._gravity_vector()

        # Store polynomial forms (cheap, exact) for any numeric evaluation.
        self.D_poly, self.C_poly, self.G_poly = D, C, G

        D_out = self._finalize(D)
        C_out = self._finalize(C)
        G_out = self._finalize(G)
        self.DCG_matrices = [D_out, C_out, G_out]

        # Compatibility attributes from the original class.
        self.Potential_energy_P = self._depoly(self._potential)
        self.total_K = self._depoly((self._qdot_vec.T * D * self._qdot_vec)[0] / 2)
        self.torques = D_out * self._qddot_vec + C_out * self._qdot_vec + G_out

    @property
    def inertia_array(self) -> list:
        """Symmetric link inertia tensors, generated symbolically if omitted."""
        if self._inertia_array is None:
            tensors = []
            for idx in range(1, self.num_joints + 1):
                ixx, ixy, ixz, iyy, iyz, izz = sp.symbols(
                    f"Ixx{idx} Ixy{idx} Ixz{idx} Iyy{idx} Iyz{idx} Izz{idx}"
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

    # ------------------------------------------------------------------ #
    # trig <-> polynomial helpers
    # ------------------------------------------------------------------ #
    def _polyize(self, expr):
        """Replace cos/sin(t_i) by c_i/s_i (structural, fast)."""
        return sp.expand(sp.sympify(expr).xreplace(self._to_poly))

    def _depoly(self, expr):
        """Convert a polynomial-space expression back to sin/cos form."""
        return sp.sympify(expr).xreplace(self._to_trig)

    def _finalize(self, matrix):
        """Optionally simplify, then convert back to trig form if requested."""
        result = matrix
        if self.simplify_results:
            result = result.applyfunc(sp.simplify)
        if self.return_trig:
            result = result.applyfunc(self._depoly)
        return result

    def _dtheta(self, expr, k):
        """Analytic d(expr)/d(theta_k) in polynomial (c/s) space.

        Uses c_k' = -s_k and s_k' = c_k, so
        d/dt_k = -s_k * d/dc_k + c_k * d/ds_k.
        """
        return -self._s[k] * sp.diff(expr, self._c[k]) + self._c[k] * sp.diff(
            expr, self._s[k]
        )

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #
    def _center_mass_jacobians(self):
        """Geometric Jacobians (Jv, Jw) at every center of mass, polynomial."""
        link_T = [self._polyize(sp.Matrix(T)) for T in self.fk.get_transformation_matrices()]
        com_T = [self._polyize(sp.Matrix(T)) for T in self.fkc.get_transformation_matrices()]

        Jv_list, Jw_list = [], []
        for link in range(self.num_joints):
            Jv = sp.zeros(3, self.num_joints)
            Jw = sp.zeros(3, self.num_joints)
            center = com_T[link + 1][:3, 3]
            for joint in range(link + 1):
                T = link_T[joint]
                axis = T[:3, 2]
                origin = T[:3, 3]
                Jv[:, joint] = axis.cross(center - origin)
                Jw[:, joint] = axis
            Jv_list.append(sp.expand(Jv))
            Jw_list.append(sp.expand(Jw))
        return Jv_list, Jw_list

    # ------------------------------------------------------------------ #
    # dynamics matrices
    # ------------------------------------------------------------------ #
    def _mass_matrix(self):
        """D(q) from translational and rotational kinetic energy, polynomial."""
        D = sp.zeros(self.num_joints)
        rotations = (
            [self._polyize(sp.Matrix(R)) for R in self.fk.get_rotation_matrices()]
            if self.fkc is not None
            else []
        )
        for i in range(self.num_joints):
            Jv = self._Jv[i]
            D += self.masses[i] * (Jv.T * Jv)
            if self.fkc is not None:
                Jw = self._Jw[i]
                R = rotations[i]
                inertia = self._polyize(sp.Matrix(self.inertia_array[i]))
                world_I = R * inertia * R.T
                D += Jw.T * world_I * Jw

        # Expand once, and reuse the upper triangle for the symmetric mirror
        # so we never expand the same entry twice.
        D = sp.Matrix(D)  # ensure mutable for item assignment
        for r in range(self.num_joints):
            for c in range(r, self.num_joints):
                val = sp.expand(D[r, c])
                D[r, c] = val
                D[c, r] = val
        return D

    def _coriolis_matrix(self, D):
        """C(q, qdot) via Christoffel symbols, polynomial differentiation."""
        n = self.num_joints
        # Pre-compute partial derivatives dD[i,j]/dt_k once (n^3 but cheap).
        dD = [[[None] * n for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                for k in range(n):
                    d = self._dtheta(D[i, j], k)
                    dD[i][j][k] = d
                    dD[j][i][k] = d  # D is symmetric

        C = sp.zeros(n)
        for i in range(n):
            for j in range(n):
                val = sp.S.Zero
                for k in range(n):
                    christoffel = (dD[i][j][k] + dD[i][k][j] - dD[j][k][i]) / 2
                    val += christoffel * self.qdot[k]
                C[i, j] = sp.expand(val)
        return C

    def _potential_energy(self):
        transforms = (
            self.fkc.get_transformation_matrices()
            if self.fkc is not None
            else self.fk.get_transformation_matrices()
        )
        total = sp.S.Zero
        for i in range(self.num_joints):
            height = self._polyize(sp.Matrix(transforms[i + 1])[2, 3])
            total += self.masses[i] * self.g * height
        return sp.expand(total)

    def _gravity_vector(self):
        return sp.Matrix(
            [sp.expand(self._dtheta(self._potential, k)) for k in range(self.num_joints)]
        )

    # ------------------------------------------------------------------ #
    # convenience: fast numeric evaluation without any trig simplification
    # ------------------------------------------------------------------ #
    def lambdified(self):
        """Return callables (D, C, G) taking numeric (q, qdot) arrays.

        Evaluates the polynomial forms directly, which is far faster than
        substituting into simplified trig expressions.
        """
        q = list(self.thetas)
        args_D_G = q
        args_C = q + list(self.qdot)
        D_fn = sp.lambdify(args_D_G, self._depoly(self.D_poly), "numpy")
        C_fn = sp.lambdify(args_C, self._depoly(self.C_poly), "numpy")
        G_fn = sp.lambdify(args_D_G, self._depoly(self.G_poly), "numpy")
        return D_fn, C_fn, G_fn
