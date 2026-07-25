import numpy as np
import sympy as sp
from manipulatorKinematics import FK_Exponential


class LagrangeDynamic:
    def __init__(
        self, thetas, fk: FK_Exponential, fkc: FK_Exponential, masses, gravity
    ):
        self.thetas = sp.Matrix(thetas)
        self.fk = fk
        self.fkc = fkc
        self.masses = sp.Matrix(masses)
        self.num_joints = len(thetas)
        self.qdot = sp.symarray("qdot", self.num_joints + 1)[1:]  # Joint velocities
        self.qddot = sp.symarray("qddot", self.num_joints + 1)[
            1:
        ]  # Joint accelerations
        self.g = gravity  # Acceleration due to gravity (m/s^2)
        self.velocities = (
            self.__compute_velocities()
        )  # Compute velocities for each link
        self.inertia_array = (
            self.__generate_intertia_matrices()
        )  # Generate symbolic inertia matrices for each link
        self.MassPoint_v_squared_all_joints = (
            self.__compute_massPoint_v_squared_all_joints()
        )  # Compute kinetic energies of each link

        self.potential_energies = (
            self.__compute_potential_energies()
        )  # Compute potential energies of each link
        self.Potential_energy_P = (
            self.__compute_total_potential_energy()
        )  # Compute total potential energy of the manipulator

        if fkc:
            self.Jv_mtx, self.Jw_mtx = (
                self.__compute_jacobian()
            )  # Compute Jacobian matrices for each link
            # Dq = self.__compute_Dq()  # Compute D(q) matrix
            # self.DCG_matrices = [Dq, self.__compute_Cq(Dq), self.__compute_Gq()]  # Compute D(q), C(q, qdot), G(q) matrices
        else:
            self.Jv_mtx = (
                self.fk.get_jacobian_linear_velocities()
            )  # Get linear velocities of each link's center of mass
            self.total_K = (
                self.__compute_total_massPoint_K()
            )  # Compute total kinetic energy of the manipulator
            self.Lagrange = (
                self.__compute_Lagrange()
            )  # Compute the Lagrangian of the manipulator
            self.dL_q, self.dL_qdot, self.dL_qdot_dt = (
                self.__compute_differential_L()
            )  # Compute the differential of the Lagrangian with respect to joint angles and joint velocities
            self.torques = (
                self.__compute_torques()
            )  # Compute the torques required at each joint
            # Dq = self.__compute_Dq()  # Compute D(q) matrix
            # self.DCG_matrices = self.__get_DCG_matrices()  # Compute D(q), C(q, qdot), G(q) matrices

        Dq = self.__compute_Dq()  # Compute D(q) matrix
        self.DCG_matrices = [
            Dq,
            self.__compute_Cq(Dq),
            self.__compute_Gq(),
        ]  # Compute D(q), C(q, qdot), G(q) matrices

    def __generate_intertia_matrices(self):
        """
        Generate symbolic inertia matrices for each link.
        """
        print("Generating inertia matrices...")
        res = []
        for i in range(self.num_joints):
            I = sp.Matrix(
                [
                    [
                        sp.symbols(f"Ixx{i+1}"),
                        sp.symbols(f"Ixy{i+1}"),
                        sp.symbols(f"Ixz{i+1}"),
                    ],
                    [
                        sp.symbols(f"Ixy{i+1}"),
                        sp.symbols(f"Iyy{i+1}"),
                        sp.symbols(f"Iyz{i+1}"),
                    ],
                    [
                        sp.symbols(f"Ixz{i+1}"),
                        sp.symbols(f"Iyz{i+1}"),
                        sp.symbols(f"Izz{i+1}"),
                    ],
                ]
            )
            res.append(I)

        return res

    def __compute_velocities(self):
        """
        Compute the linear and angular velocities of each link's center of mass.
        """
        print("Computing velocities...")
        jacobian_matrices = (
            self.fk.get_jacobian_matrices()
        )  # Get Jacobian matrices for each link
        ddot = self.qdot  # Remove the first element to match the number of joints

        res = []

        for i in range(self.num_joints):
            J = sp.Matrix(jacobian_matrices[i])
            Jv = J[:3, :]  # Extract the linear velocity part of the Jacobian
            v = Jv * sp.Matrix(ddot)  # Linear velocity
            # v = sp.trigsimp(v)
            res.append(v)
        return res

    def __compute_massPoint_v_squared_all_joints(self):
        """
        Compute the kinetic energy of the manipulator using the Lagrangian formulation.
        """
        print("Computing mass point kinetic energies...")
        if self.velocities is None or len(self.velocities) == 0:
            self.__compute_velocities()  # Compute velocities if not already computed

        v_squareds = []

        for i in range(self.num_joints):
            v = self.velocities[i]
            v_squared = (v.T * v)[
                0
            ]  # Compute the squared magnitude of the velocity vector
            # v_squared = sp.trigsimp(v_squared)  # Simplify the expression
            v_squareds.append(v_squared)

        return v_squareds

    def __compute_jacobian(self):
        """
        Compute the Jacobian matrices for each link's center of mass.
        """
        transformation_matrices = (
            self.fk.get_transformation_matrices()
        )  # Get transformation matrices for each link
        transformation_matrices_c = (
            self.fkc.get_transformation_matrices()
        )  # Get transformation matrices for each link's center of mass
        Jv_mtx = []
        Jw_mtx = []
        for i in range(1, self.num_joints + 1):
            Jv = sp.zeros(3, self.num_joints)
            Jw = sp.zeros(3, self.num_joints)
            oc_i = sp.Matrix(transformation_matrices_c[i])[
                :3, 3
            ]  # Position of the center of mass of link i
            for j in range(i):
                T_j = sp.Matrix(transformation_matrices[j])
                z_j = sp.Matrix(T_j[:3, 2])  # Orientation of the link's z-axis
                o_j = T_j[:3, 3]  # Position of the link's end-effector
                jv = z_j.cross(
                    oc_i - o_j
                )  # pyright: ignore[reportOperatorIssue] # Linear velocity Jacobian component
                jw = z_j  # Angular velocity Jacobian component

                Jv[:, j] = sp.simplify(jv)
                Jw[:, j] = sp.simplify(jw)
            Jv_mtx.append(Jv)
            Jw_mtx.append(Jw)
        return Jv_mtx, Jw_mtx

    def __compute_total_massPoint_K(self):
        """
        Compute the kinetic energy of the manipulator using the Lagrangian formulation.
        """
        print("Computing total mass point kinetic energy...")
        if self.velocities is None or len(self.velocities) == 0:
            self.__compute_velocities()  # Compute velocities if not already computed

        total_v_squared = 0
        for m, v_squared in zip(
            self.masses,
            self.MassPoint_v_squared_all_joints,  # pyright: ignore[reportArgumentType]
        ):
            mv_squared = (
                m * v_squared
            )  # Compute the kinetic energy contribution of each link

            total_v_squared += mv_squared

        total_v_squared = sp.simplify(
            total_v_squared
        )  # Simplify the total squared velocity expression

        kinetic_energies = (
            sp.Rational(1, 2) * total_v_squared
        )  # Compute the total kinetic energy of the manipulator

        return kinetic_energies

    def __compute_total_Lagrange_Dv(self):
        """
        Compute the kinetic energy of the manipulator using the Lagrangian formulation.
        """
        print("Computing total Lagrangian velocity kinetic energy...")
        Jv = self.Jv_mtx  # Get linear velocities of each link's center of mass
        Dv = sp.zeros(
            self.num_joints, self.num_joints
        )  # Initialize the total kinetic energy
        # total_m_Jv_sq = 0
        for m, jv in zip(self.masses, Jv):  # pyright: ignore[reportArgumentType]
            jv = sp.Matrix(jv)
            mJvJv = m * (
                jv.T * jv
            )  # Compute the kinetic energy contribution of each link
            Dv += mJvJv

        Dv = sp.simplify(Dv)  # Simplify the total squared velocity expression
        return Dv  # Return the scalar value of the kinetic energy

    def __compute_total_Lagrange_Dw(self):
        """
        Compute the total kinetic energy of the manipulator using the Lagrangian formulation.
        """
        print("Computing total Lagrangian rotational kinetic energy...")

        Jw = self.Jw_mtx  # Get angular velocities of each link's center of mass

        total_JwRIRJw = sp.zeros(
            self.num_joints, self.num_joints
        )  # Initialize the total kinetic energy
        for i in range(self.num_joints):
            jw = sp.Matrix(Jw[i])
            R = sp.Matrix(
                self.fk.get_rotation_matrices()[i]
            )  # Get the rotation matrix for the current link
            I = sp.Matrix(self.inertia_array[i])
            RIR = R * I * R.T  # Rotate the inertia matrix to the world frame
            JwiJw = jw.T * RIR * jw
            total_JwRIRJw += JwiJw
        Dw = sp.simplify(
            total_JwRIRJw
        )  # Compute the total kinetic energy of the manipulator
        return Dw

    def __compute_Dq(self):
        """
        Compute the total kinetic energy of the manipulator using the Lagrangian formulation.
        """
        if self.fkc:
            Dv = (
                self.__compute_total_Lagrange_Dv()
            )  # Compute the total kinetic energy of the manipulator
            Dw = (
                self.__compute_total_Lagrange_Dw()
            )  # Compute the total kinetic energy of the manipulator
            Dq = Dv + Dw
        else:
            tau = sp.Matrix(self.torques)
            Dq = tau.jacobian(
                sp.Matrix(self.qddot)
            )  # Compute the Jacobian of tau with respect to qddot

        return sp.simplify(Dq)

    def __compute_Cq(self, Dq):
        """
        Compute the Coriolis and centrifugal forces of the manipulator using the Lagrangian formulation.
        """
        D = Dq
        C = sp.zeros(self.num_joints, self.num_joints)
        Cijk = sp.MutableDenseNDimArray.zeros(
            self.num_joints, self.num_joints, self.num_joints
        )  # Initialize the Coriolis and centrifugal forces
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                for k in range(self.num_joints):
                    Cijk[i, j, k] = (
                        sp.diff(D[i, j], self.thetas[k])
                        + sp.diff(D[i, k], self.thetas[j])
                        - sp.diff(D[j, k], self.thetas[i])
                    ) / 2  # pyright: ignore[reportOperatorIssue]

        for i in range(self.num_joints):
            for j in range(self.num_joints):
                C[i, j] = sum(
                    Cijk[i, j, k] * self.qdot[k] for k in range(self.num_joints)
                )  # Compute the Coriolis and centrifugal forces

        return sp.simplify(C)

    def __compute_Gq(self):
        """
        Compute the gravitational forces of the manipulator using the Lagrangian formulation.
        """
        G = sp.zeros(self.num_joints, 1)
        for i in range(self.num_joints):
            G[i] = sp.diff(
                self.Potential_energy_P, self.thetas[i]
            )  # Compute the gravitational forces
        return sp.simplify(G)

    def __compute_potential_energies(self):
        """
        Compute the potential energy of each link's
        """
        print("Computing potential energy...")
        potential_energies = []
        if self.fkc:
            transformation_matrices = self.fkc.get_transformation_matrices()[
                1:
            ]  # Get transformation matrices for each link's center of mass
        else:
            transformation_matrices = self.fk.get_transformation_matrices()[
                1:
            ]  # Get transformation matrices for each link

        for i in range(self.num_joints):
            T = sp.Matrix(
                transformation_matrices[i]
            )  # Get the transformation matrix for the current link
            z = T[2, 3]  # Extract the scalar z-position from the transformation matrix

            m = self.masses[i]
            P = m * self.g * z  # Potential energy: m * g * h
            # P = sp.trigsimp(P)

            potential_energies.append(P)

        return potential_energies

    def __compute_total_potential_energy(self):
        """
        Compute the total potential energy of the manipulator using the Lagrangian formulation.
        """
        print("Computing total potential energy...")
        total_potential_energy = sp.Integer(0)
        for P in self.potential_energies:
            total_potential_energy += P
        return total_potential_energy

    def __compute_Lagrange(self):
        """
        Compute the Lagrangian of the manipulator using the Lagrangian formulation.
        """
        print("Computing Lagrangian...")
        L = self.total_K - self.Potential_energy_P  # Lagrangian: K - P
        return L

    def __compute_differential_L(self):
        """
        Compute the differential of the Lagrangian with respect to joint angles (q) and joint velocities (qdot).
        """
        print("Computing differential of Lagrangian...")
        L = self.Lagrange
        q = sp.Matrix(self.thetas)
        qdot = sp.Matrix(self.qdot)
        qddot = sp.Matrix(self.qddot)

        dL_dq = sp.diff(L, q)  # Compute the gradient of L with respect to q
        dL_dqdot = sp.diff(L, qdot)  # Compute the gradient of L with respect to qdot
        dL_dqdot = sp.Matrix(dL_dqdot)
        dL_dqdot_dt = (
            dL_dqdot.jacobian(q) * qdot + dL_dqdot.jacobian(qdot) * qddot
        )  # Compute the time derivative using the chain rule
        return dL_dq, dL_dqdot, dL_dqdot_dt

    def __compute_torques(self):
        """
        Compute the torques required at each joint using the Lagrangian formulation.
        """
        print("Computing torques...")
        torques = (
            self.dL_qdot_dt - self.dL_q
        )  # Compute the torques: tau = d/dt(dL/dqdot) - dL/dq
        return sp.simplify(torques)

    def __get_DCG_matrices(self):
        tau = sp.Matrix(self.torques)
        D = tau.jacobian(
            sp.Matrix(self.qddot)
        )  # Compute the Jacobian of tau with respect to qddot
        D = sp.simplify(D)
        zero_motion = {
            **{qdot: 0 for qdot in self.qdot},
            **{qddot: 0 for qddot in self.qddot},
        }
        G = tau.subs(zero_motion)  # Compute G(q) = tau(q, qdot=0, qddot=0)
        G = sp.simplify(G)

        C = (
            tau - D * sp.Matrix(self.qddot) - G
        )  # Compute C(q, qdot) = tau - D(q) * qddot - G(q)
        C = C.jacobian(
            sp.Matrix(self.qdot)
        )  # Compute the Jacobian of C with respect to qdot
        C = sp.simplify(C)
        return D, C, G
