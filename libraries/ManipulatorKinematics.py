import numpy as np
import sympy as sp
from typing import List, Optional, Tuple, Union
#=================================================
# Optimized Forward Kinematics and Inverse Kinematics Module
#=================================================

class FK_Exponential:
    """
    Forward Kinematics using exponential coordinate representation.
    Optimized for reduced computation and better memory management.
    """
    
    def __init__(self, home_trans_mtx: list, home_omegas: list, home_positions: list, thetas: list):
        """
        Initialize the FK_Exponential class.
        
        Args:
            home_trans_mtx: List of home transformation matrices (4x4)
            home_omegas: List of angular velocity vectors
            home_positions: List of position vectors
            thetas: Initial joint angles
        """
        if len(home_omegas) != len(home_positions):
            raise ValueError("The number of angular velocities and position vectors must be the same.")
        
        self.home_trans_mtx = home_trans_mtx
        self.home_omegas = home_omegas
        self.home_positions = home_positions
        self.joint_count = len(thetas)
        
        # Cache computed values
        self._velocities = None
        self._se3_matrices = None
        self._transformation_matrices = None
        self._jacobian_matrix = None
        self._jacobian_matrices = None
        self._rodrigues_rotation_matrices = None
        self._rotation_matrices = None
        self._current_thetas = None
        
        # Pre-compute velocities and SE3 matrices
        self._velocities = self._compute_velocities_fast()
        self._se3_matrices = self._create_se3_matrices_fast()
        self.set_thetas(thetas)

    # ==================== Velocity Computation ====================
    
    def _compute_velocities_fast(self) -> List:
        """Optimized velocity computation: v_i = -w_i x q_i"""
        velocities = []
        for w, q in zip(self.home_omegas, self.home_positions):
            w_mat = sp.Matrix(w)
            q_mat = sp.Matrix(q)
            v = -w_mat.cross(q_mat)
            # Use simplification selectively for symbolic expressions
            if any(isinstance(x, sp.Basic) for x in w + q):
                v = sp.trigsimp(v)
            velocities.append(v)
        return velocities

    def get_velocities(self) -> np.ndarray:
        """Get the linear velocities of the joints."""
        if self._velocities is None:
            raise ValueError("Velocities have not been computed yet.")
        return np.array(self._velocities).reshape(-1, 3)

    # ==================== SE3 Matrix Creation ====================
    
    def _create_se3_matrix_fast(self, w, v) -> sp.Matrix:
        """Create SE(3) matrix efficiently."""
        w_mat = sp.Matrix(w).reshape(3, 1)
        v_mat = sp.Matrix(v).reshape(3, 1)
        
        # Construct skew-symmetric matrix
        w_hat = sp.Matrix([
            [0, -w_mat[2], w_mat[1]],
            [w_mat[2], 0, -w_mat[0]],
            [-w_mat[1], w_mat[0], 0]
        ])
        
        se3 = sp.zeros(4, 4)
        se3[:3, :3] = w_hat
        se3[:3, 3] = v_mat
        return se3
    
    def _create_se3_matrices_fast(self) -> List[sp.Matrix]:
        """Create SE(3) matrices for all joints."""
        se3_matrices = []
        for w, v in zip(self.home_omegas, self._velocities):
            se3_matrix = self._create_se3_matrix_fast(w, v)
            # Conditional simplification
            if any(isinstance(x, sp.Basic) for x in w):
                se3_matrix = sp.trigsimp(se3_matrix)
            se3_matrices.append(se3_matrix)
        return se3_matrices

    # ==================== Exponential Map & Transformation ====================
    
    def _compute_se3_exponential(self, se3_matrix: sp.Matrix, velocity: sp.Matrix, theta) -> sp.Matrix:
        """
        Compute exp(theta * se3) using Rodrigues' formula.
        Optimized by avoiding unnecessary matrix operations.
        """
        I = sp.eye(3)
        w_skew = se3_matrix[:3, :3]
        v = sp.Matrix(velocity).reshape(3, 1)
        
        # Rodrigues' rotation formula: R = I + sin(θ)[w]^ + (1-cos(θ))[w]^2
        sin_theta = sp.sin(theta)
        cos_theta = sp.cos(theta)
        one_minus_cos = sp.Integer(1) - cos_theta
        
        R = I + sin_theta * w_skew + one_minus_cos * (w_skew * w_skew) # pyright: ignore[reportOperatorIssue]
        
        # Compute V matrix for translation
        theta_minus_sin = theta - sin_theta
        V = (I * theta + one_minus_cos * w_skew + theta_minus_sin * (w_skew * w_skew)) * v # pyright: ignore[reportOperatorIssue]
        
        result = sp.eye(4)
        result[:3, :3] = R
        result[:3, 3] = V
        
        return result
    
    def _compute_transformation_matrices(self, thetas: List) -> None:
        """
        Compute transformation matrices from SE3 exponentials and home matrices.
        Optimized with matrix chain multiplication.
        """
        # Compute exponentials for all joints
        rodrigues_matrices = []
        for se3, velocity, theta in zip(self._se3_matrices, self._velocities, thetas):
            exp_matrix = self._compute_se3_exponential(se3, velocity, theta)
            rodrigues_matrices.append(exp_matrix)
        
        self._rodrigues_rotation_matrices = rodrigues_matrices
        
        # Chain multiply and apply home transformations
        T = sp.eye(4)
        trans_matrices = [T]
        rotation_matrices = []
        
        for i in range(self.joint_count):
            R = rodrigues_matrices[i]
            T = T * R  # Cumulative product
            M_i = sp.Matrix(self.home_trans_mtx[i])
            trans = T * M_i
            # Simplify only when needed
            if any(isinstance(x, sp.Basic) for x in thetas):
                trans = sp.trigsimp(trans)
            rotation_matrices.append(trans[:3, :3])
            trans_matrices.append(trans)
        
        self._transformation_matrices = trans_matrices
        self._rotation_matrices = rotation_matrices

    # ==================== Getters & Setters ====================
    
    def set_thetas(self, thetas: List) -> None:
        """Set joint angles and recompute dependent quantities."""
        self._current_thetas = thetas
        self._compute_transformation_matrices(thetas)
        self._compute_jacobian()
        self._compute_jacobians()

    def get_transformation_matrices(self, thetas: Optional[List] = None) -> List:
        """Get transformation matrices, optionally recomputing for new thetas."""
        if thetas is not None:
            self.set_thetas(thetas)
        
        if self._transformation_matrices is None:
            raise ValueError("Transformation matrices not computed. Set thetas first.")
        return self._transformation_matrices

    def get_current_thetas(self) -> List:
        """Get current joint angles."""
        if self._current_thetas is None:
            raise ValueError("Joint angles not set yet.")
        return self._current_thetas

    def get_current_position(self, joint_index: Optional[int] = None) -> np.ndarray:
        """
        Get position of a specific joint (default: end-effector).
        
        Args:
            joint_index: 0-based joint index (None for end-effector)
        
        Returns:
            3D position vector
        """
        if self._transformation_matrices is None:
            raise ValueError("Transformation matrices not computed.")
        
        if joint_index is None:
            joint_index = len(self._transformation_matrices) - 1
        elif not (0 <= joint_index < len(self._transformation_matrices)):
            raise ValueError(f"Index {joint_index} out of range [0, {len(self._transformation_matrices)-1}]")
        
        return np.array(self._transformation_matrices[joint_index][:3, 3]).reshape(3)

    def get_transformation_matrix(self, joint_index: int) -> sp.Matrix:
        """Get transformation matrix for a specific joint."""
        if self._transformation_matrices is None:
            raise ValueError("Transformation matrices not computed.")
        if not (0 <= joint_index < len(self._transformation_matrices)):
            raise ValueError(f"Index out of range.")
        return self._transformation_matrices[joint_index]

    def get_rotation_matrices(self) -> List[sp.Matrix]:
        """Get rotation matrices for all joints."""
        if self._rotation_matrices is None:
            raise ValueError("Rotation matrices not computed.")
        return self._rotation_matrices

    # ==================== Jacobian Computation ====================
    
    def _compute_jacobian(self) -> None:
        """
        Compute the space Jacobian matrix (6 x n).

        The screw axes supplied to this class are expressed in the fixed
        (space) frame.  Each column therefore has to be transformed by the
        exponentials of the preceding joints.  Using the z-axis of each
        displayed link frame is not valid for a general POE model.
        """
        if self._transformation_matrices is None:
            raise ValueError("Transformation matrices not computed.")

        J = sp.zeros(6, self.joint_count)
        p_ee = sp.Matrix(self._transformation_matrices[-1][:3, 3])
        preceding_transform = sp.eye(4)

        for i, (omega_home, velocity_home) in enumerate( # pyright: ignore[reportGeneralTypeIssues]
            zip(self.home_omegas, self._velocities)
        ):
            rotation = preceding_transform[:3, :3]
            translation = preceding_transform[:3, 3]

            omega = rotation * sp.Matrix(omega_home)
            velocity = (
                translation.cross(omega)
                + rotation * sp.Matrix(velocity_home)
            )

            # A spatial twist [omega, velocity] gives point velocity
            # omega x p + velocity at a point p in the space frame.
            linear_velocity = omega.cross(p_ee) + velocity
            J[:, i] = sp.Matrix.vstack(linear_velocity, omega)

            preceding_transform = (
                preceding_transform * self._rodrigues_rotation_matrices[i] # pyright: ignore[reportOptionalSubscript]
            )

        self._jacobian_matrix = J

    def _compute_jacobians(self) -> None:
        """Compute individual Jacobians for each joint."""
        if self._transformation_matrices is None:
            raise ValueError("Transformation matrices not computed.")
        
        jacobians = []
        for i in range(1, self.joint_count + 1):
            J_i = self._compute_joint_jacobian(i)
            jacobians.append(J_i)
        
        self._jacobian_matrices = jacobians

    def _compute_joint_jacobian(self, joint_index: int) -> sp.Matrix:
        """Compute Jacobian for a specific joint."""
        if self._transformation_matrices is None:
            raise ValueError("Transformation matrices not computed.")
        if not (1 <= joint_index <= self.joint_count):
            raise ValueError(f"Joint index {joint_index} out of range [1, {self.joint_count}]")
        
        J = sp.zeros(6, self.joint_count)
        T_e = sp.Matrix(self._transformation_matrices[joint_index])
        p_e = T_e[:3, 3]
        # print(f"Computing Jacobian for joint {joint_index}: p_e = {p_e}")
        for i in range(joint_index):
            T_i = sp.Matrix(self._transformation_matrices[i])
            p_i = T_i[:3, 3]
            z_i = T_i[:3, 2]
            
            j = sp.Matrix.vstack(z_i.cross(p_e - p_i), z_i) # pyright: ignore[reportAttributeAccessIssue, reportOperatorIssue]
            J[:, i] = j
        
        return J

    def get_jacobian_matrix(self) -> np.ndarray:
        """Get the full Jacobian matrix (6 x n)."""
        if self._jacobian_matrix is None:
            raise ValueError("Jacobian matrix not computed.")
        return np.array(self._jacobian_matrix).reshape(6, -1)

    def get_jacobian_matrices(self) -> List[sp.Matrix]:
        """Get individual Jacobian matrices for each joint."""
        if self._jacobian_matrices is None:
            raise ValueError("Jacobian matrices not computed.")
        return self._jacobian_matrices

    def get_jacobian_linear_velocities(self) -> List[sp.Matrix]:
        """Get linear velocity part (3 x n) of Jacobian matrices."""
        if self._jacobian_matrices is None:
            raise ValueError("Jacobian matrices not computed.")
        return [J[:3, :] for J in self._jacobian_matrices]

    def get_jacobian_angular_velocities(self) -> List[sp.Matrix]:
        """Get angular velocity part (3 x n) of Jacobian matrices."""
        if self._jacobian_matrices is None:
            raise ValueError("Jacobian matrices not computed.")
        return [J[3:6, :] for J in self._jacobian_matrices]

    def is_singular(self) -> bool:
        """Check if Jacobian is singular."""
        if self._jacobian_matrix is None:
            raise ValueError("Jacobian matrix not computed.")
        Jv = self._jacobian_matrix[:3, :]
        return Jv.rank() < min(Jv.shape)


#=================================================
# Optimized Inverse Kinematics Module
#=================================================

class IK_Numerical:
    """Numerical inverse kinematics solver using Jacobian transpose method."""
    
    def __init__(self, fk: FK_Exponential):
        """
        Initialize IK solver.
        
        Args:
            fk: FK_Exponential instance
        """
        self.fk = fk

    def compute_thetas_for_position(
        self,
        desired_position: np.ndarray,
        initial_thetas: np.ndarray,
        alpha: float = 1.0,
        max_iterations: int = 100,
        epsilon: float = 1e-3,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Solve inverse kinematics using Jacobian-based iterative method.
        
        Args:
            desired_position: Target 3D position (3,)
            initial_thetas: Initial joint angle guess (n,)
            alpha: Step size/gain
            max_iterations: Maximum iterations
            epsilon: Convergence threshold
            verbose: Print iteration info
        
        Returns:
            Joint angles that reach desired position (n,)
        
        Raises:
            ValueError: If IK doesn't converge
        """
        thetas = np.array(initial_thetas, dtype=float).ravel()
        desired_position = np.array(desired_position, dtype=float).ravel()
        
        self.fk.set_thetas(thetas)
        current_position = np.array(self.fk.get_current_position(), dtype=float).ravel()
        position_error = desired_position - current_position
        norm_error = np.linalg.norm(position_error)
        
        for iteration in range(max_iterations):
            # Get Jacobian and compute pseudo-inverse
            J = np.array(self.fk.get_jacobian_matrix(), dtype=float)
            Jv = J[:3, :]  # Linear velocity part
            print(f"Jacobian (linear part) at iteration {iteration}:\n{Jv}")
            Jv_pinv = np.linalg.pinv(Jv)
            
            # Update joint angles
            delta_theta = alpha * (Jv_pinv @ position_error)
            thetas += delta_theta
            
            # Update position and error
            self.fk.set_thetas(thetas)
            current_position = np.array(self.fk.get_current_position(), dtype=float).ravel()
            position_error = desired_position - current_position
            norm_error = np.linalg.norm(position_error)
            
            if verbose:
                print(f"Iter {iteration:3d} | Pos: {current_position} | "
                      f"Thetas: {thetas} | ΔTheta: {delta_theta} | Error: {norm_error:.6f}")
            
            if norm_error < epsilon:
                if verbose:
                    print(f"✓ Converged in {iteration} iterations")
                return thetas
        
        raise ValueError(f"IK did not converge within {max_iterations} iterations. "
                        f"Final error: {norm_error:.6e}")


def create_webots_ur5e_kinematics(
    initial_thetas: Optional[List] = None
) -> Tuple[FK_Exponential, IK_Numerical]:
    """Create matching FK and IK objects for Webots R2025a ``UR5e.proto``.

    Positions are in metres and are expressed in the Webots robot base/world
    frame for a robot whose translation and rotation are both zero.  The final
    frame is the UR5e ``toolSlot`` origin.
    """
    if initial_thetas is None:
        initial_thetas = [0.0] * 6
    if len(initial_thetas) != 6:
        raise ValueError("The UR5e model requires six joint angles.")

    joint_points = [
        [0.000, 0.000, 0.163],
        [0.000, 0.138, 0.163],
        [0.425, 0.007, 0.163],
        [0.817, 0.007, 0.163],
        [0.817, 0.134, 0.163],
        [0.817, 0.134, 0.063],
    ]
    joint_axes = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]

    # Home transforms for the six successive frames.  Only the final
    # transform affects end-effector FK, but all are retained for inspection.
    identity_rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    frame_positions = joint_points[:-1] + [[0.817, 0.234, 0.063]]
    home_transforms = []
    for position in frame_positions:
        transform = [
            identity_rotation[0] + [position[0]],
            identity_rotation[1] + [position[1]],
            identity_rotation[2] + [position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
        home_transforms.append(transform)

    # At home, the wrist/tool frame is rotated pi about the base y-axis.
    home_transforms[-1][:3] = [
        [-1.0, 0.0, 0.0, 0.817],
        [0.0, 1.0, 0.0, 0.234],
        [0.0, 0.0, -1.0, 0.063],
    ]

    fk = FK_Exponential(
        home_transforms,
        joint_axes,
        joint_points,
        initial_thetas,
    )

    ik = IK_Numerical(fk)
    return fk, ik


def create_webots_ur5e_kinematics_symbolic(
    initial_thetas: Optional[List] = None,
) -> Tuple[FK_Exponential, dict]:
    """Create a symbolic FK model for the Webots R2025a ``UR5e.proto``.

    The returned ``FK_Exponential`` instance uses SymPy symbols for the link
    lengths and the joint variables so the resulting transformation matrices,
    Jacobian, and end-effector position stay symbolic until substitutions are
    applied.

    Returns:
        A tuple of ``(fk, symbols)`` where ``symbols`` contains the length and
        angle symbols grouped under the ``lengths`` and ``thetas`` keys.
    """
    length_symbols = list(
        sp.symbols(
            "base_z shoulder_y upper_arm_x elbow_y forearm_x wrist_y tool_y tool_z",
            real=True,
        )
    )
    theta_symbols = list(sp.symbols("t1:7", real=True))

    if initial_thetas is None:
        initial_thetas = theta_symbols
    if len(initial_thetas) != 6:
        raise ValueError("The UR5e model requires six joint angles.")

    base_z, shoulder_y, upper_arm_x, elbow_y, forearm_x, wrist_y, tool_y, tool_z = (
        length_symbols
    )

    joint_points = [
        [0, 0, base_z],
        [0, shoulder_y, base_z],
        [upper_arm_x, elbow_y, base_z],
        [forearm_x, elbow_y, base_z],
        [forearm_x, wrist_y, base_z],
        [forearm_x, tool_y, tool_z],
    ]
    joint_axes = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]

    identity_rotation = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    frame_positions = joint_points[:-1] + [[forearm_x, tool_y, tool_z]]
    home_transforms = []
    for position in frame_positions:
        transform = [
            identity_rotation[0] + [position[0]],
            identity_rotation[1] + [position[1]],
            identity_rotation[2] + [position[2]],
            [0, 0, 0, 1],
        ]
        home_transforms.append(transform)

    # At home, the wrist/tool frame is rotated pi about the base y-axis.
    home_transforms[-1][:3] = [
        [-1, 0, 0, forearm_x],
        [0, 1, 0, tool_y],
        [0, 0, -1, tool_z],
    ]

    fk = FK_Exponential(
        home_transforms,
        joint_axes,
        joint_points,
        initial_thetas,
    )
    return fk, {
        "lengths": length_symbols,
        "thetas": theta_symbols,
    }
