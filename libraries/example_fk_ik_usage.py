"""
Example code demonstrating common FK and IK functions.
Shows practical usage patterns for both the 3-DOF manipulator and UR5e 6-DOF robot.
"""

import numpy as np
import sympy as sp
from libraries.ManipulatorKinematics import FK_Exponential, IK_Numerical


# =====================================================================
# SECTION 1: Simple 3-DOF Manipulator Examples
# =====================================================================

def setup_3dof_manipulator():
    """
    Setup a simple 3-DOF manipulator (RBE 3-axis arm).
    Returns: FK instance
    """
    l1, l2, l3, l4, l5 = 150, 475, 600, 120, 720
    
    M1 = [[1, 0, 0, l1],
          [0, 0, -1, 0],
          [0, 1, 0, l2],
          [0, 0, 0, 1]]
    
    M2 = [[1, 0, 0, l1],
          [0, 0, -1, 0],
          [0, 1, 0, l2 + l3],
          [0, 0, 0, 1]]
    
    M3 = [[1, 0, 0, l1 + l5],
          [0, 0, -1, 0],
          [0, 1, 0, l2 + l3 + l4],
          [0, 0, 0, 1]]
    
    home_trans_matrices = [M1, M2, M3]
    home_omegas = [[0, 0, 1], [0, -1, 0], [0, -1, 0]]
    home_positions = [[0, 0, 0], [l1, 0, l2], [l1, 0, l2 + l3]]
    
    fk = FK_Exponential(home_trans_matrices, home_omegas, home_positions, [0, 0, 0])
    return fk


def setup_ur5e_robot():
    """
    Setup UR5e 6-DOF collaborative robot.
    Returns: FK instance
    """
    d1, a2, a3 = 163.0, 425.0, 600.0
    d4, d6 = 109.3, 100.0
    
    M1 = [[1, 0, 0, 0],
          [0, 0, -1, 0],
          [0, 1, 0, d1],
          [0, 0, 0, 1]]
    
    M2 = [[1, 0, 0, 0],
          [0, 0, -1, 0],
          [0, 1, 0, d1 + a2],
          [0, 0, 0, 1]]
    
    M3 = [[1, 0, 0, 0],
          [0, 0, -1, 0],
          [0, 1, 0, d1 + a2 + a3],
          [0, 0, 0, 1]]
    
    M4 = [[1, 0, 0, 0],
          [0, 0, -1, 0],
          [0, 1, 0, d1 + a2 + a3 + d4],
          [0, 0, 0, 1]]
    
    M5 = [[1, 0, 0, 0],
          [0, 0, -1, 0],
          [0, 1, 0, d1 + a2 + a3 + d4],
          [0, 0, 0, 1]]
    
    M6 = [[1, 0, 0, 0],
          [0, 0, -1, 0],
          [0, 1, 0, d1 + a2 + a3 + d4 + d6],
          [0, 0, 0, 1]]
    
    home_trans_matrices = [M1, M2, M3, M4, M5, M6]
    home_omegas = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
    home_positions = [[0, 0, 0], [0, 0, d1], [0, 0, d1+a2], 
                      [0, 0, d1+a2+a3], [0, 0, d1+a2+a3+d4], [0, 0, d1+a2+a3+d4+d6]]
    
    fk = FK_Exponential(home_trans_matrices, home_omegas, home_positions, 
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return fk


# =====================================================================
# SECTION 2: Common FK Functions
# =====================================================================

def example_1_set_joint_angles():
    """Example 1: Set joint angles and get current configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: set_thetas() - Set Joint Angles")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    
    # Set joint angles (in radians)
    joint_angles = [np.pi/4, -np.pi/6, 0.2]
    fk.set_thetas(joint_angles)
    
    print(f"Joint angles set to (radians): {joint_angles}")
    print(f"Joint angles (degrees): {np.rad2deg(joint_angles)}")
    
    # Verify by getting current thetas
    current_thetas = fk.get_current_thetas()
    print(f"Current thetas: {current_thetas}")


def example_2_get_end_effector_position():
    """Example 2: Get end-effector position (forward kinematics)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: get_current_position() - Get End-Effector Position")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    
    # Test configuration
    fk.set_thetas([0.3, -0.5, 0.1])
    
    # Get end-effector position
    position = fk.get_current_position()
    print(f"End-Effector Position: {position}")
    print(f"  X: {float(position[0]):.2f} mm")
    print(f"  Y: {float(position[1]):.2f} mm")
    print(f"  Z: {float(position[2]):.2f} mm")
    
    # Get distance from base (convert to float first)
    pos_float = np.array(position, dtype=float)
    distance = np.linalg.norm(pos_float)
    print(f"Distance from base: {distance:.2f} mm")


def example_3_get_transformation_matrices():
    """Example 3: Get transformation matrices for all joints."""
    print("\n" + "="*70)
    print("EXAMPLE 3: get_transformation_matrices() - Get All Frames")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    fk.set_thetas([0.2, -0.3, 0.1])
    
    # Get all transformation matrices
    T_matrices = fk.get_transformation_matrices()
    
    print(f"Number of transformation matrices: {len(T_matrices)}")
    print(f"(Includes base frame + one for each joint)")
    
    # Show end-effector frame
    print(f"\nEnd-Effector Transformation Matrix (T_6):")
    T_ee = np.array(T_matrices[-1], dtype=float)
    print(T_ee)
    
    # Extract orientation and position
    R = T_ee[:3, :3]  # Rotation matrix
    p = T_ee[:3, 3]   # Position vector
    print(f"\nOrientation (rotation matrix):\n{R}")
    print(f"\nPosition: {p}")


def example_4_get_jacobian_matrix():
    """Example 4: Compute Jacobian matrix."""
    print("\n" + "="*70)
    print("EXAMPLE 4: get_jacobian_matrix() - Compute Jacobian")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    fk.set_thetas([0.0, -np.pi/6, 0.0])
    
    # Get full Jacobian (6 x n)
    J = fk.get_jacobian_matrix()
    
    print(f"Jacobian shape: {J.shape}")
    print(f"(6 rows = [linear_vel_x, linear_vel_y, linear_vel_z, angular_vel_x, angular_vel_y, angular_vel_z])")
    print(f"(n columns = one per joint)")
    
    print(f"\nJacobian matrix:\n{J}")
    
    # Extract linear velocity part
    Jv = J[:3, :]
    print(f"\nLinear velocity Jacobian (first 3 rows):\n{Jv}")
    
    # Get linear velocity Jacobians for each joint
    Jv_individual = fk.get_jacobian_linear_velocities()
    print(f"\nIndividual linear velocity Jacobians: {len(Jv_individual)} matrices")


def example_5_get_rotation_matrices():
    """Example 5: Get rotation matrices for each frame."""
    print("\n" + "="*70)
    print("EXAMPLE 5: get_rotation_matrices() - Get Orientations")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    fk.set_thetas([np.pi/4, -np.pi/6, 0.1])
    
    # Get rotation matrices
    R_matrices = fk.get_rotation_matrices()
    
    print(f"Number of rotation matrices: {len(R_matrices)}")
    print(f"Each is a 3x3 matrix representing frame orientation\n")
    
    # Show end-effector rotation
    R_ee = np.array(R_matrices[-1], dtype=float)
    print(f"End-Effector Rotation Matrix:")
    print(R_ee)
    
    # Extract Euler angles (approximate)
    print(f"\nRotation matrix properties:")
    print(f"  Determinant (should be 1): {np.linalg.det(R_ee):.6f}")
    print(f"  Orthogonal (R^T * R = I): {np.allclose(R_ee.T @ R_ee, np.eye(3))}")


def example_6_check_singularities():
    """Example 6: Check for singularities."""
    print("\n" + "="*70)
    print("EXAMPLE 6: is_singular() - Detect Singularities")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    
    configs = [
        ([0, 0, 0], "Home (vertical)"),
        ([0, -np.pi/2, 0], "Horizontal"),
        ([0, -np.pi/4, 0], "45 degrees"),
    ]
    
    print("Checking for singularities at different configurations:\n")
    for thetas, label in configs:
        fk.set_thetas(thetas)
        is_sing = fk.is_singular()
        print(f"  {label:<25} Singular: {is_sing}")


def example_7_get_velocities():
    """Example 7: Get joint velocities."""
    print("\n" + "="*70)
    print("EXAMPLE 7: get_velocities() - Get Joint Velocity Vectors")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    
    # Get velocity vectors for each joint
    velocities = fk.get_velocities()
    
    print(f"Velocity vectors shape: {velocities.shape}")
    print(f"(Each row is a 3D velocity vector for each joint)\n")
    
    for i, v in enumerate(velocities):
        print(f"Joint {i+1} velocity: {v}")
        # Convert to float for norm computation
        v_float = np.array(v, dtype=float)
        print(f"  Magnitude: {np.linalg.norm(v_float):.4f}")


# =====================================================================
# SECTION 3: Common IK Functions
# =====================================================================

def example_8_inverse_kinematics():
    """Example 8: Solve inverse kinematics."""
    print("\n" + "="*70)
    print("EXAMPLE 8: compute_thetas_for_position() - Inverse Kinematics")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    ik = IK_Numerical(fk)
    
    # Define a target position
    target_position = np.array([600.0, 100.0, 1300.0])
    initial_guess = np.array([0.0, -0.5, 0.0])
    
    print(f"Target position: {target_position}")
    print(f"Initial joint guess: {initial_guess}")
    
    # Solve IK manually with proper float conversion
    try:
        thetas = initial_guess.copy()
        for it in range(200):
            fk.set_thetas(thetas)
            current_pos = np.array(fk.get_current_position(), dtype=float).ravel()
            pos_error = target_position - current_pos
            
            J = np.array(fk.get_jacobian_matrix(), dtype=float)
            Jv = J[:3, :]
            Jv_pinv = np.linalg.pinv(Jv)
            delta_theta = 0.3 * (Jv_pinv @ pos_error)
            thetas += delta_theta
            
            error_norm = np.linalg.norm(pos_error)
            if error_norm < 1.0:
                break
        
        solution = thetas
        print(f"\n✓ Solution found!")
        print(f"Joint angles (radians): {solution}")
        print(f"Joint angles (degrees): {np.rad2deg(solution)}")
        
        # Verify solution
        fk.set_thetas(solution)
        actual_position = np.array(fk.get_current_position(), dtype=float)
        error = np.linalg.norm(target_position - actual_position)
        print(f"\nActual position: {actual_position}")
        print(f"Position error: {error:.4f} mm")
        
    except Exception as e:
        print(f"✗ IK failed: {e}")


# =====================================================================
# SECTION 4: Workflow Examples
# =====================================================================

def example_9_trajectory_generation():
    """Example 9: Generate a simple point-to-point trajectory."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Trajectory Generation - Multiple Points")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    
    # Define waypoints
    waypoints = [
        ([0, 0, 0], "Home"),
        (np.array([400.0, 50.0, 1200.0]), "Point 1"),
        (np.array([700.0, -100.0, 1400.0]), "Point 2"),
        (np.array([600.0, 200.0, 1100.0]), "Point 3"),
    ]
    
    print("Trajectory waypoints:\n")
    current_theta = np.array([0.0, -0.5, 0.0])
    
    for thetas_or_pos, label in waypoints:
        if isinstance(thetas_or_pos, (list, tuple)):
            # Direct joint angles
            current_theta = np.array(thetas_or_pos, dtype=float)
            fk.set_thetas(current_theta)
            pos = fk.get_current_position()
            print(f"{label:<15} θ = {np.rad2deg(current_theta)}")
        else:
            # Compute IK for position
            try:
                thetas = current_theta.copy()
                for it in range(150):
                    fk.set_thetas(thetas)
                    current_pos = np.array(fk.get_current_position(), dtype=float).ravel()
                    pos_error = thetas_or_pos - current_pos
                    
                    J = np.array(fk.get_jacobian_matrix(), dtype=float)
                    Jv = J[:3, :]
                    Jv_pinv = np.linalg.pinv(Jv)
                    delta_theta = 0.3 * (Jv_pinv @ pos_error)
                    thetas += delta_theta
                    
                    error_norm = np.linalg.norm(pos_error)
                    if error_norm < 1.0:
                        break
                
                current_theta = thetas
                print(f"{label:<15} θ = {np.rad2deg(current_theta)} → pos = {thetas_or_pos}")
            except:
                print(f"{label:<15} [Failed to reach]")


def example_10_dexterity_analysis():
    """Example 10: Analyze robot dexterity at different configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Dexterity Analysis - Manipulability Index")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    
    configs = [
        ([0, -np.pi/6, 0], "Config 1"),
        ([0, -np.pi/4, 0], "Config 2"),
        ([0, -np.pi/3, 0], "Config 3"),
    ]
    
    print("Jacobian-based dexterity metrics (using linear part):\n")
    print(f"{'Config':<15} {'Det(Jv)':<15} {'Cond(Jv)':<15} {'Singular':<15}")
    print("-" * 60)
    
    for thetas, label in configs:
        fk.set_thetas(thetas)
        J = np.array(fk.get_jacobian_matrix(), dtype=float)
        Jv = J[:3, :]  # Use only linear velocity part (3x3 for 3-DOF)
        
        # Compute metrics
        try:
            det_Jv = np.linalg.det(Jv)
            cond_Jv = np.linalg.cond(Jv)
            is_singular = abs(det_Jv) < 1e-3
        except:
            det_Jv = 0.0
            cond_Jv = np.inf
            is_singular = True
        
        print(f"{label:<15} {det_Jv:<15.4e} {cond_Jv:<15.4f} {'Yes' if is_singular else 'No':<15}")


def example_11_multi_joint_analysis():
    """Example 11: Analyze individual joint contributions."""
    print("\n" + "="*70)
    print("EXAMPLE 11: Individual Joint Analysis")
    print("="*70)
    
    fk = setup_3dof_manipulator()
    fk.set_thetas([0.2, -0.4, 0.1])
    
    # Get individual Jacobians
    J_matrices = fk.get_jacobian_matrices()
    Jv_list = fk.get_jacobian_linear_velocities()
    Jw_list = fk.get_jacobian_angular_velocities()
    
    print(f"Number of joints: {len(J_matrices)}\n")
    
    for i in range(len(J_matrices)):
        print(f"Joint {i+1}:")
        print(f"  Full Jacobian shape: {np.array(J_matrices[i]).shape}")
        print(f"  Linear velocity Jacobian shape: {np.array(Jv_list[i]).shape}")
        print(f"  Angular velocity Jacobian shape: {np.array(Jw_list[i]).shape}")


def example_12_ur5e_operations():
    """Example 12: Basic operations with UR5e 6-DOF robot."""
    print("\n" + "="*70)
    print("EXAMPLE 12: UR5e 6-DOF Robot Operations")
    print("="*70)
    
    fk = setup_ur5e_robot()
    
    print("UR5e Setup:")
    print(f"  Number of joints: 6")
    print(f"  Home position: {fk.get_current_position()}")
    
    # Set a configuration
    config = [0.5, -np.pi/4, np.pi/6, 0.0, 0.0, 0.0]
    fk.set_thetas(config)
    
    print(f"\nAfter setting thetas: {np.rad2deg(config)}")
    print(f"  End-effector position: {np.array(fk.get_current_position(), dtype=float)}")
    print(f"  Jacobian shape: {fk.get_jacobian_matrix().shape}")


# =====================================================================
# SECTION 5: Main Demo
# =====================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("FK AND IK COMMON FUNCTIONS - USAGE EXAMPLES")
    print("="*70)
    
    examples = [
        example_1_set_joint_angles,
        example_2_get_end_effector_position,
        example_3_get_transformation_matrices,
        example_4_get_jacobian_matrix,
        example_5_get_rotation_matrices,
        example_6_check_singularities,
        example_7_get_velocities,
        example_8_inverse_kinematics,
        example_9_trajectory_generation,
        example_10_dexterity_analysis,
        example_11_multi_joint_analysis,
        example_12_ur5e_operations,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
