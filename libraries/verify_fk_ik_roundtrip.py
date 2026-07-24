"""Round-trip FK/IK validation for the example manipulator.

This script sets several joint configurations, computes the forward kinematics
position, then asks the inverse kinematics solver to recover the same pose.
It reports both joint-space and position-space error so you can confirm the
implementation is consistent.
"""

import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from libraries.ManipulatorKinematics import IK_Numerical
from libraries.example_fk_ik_usage import setup_3dof_manipulator


def run_roundtrip_check() -> None:
    fk = setup_3dof_manipulator()
    ik = IK_Numerical(fk)

    test_configs = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.25, -0.35, 0.15]),
        np.array([-0.5, 0.4, -0.2]),
    ]

    np.set_printoptions(precision=6, suppress=True)

    for index, target_thetas in enumerate(test_configs, start=1):
        fk.set_thetas(target_thetas)
        target_position = np.array(fk.get_current_position(), dtype=float)

        solved_thetas = ik.compute_thetas_for_position(
            desired_position=target_position,
            initial_thetas=target_thetas,
            alpha=0.3,
            max_iterations=50,
            epsilon=1e-6,
            verbose=False,
        )

        fk.set_thetas(solved_thetas)
        solved_position = np.array(fk.get_current_position(), dtype=float)

        theta_error = np.linalg.norm(np.asarray(solved_thetas, dtype=float) - target_thetas)
        position_error = np.linalg.norm(solved_position - target_position)

        print(f"Case {index}")
        print(f"  target thetas   = {target_thetas}")
        print(f"  solved thetas   = {np.asarray(solved_thetas, dtype=float)}")
        print(f"  theta error     = {theta_error:.3e}")
        print(f"  target position = {target_position}")
        print(f"  solved position = {solved_position}")
        print(f"  position error  = {position_error:.3e}")
        print()


if __name__ == "__main__":
    run_roundtrip_check()