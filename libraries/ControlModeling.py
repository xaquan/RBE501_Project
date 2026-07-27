import numpy as np

try:
    # Normal Webots use: imported as ``libraries.ControlModeling``.
    from .DynamicalModel.ur5e_trajectory import create_ur5e_real_dynamics
except ImportError:
    # Direct-file use: ``libraries`` itself is on sys.path.
    from DynamicalModel.ur5e_trajectory import create_ur5e_real_dynamics


class ControlModeling:
    @staticmethod
    def __pd_params_calculate(b, xi, ts, j, disturbance=0.0):
        wn = 5.3/(ts*xi)
        kp = wn**2 * j + disturbance
        kd = 2*xi*wn*j - b
        return kp, kd

    @staticmethod
    def calculate_pd_params_for_ur5e():
        """Calculate PD parameters for each joint of the UR5e robot."""
        HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
        DAMPING_RATIO = 1.0
        SETTLING_TIME = 0.5
        VISCOUS_DAMPING = 0.0

        lagrange_model = create_ur5e_real_dynamics()
        D, _, _ = lagrange_model.get_DCG_matrices(
            HOME_Q,
            np.zeros(6),
            np.zeros(6),
        )
        pd_params = []
        for i in range(len(lagrange_model.masses)):
            j = D[i, i]
            kp, kd = ControlModeling.__pd_params_calculate(
                b=VISCOUS_DAMPING,
                xi=DAMPING_RATIO,
                ts=SETTLING_TIME,
                j=j,
            )
            if kp <= 0.0 or kd <= 0.0:
                raise RuntimeError(
                    f"Calculated non-positive PD gain for joint {i + 1}: "
                    f"kp={kp}, kd={kd}."
                )
            pd_params.append((kp, kd))

        return pd_params
