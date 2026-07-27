"""Open-loop torque trajectory from UR5e home to one Cartesian point."""

from pathlib import Path
import sys

from controller import Robot
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libraries.DynamicalModel.ur5e_trajectory import quintic_trajectory
from libraries.manipulatorKinematics import create_webots_ur5e_kinematics
from libraries.ControlModeling import ControlModeling

PD = ControlModeling.calculate_pd_params_for_ur5e()

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])

TRAJECTORY_DURATION = 5.0
IK_TOLERANCE = 1e-5

position_list = [
    [0.73, 0.46, 0.0498]
]

robot = Robot()
timestep = int(robot.getBasicTimeStep())
sample_period = timestep / 1000.0

print(f"Webots timestep: {timestep} ms, sample period: {sample_period:.4f} s")

motors = []
sensors = []
for joint_name in JOINT_NAMES:
    motor = robot.getDevice(joint_name)
    if motor is None:
        raise RuntimeError(f'Cannot find motor "{joint_name}".')

    motor.setVelocity(0.0)
    motors.append(motor)

    sensor = motor.getPositionSensor()
    sensor.enable(timestep)
    sensors.append(sensor)

torque_limits = np.array(
    [motor.getMaxTorque() for motor in motors], dtype=float
)

def inverse_kinematics(position: np.ndarray) -> np.ndarray:
    """Calculate the first point's joint angles using home as the IK seed."""
    kinematics, inverse_kinematics = create_webots_ur5e_kinematics(HOME_Q)
    q_goal = inverse_kinematics.compute_thetas_for_position(
        desired_position=position,
        initial_thetas=HOME_Q,
        alpha=0.25,
        max_iterations=400,
        epsilon=IK_TOLERANCE,
        verbose=False,
    )

    return q_goal


def send_torque(torque: np.ndarray, target_q, current_q) -> None:
    """Send one open-loop torque vector to the six motors."""
    torque = apply_pd_control(torque, target_q, current_q)

    safe_torque = np.clip(torque, -torque_limits, torque_limits)
    for motor, joint_torque in zip(motors, safe_torque):
        motor.setTorque(float(joint_torque))

def apply_pd_control(torques, target_q, current_q):
    """Apply PD control to compute the torque for each joint."""
    for i in range(len(JOINT_NAMES)):
        kp, kd = PD[i]
        # position_error = target_q[i] - current_q[i]
        # velocity_error = target_qdot[i] - current_qdot[i]
        # torques[i] = kp * position_error + kd * velocity_error
        torques[i] += _PD_Controller(i, target_q[i] - current_q[i], kp, kd, sample_period)
    return torques

_pre_errors = [0.0] * len(JOINT_NAMES)

def _PD_Controller(i: int, error: float, kp: float, kd: float, dt: float) -> float:
    
    if dt <= 0.0:
        return 0.0

    derivative = (error - _pre_errors[i]) / dt
    
    # effort = self.kp * error + self.kd * derivative
    effort = kp * error + kd * derivative
    _pre_errors[i] = error

    return effort


q_goal = inverse_kinematics(np.array(position_list[0], dtype=float))
print(f"First point q (deg): {np.rad2deg(q_goal).round(2)}")
print("Calculating trajectory torque...")

trajectory = quintic_trajectory(
    start=HOME_Q,
    goal=q_goal,
    duration=TRAJECTORY_DURATION,
    sample_period=sample_period,
)
print(f"Calculated {trajectory.time.size} torque samples.")

sample_index = 0

while robot.step(timestep) != -1:
    
    if sample_index < trajectory.time.size:
        # Send exactly one planned torque sample per Webots timestep.
        current_q = np.array([sensor.getValue() for sensor in sensors], dtype=float)
        send_torque(trajectory.torque[sample_index], 
                    trajectory.position[sample_index], 
                    current_q
                    )
        sample_index += 1

        if sample_index == trajectory.time.size:
            print("Trajectory torque sequence complete.")
    else:
        # The final sample is the model's stationary holding torque.
        send_torque(trajectory.torque[-1])
