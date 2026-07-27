"""Move the UR5e through Cartesian points using model torque and PD control."""

from pathlib import Path
import sys

from controller import Robot
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libraries.ControlModeling import ControlModeling
from libraries.DynamicalModel.ur5e_trajectory import (
    Trajectory,
    quintic_trajectory,
)
from libraries.manipulatorKinematics import create_webots_ur5e_kinematics


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])

# Edit this list to change the end-effector path. Coordinates are [x, y, z]
# in metres in the robot base/world frame.
CARTESIAN_POINTS = np.array(
    [
        [0.73, 0.46, 0.0498],
        [0.65, 0.40, 0.15],
        [0.55, 0.35, 0.20],
    ],
    dtype=float,
)

TRAJECTORY_DURATION = 5.0
IK_TOLERANCE = 1e-5
IK_MAX_ITERATIONS = 400


def solve_cartesian_points() -> list[tuple[int, np.ndarray]]:
    """Convert reachable Cartesian points to joint positions."""
    kinematics, ik_solver = create_webots_ur5e_kinematics(HOME_Q)
    initial_q = HOME_Q.copy()
    joint_points = []

    for point_index, point in enumerate(CARTESIAN_POINTS):
        try:
            target_q = ik_solver.compute_thetas_for_position(
                desired_position=point,
                initial_thetas=initial_q,
                alpha=0.25,
                max_iterations=IK_MAX_ITERATIONS,
                epsilon=IK_TOLERANCE,
                verbose=False,
            )
        except ValueError as error:
            print(
                f"Skipping Cartesian point {point_index + 1} "
                f"{point.tolist()}: {error}"
            )
            continue

        kinematics.set_thetas(target_q)
        reached_point = np.asarray(
            kinematics.get_current_position(),
            dtype=float,
        ).reshape(3)
        error = np.linalg.norm(point - reached_point)
        if not np.all(np.isfinite(target_q)) or error > IK_TOLERANCE:
            print(
                f"Skipping Cartesian point {point_index + 1}: "
                f"IK error is {error:.6e} m."
            )
            continue

        joint_points.append((point_index, target_q.copy()))
        initial_q = target_q
        print(
            f"Point {point_index + 1} joint target (deg): "
            f"{np.rad2deg(target_q).round(2)}"
        )

    if not joint_points:
        raise RuntimeError("None of the Cartesian points are reachable.")

    return joint_points


def plan_trajectories(
    joint_points: list[tuple[int, np.ndarray]],
    sample_period: float,
) -> list[tuple[int, Trajectory]]:
    """Plan home-to-first and point-to-point trajectory segments."""
    trajectories = []
    start_q = HOME_Q.copy()

    for point_index, goal_q in joint_points:
        print(f"Planning trajectory to point {point_index + 1}...")
        trajectory = quintic_trajectory(
            start=start_q,
            goal=goal_q,
            duration=TRAJECTORY_DURATION,
            sample_period=sample_period,
        )
        trajectories.append((point_index, trajectory))
        start_q = goal_q

    return trajectories


def calculate_control_torque(
    model_torque: np.ndarray,
    target_q: np.ndarray,
    target_qdot: np.ndarray,
    current_q: np.ndarray,
    current_qdot: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Apply the model feed-forward torque and calculated PD feedback."""
    position_error = target_q - current_q
    velocity_error = target_qdot - current_qdot
    return (
        model_torque
        + kp * position_error
        + kd * velocity_error
    )


robot = Robot()
timestep = int(robot.getBasicTimeStep())
sample_period = timestep / 1000.0

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

torque_limits = np.asarray(
    [motor.getMaxTorque() for motor in motors],
    dtype=float,
)

pd_gains = np.asarray(
    ControlModeling.calculate_pd_params_for_ur5e(),
    dtype=float,
)
if pd_gains.shape != (len(JOINT_NAMES), 2):
    raise RuntimeError(
        f"Expected six (kp, kd) pairs, received {pd_gains.shape}."
    )
if not np.all(np.isfinite(pd_gains)):
    raise RuntimeError("The calculated PD gains are not finite.")

kp = pd_gains[:, 0]
kd = pd_gains[:, 1]

print(f"Webots timestep: {timestep} ms")
for joint_name, joint_kp, joint_kd in zip(JOINT_NAMES, kp, kd):
    print(f"{joint_name}: kp={joint_kp:.4f}, kd={joint_kd:.4f}")

joint_points = solve_cartesian_points()
trajectories = plan_trajectories(joint_points, sample_period)
print(f"Ready to execute {len(trajectories)} trajectory segments.")

segment_index = 0
sample_index = 0
previous_q = None
motion_complete = False

while robot.step(timestep) != -1:
    current_q = np.asarray(
        [sensor.getValue() for sensor in sensors],
        dtype=float,
    )

    if previous_q is None:
        current_qdot = np.zeros(len(JOINT_NAMES), dtype=float)
    else:
        current_qdot = (current_q - previous_q) / sample_period
    previous_q = current_q.copy()

    point_index, trajectory = trajectories[segment_index]

    if motion_complete:
        model_torque = trajectory.torque[-1]
        target_q = trajectory.position[-1]
        target_qdot = np.zeros(len(JOINT_NAMES), dtype=float)
    elif sample_index < trajectory.time.size:
        model_torque = trajectory.torque[sample_index]
        target_q = trajectory.position[sample_index]
        target_qdot = trajectory.velocity[sample_index]
    else:
        print(f"Reached Cartesian point {point_index + 1}.")
        if segment_index < len(trajectories) - 1:
            segment_index += 1
            sample_index = 0
            point_index, trajectory = trajectories[segment_index]
            model_torque = trajectory.torque[0]
            target_q = trajectory.position[0]
            target_qdot = trajectory.velocity[0]
        else:
            motion_complete = True
            print("All Cartesian points reached. Holding the final point.")
            model_torque = trajectory.torque[-1]
            target_q = trajectory.position[-1]
            target_qdot = np.zeros(len(JOINT_NAMES), dtype=float)

    commanded_torque = calculate_control_torque(
        model_torque=model_torque,
        target_q=target_q,
        target_qdot=target_qdot,
        current_q=current_q,
        current_qdot=current_qdot,
        kp=kp,
        kd=kd,
    )
    commanded_torque = np.clip(
        commanded_torque,
        -torque_limits,
        torque_limits,
    )

    for motor, joint_torque in zip(motors, commanded_torque):
        motor.setTorque(float(joint_torque))

    if not motion_complete and sample_index < trajectory.time.size:
        sample_index += 1
