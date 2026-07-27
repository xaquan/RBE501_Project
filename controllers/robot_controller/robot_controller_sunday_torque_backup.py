"""Move the UR5e through three Cartesian positions using planned trajectories."""

from pathlib import Path
import struct
import sys

from controller import Robot
import numpy as np


# Make the project-level libraries importable from the Webots controller.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libraries.DynamicalModel.ur5e_trajectory import quintic_trajectory
from libraries.manipulatorKinematics import create_webots_ur5e_kinematics


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Standard UR5e home configuration, in radians.
HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])

# Three stationary end-effector targets [x, y, z], in metres, expressed in
# the Webots robot base frame.
CARTESIAN_TARGETS = np.array(
    [
        [0.366269, 0.555773, 0.235061],
        [0.470301, -0.280796, 0.248677],
        [-0.134000, 0.740002, 0.167520],
    ],
    dtype=float,
)

# Position-only IK has multiple solutions. These nearby guesses select
# repeatable, joint-limit-safe solution branches for the three targets.
IK_INITIAL_GUESSES = np.deg2rad(
    [
        [48.0, -63.0, 93.0, -123.0, -87.0, 27.0],
        [-42.0, -78.0, 113.0, -128.0, -87.0, -33.0],
        [93.0, -48.0, 78.0, -123.0, -87.0, 42.0],
    ]
)

TRAJECTORY_DURATION = 5.0
TARGET_TIMEOUT = 3.0
JOINT_TOLERANCE = np.deg2rad(0.5)
IK_TOLERANCE = 1e-5
VELOCITY_FILTER = 0.25

# Torque feedback gains [N m/rad] and [N m s/rad]. The inverse-dynamics
# trajectory torque is the feed-forward term; these gains reject model and
# tracking errors.
KP_TORQUE = np.array([80.0, 100.0, 70.0, 15.0, 10.0, 5.0])
KD_TORQUE = np.array([12.0, 15.0, 10.0, 2.5, 1.5, 0.8])


robot = Robot()
timestep = int(robot.getBasicTimeStep())
sample_period = timestep / 1000.0

receiver = robot.getDevice("serial_receiver")
if receiver is None:
    raise RuntimeError('Cannot find receiver "serial_receiver".')
receiver.setChannel(1)
receiver.enable(timestep)

motors = []
sensors = []
for joint_name in JOINT_NAMES:
    motor = robot.getDevice(joint_name)
    sensor = robot.getDevice(f"{joint_name}_sensor")
    if motor is None or sensor is None:
        raise RuntimeError(f'Cannot find devices for joint "{joint_name}".')

    sensor.enable(timestep)
    motor.setTorque(0.0)  # Disable Webots PID and enter direct torque mode.
    motors.append(motor)
    sensors.append(sensor)

torque_limits = np.array(
    [motor.getMaxTorque() for motor in motors], dtype=float
)
if np.any(torque_limits <= 0.0):
    raise RuntimeError("Every joint motor must have a positive maxTorque.")


def measured_joints() -> np.ndarray:
    """Return the six measured joint angles in radians."""
    return np.array([sensor.getValue() for sensor in sensors], dtype=float)


def validate_joint_positions(q: np.ndarray) -> None:
    """Check the size, values, and Webots limits of a joint vector."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != len(JOINT_NAMES):
        raise ValueError(f"Expected 6 joint angles, received {q.size}.")
    if not np.all(np.isfinite(q)):
        raise ValueError("Joint angles must contain only finite values.")

    for joint_name, motor, angle in zip(JOINT_NAMES, motors, q):
        minimum = motor.getMinPosition()
        maximum = motor.getMaxPosition()
        has_limits = minimum != 0.0 or maximum != 0.0
        if has_limits and not minimum <= angle <= maximum:
            raise ValueError(
                f"{joint_name} target {np.rad2deg(angle):.2f} deg is outside "
                f"[{np.rad2deg(minimum):.2f}, "
                f"{np.rad2deg(maximum):.2f}] deg."
            )


def command_torques(
    desired_q: np.ndarray,
    desired_qdot: np.ndarray,
    feedforward_torque: np.ndarray,
    measured_q: np.ndarray,
    measured_qdot: np.ndarray,
) -> None:
    """Apply inverse-dynamics feed-forward torque plus PD feedback."""
    position_error = desired_q - measured_q
    velocity_error = desired_qdot - measured_qdot
    feedback_torque = (
        KP_TORQUE * position_error + KD_TORQUE * velocity_error
    )
    requested_torque = feedforward_torque + feedback_torque
    safe_torque = np.clip(
        requested_torque, -torque_limits, torque_limits
    )

    for motor, torque in zip(motors, safe_torque):
        motor.setTorque(float(torque))


def solve_target_joints() -> list[np.ndarray]:
    """Use the project IK solver to convert all Cartesian targets to q."""
    kinematics, inverse_kinematics = create_webots_ur5e_kinematics(HOME_Q)
    solutions = []

    for index, (target, initial_guess) in enumerate(
        zip(CARTESIAN_TARGETS, IK_INITIAL_GUESSES), start=1
    ):
        q_target = inverse_kinematics.compute_thetas_for_position(
            desired_position=target,
            initial_thetas=initial_guess,
            alpha=0.35,
            max_iterations=200,
            epsilon=IK_TOLERANCE,
            verbose=False,
        )
        q_target = np.asarray(q_target, dtype=float).reshape(6)
        validate_joint_positions(q_target)

        kinematics.set_thetas(q_target)
        reached_position = np.asarray(
            kinematics.get_current_position(), dtype=float
        ).reshape(3)
        position_error = np.linalg.norm(reached_position - target)
        if position_error > IK_TOLERANCE:
            raise RuntimeError(
                f"IK validation failed for target {index}: "
                f"{position_error:.3e} m."
            )

        solutions.append(q_target)
        print(
            f"Target {index}: xyz={target.round(4)}, "
            f"q(deg)={np.rad2deg(q_target).round(2)}"
        )

    return solutions


def plan_trajectories(target_joints: list[np.ndarray]):
    """Plan all rest-to-rest segments before starting target motion."""
    trajectories = []
    segment_start = HOME_Q

    for segment_goal in target_joints:
        trajectory = quintic_trajectory(
            start=segment_start,
            goal=segment_goal,
            duration=TRAJECTORY_DURATION,
            sample_period=sample_period,
        )
        trajectories.append(trajectory)
        segment_start = segment_goal

    return trajectories


def receive_camera_objects(
    detected_objects_world: dict[int, np.ndarray],
) -> None:
    """Decode every queued camera packet and retain its world position.

    The camera controller performs the camera-to-world conversion before
    packing ``(object_id, world_x, world_y, world_z)``. Therefore, these
    received coordinates must not be transformed again here.
    """
    packet_size = struct.calcsize("i3f")

    while receiver.getQueueLength() > 0:
        packet = receiver.getBytes()
        if len(packet) != packet_size:
            print(
                f"Ignoring camera packet with {len(packet)} bytes; "
                f"expected {packet_size}."
            )
            receiver.nextPacket()
            continue

        object_id, world_x, world_y, world_z = struct.unpack("i3f", packet)
        is_new_object = object_id not in detected_objects_world
        detected_objects_world[object_id] = np.array(
            [world_x, world_y, world_z], dtype=float
        )
        receiver.nextPacket()

        if is_new_object:
            print(
                f"Camera object {object_id}: world xyz="
                f"{detected_objects_world[object_id].round(4)}"
            )


# Solve and plan the stationary Cartesian targets once.
joint_targets = solve_target_joints()
target_trajectories = plan_trajectories(joint_targets)

# Motion is advanced by a state machine inside the single Webots step loop.
# This lets camera packets and future tasks be handled during every movement.
detected_objects_world: dict[int, np.ndarray] = {}
motion_state = "initializing"
planned_trajectories = []
trajectory_index = 0
trajectory_sample_index = 0
target_deadline = 0.0
previous_q = None
filtered_qdot = np.zeros(6)

validate_joint_positions(HOME_Q)
print(f"Torque limits (N m): {torque_limits.round(2)}")

while robot.step(timestep) != -1:
    receive_camera_objects(detected_objects_world)
    current_q = measured_joints()

    if previous_q is None:
        current_qdot = np.zeros(6)
    else:
        raw_qdot = (current_q - previous_q) / sample_period
        filtered_qdot = (
            VELOCITY_FILTER * raw_qdot
            + (1.0 - VELOCITY_FILTER) * filtered_qdot
        )
        current_qdot = filtered_qdot
    previous_q = current_q.copy()

    if motion_state == "initializing":
        # Position sensors become valid after the first robot step. Plan a
        # smooth torque-controlled segment from that measured pose to home.
        home_trajectory = quintic_trajectory(
            start=current_q,
            goal=HOME_Q,
            duration=TRAJECTORY_DURATION,
            sample_period=sample_period,
        )
        planned_trajectories = [home_trajectory] + target_trajectories
        motion_state = "executing"
        print("Moving to home position with torque control.")

    elif motion_state == "executing":
        trajectory = planned_trajectories[trajectory_index]

        # Apply exactly one planned torque sample per Webots step. Position
        # and velocity feedback keep the robot on the planned trajectory.
        command_torques(
            desired_q=trajectory.position[trajectory_sample_index],
            desired_qdot=trajectory.velocity[trajectory_sample_index],
            feedforward_torque=trajectory.torque[trajectory_sample_index],
            measured_q=current_q,
            measured_qdot=current_qdot,
        )
        trajectory_sample_index += 1

        if trajectory_sample_index >= trajectory.time.size:
            motion_state = "settling"
            target_deadline = robot.getTime() + TARGET_TIMEOUT

    elif motion_state == "settling":
        trajectory = planned_trajectories[trajectory_index]
        final_q = trajectory.position[-1]

        # Continue gravity compensation and feedback while checking that the
        # endpoint has physically been reached.
        command_torques(
            desired_q=final_q,
            desired_qdot=np.zeros(6),
            feedforward_torque=trajectory.torque[-1],
            measured_q=current_q,
            measured_qdot=current_qdot,
        )

        if np.max(np.abs(current_q - final_q)) <= JOINT_TOLERANCE:
            if trajectory_index == 0:
                print("Home position reached.")
            else:
                print(f"Target {trajectory_index} reached.")
            trajectory_index += 1

            if trajectory_index >= len(planned_trajectories):
                motion_state = "holding"
                print(
                    "All three targets completed. Holding final position."
                )
            else:
                trajectory_sample_index = 0
                motion_state = "executing"
                print(f"Moving to target {trajectory_index}.")
        elif robot.getTime() >= target_deadline:
            raise RuntimeError("A trajectory endpoint was not reached in time.")

    elif motion_state == "holding":
        # Direct torque commands persist, but recomputing feedback each step
        # rejects disturbances while camera data continues to be processed.
        trajectory = planned_trajectories[-1]
        command_torques(
            desired_q=trajectory.position[-1],
            desired_qdot=np.zeros(6),
            feedforward_torque=trajectory.torque[-1],
            measured_q=current_q,
            measured_qdot=current_qdot,
        )
