"""Move the UR5e to objects reported by the overhead camera."""

from pathlib import Path
import struct
import sys

from controller import Supervisor
import numpy as np


# Project libraries.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libraries.DynamicalModel.ur5e_trajectory import quintic_trajectory
from libraries.ManipulatorKinematics import create_webots_ur5e_kinematics


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Home joint angles (rad).
HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])

# Camera reports block centres.
OBJECT_APPROACH_OFFSET = np.array([0.0, 0.0, 0.20], dtype=float)
# Cup contact height for 0.1 m blocks.
OBJECT_GRASP_OFFSET = np.array([0.0, 0.0, 0.070], dtype=float)
OBJECT_REVISIT_DISTANCE = 0.01
OBJECT_REPLAN_DISTANCE = 0.02
GRASP_TIMEOUT = 1.0
LIFT_TARGET_TIMEOUT = 10.0
LIFT_JOINT_TOLERANCE = np.deg2rad(2.5)
# False: normal two-block run. True: 0 / 0.5 / 1 kg benchmark.
RUN_SEQUENTIAL_PAYLOAD_BENCHMARK = False
NORMAL_OBJECTS = (
    (np.array([0.500, -0.4995, 0.0496]), 1.0),
    (np.array([0.730, 0.4600, 0.0498]), 0.5),
)
NORMAL_DROP_CENTERS = np.array(
    [[0.366269, 0.555773, 0.05], [0.600, -0.200, 0.05]],
    dtype=float,
)
# Webots needs a positive mass; the model still uses 0 kg exactly.
BENCHMARK_SOURCE = np.array([0.500, -0.4995, 0.0496], dtype=float)
BENCHMARK_DROP = np.array([0.366269, 0.555773, 0.05], dtype=float)
BENCHMARK_PAYLOADS = (0.0, 0.5, 1.0)
PHYSICAL_ZERO_MASS = 1e-6
PLACE_DURATION = 1.0
RELEASE_SETTLE_TIME = 0.5
PAYLOAD_MOTION_STATES = {
    "lift",
    "carry_to_drop",
    "place_descend",
}
IK_MAX_ITERATIONS = 400
IK_STEP_SIZE = 0.30
IK_DAMPING = 1e-3

DOWNWARD_TOOL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=float,
)

TRAJECTORY_DURATION = 1.5
TARGET_TIMEOUT = 3.0
JOINT_TOLERANCE = np.deg2rad(0.5)
IK_TOLERANCE = 1e-5
VELOCITY_FILTER = 0.25

# PD tracking gains for the inverse-dynamics torque.
KP_TORQUE = np.array([80.0, 100.0, 70.0, 15.0, 10.0, 5.0])
KD_TORQUE = np.array([12.0, 15.0, 10.0, 2.5, 1.5, 0.8])
SIMULATION_TORQUE_LOG = PROJECT_ROOT / "torque_logs" / "simulation_command_torque.csv"


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
sample_period = timestep / 1000.0

receiver = robot.getDevice("serial_receiver")
if receiver is None:
    raise RuntimeError('Cannot find receiver "serial_receiver".')
receiver.setChannel(1)
receiver.enable(timestep)

gripper = robot.getDevice("vacuum_gripper")
if gripper is None:
    raise RuntimeError('Cannot find VacuumGripper "vacuum_gripper".')
gripper.enablePresence(timestep)
gripper.turnOff()

benchmark_box = robot.getFromDef("box1")
if benchmark_box is None:
    raise RuntimeError('Cannot find benchmark block DEF "box1".')
benchmark_translation = benchmark_box.getField("translation")
benchmark_rotation = benchmark_box.getField("rotation")
benchmark_physics = benchmark_box.getField("physics").getSFNode()
benchmark_mass = benchmark_physics.getField("mass")
normal_second_box = robot.getFromDef("box2")
if normal_second_box is None:
    raise RuntimeError('Cannot find normal-scene block DEF "box2".')
normal_second_translation = normal_second_box.getField("translation")

motors = []
sensors = []
for joint_name in JOINT_NAMES:
    motor = robot.getDevice(joint_name)
    sensor = robot.getDevice(f"{joint_name}_sensor")
    if motor is None or sensor is None:
        raise RuntimeError(f'Cannot find devices for joint "{joint_name}".')

    sensor.enable(timestep)
    motor.setTorque(0.0)  # Direct torque control.
    motors.append(motor)
    sensors.append(sensor)

torque_limits = np.array(
    [motor.getMaxTorque() for motor in motors], dtype=float
)
if np.any(torque_limits <= 0.0):
    raise RuntimeError("Every joint motor must have a positive maxTorque.")

for joint_name, kp, kd in zip(JOINT_NAMES, KP_TORQUE, KD_TORQUE):
    print(f"{joint_name}: kp={kp:.4f}, kd={kd:.4f}")


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
    """Apply Lagrange feed-forward torque plus calculated PD feedback."""
    position_error = desired_q - measured_q
    velocity_error = desired_qdot - measured_qdot
    feedback_torque = (
        KP_TORQUE * position_error + KD_TORQUE * velocity_error
    )
    requested_torque = feedforward_torque + feedback_torque
    safe_torque = np.clip(
        requested_torque, -torque_limits, torque_limits
    )

    log_simulation_torque(robot.getTime(), safe_torque)

    for motor, torque in zip(motors, safe_torque):
        motor.setTorque(float(torque))


def solve_object_joints(
    object_id: int,
    object_position: np.ndarray,
    initial_guess: np.ndarray,
    offset: np.ndarray,
    pose_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a downward-facing joint target for one camera object pose."""
    target = np.asarray(object_position, dtype=float) + offset
    kinematics, inverse_kinematics = create_webots_ur5e_kinematics(initial_guess)
    del inverse_kinematics
    q_target = np.asarray(initial_guess, dtype=float).reshape(6).copy()

    for _ in range(IK_MAX_ITERATIONS):
        kinematics.set_thetas(q_target)
        transform = np.asarray(
            kinematics.get_transformation_matrix(6), dtype=float
        )
        current_position = transform[:3, 3]
        current_rotation = transform[:3, :3]
        position_error_vector = target - current_position
        rotation_error_vector = 0.5 * sum(
            (
                np.cross(
                    current_rotation[:, axis],
                    DOWNWARD_TOOL_ROTATION[:, axis],
                )
                for axis in range(3)
            ),
            np.zeros(3),
        )
        if (
            np.linalg.norm(position_error_vector) <= IK_TOLERANCE
            and np.linalg.norm(rotation_error_vector) <= IK_TOLERANCE
        ):
            break

        pose_error = np.concatenate(
            (position_error_vector, rotation_error_vector)
        )
        jacobian = np.asarray(kinematics.get_jacobian_matrix(), dtype=float)
        damped_system = jacobian @ jacobian.T + IK_DAMPING**2 * np.eye(6)
        delta_q = IK_STEP_SIZE * jacobian.T @ np.linalg.solve(
            damped_system, pose_error
        )
        # Limit each IK step.
        q_target += np.clip(delta_q, -0.15, 0.15)
    else:
        raise ValueError(
            f"Pose IK did not converge for camera object {object_id}."
        )

    validate_joint_positions(q_target)

    kinematics.set_thetas(q_target)
    reached_transform = np.asarray(
        kinematics.get_transformation_matrix(6), dtype=float
    )
    reached_position = reached_transform[:3, 3]
    position_error = np.linalg.norm(reached_position - target)
    reached_rotation = reached_transform[:3, :3]
    orientation_error = 0.5 * sum(
        (
            np.cross(
                reached_rotation[:, axis], DOWNWARD_TOOL_ROTATION[:, axis]
            )
            for axis in range(3)
        ),
        np.zeros(3),
    )
    if (
        position_error > IK_TOLERANCE
        or np.linalg.norm(orientation_error) > IK_TOLERANCE
    ):
        raise RuntimeError(
            f"IK validation failed for camera object {object_id}: "
            f"position={position_error:.3e} m, "
            f"orientation={np.linalg.norm(orientation_error):.3e}."
        )

    print(
        f"Planning camera object {object_id}: object xyz="
        f"{np.asarray(object_position).round(4)}, {pose_name} xyz="
        f"{target.round(4)}"
    )
    return q_target, target


def next_unvisited_object(
    detected_objects_robot: dict[int, np.ndarray],
    visited_object_positions: dict[int, np.ndarray],
    completed_object_ids: set[int],
) -> tuple[int, np.ndarray] | None:
    """Choose a newly seen object, or one that has moved significantly."""
    for object_id in sorted(detected_objects_robot):
        if object_id in completed_object_ids:
            continue
        position = detected_objects_robot[object_id]
        visited_position = visited_object_positions.get(object_id)
        if (
            visited_position is None
            or np.linalg.norm(position - visited_position)
            > OBJECT_REVISIT_DISTANCE
        ):
            return object_id, position.copy()
    return None


def reset_benchmark_block(payload_mass: float) -> None:
    """Put the single benchmark block back at the common source pose."""
    physical_mass = max(payload_mass, PHYSICAL_ZERO_MASS)
    benchmark_mass.setSFFloat(physical_mass)
    benchmark_translation.setSFVec3f(BENCHMARK_SOURCE.tolist())
    benchmark_rotation.setSFRotation([0.0, 0.0, 1.0, 0.0])
    benchmark_box.resetPhysics()
    print(
        f"Reset scene for {payload_mass:.1f} kg payload "
        f"(Webots block mass {physical_mass:g} kg)."
    )


def normal_payload_mass(object_position: np.ndarray) -> float:
    """Return the configured mass for either normal-scene block."""
    reference, mass = min(
        NORMAL_OBJECTS,
        key=lambda item: np.linalg.norm(object_position - item[0]),
    )
    if np.linalg.norm(object_position - reference) > 0.20:
        return 0.5
    return mass


def configure_scene_mode() -> None:
    """Show the normal pair or prepare the single-block benchmark."""
    if RUN_SEQUENTIAL_PAYLOAD_BENCHMARK:
        reset_benchmark_block(BENCHMARK_PAYLOADS[0])
        normal_second_translation.setSFVec3f([0.0, 0.0, -1.0])
    else:
        benchmark_mass.setSFFloat(1.0)
        normal_second_translation.setSFVec3f(NORMAL_OBJECTS[1][0].tolist())


def plan_payload_motion(
    current_q: np.ndarray,
    goal_q: np.ndarray,
    payload_mass: float,
    duration: float = TRAJECTORY_DURATION,
):
    """Create a trajectory that includes the attached object's dynamics."""
    trajectory = quintic_trajectory(
        start=current_q,
        goal=goal_q,
        duration=duration,
        sample_period=sample_period,
        carried_payload_mass=payload_mass,
    )
    return trajectory


def plan_drop_approach(current_q: np.ndarray):
    """Plan the loaded move from the current pose to the active drop point."""
    drop_approach_q, _ = solve_object_joints(
        active_object_id,
        active_drop_position,
        current_q,
        OBJECT_APPROACH_OFFSET,
        "drop approach",
    )
    return plan_payload_motion(
        current_q, drop_approach_q, active_payload_mass
    )


def plan_empty_hold(current_q: np.ndarray):
    """Calculate no-payload gravity compensation after releasing a box."""
    return quintic_trajectory(
        start=current_q,
        goal=current_q,
        duration=sample_period,
        sample_period=sample_period,
    )


def log_simulation_torque(time: float, torque: np.ndarray) -> None:
    """Append the torque command applied during one Webots step."""
    if not RUN_SEQUENTIAL_PAYLOAD_BENCHMARK or benchmark_run < 0:
        return
    with SIMULATION_TORQUE_LOG.open("a", encoding="utf-8") as log_file:
        values = ",".join(f"{value:.12g}" for value in np.abs(torque))
        phase = active_motion if active_motion is not None else "startup"
        log_file.write(
            f"{time:.12g},{benchmark_run},{phase},"
            f"{active_payload_mass:.12g},{values}\n"
        )


def receive_camera_objects(
    detected_objects_robot: dict[int, np.ndarray],
) -> None:
    """Read camera packets already expressed in the robot base frame."""
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

        object_id, robot_x, robot_y, robot_z = struct.unpack("i3f", packet)
        is_new_object = object_id not in detected_objects_robot
        detected_objects_robot[object_id] = np.array(
            [robot_x, robot_y, robot_z], dtype=float
        )
        receiver.nextPacket()

        if is_new_object:
            print(
                f"Camera object {object_id}: robot xyz="
                f"{detected_objects_robot[object_id].round(4)}"
            )


# Controller state.
detected_objects_robot: dict[int, np.ndarray] = {}
visited_object_positions: dict[int, np.ndarray] = {}
completed_object_ids: set[int] = set()
object_home_positions: dict[int, np.ndarray] = {}
object_payload_masses: dict[int, float] = {}
object_at_drop: dict[int, bool] = {}
motion_state = "initializing"
active_trajectory = None
active_object_id = None
active_object_position = None
active_motion = None
active_approach_q = None
active_drop_position = None
active_payload_mass = 0.0
active_returning_home = False
trajectory_sample_index = 0
target_deadline = 0.0
grasp_deadline = 0.0
release_deadline = 0.0
picked_object_id = None
next_drop_index = 0
benchmark_payload_index = 0
benchmark_run = -1
hold_q = HOME_Q.copy()
hold_torque = np.zeros(6)
previous_q = None
filtered_qdot = np.zeros(6)

if RUN_SEQUENTIAL_PAYLOAD_BENCHMARK:
    SIMULATION_TORQUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    SIMULATION_TORQUE_LOG.write_text(
        "time,run,phase,payload_mass,tau1_cmd,tau2_cmd,tau3_cmd,"
        "tau4_cmd,tau5_cmd,tau6_cmd\n",
        encoding="utf-8",
    )

validate_joint_positions(HOME_Q)
configure_scene_mode()
print(f"Torque limits (N m): {torque_limits.round(2)}")

while robot.step(timestep) != -1:
    receive_camera_objects(detected_objects_robot)
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

    # Replan if the target moves.
    if (
        active_motion == "approach"
        and active_object_id is not None
        and active_object_position is not None
        and active_object_id in detected_objects_robot
        and np.linalg.norm(
            detected_objects_robot[active_object_id] - active_object_position
        ) > OBJECT_REPLAN_DISTANCE
    ):
        latest_object_position = detected_objects_robot[
            active_object_id
        ].copy()
        try:
            updated_q, _ = solve_object_joints(
                active_object_id,
                latest_object_position,
                current_q,
                OBJECT_APPROACH_OFFSET,
                "updated approach",
            )
        except (RuntimeError, ValueError) as error:
            print(
                f"Keeping current approach for object {active_object_id}: "
                f"{error}"
            )
        else:
            active_trajectory = quintic_trajectory(
                start=current_q,
                goal=updated_q,
                duration=TRAJECTORY_DURATION,
                sample_period=sample_period,
            )
            active_object_position = latest_object_position
            active_approach_q = updated_q
            trajectory_sample_index = 0
            motion_state = "executing"
            print(f"Replanned approach for moved object {active_object_id}.")
            continue

    if motion_state == "initializing":
        # Sensors are valid after one step.
        home_trajectory = quintic_trajectory(
            start=current_q,
            goal=HOME_Q,
            duration=TRAJECTORY_DURATION,
            sample_period=sample_period,
        )
        active_trajectory = home_trajectory
        active_motion = "home"
        motion_state = "executing"
        print("Moving to home position with torque control.")

    elif motion_state == "executing":
        trajectory = active_trajectory
        if trajectory is None:
            raise RuntimeError("No active trajectory is available to execute.")

        # Advance one trajectory sample per step.
        command_torques(
            desired_q=trajectory.position[trajectory_sample_index],
            desired_qdot=trajectory.velocity[trajectory_sample_index],
            feedforward_torque=trajectory.torque[trajectory_sample_index],
            measured_q=current_q,
            measured_qdot=current_qdot,
        )

        # Lift as soon as the vacuum attaches.
        if active_motion == "descend" and gripper.getPresence():
            if active_approach_q is None:
                raise RuntimeError("No approach pose is available for lifting.")
            active_trajectory = plan_payload_motion(
                current_q, active_approach_q, active_payload_mass
            )
            active_motion = "lift"
            trajectory_sample_index = 0
            motion_state = "executing"
            print(f"Vacuum attached to object {active_object_id}; lifting.")
            continue
        trajectory_sample_index += 1

        if trajectory_sample_index >= trajectory.time.size:
            motion_state = "settling"
            timeout = (
                LIFT_TARGET_TIMEOUT
                if active_motion in PAYLOAD_MOTION_STATES
                else TARGET_TIMEOUT
            )
            target_deadline = robot.getTime() + timeout

    elif motion_state == "settling":
        trajectory = active_trajectory
        if trajectory is None:
            raise RuntimeError("No active trajectory is available to settle.")
        final_q = trajectory.position[-1]

        # Hold the final command while settling.
        command_torques(
            desired_q=final_q,
            desired_qdot=np.zeros(6),
            feedforward_torque=trajectory.torque[-1],
            measured_q=current_q,
            measured_qdot=current_qdot,
        )

        if active_motion == "descend" and gripper.getPresence():
            if active_approach_q is None:
                raise RuntimeError("No approach pose is available for lifting.")
            active_trajectory = plan_payload_motion(
                current_q, active_approach_q, active_payload_mass
            )
            active_motion = "lift"
            trajectory_sample_index = 0
            motion_state = "executing"
            print(f"Vacuum attached to object {active_object_id}; lifting.")
            continue

        joint_tolerance = (
            LIFT_JOINT_TOLERANCE
            if active_motion in PAYLOAD_MOTION_STATES
            else JOINT_TOLERANCE
        )
        if np.max(np.abs(current_q - final_q)) <= joint_tolerance:
            hold_q = final_q.copy()
            hold_torque = trajectory.torque[-1].copy()

            if active_motion == "home":
                print("Home position reached.")
                active_trajectory = None
                active_motion = None
                motion_state = "waiting_for_object"

            elif active_motion == "benchmark_reset":
                print("Robot reset; waiting for the next benchmark detection.")
                active_trajectory = None
                active_motion = None
                motion_state = "waiting_for_object"

            elif active_motion == "approach":
                # Enable suction before descending.
                gripper.turnOn()
                try:
                    grasp_q, _ = solve_object_joints(
                        active_object_id,
                        active_object_position,
                        current_q,
                        OBJECT_GRASP_OFFSET,
                        "grasp",
                    )
                except (RuntimeError, ValueError) as error:
                    gripper.turnOff()
                    visited_object_positions[active_object_id] = (
                        active_object_position.copy()
                    )
                    print(
                        f"Skipping camera object {active_object_id}: {error}"
                    )
                    active_trajectory = None
                    active_object_id = None
                    active_object_position = None
                    active_motion = None
                    motion_state = "waiting_for_object"
                    continue

                active_trajectory = quintic_trajectory(
                    start=current_q,
                    goal=grasp_q,
                    duration=TRAJECTORY_DURATION,
                    sample_period=sample_period,
                )
                active_motion = "descend"
                trajectory_sample_index = 0
                motion_state = "executing"
                print(f"Descending to camera object {active_object_id}.")

            elif active_motion == "descend":
                active_trajectory = None
                motion_state = "verifying_grasp"
                grasp_deadline = robot.getTime() + GRASP_TIMEOUT
                print(f"Waiting for vacuum grasp of object {active_object_id}.")

            elif active_motion == "lift":
                visited_object_positions[active_object_id] = (
                    active_object_position.copy()
                )
                picked_object_id = active_object_id
                print(f"Picked up camera object {picked_object_id}.")
                if RUN_SEQUENTIAL_PAYLOAD_BENCHMARK:
                    active_drop_position = BENCHMARK_DROP.copy()
                else:
                    active_returning_home = object_at_drop.get(
                        active_object_id, False
                    )
                    if active_returning_home:
                        active_drop_position = object_home_positions[
                            active_object_id
                        ].copy()
                    else:
                        active_drop_position = NORMAL_DROP_CENTERS[
                            next_drop_index % len(NORMAL_DROP_CENTERS)
                        ].copy()
                try:
                    active_trajectory = plan_drop_approach(current_q)
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"Cannot plan drop for object {active_object_id}: "
                        f"{error}"
                    ) from error
                active_motion = "carry_to_drop"
                trajectory_sample_index = 0
                motion_state = "executing"
                print(
                    f"Carrying {active_payload_mass:.1f} kg payload to "
                    f"{active_drop_position.round(3)}."
                )

            elif active_motion == "carry_to_drop":
                try:
                    place_q, _ = solve_object_joints(
                        active_object_id,
                        active_drop_position,
                        current_q,
                        OBJECT_GRASP_OFFSET,
                        "place",
                    )
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"Cannot plan placement for object {active_object_id}: "
                        f"{error}"
                    ) from error

                active_trajectory = plan_payload_motion(
                    current_q,
                    place_q,
                    active_payload_mass,
                    duration=PLACE_DURATION,
                )
                active_motion = "place_descend"
                trajectory_sample_index = 0
                motion_state = "executing"
                print(f"Lowering object {active_object_id} for placement.")

            elif active_motion == "place_descend":
                # Release at the table.
                gripper.turnOff()
                empty_hold = plan_empty_hold(current_q)
                hold_q = current_q.copy()
                hold_torque = empty_hold.torque[-1].copy()
                active_trajectory = None
                active_motion = "release"
                release_deadline = robot.getTime() + RELEASE_SETTLE_TIME
                motion_state = "releasing"
                print(f"Released object {active_object_id} at its drop point.")

            else:
                raise RuntimeError(f"Unknown motion phase: {active_motion}")
        elif robot.getTime() >= target_deadline:
            if active_motion == "place_descend":
                # Table contact may stop the descent early.
                gripper.turnOff()
                empty_hold = plan_empty_hold(current_q)
                hold_q = current_q.copy()
                hold_torque = empty_hold.torque[-1].copy()
                active_trajectory = None
                active_motion = "release"
                release_deadline = robot.getTime() + RELEASE_SETTLE_TIME
                motion_state = "releasing"
                print(
                    f"Object {active_object_id} reached the table; released."
                )
            elif active_motion == "descend":
                # Check attachment after a blocked descent.
                hold_q = current_q.copy()
                active_trajectory = None
                motion_state = "verifying_grasp"
                grasp_deadline = robot.getTime() + GRASP_TIMEOUT
                print(
                    "Grasp descent was blocked before its exact endpoint; "
                    "checking vacuum attachment."
                )
            else:
                position_error = np.rad2deg(np.max(np.abs(current_q - final_q)))
                raise RuntimeError(
                    f"{active_motion} did not settle within "
                    f"{LIFT_TARGET_TIMEOUT if active_motion in PAYLOAD_MOTION_STATES else TARGET_TIMEOUT:.1f} s "
                    f"(max joint error {position_error:.2f} deg)."
                )

    elif motion_state == "verifying_grasp":
        command_torques(
            desired_q=hold_q,
            desired_qdot=np.zeros(6),
            feedforward_torque=hold_torque,
            measured_q=current_q,
            measured_qdot=current_qdot,
        )

        if gripper.getPresence():
            if active_approach_q is None:
                raise RuntimeError("No approach pose is available for lifting.")
            active_trajectory = plan_payload_motion(
                current_q, active_approach_q, active_payload_mass
            )
            active_motion = "lift"
            trajectory_sample_index = 0
            motion_state = "executing"
            print(f"Vacuum attached to object {active_object_id}; lifting.")
        elif robot.getTime() >= grasp_deadline:
            gripper.turnOff()
            visited_object_positions[active_object_id] = (
                active_object_position.copy()
            )
            print(f"Vacuum failed to attach to object {active_object_id}.")
            active_object_id = None
            active_object_position = None
            active_motion = None
            motion_state = "waiting_for_object"

    elif motion_state == "releasing":
        command_torques(
            desired_q=hold_q,
            desired_qdot=np.zeros(6),
            feedforward_torque=hold_torque,
            measured_q=current_q,
            measured_qdot=current_qdot,
        )
        if robot.getTime() >= release_deadline:
            if not RUN_SEQUENTIAL_PAYLOAD_BENCHMARK:
                completed_object_ids.add(active_object_id)
                object_at_drop[active_object_id] = not active_returning_home
                if not active_returning_home:
                    next_drop_index += 1
                print(f"Object {active_object_id} placed. Looking for next object.")
                active_object_id = None
                active_object_position = None
                active_drop_position = None
                active_approach_q = None
                active_motion = None
                active_payload_mass = 0.0
                active_returning_home = False
                picked_object_id = None
                if (
                    len(object_home_positions) >= len(NORMAL_DROP_CENTERS)
                    and len(completed_object_ids) >= len(object_home_positions)
                ):
                    completed_object_ids.clear()
                    next_drop_index = 0
                    print("Starting the next pick-and-place cycle.")
                motion_state = "waiting_for_object"
                continue

            print(f"Completed {active_payload_mass:.1f} kg benchmark run.")
            benchmark_payload_index += 1
            benchmark_run = -1
            active_object_id = None
            active_object_position = None
            active_drop_position = None
            active_approach_q = None
            active_motion = None
            active_payload_mass = 0.0
            picked_object_id = None
            if benchmark_payload_index >= len(BENCHMARK_PAYLOADS):
                print("All 0, 0.5, and 1.0 kg benchmark runs are complete.")
                motion_state = "waiting_for_object"
                continue

            reset_benchmark_block(BENCHMARK_PAYLOADS[benchmark_payload_index])
            detected_objects_robot.clear()
            visited_object_positions.clear()
            completed_object_ids.clear()
            active_trajectory = quintic_trajectory(
                start=current_q,
                goal=HOME_Q,
                duration=TRAJECTORY_DURATION,
                sample_period=sample_period,
            )
            active_motion = "benchmark_reset"
            trajectory_sample_index = 0
            motion_state = "executing"

    elif motion_state == "waiting_for_object":
        if (
            RUN_SEQUENTIAL_PAYLOAD_BENCHMARK
            and benchmark_payload_index >= len(BENCHMARK_PAYLOADS)
        ):
            command_torques(
                desired_q=hold_q,
                desired_qdot=np.zeros(6),
                feedforward_torque=hold_torque,
                measured_q=current_q,
                measured_qdot=current_qdot,
            )
            continue

        object_to_visit = next_unvisited_object(
            detected_objects_robot,
            visited_object_positions,
            completed_object_ids,
        )
        if object_to_visit is None:
            # Hold until a target appears.
            command_torques(
                desired_q=hold_q,
                desired_qdot=np.zeros(6),
                feedforward_torque=hold_torque,
                measured_q=current_q,
                measured_qdot=current_qdot,
            )
            continue

        object_id, object_position = object_to_visit
        try:
            target_q, _ = solve_object_joints(
                object_id,
                object_position,
                current_q,
                OBJECT_APPROACH_OFFSET,
                "approach",
            )
        except (RuntimeError, ValueError) as error:
            # Wait for a new camera reading.
            visited_object_positions[object_id] = object_position.copy()
            print(f"Skipping camera object {object_id}: {error}")
            continue

        active_trajectory = quintic_trajectory(
            start=current_q,
            goal=target_q,
            duration=TRAJECTORY_DURATION,
            sample_period=sample_period,
        )
        active_object_id = object_id
        active_object_position = object_position
        object_home_positions.setdefault(object_id, object_position.copy())
        if RUN_SEQUENTIAL_PAYLOAD_BENCHMARK:
            active_payload_mass = BENCHMARK_PAYLOADS[benchmark_payload_index]
            benchmark_run = benchmark_payload_index
        else:
            active_payload_mass = object_payload_masses.setdefault(
                object_id, normal_payload_mass(object_position)
            )
            benchmark_run = -1
        active_approach_q = target_q
        active_motion = "approach"
        trajectory_sample_index = 0
        motion_state = "executing"
        print(
            f"Object {object_id} payload model: "
            f"{active_payload_mass:.1f} kg."
        )
