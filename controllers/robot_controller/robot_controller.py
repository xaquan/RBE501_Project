"""Move the UR5e to objects reported by the overhead camera."""

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
from libraries.ManipulatorKinematics import create_webots_ur5e_kinematics


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

# Camera positions refer to object centres.
OBJECT_APPROACH_OFFSET = np.array([0.0, 0.0, 0.20], dtype=float)
# Tool height for cup contact with the 0.1 m boxes.
OBJECT_GRASP_OFFSET = np.array([0.0, 0.0, 0.070], dtype=float)
OBJECT_REVISIT_DISTANCE = 0.01
OBJECT_REPLAN_DISTANCE = 0.02
GRASP_TIMEOUT = 1.0
LIFT_TARGET_TIMEOUT = 6.0
LIFT_JOINT_TOLERANCE = np.deg2rad(1.5)
# Initial box centres and masses; recognition IDs vary between runs.
KNOWN_OBJECTS = (
    (np.array([0.500, -0.4995, 0.0496]), 1.0),
    (np.array([0.730, 0.4600, 0.0498]), 0.5),
)
# Tabletop drop locations.
DROP_OBJECT_CENTERS = np.array(
    [[0.366269, 0.555773, 0.05], [0.470301, -0.280796, 0.05]],
    dtype=float,
)
PLACE_DURATION = 4.0
RELEASE_SETTLE_TIME = 0.5
PAYLOAD_MOTION_STATES = {"lift", "carry_to_drop", "place_descend"}
IK_MAX_ITERATIONS = 400
IK_STEP_SIZE = 0.30
IK_DAMPING = 1e-3

# Tool orientation that keeps the cup facing down.
DOWNWARD_TOOL_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=float,
)

TRAJECTORY_DURATION = 3.5
TARGET_TIMEOUT = 3.0
JOINT_TOLERANCE = np.deg2rad(0.5)
IK_TOLERANCE = 1e-5
VELOCITY_FILTER = 0.25

# Fixed positive torque-feedback gains. LagrangeDynamicReal supplies the
# inverse-dynamics feed-forward torque; these gains correct tracking error.
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

gripper = robot.getDevice("vacuum_gripper")
if gripper is None:
    raise RuntimeError('Cannot find VacuumGripper "vacuum_gripper".')
gripper.enablePresence(timestep)
gripper.turnOff()

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
        # Keep each IK update bounded.
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


def payload_mass_for_camera_object(object_position: np.ndarray) -> float:
    """Return the configured mass of the nearest original scene object."""
    position = np.asarray(object_position, dtype=float)
    reference_position, mass = min(
        KNOWN_OBJECTS,
        key=lambda item: np.linalg.norm(position - item[0]),
    )
    distance = np.linalg.norm(position - reference_position)
    if distance > 0.20:
        print(
            f"No mass reference within 0.20 m of {position.round(3)}; "
            "using 0.5 kg."
        )
        return 0.5
    return mass


def plan_payload_motion(
    current_q: np.ndarray,
    goal_q: np.ndarray,
    payload_mass: float,
    duration: float = TRAJECTORY_DURATION,
):
    """Create a trajectory that includes the attached object's dynamics."""
    return quintic_trajectory(
        start=current_q,
        goal=goal_q,
        duration=duration,
        sample_period=sample_period,
        carried_payload_mass=payload_mass,
    )


def plan_empty_hold(current_q: np.ndarray):
    """Calculate no-payload gravity compensation after releasing a box."""
    return quintic_trajectory(
        start=current_q,
        goal=current_q,
        duration=sample_period,
        sample_period=sample_period,
    )


def receive_camera_objects(
    detected_objects_robot: dict[int, np.ndarray],
) -> None:
    """Decode packets and retain each position in the robot base frame.

    The camera controller performs the camera-to-robot conversion before
    packing ``(object_id, robot_x, robot_y, robot_z)``. Therefore, these
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
# Per-object state for the shuttle cycle.
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
hold_q = HOME_Q.copy()
hold_torque = np.zeros(6)
previous_q = None
filtered_qdot = np.zeros(6)

validate_joint_positions(HOME_Q)
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

    # Refresh a target that moves during the approach.
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
        # Sensors are valid after the first step.
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

        # One trajectory sample per simulation step.
        command_torques(
            desired_q=trajectory.position[trajectory_sample_index],
            desired_qdot=trajectory.velocity[trajectory_sample_index],
            feedforward_torque=trajectory.torque[trajectory_sample_index],
            measured_q=current_q,
            measured_qdot=current_qdot,
        )

        # Lift as soon as the vacuum link is created.
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

        # Hold the final sample while the joints settle.
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

            elif active_motion == "approach":
                # Enable suction before the contact move.
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
                active_returning_home = bool(
                    object_home_positions.get(active_object_id) is not None
                    and active_object_id in object_home_positions
                    and object_at_drop.get(active_object_id, False)
                )
                if active_returning_home:
                    active_drop_position = object_home_positions[
                        active_object_id
                    ].copy()
                else:
                    active_drop_position = DROP_OBJECT_CENTERS[
                        next_drop_index % len(DROP_OBJECT_CENTERS)
                    ].copy()
                try:
                    drop_approach_q, _ = solve_object_joints(
                        active_object_id,
                        active_drop_position,
                        current_q,
                        OBJECT_APPROACH_OFFSET,
                        "drop approach",
                    )
                except (RuntimeError, ValueError) as error:
                    raise RuntimeError(
                        f"Cannot plan drop for object {active_object_id}: {error}"
                    ) from error

                active_trajectory = plan_payload_motion(
                    current_q, drop_approach_q, active_payload_mass
                )
                active_motion = "carry_to_drop"
                trajectory_sample_index = 0
                motion_state = "executing"
                print(
                    f"Carrying object {active_object_id} "
                    f"{'home' if active_returning_home else 'to drop'} "
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
                # Release once the box reaches the table.
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
                # Table contact can stop the final part of the descent.
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
                # Check for a vacuum link after a blocked grasp descent.
                hold_q = current_q.copy()
                active_trajectory = None
                motion_state = "verifying_grasp"
                grasp_deadline = robot.getTime() + GRASP_TIMEOUT
                print(
                    "Grasp descent was blocked before its exact endpoint; "
                    "checking vacuum attachment."
                )
            else:
                raise RuntimeError("A trajectory endpoint was not reached in time.")

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
            # Start the return leg after both boxes have moved.
            if (
                len(object_home_positions) >= len(DROP_OBJECT_CENTERS)
                and len(completed_object_ids) >= len(object_home_positions)
            ):
                completed_object_ids.clear()
                next_drop_index = 0
                print("Starting the next pick-and-place cycle.")
            motion_state = "waiting_for_object"

    elif motion_state == "waiting_for_object":
        object_to_visit = next_unvisited_object(
            detected_objects_robot,
            visited_object_positions,
            completed_object_ids,
        )
        if object_to_visit is None:
            # Hold position until a target is available.
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
            # Skip this reading until the camera reports a new position.
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
        object_payload_masses.setdefault(
            object_id, payload_mass_for_camera_object(object_position)
        )
        active_payload_mass = object_payload_masses[object_id]
        active_approach_q = target_q
        active_motion = "approach"
        trajectory_sample_index = 0
        motion_state = "executing"
        print(
            f"Object {object_id} payload model: "
            f"{active_payload_mass:.1f} kg."
        )
