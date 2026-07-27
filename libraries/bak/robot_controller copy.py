from controller import Robot
import math
import struct

from pathlib import Path
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Make the project-level libraries package importable from this controller.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from libraries.manipulatorKinematics import create_webots_ur5e_kinematics

robot = Robot()
timestep = int(robot.getBasicTimeStep())

receiver = robot.getDevice("serial_receiver")
receiver.setChannel(1)  # Matches the emitter in the camera robot.
receiver.enable(timestep)

ee_gps = robot.getDevice("ee_gps")
ee_gps.enable(timestep)

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

motors = []
position_sensors = []
for name in joint_names:
    motor = robot.getDevice(name)
    if motor is None:
        raise RuntimeError(f'Cannot find motor "{name}"')

    # Webots position control: setPosition() supplies the target and
    # setVelocity() limits how fast the joint moves toward that target.
    motor.setVelocity(0.5)  # rad/s
    motors.append(motor)

    sensor = robot.getDevice(f"{name}_sensor")
    sensor.enable(timestep)
    position_sensors.append(sensor)


kinematics, inverse_kinematics = create_webots_ur5e_kinematics()


def move_to_joint_positions(target_degrees):
    """Command all six UR5e joints to the requested angles."""
    if len(target_degrees) != len(motors):
        raise ValueError(
            f"Expected {len(motors)} joint angles, got {len(target_degrees)}"
        )

    for name, motor, target_degree in zip(
        joint_names, motors, target_degrees
    ):
        target_radian = math.radians(target_degree)
        minimum = motor.getMinPosition()
        maximum = motor.getMaxPosition()

        # A [0, 0] range means that Webots treats the joint as continuous.
        has_limits = minimum != 0.0 or maximum != 0.0
        if has_limits and not minimum <= target_radian <= maximum:
            raise ValueError(
                f"{name}: {target_degree} degrees is outside the "
                f"[{math.degrees(minimum):.1f}, "
                f"{math.degrees(maximum):.1f}] degree range"
            )

        motor.setPosition(target_radian)


def report_end_effector_position():
    """Compare POE forward kinematics with the simulated toolSlot position."""
    measured_joint_angles = [sensor.getValue() for sensor in position_sensors]
    kinematics.set_thetas(measured_joint_angles)

    fk_position = np.asarray(
        kinematics.get_current_position(), dtype=float
    ).reshape(3)
    gps_position = np.asarray(ee_gps.getValues(), dtype=float)
    position_error = np.linalg.norm(fk_position - gps_position)

    print(f"Measured joints (deg): {np.rad2deg(measured_joint_angles).round(2)}")
    print(f"EE position from FK (m): {fk_position.round(6)}")
    print(f"EE position from GPS (m): {gps_position.round(6)}")
    print(f"FK/GPS position error (m): {position_error:.6e}")


# Each row is one reference pose. Joint order:
# shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.
REFERENCE_JOINT_POSITIONS_DEG = [
    [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    [45.0, -60.0, 90.0, -120.0, -90.0, 30.0],
    [-45.0, -75.0, 110.0, -125.0, -90.0, -30.0],
    [90.0, -45.0, 75.0, -120.0, -90.0, 45.0],
]
IK_INITIAL_OFFSET_DEG = np.array([4.0, -4.0, 4.0, -4.0, 4.0, -4.0])
IK_POSITION_TOLERANCE_M = 1e-6
MOVE_INTERVAL_SECONDS = 5.0


def solve_and_check_ik(reference_degrees):
    """Create an FK target, solve it with IK, and verify the result."""
    reference_radians = np.deg2rad(reference_degrees)
    kinematics.set_thetas(reference_radians)
    target_position = np.asarray(
        kinematics.get_current_position(), dtype=float
    ).reshape(3)

    initial_guess = reference_radians + np.deg2rad(IK_INITIAL_OFFSET_DEG)
    ik_angles = inverse_kinematics.compute_thetas_for_position(
        desired_position=target_position,
        initial_thetas=initial_guess,
        alpha=0.35,
        max_iterations=150,
        epsilon=IK_POSITION_TOLERANCE_M,
        verbose=False,
    )

    kinematics.set_thetas(ik_angles)
    ik_position = np.asarray(
        kinematics.get_current_position(), dtype=float
    ).reshape(3)
    ik_error = np.linalg.norm(ik_position - target_position)
    if ik_error >= IK_POSITION_TOLERANCE_M:
        raise RuntimeError(f"IK validation failed: error={ik_error:.3e} m")

    ik_degrees = np.rad2deg(ik_angles)
    print(f"Reference angles (deg): {np.round(reference_degrees, 3)}")
    print(f"FK target position (m): {target_position.round(6)}")
    print(f"IK returned angles (deg): {ik_degrees.round(3)}")
    print(f"IK position error (m): {ik_error:.3e}")
    return ik_degrees.tolist()


# Compute and validate every IK solution before starting the motion.
IK_JOINT_POSITIONS_DEG = [
    solve_and_check_ik(reference)
    for reference in REFERENCE_JOINT_POSITIONS_DEG
]

current_pose_index = 0
next_move_time = robot.getTime() + MOVE_INTERVAL_SECONDS
move_to_joint_positions(IK_JOINT_POSITIONS_DEG[current_pose_index])
print(
    f"Moving to IK pose {current_pose_index + 1}: "
    f"{np.round(IK_JOINT_POSITIONS_DEG[current_pose_index], 3)}"
)

detected_objects = {}
while robot.step(timestep) != -1:
    if robot.getTime() >= next_move_time:
        report_end_effector_position()

        current_pose_index = (
            current_pose_index + 1
        ) % len(IK_JOINT_POSITIONS_DEG)
        target = IK_JOINT_POSITIONS_DEG[current_pose_index]
        move_to_joint_positions(target)
        print(
            f"Moving to IK pose {current_pose_index + 1}: "
            f"{np.round(target, 3)}"
        )
        next_move_time += MOVE_INTERVAL_SECONDS

    if receiver.getQueueLength() > 0:
        detected_objects = {}

    while receiver.getQueueLength() > 0:
        data = receiver.getBytes()
        obj_id, x, y, z = struct.unpack('i3f', data)
        receiver.nextPacket()

        # The joint target remains active while camera data is received.
        detected_objects[obj_id] = [x, y, z]
        # print(f"Object {obj_id}: ({x:.3f}, {y:.3f}, {z:.3f})")
        # target_pose = [x, y, z]
        # joint_angles = your_ik_solver(target_pose)
        # move_to_joint_positions(joint_angles)
