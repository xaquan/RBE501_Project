from controller import Robot, Supervisor
import math
from typing import cast
import struct

robot = Robot()
timestep = int(robot.getBasicTimeStep())

receiver = robot.getDevice("serial_receiver")
receiver.setChannel(1)  # matches the Emitter above
receiver.enable(timestep)


joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

motors = []

for name in joint_names:
    motor = robot.getDevice(name)

    if motor is None:
        raise RuntimeError(f'Cannot find motor "{name}"')

    motor.setVelocity(0.5)
    motors.append(motor)


# Initial pose in radians
initial_position = [
    0.0,
    math.pi / 2,
    -math.pi / 2,
    math.pi / 2,
    math.pi / 2,
    0.0,
]

for motor, position in zip(motors, initial_position):
    motor.setPosition(position)


start_time = robot.getTime()

print("Robot controller loaded.")

detected_object = {}

while robot.step(timestep) != -1:
    elapsed_time = robot.getTime() - start_time

    # Wait two seconds before starting the motion
    if elapsed_time < 2.0:
        continue

    # Slowly move the shoulder and elbow
    shoulder_position = -math.pi / 2 + 0.35 * math.sin(elapsed_time)
    elbow_position = math.pi / 2 + 0.45 * math.sin(elapsed_time)

    motors[1].setPosition(shoulder_position)
    motors[2].setPosition(elbow_position)
    
    if receiver.getQueueLength() > 0:
        detected_object = {}
        
    while receiver.getQueueLength() > 0:
        data = receiver.getBytes()
        obj_id, x, y, z = struct.unpack('i3f', data)
        receiver.nextPacket()

        # print(f"Received object {obj_id} at ({x:.3f}, {y:.3f}, {z:.3f})")
        detected_object[obj_id] = [x, y, z]
        # target_pose = [x, y, z]
        # joint_angles = your_ik_solver(target_pose)
        # move arm to joint_angles   