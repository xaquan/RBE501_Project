from controller import Robot
import math


robot = Robot()
timestep = int(robot.getBasicTimeStep())

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
    -math.pi / 2,
    math.pi / 2,
    -math.pi / 2,
    -math.pi / 2,
    0.0,
]

for motor, position in zip(motors, initial_position):
    motor.setPosition(position)


start_time = robot.getTime()

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