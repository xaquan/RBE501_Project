from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

gripper = robot.getDevice("vacuum gripper")
gripper.enablePresence(timestep)
gripper.turnOn()

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

motors = []
sensors = []
for name in joint_names:
    motor = robot.getDevice(name)
    if motor is None:
        raise RuntimeError(f'Cannot find motor "{name}"')
    motor.setVelocity(0.5)
    motors.append(motor)

    sensor = motor.getPositionSensor()
    if sensor is None:
        raise RuntimeError(f'Motor "{name}" has no position sensor')
    sensor.enable(timestep)
    sensors.append(sensor)

# Read waypoints from file 
# Each line: pos1 pos2 pos3 pos4 pos5 pos6  torque1 torque2 torque3 torque4 torque5 torque6
waypoints = []
with open("waypoints.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 12:
            raise ValueError(f"Expected 12 values per line, got {len(values)}: {line}")
        positions = values[0:6]
        torques = values[6:12]
        waypoints.append((positions, torques))

TOLERANCE = 0.01  # rad, how close counts as "arrived"

current_wp = 0
target_set = False

while robot.step(timestep) != -1:
    if current_wp >= len(waypoints):
        break  # done with all waypoints

    positions, torques = waypoints[current_wp]

    # Send the target once when we start this waypoint
    if not target_set:
        for motor, pos, tq in zip(motors, positions, torques):
            motor.setAvailableTorque(tq)
            motor.setPosition(pos)
        target_set = True

    # Check if all joints reached their target
    reached = all(
        abs(sensor.getValue() - pos) < TOLERANCE
        for sensor, pos in zip(sensors, positions)
    )

    if reached:
        current_wp += 1
        target_set = False

print("All waypoints reached.")