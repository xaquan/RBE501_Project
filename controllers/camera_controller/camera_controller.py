from controller import Robot
import math
import struct

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("overhead_camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

emitter = robot.getDevice("serial_emitter")
POSITION_CHANGE_TOLERANCE = 0.001  # metres

# Fixed poses from UR5.wbt. Webots Camera recognition positions use the
# camera's local [forward, left, up] axes. The camera is a child of a rig,
# so both rotations and the child translation must be included.
CAMERA_RIG_WORLD_POSITION = (0.0, 0.0, 1.8)
CAMERA_RIG_Y_ROTATION = 1.5708
CAMERA_LOCAL_POSITION = (0.0, -1.24906e-6, 0.17)
CAMERA_LOCAL_X_ROTATION = -3.1415853071795863

# The UR5e base has zero translation and rotation in UR5.wbt, so its base
# coordinates currently equal world coordinates.
ROBOT_BASE_WORLD_POSITION = (0.0, 0.0, 0.0)

cos_y = math.cos(CAMERA_RIG_Y_ROTATION)
sin_y = math.sin(CAMERA_RIG_Y_ROTATION)
cos_x = math.cos(CAMERA_LOCAL_X_ROTATION)
sin_x = math.sin(CAMERA_LOCAL_X_ROTATION)

# R_camera_to_world = Ry(rig) @ Rx(camera).
camera_to_world = (
    (cos_y, sin_y * sin_x, sin_y * cos_x),
    (0.0, cos_x, -sin_x),
    (-sin_y, cos_y * sin_x, cos_y * cos_x),
)

local_x, local_y, local_z = CAMERA_LOCAL_POSITION
camera_world_position = (
    CAMERA_RIG_WORLD_POSITION[0] + cos_y * local_x + sin_y * local_z,
    CAMERA_RIG_WORLD_POSITION[1] + local_y,
    CAMERA_RIG_WORLD_POSITION[2] - sin_y * local_x + cos_y * local_z,
)

# Last transmitted robot-base position for each recognized object.
last_sent_positions = {}

print("Camera Loaded...")

while robot.step(timestep) != -1:
    objects = camera.getRecognitionObjects()
    visible_object_ids = set()

    for obj in objects:
        obj_id = obj.getId()
        visible_object_ids.add(obj_id)

        # Recognition position is relative to the camera, not the robot.
        cam_x, cam_y, cam_z = obj.getPosition()

        world_x = (
            camera_world_position[0]
            + camera_to_world[0][0] * cam_x
            + camera_to_world[0][1] * cam_y
            + camera_to_world[0][2] * cam_z
        )
        world_y = (
            camera_world_position[1]
            + camera_to_world[1][0] * cam_x
            + camera_to_world[1][1] * cam_y
            + camera_to_world[1][2] * cam_z
        )
        world_z = (
            camera_world_position[2]
            + camera_to_world[2][0] * cam_x
            + camera_to_world[2][1] * cam_y
            + camera_to_world[2][2] * cam_z
        )

        robot_x = world_x - ROBOT_BASE_WORLD_POSITION[0]
        robot_y = world_y - ROBOT_BASE_WORLD_POSITION[1]
        robot_z = world_z - ROBOT_BASE_WORLD_POSITION[2]
        robot_position = (robot_x, robot_y, robot_z)

        previous_position = last_sent_positions.get(obj_id)
        position_changed = (
            previous_position is None
            or math.dist(robot_position, previous_position)
            > POSITION_CHANGE_TOLERANCE
        )

        if not position_changed:
            continue

        message = struct.pack('i3f', obj_id, robot_x, robot_y, robot_z)
        emitter.send(message)
        last_sent_positions[obj_id] = robot_position
        print(
            f"Object {obj_id} changed: "
            f"robot xyz=({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})"
        )

    # If an object leaves the image, forget its old position. It will be sent
    # as a new detection if it becomes visible again.
    disappeared_ids = set(last_sent_positions) - visible_object_ids
    for obj_id in disappeared_ids:
        del last_sent_positions[obj_id]
