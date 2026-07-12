from controller import Robot
import struct

robot = Robot()
timestep = int(robot.getBasicTimeStep())
# supervisor = Supervisor()

camera = robot.getDevice("overhead_camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

camPos = [0, 0, 1.8]

emitter = robot.getDevice("serial_emitter")
CAMERA_HEIGHT = 1.8

print("Camera Loaded...")

while robot.step(timestep) != -1:
    image = camera.getImage()
    objects = camera.getRecognitionObjects()
    
    for obj in objects:
        obj_id = obj.getId()
        # position is relative to the CAMERA's coordinate frame, not world!
        cam_x, cam_y, cam_z = obj.getPosition()

        # Camera points straight down (rotated 180° about X), mounted at
        # world height CAMERA_HEIGHT above the origin plane. Convert:
        world_x = cam_x
        world_y = -cam_y   # sign flips depending on your camera's rotation axis
        world_z = -cam_z + CAMERA_HEIGHT

        message = struct.pack('i3f', obj_id, world_x, world_y, world_z)
        emitter.send(message)
        # print(f"Object {obj_id} at world pos: ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")