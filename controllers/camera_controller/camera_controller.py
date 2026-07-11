from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("top_camera")
camera.enable(timestep)

print("Camera Loading...")

while robot.step(timestep) != -1:
    image = camera.getImage()

    if image is not None:
        pass