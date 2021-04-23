import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import random
import airsim
import pandas as pd


def to_xyz(pos):
    return np.array([pos.x_val, pos.y_val, pos.z_val])


def saveWorld(client):
    cones_left = client.simListSceneObjects('FloatingActor($|[^a-zA-Z]+)')
    cones_right = client.simListSceneObjects('FloatingActorYellow($|[^a-zA-Z]+)')
    print(f"Всего конусов: \n левых: {len(cones_left)}, правых: {len(cones_right)} ")
    f = open("world.dat", 'a')
    for i in range(len(cones_left)):
        pos = client.simGetObjectPose(cones_left[i]).position
        xyz = to_xyz(pos)
        f.write(f"{i} {xyz[0]} {xyz[1]} left\n")
    for i in range(len(cones_right)) :
        pos = client.simGetObjectPose(cones_right[i]).position
        xyz = to_xyz(pos)
        f.write(f"{i} {xyz[0]} {xyz[1]} right\n")
    f.close()


client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())

start_data = pd.read_csv("start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0] / 100, 0)
heading = airsim.utils.to_quaternion(0, 0, 0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []  # store trials here (n)
ys = []  # store relative frequency here
rs = []  # for theoretical probability

saveWorld(client)
# client.simGetVehiclePose().orientation.
# This function is called periodically from FuncAnimation
def animate(i, xs, ys, client):
    # Aquire and parse data from serial port
    vehiclePose = client.simGetVehiclePose()
    position = vehiclePose.position
    orientation = airsim.to_eularian_angles(vehiclePose.orientation)
    print(orientation[0], orientation[1], orientation[2])
    # Add x and y to lists
    xs.append(position.x_val)
    ys.append(position.y_val)
    # rs.append(0.5)

    # Limit x and y lists to 20 items
    # xs = xs[-20:]
    # ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys, label="Experimental Probability")
    # ax.quiver
    yaw = orientation[2]
    new_x = np.cos(yaw)
    new_y = np.sin(yaw)
    ax.quiver(position.x_val, position.y_val, new_x, new_y)
    # ax.plot(xs, rs, label="Theoretical Probability")
    ax.set_ylim([-200, 200])
    ax.set_xlim([-200, 200])
    # ax.set_ylim(np.min(ys) - np.std(ys),np.max(ys) + np.std(ys))
    # ax.set_xlim(np.min(xs) - np.std(xs), np.max(xs) + np.std(xs))
    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('This is how I roll...')
    plt.ylabel('Relative frequency')
    plt.legend()
    # plt.axis([1, None, 0, 1.1])  # Use for arbitrary number of trials
    # plt.axis([1, 100, 0, 1.1]) #Use for 100 trial demo

# Set up plot to call animate() function periodically
# ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys,client), interval=10)
# plt.show()
