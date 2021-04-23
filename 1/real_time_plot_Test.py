import matplotlib.pyplot as plt
import numpy as np
import airsim
import pandas as pd
import time

plt.style.use('ggplot')


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    line1.set_xdata(x_vec)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    if np.min(x_vec) <= line1.axes.get_xlim()[0] or np.max(x_vec) >= line1.axes.get_xlim()[1]:
        plt.xlim([np.min(x_vec) - np.std(x_vec), np.max(x_vec) + np.std(x_vec)])

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())

#x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)



start_data = pd.read_csv("../start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0]/100, 0)
heading = airsim.utils.to_quaternion(0,0,0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)

X = [position.x_val]
Y = [position.y_val]
size = 100
x_vec = np.array(X)
y_vec = np.array(Y)
line1 = []

plt.ion()
hl, = plt.plot([position.x_val], [position.y_val])
#### Поробоавать снова!!!!!!!!!!!!!!!!!!
def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data))
    hl.set_ydata(np.append(hl.get_ydata(), new_data))
    plt.draw()
    plt.pause(0.02)

while True:
    update_line(hl,[[position.x_val],[position.y_val]])

# plt.ion()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#line1, = ax.plot(X, Y, 'r-')
#plt.draw()
#while True:
#    position = client.simGetVehiclePose().position
#    print(f"X={position.x_val}, Y={position.y_val}, Z={position.z_val}")
#    X.append(position.x_val)
#    Y.append(position.y_val)
#    line1.set_ydata(Y)
#    line1.set_xdata(X)
#    # line1.set_xdata(position.x_val)
#    plt.draw()
#    plt.pause(0.02)
#plt.ioff()
#plt.show()
# plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y, 'r-')
# plt.draw()

#for phase in np.linspace(0, 10*np.pi, 500):
#    line1.set_ydata(np.sin(x + phase))
#    plt.draw()
#    plt.pause(0.02)

# plt.ioff()
# plt.show()