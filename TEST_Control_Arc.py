import airsim
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

client = airsim.CarClient()
client.confirmConnection()
# Могу ли я управлять машиной из кода?
start_data = pd.read_csv("start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0] / 100, -1)
heading = airsim.utils.to_quaternion(0, 0, 0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)
time.sleep(2)
theta = np.linspace(0,2*np.pi)
r = 10