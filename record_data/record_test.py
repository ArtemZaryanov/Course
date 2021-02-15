import airsim
import numpy as np
from record_data import lidar_car_data as lidar
from record_data import camera_car_data as camera
import pandas as pd
import time

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
start_data = pd.read_csv("../start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0] / 100, 0)
heading = airsim.utils.to_quaternion(0, 0, 0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)

# Camera

cameraTest = camera.CameraRecord(client)

# Lidar

lidarTest = lidar.LidarTest(client)

# record
is_start_record = True
if is_start_record:
    cameraTest.start_recording()
    lidarTest.start_recording()

while True:

    # print(np.degrees(airsim.to_eularian_angles(client.getImuData().orientation)))
    time.sleep(0.5)
    if is_start_record:
        cameraTest.record()
        # time.sleep(0.025)
        lidarTest.record()
        # Необходимо!!! Иначе ошибки при записи
