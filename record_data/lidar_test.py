import airsim
from record_data import lidar_car_data as lidar
import pandas as pd
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
start_data = pd.read_csv("../start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0]/100, 0)
heading = airsim.utils.to_quaternion(0,0,0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)



lidarTest = lidar.LidarTest(client)
client.simGetLidarSegmentation()
lidarTest.start_recording()
while True:
    lidarTest.record()