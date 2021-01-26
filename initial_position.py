import airsim
import pandas as pd
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())
start_data = pd.read_csv("start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0]/100, 0)
heading = airsim.utils.to_quaternion(0,0,0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)
lidar_data = client.getLidarData()
lidar_data.point_cloud
client.enableApiControl(False)