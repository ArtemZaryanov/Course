from PIL import Image
import airsim
from record_data import camera_car_data as camera
import pandas as pd
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
start_data = pd.read_csv("../start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0]/100, 0)
heading = airsim.utils.to_quaternion(0,0,0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)
cameraTest = camera.CameraRecord(client)
client.simGetLidarSegmentation()
cameraTest.start_recording()
while True:
    cameraTest.record()