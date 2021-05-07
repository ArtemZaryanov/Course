import airsim
import time
import airsim.utils
import numpy as np
from record_data import camera_car_data as camera
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
controls = airsim.CarControls()
controls.handbrake = True
client.setCarControls(controls)
print(client.isApiControlEnabled())
object_name = 'ChessBoard'
if len(client.simListSceneObjects(object_name)) != 0:
    print("object is find ")
else:
    raise IOError("object not find")

print(client.simListSceneObjects('ChessBoard'))
chess_board_pos = client.simGetObjectPose(object_name).position
# Ось напралвена вниз!!!!
print("camera\n", client.simGetCameraInfo("front_center").pose)
print("chess_board_pos_init", chess_board_pos)
t = 0
pose_o = airsim.Pose()
pose_o_c = airsim.Pose()
pose_o.position = chess_board_pos
pose_o_c.position = pose_o.position
dx = 0.1  # ->1 см
dy = 0.1  # ->1 см
dz = 0.1  # ->5 см

ddx = [0*i * dx for i in range(-2, 20, 1)]
ddy = [i * dy for i in range(-15, 0, 1)]
ddz = [i * dz for i in range(-6,-2, 1)]
X = np.array([pose_o.position.x_val]*5)#[pose_o.position.x_val for i in range(-15, 0, 1)]
Y = pose_o.position.y_val +np.random.random(5)#[pose_o.position.y_val+ i * dy for i in range(-15, 0, 1)]
Z = pose_o.position.z_val -np.random.random(5)#[pose_o.position.z_val + i * dz for i in range(-15,0, 1)]
# client.startRecording()
cameraTest = camera.CameraRecord(client)
client.simGetLidarSegmentation()
cameraTest.start_recording()
for z in Z:
    for y in Y:
        # pose_o.position.x_val
        pose_o.position.y_val = y
        client.simSetObjectPose(object_name, pose_o)
        cameraTest.record(airsim.ImageType.Scene)
        time.sleep(1)
    pose_o.position.y_val = 0
    pose_o.position.z_val = z
print("Конец цикла")
#client.stopRecording()
print("Запись данных в файл")
f = open(f"{object_name}_pos.dat", 'w')
f.write(f"1\n")
for z in Z:
    for y in Y:
        f.write(f"{X[0]} {y} {z}\n")
f.close()