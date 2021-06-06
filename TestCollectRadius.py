import airsim
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from CurveController import VelocityControl

def write_to_file_sync(data_file, str):
    data_file.write(str)
    data_file.flush()
    os.fsync(data_file.fileno())
def CarState(client):
    State = client.getCarState()
    pos = State.kinematics_estimated.position.to_numpy_array()
    velocity = State.speed
    kinematics_estimated = State.kinematics_estimated
    return pos, velocity, kinematics_estimated

def A_velocity(kP, kD, kI, client, PID_Velocity):
    # Получаем steering и throttle
    _, v, kinematics = CarState(client)
    linear_accel = kinematics.linear_acceleration.get_length()
    _e_v_p = PID_Velocity.e_velocity_p(v, min_velocity, max_velocity)
    _e_v_d = PID_Velocity.e_velocity_d(linear_accel, v, min_velocity, max_velocity)
    _e_v_i = PID_Velocity.e_velocity_i(v, min_velocity, max_velocity, 0.01)
    # Установили параметры
    PID_Velocity.setControllerParams(kP, kD, kI)
    correction_v = PID_Velocity.PIDController(_e_v_p, _e_v_d, _e_v_i)
    # print(correction_v)
    throttle = client.getCarControls().throttle * (1 + correction_v)
    if throttle > 1:
        throttle = 1
    if throttle < -1:
        throttle = -1
    steering = client.getCarControls().steering
    return _e_v_p + _e_v_d + _e_v_i, throttle, steering
    # TODO Доделать Twiddle алгоритм для контролерра скорости


client = airsim.CarClient()
client.confirmConnection()
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())
client.enableApiControl(True)
#start_data = pd.read_csv("start_data.csv")
#position = airsim.Vector3r(start_data.X[0], start_data.Y[0] / 100, -1)
#heading = airsim.utils.to_quaternion(0, 0, 0)
#pose = airsim.Pose(position, heading)
#client.simSetVehiclePose(pose, True)
time.sleep(2)
steering_reg = open(f"steering_reg.dat", 'w')
steering_reg_cont = open(f"steering_reg_con_2.dat", 'w')
steering_reg.write("v r w steering\n")
steering_reg_cont.write("v r w steering\n")
# Скорость
kp_velocity = 0
kd_velocity = 0
ki_velocity = 0
erros_velocity_i = []
max_velocity = 8
min_velocity = 4
v0 = 0
e0_velocity = min_velocity
error_velocity_i_0 = e0_velocity
is_velocity_control = True
PID_Velocity = VelocityControl(client, e0_velocity, v0, error_velocity_i_0)
client.setCarControls(airsim.CarControls(throttle=1,steering=0))
time.sleep(1)
client.setCarControls(airsim.CarControls(throttle=0.5, steering=0))
time.sleep(2)
num = 400
steerings = 2*np.random.rand(num)-1
start_time = time.time()
k = 0
throtle = 0
vv = False
# Почти 5 км. Пдождать после первого

# Карту сверху нарисовать!!!!!!!!!!!!!!!!!!!
while True:
    car_state = client.getCarState()
    #s = 2*np.random.rand(1)[0]-1
    w = car_state.kinematics_estimated.angular_velocity
    client.getImuData().angular_velocity
    vel = client.getCarState().speed
    if vel>10:
        throtle = -1
        vv = True
    else:
        throtle = 1
        vv = False
    if k>num-1:
        break
    s = steerings[k]

    client.setCarControls(airsim.CarControls(throttle=0.5, steering=s))
    if time.time() - start_time > 3:
        start_time = time.time()
        k= k+1
        r = vel / w.get_length()
        write_to_file_sync(steering_reg_cont, f"{vel} {r} {w.get_length()} {s}\n")
    #if w != 0 and client.getCarControls().steering != 0:

