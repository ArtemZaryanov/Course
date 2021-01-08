# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import airsim.utils
import numpy as np
import pandas as pd
import SplineRoad as SR
import CurveController as Controller
from Driver import Driver
import matplotlib.pyplot as plt
from PIL import Image
#import keyboard
import os
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Генерация трассы
def generate_track(return_function=False):
    FR = SR.FunctionRoad()
    FR.generate_data()
    FR.move_data()  # Потом убрать
    if return_function:
        return SR.get_function(), None
    else:
        return None


def CarState(client):
    State = client.getCarState()
    pos = State.kinematics_estimated.position.to_numpy_array()
    velocity = State.speed
    kinematics_estimated = State.kinematics_estimated
    return pos, velocity, kinematics_estimated


# Возврат в начальное состояние и установка автомобиля
# По направлению нормали к трассе
def reset_environment(client, RoadFunction=None):
    client.reset()
    # client.simSetVehiclePose(home_orient,True)
    # client.simSetVehiclePose(home_pose,True)
    # SimMove(1, 0.1, client, delta)


def SimMove(t, s, client, car_controls, delta):
    car_controls.throttle = t
    car_controls.steering = s
    client.setCarControls(car_controls)
    time.sleep(delta)

# Управление с помощью PID Controller
def sign_steering(posAgent, p, function, steering):
    # Определение знака поворота
    # Вектор от точки нормали до позиции автомобиля
    # BP = np.array([p[0] - posAgent[0], p[1] - posAgent[1]])
    # Вектор касательной
    # tan = derivative(function, p[0])
    # C = np.array([tan / np.sqrt(1 + tan ** 2), 1 / np.sqrt(1 + tan ** 2)])
    print(f"l = {SR.distance_to_point(posAgent)}")
    # return -np.sign(np.cross(10*C, BP))
    l_left, _ = SR.distance_to_point_l(posAgent)
    l, _ = SR.distance_to_point(posAgent)
    l_right, _ = SR.distance_to_point_r(posAgent)
    print(f"l_lef={l_left}, l = {l}, l_right = {l_right} ")

    if l_left > l_right:
        return -1
    if l_left < l_right:
        return 1
    else:
        return 0


def A_velocity(kP, kD, kI, client, PID_Velocity):
    # Получаем steering и throttle
    _, v, kinematics = CarState(client)
    linear_accel = kinematics.linear_acceleration.get_length()
    _e_v_p = PID_Velocity.e_velocity_p(v, min_velocity, max_velocity)
    _e_v_d = PID_Velocity.e_velocity_d(linear_accel, v, min_velocity, max_velocity)
    _e_v_i = PID_Velocity.e_velocity_i(v, min_velocity, max_velocity)
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


def A_curve(kP, kD, kI, client, errors_curve_i, PID_Curve):
    _e_c_p, p = PID_Curve.e_curve_p(client)
    _e_c_d = PID_Curve.e_curve_d(client)
    _e_c_i = PID_Curve.e_curve_i(client, delta, errors_curve_i)
    # Установили параметры
    PID_Curve.setControllerParams(kP, kD, kI)
    correction_s = PID_Curve.PIDController(_e_c_p, _e_c_d, _e_c_i)
    throttle = client.getCarControls().throttle
    pos = client.getCarState().kinematics_estimated.position.to_numpy_array()
    function = SR.get_function()
    p = np.array([p, function(p)])
    steering = correction_s * sign_steering(pos, p, function, client.getCarControls().steering)
    print(f"errors e_c_p={_e_c_p,},e_c_d={_e_c_d},e_c_i={_e_c_i},sign={sign_steering(pos, p, function, steering)}")
    return _e_c_p + _e_c_i, throttle, steering


# TODO Добавить генератор трасс и получить его нормали. Для визуализации можно использовать просто matplotlib,
#  а UE4 использовать только для демонтрации проекта
# generate_track()
# Пути к файлам
path_project = "D:/Users/user/PycharmProjects/Course/"
path_model  = "D:/Users/user/PycharmProjects/Course/"
# Параметры симулятора
delta = 0.01
# Driver
SimpleCNN = Driver()
SimpleCNN.model_CNN_init(path_model + "SimpleCNN")
time.sleep(10)
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())


# PID регуляторы
# Трасса
kp_curve = 0
kd_curve = 0
ki_curve = 0
erros_curve_i = []
is_curve_control = False
PID_Curve = Controller.CurveControl()
# Скорость
kp_velocity = 0
kd_velocity = 0
ki_velocity = 0
erros_velocity_i = []
max_velocity = 20
min_velocity = 10
v0 = 0
e0_velocity = min_velocity
error_velocity_i_0 = e0_velocity
is_velocity_control = False
PID_Velocity = Controller.VelocityControl(e0_velocity, v0, error_velocity_i_0)

# Для SimMove
car_controls = client.getCarControls("PhysXCar")


# Получить изображение с камеры. Возвращает numpy массив
def get_sim_image(client):
    responses = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])
    # print("Type %d, size %d" % (responses[0].image_type, len(responses[0].image_data_uint8)))
    img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)

    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    print(img_rgb.shape)
    return img_rgb


# Проехать по уже известным steering и throttle
def read_true_input(path_project, file_name):
    return pd.read_csv(path_project + file_name)


def move_on_true_input(true_input, client, car_controls, delta):
    # for по всему DF
    # keyboard.send("R")
    for d,th, st in true_input.values:
        SimMove(th, st, client, car_controls, d)
    # Остановиться
    # keyboard.send("R")
    SimMove(0, 0, client, car_controls, delta)



true_input = read_true_input(path_project, "true_input.csv")
# SimMove(1, 0, client, delta)
time.sleep(1)
# keyboard.send("R")
# TODO  Полность почистить код и заново снять эталонные данные
while True:
    # time.sleep(1)
    # SimMove(1, 0, client, car_controls, delta)
    # print(kp_velocity, kd_velocity, ki_velocity)
    # posAgent, speedAgent, _ = CarState(client)
    # move_on_true_input(true_input, client, car_controls, 0.05)
    # get camera images from the car

    steering = SimpleCNN.Control_CNN(get_sim_image(client))
    # print(type(steering))
    SimMove(1,int(steering),client,car_controls,0.01)
    # get camera images from the car
    # airsim.write_file('py1.png', responses[0].image_data_float)
"""
    if client.simGetCollisionInfo().has_collided:
        reset_environment(client)
        SimMove(0.5, 0.001, client, delta)
    if is_velocity_control:
        _, t, s = A_velocity(0.1, 1, 1, client, PID_Velocity)

        SimMove(t, s, client, delta)
    if is_curve_control:
        _, t, s = A_curve(0.0003, 0.0002, 0.1, client, erros_curve_i, PID_Curve)
        SimMove(t, s, client, delta)
"""
