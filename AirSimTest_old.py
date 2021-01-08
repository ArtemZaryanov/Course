# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import airsim.utils
import numpy as np
import pandas as pd
import SplineRoad as SR
import CurveController as Controller
import matplotlib.pyplot as plt
from scipy.misc import derivative


def sign_steering(posAgent, p, function, steering):
    # Определение знака поворота
    # Вектор от точки нормали до позиции автомобиля
    # BP = np.array([p[0] - posAgent[0], p[1] - posAgent[1]])
    # Вектор касательной
    # tan = derivative(function, p[0])
    # C = np.array([tan / np.sqrt(1 + tan ** 2), 1 / np.sqrt(1 + tan ** 2)])
    print(f"l = {SR.distance_to_point(posAgent)}")
    # return -np.sign(np.cross(10*C, BP))
    l_left,_ = SR.distance_to_point_l(posAgent)
    l,_ = SR.distance_to_point(posAgent)
    l_right,_ = SR.distance_to_point_r(posAgent)
    print(f"l_lef={l_left}, l = {l}, l_right = {l_right} ")

    if l_left>l_right:
        return  -1
    if l_left<l_right:
        return 1
    else:
        return  0
def curve_control():
    pass


# Генерация трассы
def generate_track(return_function=False):
    FR = SR.FunctionRoad()
    FR.generate_data()
    FR.move_data()  # Потом убрать
    if return_function:
        return SR.get_function(), None
    else:
        return None


# Вывод графиков

# Возврат в начальное состояние и установка автомобиля
# По направлению нормали к трассе
def reset_environment(client, RoadFunction=None):
    client.reset()
    # client.simSetVehiclePose(home_orient,True)
    # client.simSetVehiclePose(home_pose,True)
    # SimMove(1, 0.1, client, delta)


def CarState(client):
    State = client.getCarState()
    pos = State.kinematics_estimated.position.to_numpy_array()
    velocity = State.speed
    kinematics_estimated = State.kinematics_estimated
    return pos, velocity, kinematics_estimated


def SimMove(t, s, client, delta):
    car_controls.throttle = t
    car_controls.steering = s
    client.setCarControls(car_controls)
    time.sleep(delta)


# TODO Добавить генератор трасс и получить его нормали. Для визуализации можно использовать просто matplotlib,
#  а UE4 использовать только для демонтрации проекта
# generate_track()
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())

# Параметры симулятора
delta = 0.01

# PID
kp_curve = 0
kd_curve = 0
ki_curve = 0
erros_curve_i = []

kp_velocity = 0
kd_velocity = 0
ki_velocity = 0
erros_velocity_i = []
max_velocity = 20
min_velocity = 10
is_velocity_control = False

# Ошибки в начлаьный момоент времени
#
v0 = 0
e0 = min_velocity
home_pose = client.getCarState().kinematics_estimated.position
l_past, _ = SR.distance_to_point([home_pose.x_val, home_pose.y_val])

# Непрерывнуые фцнкции ошибок
def e_velocity_p(v: float, v_min: float, v_max: float):
    v_mm_half = (v_min + v_max) / 2
    if v <= v_min:
        return (e0 / ((v0 - v_max) ** 2)) * (v - v_min) ** 2
    if (v_min <= v) and (v_max >= v):
        return np.sin((v - v_min) ** 2 * (v_max - v) ** 2 / ((v_max - v_min) ** 4))  # Было нуль
    if v_max <= v:
        return -(v - v_max) ** 2


def e_velocity_d(a: float, v: float, v_min: float, v_max: float):
    v_mm_half = (v_min + v_max) / 2
    if v <= v_min:
        return 2 * a * (e0 / ((v0 - v_max) ** 2)) * (v - v_min) ** 2
    if (v_min <= v) and (v_max >= v):
        return 2 * a / ((v_max - v_min) ** 4) * (
                (v - v_min) * (v_max - v) ** 2
                -
                (v - v_min) ** 2 * (v_max - v)
        ) * np.cos((v - v_min) ** 2 * (v_max - v) ** 2 / ((v_max - v_min) ** 4))

    if v_max <= v:
        return -2 * a * (v - v_max) ** 2


def e_velocity_i(v: float, v_min: float, v_max: float):
    e_p = e_velocity_p(v, v_min, v_max)
    erros_velocity_i.append(e_p)
    e_v_i = sum(erros_velocity_i) * delta
    return e_v_i


def e_curve_p(client):
    pos, _, _ = CarState(client)
    l, p = SR.distance_to_point(pos)
    return l, p


def e_curve_d(client):
    pos, _, kinematics = CarState(client)
    l, p = e_curve_p(client)
    vx = kinematics.linear_velocity.x_val
    vy = kinematics.linear_velocity.y_val
    f = SR.get_function()
    df = derivative(SR.get_function(), pos[0])
    _e = (2 / (l ** 2 + 1)) * ((pos[0] - p) * (vx) + (pos[1] - f(p)) * (vy))
    if abs(_e) > 10000:
        return 0
    else:
        return _e


def e_curve_i(client, delta,erros_curve_i):
    e_p, _ = e_curve_p(client)
    erros_curve_i.append(e_p)
    if len(erros_curve_i)>100:
        buff = erros_curve_i[-1]
        erros_curve_i = [buff]
    return sum(erros_curve_i) * delta

def A_velocity(kP, kD, kI, client):
    # Получаем steering и throttle
    _, v, kinematics = CarState(client)
    linear_accel = kinematics.linear_acceleration.get_length()
    _e_v_p = e_velocity_p(v, min_velocity, max_velocity)
    _e_v_d = e_velocity_d(linear_accel, v, min_velocity, max_velocity)
    _e_v_i = e_velocity_i(v, min_velocity, max_velocity)
    # Установили параметры
    speed_controller.setControllerParams(kP, kD, kI)
    correction_v = speed_controller.PIDController(_e_v_p, _e_v_d, _e_v_i)
    # print(correction_v)
    throttle = client.getCarControls().throttle * (1 + correction_v)
    if throttle > 1:
        throttle = 1
    if throttle < -1:
        throttle = -1
    steering = client.getCarControls().steering
    return _e_v_p + _e_v_d + _e_v_i, throttle, steering
    # TODO Доделать Twiddle алгоритм для контролерра скорости

def A_curve(kP, kD, kI, client, error_curve_i):
    _e_c_p, p = e_curve_p(client)
    _e_c_d = e_curve_d(client)
    _e_c_i = e_curve_i(client, delta,erros_curve_i)
    # Установили параметры
    curve_controller.setControllerParams(kP, kD, kI)
    correction_s = curve_controller.PIDController(_e_c_p, _e_c_d, _e_c_i)
    throttle = client.getCarControls().throttle
    pos = client.getCarState().kinematics_estimated.position.to_numpy_array()
    function = SR.get_function()
    p = np.array([p, function(p)])
    steering = correction_s * sign_steering(pos, p, function, client.getCarControls().steering)
    #if steering > 0.1:
    #    steering = 0.1
    #if steering < -0.1:
    #    steering = -0.1
    print(f"errors e_c_p={_e_c_p,},e_c_d={_e_c_d},e_c_i={_e_c_i},sign={sign_steering(pos, p, function, steering)}")
    return _e_c_p + _e_c_i, throttle, steering


# Проехать по уже известным steering и throttle
def read_true_input(path_project,file_name):
    return pd.read_csv(path_project,file_name)


def move_on_true_input(true_input,client,delta):
    # for по всему DF
    for st,th in true_input.values:
        SimMove(1,1,client,delta)
    # Остановиться
    SimMove(0,0,client,delta)
best_error_v_p = e_velocity_p(0, min_velocity, max_velocity)
current_steering_sign = 1
home_orient = client.getCarState().kinematics_estimated.orientation
car_controls = airsim.CarControls()
car_controls.throttle = 1
car_controls.steering = 1
# print(f"TruePose",client.simGetTrueVehiclePose("PhysXCar").position)
speed_controller = Controller.Controller()
curve_controller = Controller.Controller()
car_controls = client.getCarControls("PhysXCar")
is_curve_control = True

t0 = time.time()
SimMove(1, 0, client, delta)
while True:
    # print(kp_velocity, kd_velocity, ki_velocity)
    posAgent, speedAgent, _ = CarState(client)
    if client.simGetCollisionInfo().has_collided:
        reset_environment(client)
        SimMove(0.5, 0.001, client, delta)
    if is_velocity_control:
        _, t, s = A_velocity(0.1, 1, 1, client)

        SimMove(t, s, client, delta)
    if is_curve_control:
        _, t, s = A_curve(0.0003,0.0002,0.1, client,erros_curve_i)
        SimMove(t, s, client, delta)
    # print("l=",SR.distance_to_point(posAgent))
    # velocity = np.append(velocity,[[time.time() - t0,speedAgent]],axis=0)
    # trottles = np.append(trottles, [[time.time() - t0, car_controls.throttle]], axis=0)
    # steerings = np.append(steerings, [[time.time() - t0, car_controls.throttle]], axis=0)
