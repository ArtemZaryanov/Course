# ready to run example: PythonClient/car/hello_car.py
import airsim
import time
import airsim.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TODO переписать под SplineRoadNew. Далее SplineRoadNew->SplineRoad
# import SplineRoad as SR
import CurveController as Controller
import cv2
from Driver import DriverCNN
from Driver import DriverPID
from Driver import ArcDriver
from Driver import TrackType
from Driver import SimpleDriver
from record_data import lidar_car_data as lidar
# import keyboard
import os
import tensorflow as  tf
import warnings

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DELTA_RECORDING = 0.05
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
path_project = ROOT_DIR  # "D:/Users/user/PycharmProjects/Course/"
path_model = ROOT_DIR  # "D:/Users/user/PycharmProjects/Course/"
# Параметры симулятора
delta = 0.01
# Ограничим выделяемую память
# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
is_simple_CNN_driver = False
is_PID_driver = False
is_simple_driver = False
is_real_plot = False
is_arc_drive = True
is_collect_data = False
if is_simple_driver or is_simple_CNN_driver or is_arc_drive:
    client.enableApiControl(True)
else:
    client.enableApiControl(False)
# Могу ли я управлять машиной из кода?
print(client.isApiControlEnabled())
# Driver
if is_simple_CNN_driver:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 5)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    SimpleCNN = DriverCNN()
    SimpleCNN.model_CNN_init(path_model + "\SimpleCNN")
    # Для получнеия ошибок
    time.sleep(2)
if is_simple_driver or is_collect_data or is_simple_CNN_driver:
    arcDriver = ArcDriver(client)
# TODO Поменять для отчета имена  SimplePID{CNN} для надежности
if is_PID_driver:
    # Поменять потом!
    SimplePID = DriverPID(TrackType.Standard, 0.1, 0.003, 0.002, 0.1, client)
if is_arc_drive:
    arcDriver = ArcDriver(client)
if is_simple_driver:
    X = [880, 380, 380, 880]
    Y = [-80, -10, 10, 100]
    x = [894, 916, 1000, 1039]
    y = [622, 866, 895, 621]
    control_points = (X, Y, x, y)
    simpleDriver = SimpleDriver()
    simpleDriver.model_init(control_points)

# PID регуляторы
# Трасса
kp_curve = 0
kd_curve = 0
ki_curve = 0
erros_curve_i = []
is_curve_control = False
# PID_Curve = Controller.CurveControl()
# Скорость
kp_velocity = 0
kd_velocity = 0
ki_velocity = 0
erros_velocity_i = []
max_velocity = 8
min_velocity = 5
v0 = 0
e0_velocity = min_velocity
error_velocity_i_0 = e0_velocity
is_velocity_control = True
PID_Velocity = Controller.VelocityControl(client, e0_velocity, v0, error_velocity_i_0)

# Для SimMove
car_controls = client.getCarControls("PhysXCar")


# Lidar

# lidarTest = lidar.LidarTest(client)
# client.simGetLidarSegmentation()
# lidarTest.start_recording()


# Получить изображение с камеры. Возвращает numpy массив
def get_sim_image(client):
    responses = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])
    # print("Type %d, size %d" % (responses[0].image_type, len(responses[0].image_data_uint8)))
    img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)

    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    return img_rgb


# Не актуально
# Проехать по уже известным steering и throttle
def read_true_input(path_project, file_name):
    return pd.read_csv(path_project + file_name)


# Не актуально
def move_on_true_input(true_input, client, car_controls, delta):
    # for по всему DF
    # keyboard.send("R")
    for d, th, st in true_input.values:
        SimMove(th, st, client, car_controls, d)
    # Остановиться
    # keyboard.send("R")
    SimMove(0.2, 0, client, car_controls, delta)


# true_input = read_true_input(path_project, "true_input.csv")
# SimMove(1, 0, client, delta)
time.sleep(1)
# keyboard.send("R")
start_data = pd.read_csv("start_data.csv")
position = airsim.Vector3r(start_data.X[0]/ 100, start_data.Y[0] / 100, -1)
heading = airsim.utils.to_quaternion(0, 0, - 1.5)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)
SimMove(0.5, 0, client, car_controls, 0.05)
# TODO  Полность почистить код и заново снять эталонные данные
if is_collect_data:
    print("collect_data")
    time.sleep(2)
# client.startRecording()

# real_plot
# Create figure for plotting
fig, ax = plt.subplots(2, 1)
# ax = fig.add_subplot(1, 1, 1)
xs = []  # store trials here (n)
ys = []  # store relative frequency here
rs = []  # for theoretical probability
steerings = []
throttles = []


def animate(i, xs, ys, client):
    throttle, steering = SimpleDrive(client)
    # Aquire and parse data from serial port
    vehiclePose = client.simGetVehiclePose()
    position = vehiclePose.position
    orientation = airsim.to_eularian_angles(vehiclePose.orientation)
    # Add x and y to lists
    xs.append(position.x_val)
    ys.append(position.y_val)
    steerings.append(steering)
    throttles.append(throttle)
    # rs.append(0.5)
    # Limit x and y lists to 20 items
    # xs = xs[-20:]
    # ys = ys[-20:]
    # Draw x and y lists
    ax[0].clear()
    ax[1].clear()
    # ax.plot(xs, ys, label="Experimental Probability")
    # ax.quiver
    yaw = orientation[2]
    new_x = np.cos(yaw)
    new_y = np.sin(yaw)
    # ax.quiver(position.x_val, position.y_val, new_x, new_y)
    ff = simpleDriver.get_objective_func()
    x = np.linspace(0, 1000)
    ax[0].plot(x, list(map(ff, x)))
    # Steering
    ax[1].plot(steerings)

    # Для конусов
    # cone_center  = simpleDriver.get_data()
    # print(cone_center)
    # ax.scatter(cone_center[:,0],cone_center[:,1])
    # ax.plot(xs, rs, label="Theoretical Probability")
    # ax.set_ylim([-40 , +40])
    # ax.set_xlim([-40, +40])
    # ax.set_ylim(np.min(ys) - np.std(ys),np.max(ys) + np.std(ys))
    # ax.set_xlim(np.min(xs) - np.std(xs), np.max(xs) + np.std(xs))
    # Format plot
    # ax[0].xticks(rotation=45, ha='right')
    # ax[0].subplots_adjust(bottom=0.30)
    # ax[0].title('This is how I roll...')
    # ax[0].ylabel('Relative frequency')
    # plt.legend()
    # plt.axis([1, None, 0, 1.1])  # Use for arbitrary number of trials
    # plt.axis([1, 100, 0, 1.1]) #Use for 100 trial demo


def SimpleDrive(client):
    if not client.simIsPause():
        start_time = time.time()

        time_stamp_lidar_data, lidar_data = lidarTest.get_lidar_data()
        # time_stamp_camera_data, \
        image = get_sim_image(client)
        # plt.imshow(image)
        # plt.show()
        image_norm = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = (image_norm * 255).astype(np.uint8)
        # Для котнтроля опорных цветов
        # print(np.unique(img.reshape(-1, img.shape[2]), axis=0, return_counts=True))
        steering = simpleDriver.Control_Simple(img, lidarTest.get_lidar_data()[1].T)
        print("- -- %s seconds ---" % (time.time() - start_time))
        throtle = 0.5
        SimMove(throtle, steering[0], client, car_controls, 0.1)
        return throtle, steering[0],


if is_real_plot:
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys, client), interval=10)
    plt.show()


def write_to_file_sync(data_file, str):
    data_file.write(str)
    data_file.flush()
    os.fsync(data_file.fileno())

if is_simple_CNN_driver:
    ping_CNN_file = open(f"ping_CNN.dat", 'w')
    cone_errors_CNN = open(f"error_cone_CNN.dat", 'w')
    lmin_errors_CNN = open(f"lmin_error_CNN.dat", 'w')
    correct_cone_errors_CNN = open(f"correct_error_cone_CNN.dat", 'w')
    vehicle_path_CNN = open(f"vehicle_path_CNN.dat", 'w')
    vehicle_path_CNN.write("x y z pitch roll yaw\n")
if is_arc_drive:
    ping_arc = open(f"ping_arc.dat", 'w')
    arcs = open(f"arcs.dat",'w')
    arcs.write("r y z pitch roll yaw\n")
    cone_errors_arc = open(f"error_cone_arc.dat", 'w')
    lmin_errors_arc = open(f"lmin_error_arc.dat", 'w')
    correct_cone_errors_arc = open(f"correct_error_cone_arc.dat", 'w')
    vehicle_path_arc = open(f"vehicle_path_arc.dat", 'w')
    vehicle_path_arc.write("x y z pitch roll yaw r steering\n")
if is_PID_driver:
    ping_PID_file = open(f"ping_PID.dat", 'w')
# ht_CNN = open(f"ht_CNN.dat", 'w')
cone_left, cone_right = arcDriver.get_world_cone()
world_cone_left = open(f"world_cone_left.dat", 'w')
world_cone_right = open(f"world_cone_right.dat", 'w')
for c in cone_left:
    x, y = c
    write_to_file_sync(world_cone_left, f"{x} {y}\n")
for c in cone_right:
    x, y = c
    write_to_file_sync(world_cone_right, f"{x} {y}\n")


def write_to_file_sync(data_file, str):
    data_file.write(str)
    data_file.flush()
    os.fsync(data_file.fileno())
while True and not is_real_plot:
    print("is_real_plot")
    # SimMove(0.5, 0, client, car_controls, delta)
    # print(kp_velocity, kd_velocity, ki_velocity)
    # posAgent, speedAgent, _ = CarState(client)
    # move_on_true_input(true_input, client, car_controls, 0.05)
    # get camera images from the car
    # Сбор данных
    # if is_collect_data:
    # Сбор данных с камеры
    #    if client.isRecording():
    #        get_sim_image(client)
    #        lidarTest.record()
    # time.sleep(DELTA_RECORDING)
    if is_collect_data:
        start_time = time.time()
        ping = time.time() - start_time
        write_to_file_sync(ping_CNN_file, f"{ping}\n")
        h_error = arcDriver.get_L_mint()
        write_to_file_sync(lmin_errors_CNN, f"{h_error}\n")
        e_cone = arcDriver.get_cone_errors()
        correct_e_cone = arcDriver.get_correct_cone_errors()
        write_to_file_sync(cone_errors_CNN, f"{e_cone}\n")
        write_to_file_sync(correct_cone_errors_CNN, f"{correct_e_cone}\n")
        pose = client.simGetVehiclePose().position.to_numpy_array()[0:3]
        pitch, roll, yaw = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
        write_to_file_sync(vehicle_path_CNN, f"{pose[0]} {pose[1]} {pose[2]} {pitch} {roll} {yaw}\n")

    if is_PID_driver:
        start_time = time.time()
        _, throtle, steering = SimplePID.ControlPID()
        if is_velocity_control:
            _, throtle, _ = A_velocity(0.1, 1, 0, client, PID_Velocity)
        ping = time.time() - start_time
        ping_PID_file.write(f"{ping}\n")
        SimMove(0.5, steering, client, car_controls, 0.001)
    if is_arc_drive:
        start_time = time.time()
        steering,r = arcDriver.Control_Arc()
        if is_velocity_control:
            _, throtle, _ = A_velocity(0.1, 1, 0, client, PID_Velocity)
        ping = time.time() - start_time
        ping_arc.write(f"{ping}\n")
        h_error = arcDriver.get_L_mint()
        write_to_file_sync(lmin_errors_arc, f"{h_error}\n")
        # ping_CNN_file.write(f"{ping}\n")
        e_cone = arcDriver.get_cone_errors()
        correct_e_cone = arcDriver.get_correct_cone_errors()
        write_to_file_sync(cone_errors_arc, f"{e_cone}\n")
        write_to_file_sync(correct_cone_errors_arc, f"{correct_e_cone}\n")
        pose = client.simGetVehiclePose().position.to_numpy_array()
        pitch,roll,yaw  = airsim.to_eularian_angles(client.getImuData().orientation)
        write_to_file_sync(vehicle_path_arc, f"{pose[0]} {pose[1]} {pose[2]} {pitch} {roll} {yaw} {r} {steering}\n")
        SimMove(0.5, steering, client, car_controls, 0.001)
    if is_simple_CNN_driver:

        start_time = time.time()
        if is_velocity_control:
            _, throtle, _ = A_velocity(0.1, 1, 0, client, PID_Velocity)
        steering = SimpleCNN.Control_CNN(get_sim_image(client))
        ping = time.time() - start_time
        write_to_file_sync(ping_CNN_file, f"{ping}\n")

        h_error = arcDriver.get_L_mint()
        write_to_file_sync(lmin_errors_CNN, f"{h_error}\n")
        # ping_CNN_file.write(f"{ping}\n")
        e_cone = arcDriver.get_cone_errors()
        correct_e_cone = arcDriver.get_correct_cone_errors()
        write_to_file_sync(cone_errors_CNN, f"{e_cone}\n")
        write_to_file_sync(correct_cone_errors_CNN, f"{correct_e_cone}\n")
        pose = client.simGetVehiclePose().position.to_numpy_array()[0:2]
        write_to_file_sync(vehicle_path_CNN, f"{pose[0]} {pose[1]}\n")
        # cone_errors_CNN.write(f"{e_cone}\n")

        print("errors=", e_cone)
        # write_to_file_sync(ping_CNN_file,f"{ping}")
        # print("- -- %s seconds ---" % ping)
        SimMove(throtle, int(steering), client, car_controls, 0.001)

    if is_simple_driver and not client.simIsPause():
        start_time = time.time()

        time_stamp_lidar_data, lidar_data = lidarTest.get_lidar_data()
        # time_stamp_camera_data, \
        image = get_sim_image(client)
        # plt.imshow(image)
        # plt.show()
        image_norm = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = (image_norm * 255).astype(np.uint8)
        # Для котнтроля опорных цветов
        # print(np.unique(img.reshape(-1, img.shape[2]), axis=0, return_counts=True))
        steering = simpleDriver.Control_Simple(img, lidarTest.get_lidar_data()[1].T)
        print("- -- %s seconds ---" % (time.time() - start_time))
        throtle = 0.5
        SimMove(throtle, -int(steering), client, car_controls, 0.2)
    # get camera images from the car
    # airsim.write_file('py1.png', responses[0].image_data_float)
