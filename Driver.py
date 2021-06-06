from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from SplineRoadNew import SplineRoad as SplineRoad
import airsim
import scipy
import os
from CameraProcessing import CameraDataProccesing
from CameraProcessing import LidarDataProccesing
from CameraProcessing import PinholeCamera
from CurveController import CurveControl
from CurveController import Controller


def getRandomNumber():
    return 4


class TrackType:
    Standard = 0
    Epicycloid = 1
    Eight = 2
    Random = 3
    PolarFunction = 4
    Spline = 5


class DriverPID:
    def __init__(self, track_type, delta, kP, kD, kI, client):

        self.PID_Curve = CurveControl(track_type, client)
        self.SR = self.PID_Curve.getSplineRoadObject()
        self.setControllerParams(kP, kD, kI)
        self.client = client
        self.delta = delta
        self.errors_curve_i = [0]

    def setControllerParams(self, kP, kD, kI):
        self.PID_Curve.setControllerParams(kP, kD, kI)

    def sign_steering(self, posAgent, p, function, steering):
        # Определение знака поворота
        # Вектор от точки нормали до позиции автомобиля
        # BP = np.array([p[0] - posAgent[0], p[1] - posAgent[1]])
        # Вектор касательной
        # tan = derivative(function, p[0])
        # C = np.array([tan / np.sqrt(1 + tan ** 2), 1 / np.sqrt(1 + tan ** 2)])
        print(f"l = {self.PID_Curve.SR.distance_to_point(posAgent)}")
        # return -np.sign(np.cross(10*C, BP))
        l_left, _ = self.SR.distance_to_point_l(posAgent)
        l, _ = self.SR.distance_to_point(posAgent)
        l_right, _ = self.SR.distance_to_point_r(posAgent)
        print(f"l_lef={l_left}, l = {l}, l_right = {l_right} ")

        if l_left > l_right:
            return -1
        if l_left < l_right:
            return 1
        else:
            return 0

    # _, t, s = A_curve(0.0003, 0.0002, 0.1, client, erros_curve_i)
    def A_curve(self):
        _e_c_p, p = self.PID_Curve.e_curve_p()
        _e_c_d = self.PID_Curve.e_curve_d()
        _e_c_i = self.PID_Curve.e_curve_i(self.delta, self.errors_curve_i)
        self.errors_curve_i.append(_e_c_i)
        correction_s = self.PID_Curve.PIDController(_e_c_p, _e_c_d, _e_c_i)
        throttle = self.client.getCarControls().throttle
        pos = self.client.getCarState().kinematics_estimated.position.to_numpy_array()
        function = self.SR.get_function()
        p = np.array([p, function(p)])
        steering = correction_s * self.sign_steering(pos, p, function, self.client.getCarControls().steering)
        print(
            f"errors e_c_p={_e_c_p,},e_c_d={_e_c_d},e_c_i={_e_c_i},sign={self.sign_steering(pos, p, function, steering)}")
        return _e_c_p + _e_c_i, throttle, steering

    def ControlPID(self):
        return self.A_curve()


class DriverCNN:

    def __init__(self):
        self.model_CNN = None
        pass

    # После инициализации модели. Необходимо подать картинку,
    # чтобы получить  вектор управления
    def Control_CNN(self, image: np.ndarray):
        # assert self.model_CNN is None, "model is None"
        return self.model_CNN.predict([image[np.newaxis]]).argmax() - 1  # Пока так!!!!!

    # Иннициализация модели
    def model_CNN_init(self, path):
        self.model_CNN = keras.models.load_model(path)
        self.model_CNN.summary()
        print(self.model_CNN)

    # Преобразование выходов сети
    def _convert_CNN(self, output):
        pass


class ArcDriver:
    # Надо получить сразу все конусы среды
    def __init__(self, client,steering_func = None):
        self.client = client
        self.cones_left = None
        self.cones_right = None
        self.r0 = 0
        self.get_all_cone()
        self.a = 7.73240547
        self.b = -4.09307997
        self.steering_func_p = lambda r:np.arctan(self.a/(self.b/2 + r))/(np.pi/2)
        self.steering_func_n = lambda r: np.arctan(self.a / (self.b / 2 + r)) / (np.pi / 2)
        self.lidar_R = 16
        self.x0 = [0,0,0]


    # Эммуляция работы Лидара. Потом убрать.
    # Находим все конусы в радиусе R от автомобиля
    def get_world_cone(self):
        return self.cones_left,self.cones_right

    def get_all_cone(self):
        b_l = lambda n: self.client.simGetObjectPose(n).position.to_numpy_array()[0:2]
        self.cones_left = np.array(list(map(b_l, self.client.simListSceneObjects('FloatingActor_($|[^a-zA-Z]+)'))))
        self.cones_right = np.array(
            list(map(b_l, self.client.simListSceneObjects('FloatingActorYellow_($|[^a-zA-Z]+)'))))

    def get_cone_arc(self):
        R = 16
        FOV = 180
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        orientation = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        yaw = orientation[2]
        arc_pnts = self.get_arc_pnts(R, np.radians(FOV), yaw, pos_a[0], pos_a[1])

        cones_left_in_circle, cones_right_in_circle = self.get_cone(R)
        cones_left_in_arc = self.find_in_contour(
            cones_left_in_circle, arc_pnts, R, np.radians(FOV), pos_a[0], pos_a[1], yaw)
        cones_right_in_arc = self.find_in_contour(
            cones_right_in_circle, arc_pnts, R, np.radians(FOV), pos_a[0], pos_a[1], yaw)
        return cones_left_in_arc, cones_right_in_arc
    def get_L_mint(self):
        cones_left_in_arc, cones_right_in_arc = self.get_cone(16)
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        cones_in_arc = np.concatenate([cones_left_in_arc, cones_right_in_arc], axis=0)
        L_min = np.max((np.linalg.norm(cones_in_arc - pos_a, axis=1)) ** 2, axis=0)
        return L_min
    def get_h_errors(self):
        cones_left_in_arc, cones_right_in_arc = self.get_cone(16)
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        cones_in_arc = np.concatenate([cones_left_in_arc, cones_right_in_arc], axis=0)

        h_errors = np.prod((np.linalg.norm(cones_in_arc - pos_a, axis=1)) ** 2, axis=0) / cones_in_arc.shape[0]
        return h_errors

    def get_correct_cone_errors(self):
        cones_left_in_arc, cones_right_in_arc = self.get_cone(16)
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        cones_in_arc = np.concatenate([cones_left_in_arc, cones_right_in_arc], axis=0)
        rr = np.linalg.norm(cones_in_arc - pos_a, axis=1)
        correct_cone_errors = np.mean((rr/(self.lidar_R - rr)), axis=0)
        return correct_cone_errors

    def get_cone_errors(self):
        cones_left_in_arc, cones_right_in_arc = self.get_cone(16)
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        cones_in_arc = np.concatenate([cones_left_in_arc, cones_right_in_arc], axis=0)

        cone_errors = np.mean((np.linalg.norm(cones_in_arc - pos_a, axis=1)) ** 1/2, axis=0)
        return cone_errors

    def rotatePoint(self,centerPoint, point, angle):

        """Rotates a point around another centerPoint. Angle is in degrees.
        Rotation is counter-clockwise"""
        import math
        angle = math.radians(angle)
        temp_point = point[0] - centerPoint[0], point[1] - centerPoint[1]
        temp_point = (temp_point[0] * math.cos(angle) - temp_point[1] * math.sin(angle),
                      temp_point[0] * math.sin(angle) + temp_point[1] * math.cos(angle))
        temp_point = temp_point[0] + centerPoint[0], temp_point[1] + centerPoint[1]
        return temp_point

    def distance_multi(self, c_l, c_r, g):
        a, b, r = g
        f = 0
        # Векторизовать
        for cc_ in c_l:
            f = f + ((np.linalg.norm(
                cc_ - np.array([a, b]))) ** 2 - r ** 2)  # *np.exp(-np.linalg.norm(cc_ - np.array([0,r]))**4)
        for cc_ in c_r:
            f = f + (np.linalg.norm(
                cc_ - np.array([a, b])) ** 2 - r ** 2)  # *np.exp(-np.linalg.norm(cc_ - np.array([0,r]))**4)
        return f

    def optimization_my(self, yaw, c_l, c_r):
        plot = False
        from scipy import optimize
        def near_fine(d):
            # if d<0:
            #    raise ValueError("Wrong d=%f"%(d))
            return 1 * np.exp(-2*d * d / 5.5)

        #def near_fine(d):
            # if d<0:
            #    raise ValueError("Wrong d=%f"%(d))
        #    return 1 * np.exp(-d * d / 9 * 1)

        def distance_my(c_l, c_r, r):
            f = 0
            if r > 0:
                for cc_ in c_l:
                    d1 = np.hypot(cc_[0], cc_[1] - r) - r
                    if d1 > 0:
                        #if plot: plt.scatter(cc_[0], cc_[1])
                        f += d1
                    # else:
                    f += near_fine(-d1)
                for cc_ in c_r:
                    d2 = np.hypot(cc_[0], cc_[1] - r) - r
                    if d2 < 0:
                        #if plot: plt.scatter(cc_[0], cc_[1])
                        f += -d2
                    #            else:
                    #                f+=near_fine(d2)
                    f += near_fine(d2)
            else:
                for cc_ in c_l:
                    d1 = np.hypot(cc_[0], cc_[1] - r) + r
                    if d1 < 0:
                        #if plot: plt.scatter(cc_[0], cc_[1])
                        f += -d1
                    # lse:
                    #   f+=near_fine(d1)
                    f += near_fine(d1)
                for cc_ in c_r:
                    d2 = np.hypot(cc_[0], cc_[1] - r) + r
                    if d2 > 0:
                        #if plot: plt.scatter(cc_[0], cc_[1])
                        f += d2
                    # lse:
                    #   f+=near_fine(-d2)

                    f += near_fine(-d2)

            #       if cc_[0]**2+(cc_[1]-r)**2<r**2:
            #            f = f + (cc_[0]**2+(cc_[1]-r)**2)
            #    for cc_ in c_r:
            #        if ((cc_[0]**2+(cc_[1]-r)**2)**(1))>r**2:
            #            f = f + (cc_[0]**2+(cc_[1]-r)**2)
            return f

        v_my = lambda r: distance_my(c_l, c_r, r)
        result = optimize.minimize(v_my,[0],
                            method ='Nelder-mead',options={'xatol':10**(-4),'initial_simplex':np.array([[-100],[100]])})
        r = result.x
        d = 1
        if r>0:
            d=1
        else:
            d=-1

        return r[0],d
    def optimization_raz(self, yaw, c_l, c_r):
        from scipy import optimize
        def distance_razim_1(c_l, c_r, rr):
            f = 0
            r = rr
            for cc_ in c_l:
                w = 1 / (1 + np.exp((cc_[0] ** 2 + cc_[1] ** 2 - 5) / 2))
                f = f + (((r - cc_[0]) ** 2 + cc_[1] ** 2 - r ** 2)) * 1
            for cc_ in c_r:
                w = 1 / (1 + np.exp((cc_[0] ** 2 + cc_[1] ** 2 - 5) / 2))
                f = f + (((r - cc_[0]) ** 2 + cc_[1] ** 2 + r ** 2)) * 1
            return f
        def distance_razim_1__(c_l, c_r, r):
            f = 0
            for cc_ in c_l:
                f = f + (np.linalg.norm(
                    cc_ - np.array([r, 0])) ** 2 - r ** 2)/np.linalg.norm(cc_)**2  # *np.exp(-np.linalg.norm(cc_ - np.array([0,r]))**4)
            for cc_ in c_r:
                f = f + (np.linalg.norm(
                    cc_ - np.array([r, 0])) ** 2 - r ** 2)/np.linalg.norm(cc_)**2  # *np.exp(-np.linalg.norm(cc_ - np.array([0,r]))**4)
            return f

        v1 = lambda r: distance_razim_1(c_l, c_r, r)

        def distance_razim_2(c_l, c_r, r):
            f = 0
            for cc_ in c_l:
                w = 1 / (1 + np.exp((cc_[0] ** 2 + cc_[1] ** 2 - 5) / 2))
                f = f + (((r - cc_[1]) ** 2 + cc_[0] ** 2 - r ** 2)) * 1
            for cc_ in c_r:
                w = 1 / (1 + np.exp((cc_[0] ** 2 + cc_[1] ** 2 - 5) / 2))
                f = f + (((r - cc_[1]) ** 2 + cc_[0] ** 2 + r ** 2)) * 1
            return f
        def distance_razim_2__(c_l, c_r, r):
            f = 0
            for cc_ in c_l:
                f = f + ((np.linalg.norm(
                    cc_ - np.array([0, r]))) ** 2 - r ** 2)/np.linalg.norm(cc_)**2  # *np.exp(-np.linalg.norm(cc_ - np.array([0,r]))**4)
            for cc_ in c_r:
                f = f + (np.linalg.norm(
                    cc_ - np.array([0, r])) ** 2 - r ** 2)/np.linalg.norm(cc_)**2  # *np.exp(-np.linalg.norm(cc_ - np.array([0,r]))**4)
            return f

        v2 = lambda r: distance_razim_2(c_l, c_r, r)

        result_1 = optimize.minimize(v1, [0])
        result_2 = optimize.minimize(v2, [0])
        if result_2.fun > result_1.fun:
            r = result_1.x
        else:
            r = result_2.x
        d = 1
        if r<0:
            d = d*(-1)
        else:
            d = d*(1)
        return r[0],d
    def optimization(self,yaw,c_l,c_r):
        from scipy import optimize
        def objective(c_l, c_r, g):
            a, b, r = g
            f = 0
            for cc_ in c_l:
                f = f + ((np.linalg.norm(cc_ - np.array([a, b]))) ** 2 - r ** 2)#*np.linalg.norm(cc_)
            for cc_ in c_r:
                f = f + (np.linalg.norm(cc_ - np.array([a, b])) ** 2 - r ** 2) #*np.linalg.norm(cc_)
            return f
        R = self.lidar_R
        con_1_1 = lambda g: g[0] ** 2 + g[1] ** 2 - g[2] ** 2  # Чтобы проходило через (0,0)
        nlc_1_1 = optimize.NonlinearConstraint(con_1_1, 0, 0)
        con_1_2 = lambda g: np.sqrt(g[0] ** 2 + g[1] ** 2) - R  # Чтобы окруность поврота лежала вне круга лидара
        nlc_1_2 = optimize.NonlinearConstraint(con_1_2, 0, np.inf)

        new_x = np.cos(yaw)
        new_y = np.sin(yaw)
        con_1_3 = lambda g: g[0] * new_x + g[1] * new_y
        nlc_1_3 = optimize.NonlinearConstraint(con_1_3, 0, 0)  # Направление аавтомобиля есть касательная к окружности

        con_1_4 = lambda g: g[2]
        nlc_1_4 = optimize.NonlinearConstraint(con_1_4, R, np.inf)
        # Еще раз проверить!!!!!!!!!!! Нужно чтобы для всех левых он дал больше  R а для остальны

        con_1_5 = lambda g: (np.linalg.norm(c_l - np.array([g[0], g[1]]), axis=1) - g[2])
        nlc_1_5 = optimize.NonlinearConstraint(con_1_5, 0, np.inf)

        con_1_6 = lambda g: (np.linalg.norm(c_r - np.array([g[0], g[1]]), axis=1) - g[2])
        nlc_1_6 = optimize.NonlinearConstraint(con_1_6,-np.inf, 0)

        con_2_5 = lambda g: (np.linalg.norm(c_l - np.array([g[0], g[1]]), axis=1) - g[2])
        nlc_2_5 = optimize.NonlinearConstraint(con_2_5, -np.inf, 0)

        con_2_6 = lambda g: (np.linalg.norm(c_r - np.array([g[0], g[1]]), axis=1) - g[2])
        nlc_2_6 = optimize.NonlinearConstraint(con_2_6, 0, np.inf)

        v3_1 = lambda g: objective(c_l, c_r, g)
        v3_2 = lambda g: objective(c_l, c_r, g)

        result_3_1 = optimize.minimize(v3_1, [0,0,0],
                                       constraints=(nlc_1_1,nlc_1_3, nlc_1_4, nlc_1_5, nlc_1_6))
        result_3_2 = optimize.minimize(v3_2, [0,0,0],
                                       constraints=(nlc_1_1, nlc_1_3, nlc_1_4, nlc_2_5, nlc_2_6))
        direct = -1
        if result_3_1.success:
            if result_3_2.success:
                if result_3_1.fun < result_3_2.fun:
                    a, b, r = result_3_1.x
                    direct = direct * (1)
                else:
                    a, b, r = result_3_2.x
                    direct = direct * (-1)
            else:
                a, b, r = result_3_1.x
                direct = direct * (1)
        else:
            a, b, r = result_3_2.x
            direct = direct*(-1)
        return  r, direct

    def rotate_point(self,cx, cy, angle, pp):
        import math
        s = math.sin(angle)
        c = math.cos(angle)
        p = pp.copy()
        # translate point back to origin:
        p[0] -= cx
        p[1] -= cy

        # rotate point
        xnew = p[0] * c - p[1] * s
        ynew = p[0] * s + p[1] * c

        p[0] = xnew + cx
        p[1] = ynew + cy
        return p
    def Control_Arc(self):
        # Нужны локлаьные координаты!!!
        cones_right_in, cones_left_in = self.get_cone(self.lidar_R)
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        orientation = self.client.getCarState().kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        cones_left_in =cones_left_in -  pos_a
        cones_right_in = cones_right_in - pos_a
        new_x = np.cos(float(yaw))
        new_y = np.sin(float(yaw))
        angle = -yaw
        cones_left_in_rot = np.array([self.rotate_point(0, 0, angle, point) for point in cones_left_in])
        cones_right_in_rot = np.array([self.rotate_point(0, 0, angle, point) for point in cones_right_in])
        c_l  = cones_left_in_rot.copy()
        c_r = cones_right_in_rot.copy()
        #if flag == 1 or flag == 2:
            # cond = lambda  c: np.arccos(c[:,0]/np.linalg.norm(c,axis=1))
            # ind_l = np.where((cond(c_l)>0) & (cond(c_l)<np.pi/2))
        ind_l = np.where(c_l[:, 0] > 0)
            #c_l = c_l[ind_l]
            # ind_r = np.where((c_r)>-3*np.pi/2) & (cond(c_r)<0))
        ind_r = np.where(c_r[:, 0] > 0)
            #c_r = c_r[ind_r]
        #if flag == 3 or flag == 4:
            # cond = lambda  c: np.arccos(c[:,0]/np.linalg.norm(c,axis=1))
            # ind_l = np.where((cond(c_l)>np.pi/2) & (cond(c_l)<np.pi))
        #    ind_l = np.where(c_l[:, 0] < 0)
            #c_l = c_l[ind_l]
            #ind_r = np.where((cond(c_r)<np.pi) & (cond(c_r)>-3*np.pi/2))
        #    ind_r = np.where(c_r[:,0] < 0)
            #c_r = c_r[ind_r]
        #if np.degrees(-angle) > np.degrees(np.pi - angle):
        #    cond = lambda c: np.arccos(c[:, 0] / np.linalg.norm(c, axis=1))
        #    ind_l = np.where((cond(c_l) < 3 * np.pi / 2) & (cond(c_l) > np.pi / 2))
        #    c_l = c_l[ind_l]
        #    ind_r = np.where((cond(c_r) < 3 * np.pi / 2) & (cond(c_r) > np.pi / 2))
        #    c_r = c_r[ind_r]
        #    print("-angle")
        #else:
        #    cond = lambda c: np.arccos(c[:, 0] / np.linalg.norm(c, axis=1))
        #    ind_l = np.where((cond(c_l) > -3 * np.pi / 2) & (cond(c_l) < np.pi / 2))
        #    c_l = c_l[ind_l]
        #    ind_r = np.where((cond(c_r) > -3 * np.pi / 2) & (cond(c_r) < np.pi / 2))
        #    c_r = c_r[ind_r]
        #    print("pi-angle")
        # cond = lambda c: np.arccos(c[:, 0] / np.linalg.norm(c, axis=1))
        # ind_l = np.where((cond(c_l) > -3 * np.pi / 2) & (cond(c_l) < np.pi / 2))
        # c_l = c_l[ind_l]
        # ind_r = np.where((cond(c_r) > -3 * np.pi / 2) & (cond(c_r) < np.pi / 2))
        # c_r = c_r[ind_r]
        c_ =np.concatenate([c_l,c_r],axis=0)
        r,d = self.optimization_my(yaw,c_l,c_r)
        a = 15.08905543
        b = 20.14829197
        if d>0:
            s = self.steering_func_p(r)
            print("+")
        else:
            s = self.steering_func_n(r)
            print("-")
        #if r>80:
        #    s  = 0
        print(f"Оптимальный радиус поворота {r}")
        print(f"Оптимальный угол поворота в AirSim {s}")

        # DrawLine Airsim!!!!
        return s,r

    def get_cone(self, R):
        cones_left_in = []
        cones_right_in = []
        pos_a = self.client.simGetVehiclePose().position.to_numpy_array()[0:2]
        cones_left_in = self.cones_left[np.where(np.linalg.norm(self.cones_left - pos_a, axis=1) < R)]
        cones_right_in = self.cones_right[np.where(np.linalg.norm(self.cones_right - pos_a, axis=1) < R)]
        return np.array(cones_left_in), np.array(cones_right_in)
    # получить точки дуги радиуса r с углом phi с углом alpha(в радианах) относительно OX в точке (x0,y0)
    def get_arc_pnts(self, r, phi, alpha, x0, y0, N=100):
        # начальная точка
        arc_pnts = [[x0, y0]]
        # первый край
        arc_pnts.append([x0 + r * np.cos(alpha - phi / 2), y0 + r * np.sin(alpha - phi / 2)])
        # дуга
        theta = np.linspace(alpha - phi / 2, alpha + phi / 2, num=N)
        arc_pnts = arc_pnts + [[x0 + r * np.cos(t), y0 + r * np.sin(t)] for t in theta]
        # второй край
        arc_pnts.append([x0 + r * np.cos(alpha + phi / 2), y0 + r * np.sin(alpha + phi / 2)])
        # возвращаемся

        arc_pnts.append(arc_pnts[0])
        return np.array(arc_pnts)

    # отобрать все точки, которые входят в данный сектор радиуса r и углом phi и направлением alpha
    def find_in_contour(self, points, arc_pnts, r, phi, x0, y0, alpha, epsilon_1=10 ** (-2), epsilon_2=10 ** (-2)):
        # 2 точка и предпоследняя это границы, тогда вектор направления
        arc_pnts_ = arc_pnts
        xd2 = (arc_pnts_[1][0] + arc_pnts_[-2][0]) / 2
        yd2 = (arc_pnts_[1][1] + arc_pnts_[-2][1]) / 2
        direct_vec = [xd2, yd2]
        # по расстоянию
        inds_r = np.where(np.linalg.norm(points - arc_pnts[0], axis=1) <= r * (1 + epsilon_1))
        points_ir = points[inds_r]
        # по углу
        angles = np.arccos(np.matmul(points_ir - arc_pnts[0], direct_vec - arc_pnts[0])
                           / (np.linalg.norm(points_ir - arc_pnts[0], axis=1) * np.linalg.norm(
            direct_vec - arc_pnts[0])))
        # & поэлеиентное и and сравнивает весь
        ind_a = np.where((angles <= phi * (epsilon_2 + 1) / 2) & (-phi * (epsilon_2 + 1) / 2 <= angles))
        points_a = points_ir[ind_a]
        return points_a


class SimpleDriver:
    def __init__(self):
        self.pinholeCamera = None
        self.cameraDataProccesing = None
        self.lidar_data_proccesing = None
        self.cone_center = None
        self.objective_func = None
        self.FOV_r = None
        self.DOV = None
        self.is_init = False
        pass

    def get_data(self):
        return self.cone_center

    def get_objective_func(self):
        return self.objective_func

    def Control_Simple(self, image: np.ndarray, lidar_data):
        # Разделили на картинке левые и правые конусы(они отличаются по цвету)
        img_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        img_tresh_low, img_tresh_high = self.cameraDataProccesing.color_clustering(img_HSV)
        # Получаем прямоугольники, ограничивающие контуры
        cntr_ps_low, bounding_rects_low, cones_low = self.cameraDataProccesing.get_bounding_rect(img_tresh_low, False)
        cntr_ps_high, bounding_rects_high, cones_high = self.cameraDataProccesing.get_bounding_rect(img_tresh_high,
                                                                                                    False)
        # Берем только первые две пары передних
        max_area_ind = lambda cones: np.argmax(list(map(lambda ch: cv2.contourArea(ch), cones)))
        n_max_area_ind = lambda cones: np.flip(np.argsort(list(map(lambda ch: cv2.contourArea(ch), cones))))[0:2]

        ind_low = n_max_area_ind(cones_low)
        ind_high = n_max_area_ind(cones_high)

        cntr_ps = []
        for ind in ind_low:
            # contur1 = np.append(cones_low[ind], [cones_high[ind][0]], axis=0)
            contur1 = cones_low[ind]
            cntr_ps.append([contur1[:, :, 0].mean(), contur1[:, :, 1].mean()])
            # cntr_ps.append((cntr_ps_low[ind]))
        for ind in ind_high:
            # contur1 = np.append(cones_high[ind], cones_high[ind][0], axis=0)
            contur1 = cones_high[ind]
            cntr_ps.append([contur1[:, :, 0].mean(), contur1[:, :, 1].mean()])
            # cntr_ps.append((cntr_ps_high[ind]))

        cntr_ps = np.array(cntr_ps)

        # Используя опорные цвета сегментации AirSim  ставим метки. Нужны для SVM
        # Нужно подавать RGB изображение
        labels = self.cameraDataProccesing.get_label(image, cntr_ps)

        # lidar_data_proccesing
        # lidar_data_proccesing.plot_disntane_function(lidar_data[index])
        print(np.array(lidar_data).shape)
        cone_center, dist = self.lidar_data_proccesing.get_cone_info(lidar_data)
        approx_cone_center = self.lidar_data_proccesing.approx_data(cone_center)
        self.cone_center = approx_cone_center
        # lidar_data_proccesing.visulation_cone(approx_cone_center)

        # PinholeCamera
        space_cooord = []
        for pnt in cntr_ps:
            coord = self.pinholeCamera.get_spaces_coord(pnt) / 100
            # coord = rotate_point(0,0,0*orientation.Z[index],coord)
            coord = [coord[0], coord[1] - 0 * 230 / 100]
            # print(coord)
            space_cooord.append(coord)

        # SVM
        Y_train = labels
        X_train = space_cooord[0:len(Y_train)]
        X_test = approx_cone_center

        from sklearn import svm

        C = 1.0  # = self._alpha in our algorithm
        model1 = svm.SVC(kernel='linear', C=C)
        # model1 = svm.LinearSVC(C=C, max_iter=10000)
        # model1 = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        # model1 = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)

        model1.fit(X_train, Y_train)
        y_predict = model1.predict(X_test)

        # distance_0 = self.lidar_data_proccesing.calculate_distance_cones(approx_cone_center[y_predict == 0])
        # distance_1 = self.lidar_data_proccesing.calculate_distance_cones(approx_cone_center[y_predict == 1])

        # Находим оптимальный угол поворота
        from scipy.optimize import minimize
        # TODO построить график
        # Веткоризовать
        def objective(pnts, r):
            # v = 0
            # for pnt in pnts:
            #    v = v + ((pnt[0] ** 2 + (pnt[1] - r) ** 2) ** (1 / 2) - r) ** 2
            # return -v
            return -np.sum((pnts[:, 0] ** 2 - ((pnts[:, 1] - r) ** 2) ** (1 / 2) - r) ** 2)

        # № Диапозон указать
        # TODO  две задачи оптимизации при +-r у одноо полоиждтельный и другой отрицательй (уже не надо скоорее  всего)
        # TODO  Не учитвать те которые не в угле обзоре
        # График
        print(approx_cone_center.shape)
        f = lambda r: objective(approx_cone_center, r)
        self.objective_func = f
        result = minimize(f, [0], method='nelder-mead')
        s = np.arctan(1 / (result.x))
        print(f"Оптимальный радиус поворота {result.x}")
        print(f"Оптимальный угол поворота в AirSim {s}")
        return s

    def model_init(self, control_point, FOV_d=120, DOV=16):
        self.pinholeCamera = PinholeCamera(control_point)
        self.cameraDataProccesing = CameraDataProccesing(True, 0)
        self.lidar_data_proccesing = LidarDataProccesing(False)
        self.FOV_r = np.radians(FOV_d)
        self.DOV = DOV
        self.is_init = True
