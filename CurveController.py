# Данный класс реализвует простейший вариант автопилота автомобиля,предназначенный для движения по заданной кривой
"""
Нам даны:
    Кривая, заданная в виде функции f(x). В двльнейшем планируется перейти на сплайн

Идея следующая:
+- Трасса разделена на сегменты
+- Контроллер движения условно состоит из 3 стадий:
    1) Движение вперед. На участках трассы близких к прямым машина движется прямо к целоевой точке
    2) Поворот. На данной стадии происходит поворот машины с минимальным линейным движением
    3)Собственно поворот с движением
Данные стадии зависимы друг от друга. 3=1+2, но при этом 1 и 2 достаточно, но тогда путь модет быть не оптимальным
При правильно подобранных параметрах все три стадии должны дать в итоге оптимальный путь, который и будет браться в качестве baseline
Основная проблема. Ошибка , когда машина модет не находится на трассе. Частично это решает вводи эпсилон-окрестности вокруг активной точки
Машина должна двигаться непрерывно без рывков и прочее.
Нельзя и двигаться постоянно. Нужен контроль скоростию Ввод оптимальной скорости. Подбор силы и времени тромоза, чтобы скорость упала к следующим итерациям  или эе ввод передач
"""

# Если возникнут проблемы https://github.com/microsoft/AirSim/pull/2243
import numpy as np
import SplineRoad as SR
from scipy.misc import derivative


class Controller:

    def __init__(self, kProportional=0, kDerivative=0, kIntegral=0):
        self._kp = kProportional
        self._kd = kDerivative
        self._ki = kIntegral

    # error -отклонение от трассы
    # Подаем на вход steering, throttle
    # Получем их исправленные значения
    def PIDController(self, e=0.0, de=0.0, ie=0.0):
        """ return correct"""
        correct = self._kp * e + self._kd * de + self._ki * ie
        return correct

    def getControllerParams(self):
        return self._kp, self._kd, self._ki

    def setControllerParams(self, kp, kd, ki):
        self._kp = kp
        self._kd = kd
        self._ki = ki

    def CarState(self, client):
        State = client.getCarState()
        pos = State.kinematics_estimated.position.to_numpy_array()
        velocity = State.speed
        kinematics_estimated = State.kinematics_estimated
        return pos, velocity, kinematics_estimated


class VelocityControl(Controller):
    def __init__(self,e0,v0,error_velocity_i_0):
        super().__init__()
        self.e0 = e0
        self.v0 = v0
        self.errors_velocity_i = [error_velocity_i_0]

    def e_velocity_p(self,v: float, v_min: float, v_max: float):
        v_mm_half = (v_min + v_max) / 2
        if v <= v_min:
            return (self.e0 / ((self.v0 - v_max) ** 2)) * (v - v_min) ** 2
        if (v_min <= v) and (v_max >= v):
            return np.sin((v - v_min) ** 2 * (v_max - v) ** 2 / ((v_max - v_min) ** 4))  # Было нуль
        if v_max <= v:
            return -(v - v_max) ** 2

    def e_velocity_d(self,a: float, v: float, v_min: float, v_max: float):
        v_mm_half = (v_min + v_max) / 2
        if v <= v_min:
            return 2 * a * (self.e0 / ((self.v0 - v_max) ** 2)) * (v - v_min) ** 2
        if (v_min <= v) and (v_max >= v):
            return 2 * a / ((v_max - v_min) ** 4) * (
                    (v - v_min) * (v_max - v) ** 2
                    -
                    (v - v_min) ** 2 * (v_max - v)
            ) * np.cos((v - v_min) ** 2 * (v_max - v) ** 2 / ((v_max - v_min) ** 4))

        if v_max <= v:
            return -2 * a * (v - v_max) ** 2

    def e_velocity_i(self,v: float, v_min: float, v_max: float,delta):
        e_p = self.e_velocity_p(v, v_min, v_max)
        self.errors_velocity_i.append(e_p)
        e_v_i = sum(self.errors_velocity_i) * delta
        return e_v_i


class CurveControl(Controller):
    def __init__(self):
        super().__init__()

    def e_curve_p(self, client):
        pos, _, _ = self.CarState(client)
        l, p = SR.distance_to_point(pos)
        return l, p

    def e_curve_d(self, client):
        pos, _, kinematics = self.CarState(client)
        l, p = self.e_curve_p(client)
        vx = kinematics.linear_velocity.x_val
        vy = kinematics.linear_velocity.y_val
        f = SR.get_function()
        df = derivative(SR.get_function(), pos[0])
        _e = (2 / (l ** 2 + 1)) * ((pos[0] - p) * (vx) + (pos[1] - f(p)) * (vy))
        if abs(_e) > 10000:
            return 0
        else:
            return _e

    def e_curve_i(self, client, delta, erros_curve_i):
        e_p, _ = self.e_curve_p(client)
        erros_curve_i.append(e_p)
        if len(erros_curve_i) > 100:
            buff = erros_curve_i[-1]
            erros_curve_i = [buff]
        return sum(erros_curve_i) * delta