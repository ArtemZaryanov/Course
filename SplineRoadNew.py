import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.misc import derivative
import pandas as pd
import os
import subprocess
def _koefx(s, x0, derivate):
    return s * np.sqrt(1 / (1 + np.square(1 / derivate)))


def _coef_y(s, x0, derivate):
    return -1 * (1 / derivate) * _koefx(s, x0, derivate)


class SplineRoad:
    def __init__(self, s=0.20, a=10000, b=10000, start=0, end=1, count_cone=100):
        self.s = s
        self.a = a
        self.b = b
        self._maxfunc = None
        self.cs = None
        self.cs_norm = None
        self.xxx = None
        self.yyy = None
        self.start = start
        self.end = end
        self.count_cone = count_cone
        self.xxc = None
        self.yyc = None
        self.lcx = None
        self.lcy = None
        self.rcx = None
        self.rcy = None
        self.is_generated_data = False

    def _function(self, xx):
        return self.cs_norm(xx)

    def epicycloid_track_generate_data(self,s = 0.05,k=4, output=True):
        theta = np.linspace(0,2*np.pi,2*self.count_cone)
        r = 1
        x = r * (k + 1) * (np.cos(theta) - (np.cos((k + 1) * theta)) / (k + 1))
        y = r * (k + 1) * (np.sin(theta) - (np.sin((k + 1) * theta)) / (k + 1))
        xi = x * (1 - s)
        yi = y * (1 - s)
        self.lcx = x
        self.lcy = y
        self.rcx = xi
        self.rcy = yi
        self.is_generated_data = True
        if output:
            return np.array([[self.lcx, self.lcy], [self.rcx, self.rcy]]), np.array(
                [self.xxc, self.yyc])


    def standard_track_generate_data(self,s = 0.2,output = True):
        # Полуокружности
        thetaR = np.linspace(0, np.pi, 3 * self.count_cone // 10)
        cRx = np.sin(thetaR) + 1
        cRy = np.cos(thetaR)
        cRxi = (1 - s) * np.sin(thetaR) + 1
        cRyi = (1 - s) * np.cos(thetaR)
        thetaL = np.linspace(np.pi, 2 * np.pi, 3 * self.count_cone // 10)
        cLx = np.sin(thetaL) - 1
        cLy = np.cos(thetaL)
        cLxi = (1 - s) * np.sin(thetaL) - 1
        cLyi = (1 - s) * np.cos(thetaL)

        lx = np.linspace(-1, 1, self.count_cone // 5)
        lUy = np.ones(lx.shape[0]) * cLy[-1]
        lUyi = np.ones(lx.shape[0]) * cLyi[-1]
        lDy = np.ones(lx.shape[0]) * cLy[0]
        lDyi = np.ones(lx.shape[0]) * cLyi[0]

        self.lcx = np.concatenate([cLx, lx, lx, cRx])
        self.lcy = np.concatenate([cLy, lUy, lDy, cRy])
        self.rcx = np.concatenate([cLxi, lx, lx, cRxi])
        self.rcy = np.concatenate([cLyi, lUyi, lDyi, cRyi])

        start_point = np.array([self.transform_x(0),self.transform_y (1 + 0.75) / 2])
        start_direct = 0
        start_data = pd.DataFrame(
            {'X': [start_point[0]],'Y': [start_point[1]], "direct":[start_direct]})
        start_data.to_csv("start_data.csv")
        self.is_generated_data = True
        if output:
            return np.array([[self.lcx, self.lcy], [self.rcx, self.rcy]]), np.array(
                [self.xxc, self.yyc])

    def splev_spline_data(self,x, y,point_count):
        # append the starting x,y coordinates
        x_ = np.r_[x, x[0]]
        y_ = np.r_[y, y[0]]

        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = splprep([x_, y_], s=0, per=True)

        # evaluate the spline fits for 1000 evenly spaced distance values
        xi_, yi_ = splev(np.linspace(0, 1, point_count), tck)
        return xi_, yi_

    def track_eight_generate_data(self, output = True):
        x = np.array([-1, -1.2, -1, -0.1])
        y = np.array([-0.8, 0, 0.8, 0])
        xi, yi = self.splev_spline_data(x, y,self.count_cone//2)
        x1 = np.array([1, 1.2, 1, 0.1
                       ])
        y1 = np.array([-0.8, 0, 0.8, 0])
        x1i, y1i = self.splev_spline_data(x1, y1,self.count_cone//2)

        x2 = np.array([-1, -1,
                       0,
                       1, 1, 0
                       ])
        y2 = np.array([-1, 1,
                       0.6,
                       1, -1, -0.6])
        x2i, y2i = self.splev_spline_data(x2, y2,self.count_cone)

        self.lcx = np.concatenate([x2i])
        self.lcy = np.concatenate([y2i])

        self.rcx = np.concatenate([xi, x1i])
        self.rcy = np.concatenate([y1i, y1i])
        self.is_generated_data = True
        if output:
            return np.array([[self.lcx, self.lcy], [self.rcx, self.rcy]]), np.array(
                [self.xxc, self.yyc])

    def random_track_generate_data(self, output=True):
        x1 = np.array([-1, -0.8,
                       -0.6, -0.4,
                       -0.2, 0,
                       0.2, 0.4,
                       0.6, 0.8,
                       1, 0.8,
                       0.6, 0.4,
                       0.2, 0,
                       -0.2, -0.4,
                       -0.6, -0.8
                       ])
        y1 = np.array([0, 0.8,
                       0.4, 0.8,
                       0.4, 0.8,
                       0.4, 0.8,
                       0.4, 0.8,
                       0, -0.8,
                       -0.4, -0.8,
                       -0.4, -0.8,
                       -0.4, -0.8,
                       -0.4, -0.8,
                       ])
        x1i, y1i = self.splev_spline_data(x1, y1,self.count_cone*2)

        x2 = np.array([-0.9, -0.8,
                       -0.6, -0.4,
                       -0.2, 0,
                       0.2, 0.4,
                       0.6, 0.8,
                       0.9, 0.8,
                       0.6, 0.4,
                       0.2, 0,
                       -0.2, -0.4,
                       -0.6, -0.8
                       ])
        y2 = np.array([0, 0.6,
                       0.1, 0.6,
                       0.1, 0.6,
                       0.1, 0.6,
                       0.1, 0.6,
                       0, -0.6,
                       -0.1, -0.6,
                       -0.1, -0.6,
                       -0.1, -0.6,
                       -0.1, -0.6,
                       ])
        x2i, y2i = self.splev_spline_data(x2, y2,self.count_cone*2)
        self.lcx = np.concatenate([x1i])
        self.lcy = np.concatenate([y1i])

        self.rcx = np.concatenate([x2i])
        self.rcy = np.concatenate([y2i])

        self.is_generated_data = True
        if output:
            return np.array([[self.lcx, self.lcy], [self.rcx, self.rcy]]), np.array(
                [self.xxc, self.yyc])

    def polar_function_generate_data(self, output=True):

        r = np.arange(0, 2, 0.01)
        theta = 2 * np.pi * r
        # В UnrealEngine4 в см
        # Конусы
        self.xxc = r * np.cos(theta)
        self.ycc = r * np.sin(theta)

        self.lcx = r * np.cos(theta - self.s)
        self.lcy = r * np.sin(theta - self.s)

        self.rcx = r * np.cos(theta + self.s)
        self.rcy = r * np.sin(theta + self.s)
        self.is_generated_data = True
        if output:
            return np.array([[self.lcx, self.lcy], [self.rcx, self.rcy]]), np.array(
                [self.xxc, self.ycc])

    def generate_data(self, output=True):
        # 1 Генерация случайных данных
        # np.random.seed(1000)
        data_x = np.linspace(0,1,7,endpoint=True)
        data_y = np.concatenate([[0], np.random.rand(5), [1]])
        self.xxx = np.linspace(0, 1, 1000)
        self.cs = CubicHermiteSpline(data_x, data_y,np.zeros(7)+1)
        self._maxfunc = np.max(self.cs(self.xxx))
        self.cs_norm = lambda x: self.cs(x)
        self.xxx = self.a * self.xxx
        # В UnrealEngine4 в см
        # Конусы

        self.xxc = np.linspace(self.start, self.end, self.count_cone)[0::]
        self.ycc = self._function(self.xxc)
        deriv = derivative(self._function, self.xxc)
        self.lcx = self.xxc  # + _koefx(self.s, self.xxc, deriv)
        self.lcy = self._function(self.xxc) + self.s / 2  # _coef_y(self.s, self.xxc, deriv)
        self.rcx = self.xxc  # - _koefx(self.s, self.xxc, deriv)
        self.rcy = self._function(self.xxc) - self.s / 2  # - _coef_y(self.s, self.xxc, deriv)
        self.is_generated_data = True
        if output:
            return np.array([[self.lcx, self.lcy], [self.rcx, self.rcy]]), np.array(
                [self.xxc, self._function(self.xxc)])
    def move_data(self):
        assert self.is_generated_data == True, "no data is generated"
        # data_left_cone = pd.DataFrame({'X': self.transform_x(self.lcx-self.s), 'Y': self.transform_y(self.lcy-self.s/2), 'isPhysics': False})
        # data_right_cone = pd.DataFrame({'X': self.transform_x(self.rcx +self.s), 'Y': self.transform_y(self.rcy + self.s/2), 'isPhysics': False})
        # data_central =    pd.DataFrame({'X':self.transform_x(self.xxc), 'Y':self.transform_y(self.ycc), 'isPhysics': False})
        data_left_cone = pd.DataFrame({'X': self.transform_x(self.lcx), 'Y': self.transform_y(self.lcy), 'isPhysics': False})
        if self.xxc is not None:
            data_central = pd.DataFrame({'X': self.transform_x(self.xxc), 'Y': self.transform_y(self.ycc), 'isPhysics': False})
        else:
            data_central = pd.DataFrame(
                {'X': self.transform_x([0]), 'Y': self.transform_y([0]), 'isPhysics': False})
        data_right_cone = pd.DataFrame({'X': self.transform_x(self.rcx), 'Y': self.transform_y(self.rcy), 'isPhysics': False})
        data_left_cone.to_csv("data_left_cone.csv")
        data_right_cone.to_csv("data_right_cone.csv")
        data_central.to_csv("data_central.csv")
        path_from = os.getcwd() + "\\*.csv "
        path_to = "D:\\Users\\user\\Documents\\Course\\MyProject2\\Content"
        cmd = "copy" + " " + path_from + " " + path_to + "/y"
        # print(cmd)
        returned_output = subprocess.check_output(cmd,
                                                  shell=True)  # returned_output содержит вывод в виде строки байтов

        print('Результат выполнения команды:', returned_output.decode("CP866"))  # Преобразуем байты в строку

    def plot_cones(self, plot_func: bool = True, accuracy: int = 200):
        assert self.is_generated_data == True, "no data is generated"
        if plot_func:
            xx = np.linspace(self.start, self.end, accuracy)
            yy = self._function(xx)
            yy1 = yy + self.s / 2
            yy2 = yy - self.s / 2
            plt.plot(xx, yy1, c='Black')
            plt.plot(xx, yy2, c='Black')
            plt.plot(xx, yy, c="Blue")

        plt.plot(self.transform_x(self.lcx), self.transform_y(self.lcy), 'o', c='Red')
        if self.xxc is not None:
            plt.plot(self.transform_x(self.xxc), self.transform_y(self.ycc), 'o', c='Red')
        plt.plot(self.transform_x(self.rcx), self.transform_y(self.rcy), 'o', c='Red')
        plt.show()

    def transform_x(self,k):
        return self.a*k
    def transform_y(self,k):
        return self.b*k