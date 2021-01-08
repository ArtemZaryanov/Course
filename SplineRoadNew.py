import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.misc import derivative
import pandas as pd
import os
import subprocess
def _koefx(s, x0, derivate):
    return s * np.sqrt(1 / (1 + np.square(1 / derivate)))


def _coef_y(s, x0, derivate):
    return -1 * (1 / derivate) * _koefx(s, x0, derivate)


class SplineRoad:
    def __init__(self, s=0.25, a=80000, b=10000, start=0, end=1, count_cone=100):
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

    def generate_data(self, output=True):
        # 1 Генерация случайных данных
        np.random.seed(1000)
        data_x = np.concatenate([[0], np.sort(np.random.rand(5)), [1]])
        data_y = data_x**2*np.random.
        self.xxx = np.linspace(0, 1, 1000)
        self.cs = CubicSpline(data_x, data_y)
        self._maxfunc = np.max(self.cs(self.xxx))
        self.cs_norm = lambda x: self.cs(x)
        self.xxx = self.a * self.xxx
        # В UnrealEngine4 в см
        # Конусы

        self.xxc = np.linspace(self.start, self.end, self.count_cone)[0::]
        self.ycc = self._function(self.xxc)
        print(self.ycc)
        deriv = derivative(self._function, self.xxc)
        self.lcx = self.xxc + _koefx(self.s, self.xxc, deriv)
        self.lcy = self._function(self.xxc) + _coef_y(self.s, self.xxc, deriv)
        self.rcx = self.xxc - _koefx(self.s, self.xxc, deriv)
        self.rcy = self._function(self.xxc) - _coef_y(self.s, self.xxc, deriv)
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
        data_right_cone = pd.DataFrame({'X': self.transform_x(self.rcx), 'Y': self.transform_y(self.rcy), 'isPhysics': False})
        data_central =    pd.DataFrame({'X':self.transform_x(self.xxc), 'Y':self.transform_y(self.ycc), 'isPhysics': False})
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
        xx = np.linspace(self.start, self.end, accuracy)
        yy = self._function(xx)
        yy1 = yy + self.s / 2
        yy2 = yy - self.s / 2
        if plot_func:
            plt.plot(xx, yy1, c='Black')
            plt.plot(xx, yy2, c='Black')
            plt.plot(xx, yy, c="Blue")

        plt.plot(self.transform_x(self.lcx), self.transform_y(self.lcy), 'o', c='Red')
        plt.plot(self.transform_x(self.xxc), self.transform_y(self.ycc), 'o', c='Red')
        plt.plot(self.transform_x(self.rcx), self.transform_y(self.rcy), 'o', c='Red')
        plt.show()

    def transform_x(self,k):
        return self.a*k
    def transform_y(self,k):
        return self.b*k