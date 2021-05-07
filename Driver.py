from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import SplineRoadNew
import os
from CameraProcessing import CameraDataProccesing
from CameraProcessing import LidarDataProccesing
from CameraProcessing import PinholeCamera

def getRandomNumber():
    return 4

class TrackType:
    Standart = 0
    Epicycloid = 1
    Eight = 2
    Random = 3
    PolarFunction = 4
    SplevSpline = 5
class DriverPID:
    def __init__(self,type_trace):
        track_function = None
        track_points = None
        SR = SplineRoadNew.SplineRoad(count_cone=150)
        if type_trace == TrackType.Standart:
            SR.standard_track_generate_data(s=0.07, a=0)


class DriverCNN:

    def __init__(self):
        self.model_CNN = None
        pass

    # После инициализации модели. Необходимо подать картинку,
    # чтобы получить  вектор управления
    def Control_CNN(self, image: np.ndarray):
        # assert self.model_CNN is None, "model is None"
        return self.model_CNN.predict_proba([image[np.newaxis]]).argmax() - 1  # Пока так!!!!!

    # Иннициализация модели
    def model_CNN_init(self, path):
        self.model_CNN = keras.models.load_model(path)
        self.model_CNN.summary()
        print(self.model_CNN)

    # Преобразование выходов сети
    def _convert_CNN(self, output):
        pass


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
