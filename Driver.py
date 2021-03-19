from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os
from CameraProcessing import CameraDataProccesing
from CameraProcessing import LidarDataProccesing
from CameraProcessing import PinholeCamera


def getRandomNumber():
    return 4


class Driver:

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
        self.is_init = False
        pass

    def Control_Simple(self, image: np.ndarray, lidar_data):
        # Разделили на картинке левые и правые конусы(они отличаются по цвету)
        img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
            cntr_ps.append((cntr_ps_low[ind]))
        for ind in ind_high:
            cntr_ps.append((cntr_ps_high[ind]))
        cntr_ps = np.array(cntr_ps)

        # Используя опорные цвета сегментации AirSim  ставим метки. Нужны для SVM
        labels = self.cameraDataProccesing.get_label(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cntr_ps)

        # lidar_data_proccesing
        # lidar_data_proccesing.plot_disntane_function(lidar_data[index])
        cone_center, dist = self.lidar_data_proccesing.get_cone_info(lidar_data)
        approx_cone_center = self.lidar_data_proccesing.approx_data(cone_center)
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
        X_train = space_cooord
        Y_train = labels
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

        def objective(Nt, pnts, r):
            v = 0
            for pnt in pnts:
                v = v + ((pnt[0] ** 2 + (pnt[1] - r) ** 2) ** (1 / 2) - r) ** 2
            return v

        f = lambda r: objective(approx_cone_center.shape[0], approx_cone_center, r)
        result = minimize(f, 0, method='nelder-mead')
        phi = 1 / np.tan(result.x)
        print(f"Оптимальный угол поворота {phi}")
        return phi

    def model_init(self, control_point):
        self.pinholeCamera = PinholeCamera(control_point)
        self.cameraDataProccesing = CameraDataProccesing(True, 0)
        self.lidar_data_proccesing = LidarDataProccesing(False)
        self.is_init = True
