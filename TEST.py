import CurveController
import SplineRoad
# import SplineRoadNew
from Driver import Driver
import cv2
import os
from tensorflow import keras
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
path_model  = "D:/Users/user/PycharmProjects/Course/"
# Параметры симулятора
delta = 0.01
# Driver
SimpleCNN = Driver()
SimpleCNN.model_CNN_init(path_model + "SimpleCNN")
image = cv2.imread("img_0_5_1608409809137534600.png")
print(SimpleCNN.Control_CNN(image))
def
# SimpleCNN.Control_CNN()
#SR = SplineRoadNew.SplineRoad()
#SR.generate_data()
#SR.plot_cones(plot_func=False)
# SR.move_data()
# FR = SplineRoad.FunctionRoad()
# FR.generate_data()
# FR.plot_cones()
# FR.move_data()
