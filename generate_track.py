import SplineRoadNew
# from Driver import Driver
# from tensorflow import keras
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# path_model  = "D:/Users/user/PycharmProjects/Course/"
# Параметры симулятора
delta = 0.01
# Driver
#SimpleCNN = Driver()
#SimpleCNN.model_CNN_init(path_model + "SimpleCNN")
#image = cv2.imread("img_0_5_1608409809137534600.png")
#print(SimpleCNN.Control_CNN(image))
# SimpleCNN.Control_CNN()
SR = SplineRoadNew.SplineRoad(count_cone=100)
# SR.random_track_generate_data()
#SR.random_track_generate_data()
# SR.standard_track_generate_data(s= 0.09, a = 0.15)
#SR.track_eight_generate_data(s=0.1)
SR.s_track_generate_data(s=0.09)
# SR.test_track(s=0.05)
# SR.track_eight_generate_data()
# SR.polar_function_generate_data()
SR.plot_cones(plot_func=False)
# SR.move_data()
# FR = SplineRoad.FunctionRoad()
# FR.generate_data()
# FR.plot_cones()
SR.move_data()
