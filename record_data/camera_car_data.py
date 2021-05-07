# https://github.com/microsoft/AirSim/blob/master/PythonClient/car/car_lidar.py
# Python client example to get Lidar data from a car
#

import airsim
from PIL import Image

import os
import datetime
import time
import pprint
import numpy


# Makes the drone fly and get Lidar data
class CameraRecord:

    def __init__(self,client):
        self.ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"image_record")
        # connect to the AirSim simulator
        self.client = client
        self.path = None
        self.path_data = None
        self.is_start_recording = False
        self.data_file = None

    def record(self,ImageType=5):
        assert self.is_start_recording == True, "The recording didn't start"
        time_stamp, image = self.get_sim_image(ImageType)
        #time_stamp_s, time_stamp_ns = str(time.time()).split('.')
        # time_stamp = time_stamp_s + time_stamp_ns + "00"
        # self.write_to_file_sync(f"{lidarData.pose.position.x_val} {lidarData.pose.position.y_val} {lidarData.pose.position.z_val} {lidarData.time_stamp}\n")
        self.write_CameraImage_to_disk(image,os.path.join(self.path_data,time_stamp))

    def get_sim_image(self,ImageType=5):
        time_stamp_s, time_stamp_ns = str(time.time()).split('.')
        time_stamp = time_stamp_s + time_stamp_ns + "00"
        responses = self.client.simGetImages([
            airsim.ImageRequest(0, ImageType, False, False)])
        # print("Type %d, size %d" % (responses[0].image_type, len(responses[0].image_data_uint8)))
        img1d = numpy.fromstring(responses[0].image_data_uint8, dtype=numpy.uint8)

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        return time_stamp, img_rgb

    def start_recording(self):
        # Создать папку дата.время
        # Сохранять там файлы .npy в формате time_stamp
        # print(ROOT_DIR)
        date = datetime.datetime.now().__format__("%Y-%d-%m-%H-%M-%S")
        # print(os.mkdir(os.path.join(ROOT_DIR, date)))
        self.path = os.path.join(self.ROOT_DIR, date)
        os.mkdir(self.path)
        self.path_data = os.path.join(self.path, "data")
        os.mkdir(self.path_data)
        self.data_file = open(os.path.join(self.path,"image_record.txt"),'w')
        # Заголовок
        self.write_to_file_sync("X Y Z TimeStamp\n")
        self.is_start_recording = True

    def write_to_file_sync(self,str):
        self.data_file.write(str)
        self.data_file.flush()
        os.fsync(self.data_file.fileno())

    def stop_recording(self):
        pass

    def write_CameraImage_to_disk(self, image,name="point"):
        im = Image.fromarray(image)
        im.save(name + '.png')

    def stop(self):

        airsim.wait_key('Press any key to reset to original state')

        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")


class CameraDataProccesing:
    pass

