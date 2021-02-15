# https://github.com/microsoft/AirSim/blob/master/PythonClient/car/car_lidar.py
# Python client example to get Lidar data from a car
#

import airsim

import os
import datetime
import time
import pprint
import numpy


# Makes the drone fly and get Lidar data
class LidarTest:

    def __init__(self,client):

        self.ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"lidar_record")
        # connect to the AirSim simulator
        self.client = client
        self.path = None
        self.path_data = None
        self.is_start_recording = False
        self.data_file = None

    def record(self):
        assert self.is_start_recording == True, "The recording didn't start"
        time_stamp_s, time_stamp_ns = str(time.time()).split('.')
        time_stamp = time_stamp_s + time_stamp_ns + "00"
        lidarData = self.client.getLidarData()
        if (len(lidarData.point_cloud) < 3):
            print("\tNo points received from Lidar data")
        else:
            # print("\tReading %d: time_stamp: %d number_of_points: %d" % (i, lidarData.time_stamp, len(points)))
            # print("\t\tlidar position: %s" % (pprint.pformat(lidarData.pose.position)))
            # print("\t\tlidar orientation: %s" % (pprint.pformat(lidarData.pose.orientation)))
            points = self.parse_lidarData(lidarData)
            # self.write_to_file_sync(f"{lidarData.pose.position.x_val} {lidarData.pose.position.y_val} {lidarData.pose.position.z_val} {lidarData.time_stamp}\n")
            self.write_lidarData_to_disk(points,os.path.join(self.path_data,str(time_stamp)))

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
        self.data_file = open(os.path.join(self.path,"lidar_record.txt"),'w')
        # Заголовок
        self.write_to_file_sync("X Y Z TimeStamp\n")
        self.is_start_recording = True

    def write_to_file_sync(self,str):
        self.data_file.write(str)
        self.data_file.flush()
        os.fsync(self.data_file.fileno())

    def stop_recording(self):
        pass

    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
        points = numpy.reshape(points, (int(points.shape[0] / 3), 3))

        return points

    def write_lidarData_to_disk(self, points,name="point"):
        numpy.save(name,points)
        # TODO
        #print("not yet implemented")

    def stop(self):

        airsim.wait_key('Press any key to reset to original state')

        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")


class LidarDataProccesing:
    pass

