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

        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # connect to the AirSim simulator
        self.client = client
        self.path = None
        self.path_data = None
        self.is_start_recording = False
        self.lidar_record = []

    def record(self):
        assert self.is_start_recording == True, "The recording didn't start"
        lidarData = self.client.getLidarData()
        if (len(lidarData.point_cloud) < 3):
            print("\tNo points received from Lidar data")
        else:
            # print("\tReading %d: time_stamp: %d number_of_points: %d" % (i, lidarData.time_stamp, len(points)))
            # print("\t\tlidar position: %s" % (pprint.pformat(lidarData.pose.position)))
            # print("\t\tlidar orientation: %s" % (pprint.pformat(lidarData.pose.orientation)))
            points = self.parse_lidarData(lidarData)
            self.lidar_record.append([lidarData.time_stamp,lidarData.pose.position.x_val,
                                      lidarData.pose.position.y_val,
                                      lidarData.pose.position.z_val])
            self.write_lidarData_to_disk(points,os.path.join(self.path_data,str(lidarData.time_stamp)))

    def start_recording(self):
        # Создать папку дата.время
        # Сохранять там файлы .npy в формате time_stamp
        # print(ROOT_DIR)
        date = datetime.datetime.now().__format__("%Y-%d-%H-%M-%S")
        # print(os.mkdir(os.path.join(ROOT_DIR, date)))
        self.path = os.path.join(self.ROOT_DIR, date)
        os.mkdir(self.path)
        self.path_data = os.path.join(self.path, "data")
        os.mkdir(self.path_data)
        self.is_start_recording = True

    def stop_recording(self):
        numpy.save(os.path.join(self.path,"lidar_record")
                   ,numpy.array(self.lidar_record))

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

