import airsim
import numpy as np
import os
import datetime

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
# Могу ли я управлять машиной из кода?
