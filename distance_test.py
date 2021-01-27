import airsim
import numpy as np
import os
import datetime

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
client.timeout_value = 5
# Могу ли я управлять машиной из кода?
while True:
    print(client.ping())
print("Stop!")