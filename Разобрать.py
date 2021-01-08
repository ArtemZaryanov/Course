# from SplineRoad import function
def direct_vector(client, object_name: str):
    # Двигаемся на малое расстояние, чтобы полуить текущий вектор направления. Его нужно определить лишь один раз\
    car_controls = airsim.CarControls()
    # car_state = client.getCarState()
    pos0 = client.simGetObjectPose(object_name).position.to_numpy_array()
    car_controls.throttle = 1
    client.setCarControls(car_controls)
    time.sleep(0.5)
    pos1 = client.simGetObjectPose(object_name).position.to_numpy_array()
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    car_controls = None
    return pos1 - pos0


def move_to_point_direct(client, object_name: str):
    car_controls = airsim.CarControls()
    car_controls.throttle = 1
    car_controls.steering = 0.1
    client.setCarControls(car_controls)
    time.sleep(2)
    car_controls.throttle = 0
    car_controls.steering = 0
    car_controls.handbrake = True
    client.setCarControls(car_controls)
    time.sleep(5)