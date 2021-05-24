import airsim
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Данные по лидару потом использваоть AirSim иначе долго считает
def get_cone(R):
    cones_left_in = []
    cones_right_in = []
    pos_a = client.simGetVehiclePose().position.to_numpy_array()[0:2]
    print("---",pos_a)
    cones_left_in = cones_left[np.where(np.linalg.norm(cones_left - pos_a, axis=1, ord=2) < R)]
    cones_right_in = cones_right[np.where(np.linalg.norm(cones_right - pos_a, axis=1, ord=2) < R)]
    # TODO можно соератить в два раза если размеры одинаковые
    # Ищем все левые конусы
    #for pose_c in cones_left:
        # pose_c = client.simGetObjectPose(cone_names).position.to_numpy_array()[0:2]
    #    if np.sqrt((pose_c[0]-pos_a[0])**2+(pose_c[1]-pos_a[1])**2) < R:
    #        cones_left_in.append(pose_c)
    # Ищем все правые конусы
    #for pose_c in cones_right:
    #    if np.sqrt((pose_c[0]-pos_a[0])**2+(pose_c[1]-pos_a[1])**2) < R:
    #        cones_right_in.append(pose_c)
    return np.array(cones_left_in), np.array(cones_right_in)


# получить точки дуги радиуса r с углом phi с углом alpha(в радианах) относительно OX в точке (x0,y0)
def get_arc_pnts(r, phi, alpha, x0, y0, N=100):
    # начальная точка
    arc_pnts = [[x0, y0]]
    # первый край
    arc_pnts.append([x0 + r * np.cos(alpha - phi / 2), y0 + r * np.sin(alpha - phi / 2)])
    # дуга
    theta = np.linspace(alpha - phi / 2, alpha + phi / 2, num=N)
    arc_pnts = arc_pnts + [[x0 + r * np.cos(t), y0 + r * np.sin(t)] for t in theta]
    # второй край
    arc_pnts.append([x0 + r * np.cos(alpha + phi / 2), y0 + r * np.sin(alpha + phi / 2)])
    # возвращаемся

    arc_pnts.append(arc_pnts[0])
    return np.array(arc_pnts)


# отобрать все точки, которые входят в данный сектор радиуса r и углом phi и направлением alpha
def find_in_contour(points, arc_pnts, r, phi, x0, y0, alpha, epsilon_1=10 ** (-1), epsilon_2=10 ** (-1)):
    # 2 точка и предпоследняя это границы, тогда вектор направления
    arc_pnts_ = arc_pnts
    xd2 = (arc_pnts_[1][0] + arc_pnts_[-2][0]) / 2
    yd2 = (arc_pnts_[1][1] + arc_pnts_[-2][1]) / 2
    direct_vec = [xd2, yd2]
    # по расстоянию
    print(direct_vec)
    inds_r = np.where(np.linalg.norm(points - arc_pnts[0], axis=1) <= r * (1 + epsilon_1))
    points_ir = points[inds_r]
    # по углу
    angles = np.arccos(np.matmul(points_ir - arc_pnts[0], direct_vec - arc_pnts[0])
                       / (np.linalg.norm(points_ir - arc_pnts[0], axis=1) * np.linalg.norm(direct_vec - arc_pnts[0])))
    # & поэлеиентное и and сравнивает весь
    ind_a = np.where((angles <= phi * (epsilon_2 + 1) / 2) & (-phi * (epsilon_2 + 1) / 2 <= angles))
    points_a = points_ir[ind_a]
    return points_a


client = airsim.CarClient()
client.confirmConnection()
# Могу ли я управлять машиной из кода?
start_data = pd.read_csv("start_data.csv")
position = airsim.Vector3r(start_data.X[0], start_data.Y[0] / 100, -1)
heading = airsim.utils.to_quaternion(0, 0, 0)
pose = airsim.Pose(position, heading)
client.simSetVehiclePose(pose, True)
time.sleep(2)
print(client.isApiControlEnabled())
pos_a = client.simGetVehiclePose().position.to_numpy_array()[0:2]
b_l = lambda n: client.simGetObjectPose(n).position.to_numpy_array()[0:2]
cones_left = np.array(list(map(b_l, client.simListSceneObjects('FloatingActor_($|[^a-zA-Z]+)'))))
cones_right = np.array(list(map(b_l, client.simListSceneObjects('FloatingActorYellow_($|[^a-zA-Z]+)'))))
start_time = time.time()
# print(cones_left)
R = 16
FOV = 180
orientation  = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
yaw = orientation[2]
arc_pnts = get_arc_pnts(R,np.radians(FOV),yaw,pos_a[0],pos_a[1])

cones_left_in_circle, cones_right_in_circle = get_cone(R)
cones_left_in_arc = find_in_contour(cones_left_in_circle,arc_pnts,R,np.radians(FOV),pos_a[0],pos_a[1],yaw)
cones_right_in_arc = find_in_contour(cones_right_in_circle,arc_pnts,R,np.radians(FOV),pos_a[0],pos_a[1],yaw)
print("- -- %s seconds ---" % (time.time() - start_time))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(cones_left[:, 0], cones_left[:, 1], 'o')
ax.plot(cones_right[:, 0], cones_right[:, 1], 'o')
ax.plot(pos_a[0], pos_a[1], 'o')
ax.plot(arc_pnts[:,0],arc_pnts[:,1])
theta = np.linspace(0, 2 * np.pi)
# ax.plot(pos_a[0] + R * np.cos(theta), pos_a[1] + R * np.sin(theta))
ax.plot(cones_left_in_arc[:,0],cones_left_in_arc[:,1],'o',c="Red")
ax.plot(cones_right_in_arc[:,0],cones_right_in_arc[:,1],'o',c="Red")
plt.show()
print(f"Всего конусов в радиусе {R}: левых {cones_left_in_arc.shape[0]}, правых {cones_right_in_arc.shape[0]}")
print(np.concatenate([cones_left_in_arc,cones_right_in_arc],axis=0).shape)