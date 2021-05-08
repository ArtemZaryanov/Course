import numpy as np
import matplotlib.pyplot as plt
import cv2
import SplineRoadNew
from scipy.interpolate import CubicSpline
SR = SplineRoadNew.SplineRoad(count_cone=150)
_, track_points = SR.standard_track_generate_data(s=0.07, a=0)
# print(track_points[0,:])
fig, ax = plt.subplots(2,1)
x = track_points[0,:]
y = track_points[1,:]
print(np.arange(0,x.shape[0]).shape)
ax[0].plot(x,y,'o')

tt = np.arange(0,x.shape[0])
cs = CubicSpline(np.arange(0,x.shape[0]), np.c_[x,y], bc_type='periodic')
cs.derivative()
# new_points = splev(u, tck)
ax[1].plot(cs(tt)[:, 0], cs(tt)[:, 1], 'o')
plt.show()