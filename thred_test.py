import numpy as np
import time
import matplotlib.pyplot as plt
# x = np.linspace(0, 10, 100)
# y = np.cos(x)

plt.ion()

figure, ax = plt.subplots(figsize=(8, 6))
line1, = ax.plot([], [])

plt.title("Dynamic Plot of sinx", fontsize=25)

plt.xlabel("X", fontsize=18)
plt.ylabel("sinX", fontsize=18)

while(True):
    updated_y = np.random.random()

    line1.set_xdata(np.random.random())
    line1.set_ydata(updated_y)
    print(np.random.random(),updated_y)
    figure.canvas.draw()

    figure.canvas.flush_events()
    time.sleep(0.1)


