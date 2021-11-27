from matplotlib import pyplot as plt
import numpy as np

t = np.arange(10000)

initial_lr = 0.0001

k = 2
decayed_rate = initial_lr * 0.9**(t / 200)

plt.plot(t, decayed_rate)

plt.show()
