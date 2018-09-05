import numpy as np

f = np.loadtxt("temp.dat")

rmse = f[:, 1]
print(np.mean(rmse))


