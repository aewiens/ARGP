import numpy as np

test = np.loadtxt("ccsd-t-5z.dat", delimiter=',', skiprows=1)

E = test[::10, :]

print(len(E))

print(E[0])
