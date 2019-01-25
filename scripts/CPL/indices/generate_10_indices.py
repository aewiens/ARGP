#!/usr/bin/env python3
import os
import numpy as np
 
np.random.seed(1)
 
# Load ab initio surface
E0 = np.loadtxt("/Users/averywiens/git/NARGP-manuscript/third-submission/N2/surfaces/cas-631g.tab")
E = np.loadtxt("surfaces/mrci-pcv5z.tab")
 
# Level 0 training set
train0 = np.random.randint(0, len(E), size=30)
np.save("train0_10.npy", train0)
T0 = np.array([E0[i] for i in train0])
X0, Y0 = np.split(T0, 2, axis=1)
 
# Level 2 training set
train = np.random.choice(train0, size=10)
np.save("train1_10.npy", train)
test = np.array([i for i in range(len(E)) if i not in train])
