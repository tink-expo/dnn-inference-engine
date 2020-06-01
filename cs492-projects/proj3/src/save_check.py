import numpy as np

for i in range(40):
    p1 = './intermediate-1/layer_{}.npy'.format(i)
    p2 = './intermediate/layer_{}.npy'.format(i)
    print(abs(np.load(p1) - np.load(p2)).max())