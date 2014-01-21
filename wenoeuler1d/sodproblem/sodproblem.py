import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.getcwd() + '/src')
print(sys.path)

import weno3 as w3

N = 1000
GAMMA = 1.4
T = 0.25
lb = 0.0
rb = 1.0
p = np.zeros(N)
u = np.zeros(N)
rho = np.zeros(N)
ic = np.zeros(3 * N)

ws = w3.Weno3(lb, rb, N, GAMMA)
x = ws.get_x_center()

for i in range(0, N):
    if x[i] < 0.5:
        p[i] = 1.0
        rho[i] = 1.0
    else:
        p[i] = 0.1
        rho[i] = 0.125

    u[:] = 0.0

ic[0:N] = rho
ic[N:2*N] = u
ic[2*N:3*N] = p

solution = ws.integrate(ic, T)
plt.plot(x, solution)
plt.show()
