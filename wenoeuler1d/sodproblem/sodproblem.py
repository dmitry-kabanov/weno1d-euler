import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.getcwd())

import wenoeuler1d.weno5solver.weno3 as w3
import wenoeuler1d.weno5solver.riemannSolver as rs

np.seterr(all='raise')

N = 100
GAMMA = 1.4
T = 0.2
lb = 0.0
rb = 1.0
discontinuity_position = 0.3
p = np.zeros(N)
u = np.zeros(N)
rho = np.zeros(N)
ic = np.zeros(3 * N)

ws = w3.Weno3(lb, rb, N, GAMMA, cfl_number=0.3)
x = ws.get_x_center()

sod_left = {'rho': 1.0, 'u': 0.75, 'p': 1.0}
sod_right = {'rho': 0.125, 'u': 0.0, 'p': 0.1}

for i in range(0, N):
    if x[i] < discontinuity_position:
        rho[i] = sod_left["rho"]
        u[i] = sod_left["u"]
        p[i] = sod_left["p"]
    else:
        rho[i] = sod_right["rho"]
        u[i] = sod_right["u"]
        p[i] = sod_right["p"]

ic[0:N] = rho
ic[N:2*N] = u
ic[2*N:3*N] = p

solution = ws.integrate(ic, T)
rho = solution[0:N]
u = solution[N:2*N] / rho
etotal = solution[2*N:3*N] / rho
pressure = (etotal - u**2 / 2.0) * (GAMMA - 1.0) * rho

riemannSolver = rs.riemannSolver()
riemannSolver.solve(sod_left, sod_right, gamma=GAMMA)
rho_rs = np.zeros(x.size)
u_rs = np.zeros(x.size)
p_rs = np.zeros(x.size)

for i in range(x.size):
    coord = x[i] - discontinuity_position
    rho_rs[i], u_rs[i], p_rs[i] = riemannSolver.sample_solution(coord / T)

plt.plot(x, pressure, 'o')
plt.plot(x, p_rs, '-')
plt.show()
