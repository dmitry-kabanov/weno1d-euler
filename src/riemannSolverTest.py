import os
import numpy as np
import matplotlib.pyplot as plt
import riemannSolver as rs


def solve_riemann_problem(leftSide, rightSide, time, title, bounds):
    riemannSolver = rs.riemannSolver()
    rho0, u0, p0 = riemannSolver.solve(leftSide, rightSide, gamma)

    print("{0} solution:".format(title.capitalize()))
    print("Rho = {}".format(rho0))
    print("U   = {}".format(u0))
    print("P   = {}".format(p0))

    x = np.linspace(lb, rb, n)
    rho = np.zeros(x.size)
    u = np.zeros(x.size)
    p = np.zeros(x.size)

    for i in range(x.size):
        rho[i], u[i], p[i] = riemannSolver.sample_solution(x[i] / time)

    plt.plot(x, p)
    plt.xlim([lb, rb])
    plt.ylim(bounds["p"])
    plt.savefig(dir + title + "_pressure.eps")

dir = "images/riemann-solver-tests/"
if not os.path.exists(dir):
    os.makedirs(dir)

# Specific heats ratio (the same for all tests)
gamma = 1.4
# Left boundary
lb = -0.5
# Right boundary
rb = 0.5
# Number of mesh points
n = 400

sod_left = {'rho': 1.0, 'u': 0.0, 'p': 1.0}
sod_right = {'rho': 0.125, 'u': 0.0, 'p': 0.1}
time = 0.25
title = "test1"
bounds = {'p': [0.0, 1.1]}
solve_riemann_problem(sod_left, sod_right, time, title, bounds)

test2_left = {'rho': 1.0, 'u': -2.0, 'p': 0.4}
test2_right = {'rho': 1.0, 'u': 2.0, 'p': 0.4}
time = 0.15
title = "test2"
bounds = {'p': [-0.1, 0.5]}
#solve_riemann_problem(test2_left, test2_right, time, title, bounds)

test3_left = {'rho': 1.0, 'u': 0.0, 'p': 1000.0}
test3_right = {'rho': 1.0, 'u': 0.0, 'p': 0.01}
time = 0.012
title = "test3"
bounds = {'p': [0.0, 1000.0]}
solve_riemann_problem(test3_left, test3_right, time, title, bounds)

test4_left = {'rho': 1.0, 'u': 0.0, 'p': 0.01}
test4_right = {'rho': 1.0, 'u': 0.0, 'p': 100.0}
time = 0.035
title = "test4"
bounds = {'p': [0.0, 110.0]}
solve_riemann_problem(test4_left, test4_right, time, title, bounds)

test5_left = {'rho': 5.99924, 'u': 19.5975, 'p': 460.894}
test5_right = {'rho': 5.99242, 'u': -6.19633, 'p': 46.0950}
time = 0.035
title = "test5"
bounds = {'p': [0.0, 2000.0]}
solve_riemann_problem(test3_left, test3_right, time, title, bounds)
