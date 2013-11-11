import numpy as np
import matplotlib.pyplot as plt
import advection as advec
import advection_util as au
import bootstrap as bs

bs.bootstrap()

N = [10, 20, 40, 80, 160, 320]
T = 1.0

errorsList = []
dxList = []
for n in N:
    w = au.createWeno3Solver(n)
    x_center = w.get_x_center()
    x_boundary = w.get_x_boundary()
    u0 = advec.initial_condition_sine(x_center, x_boundary)
    solution = w.integrate(u0, T)
    exact = advec.exact_solution_sine(x_center - advec.CHAR_SPEED * T, T, n, x_boundary - advec.CHAR_SPEED * T)
    dx = w.get_dx()
    errorsList.append(np.linalg.norm(exact - solution, np.Inf))
    dxList.append(dx)


plt.figure(figsize=[10, 7.5])
plt.loglog(dxList, errorsList, '-o', label=r'$\|error\|_1$')
plt.loglog(dxList, [1e4* dx**5 for dx in dxList], '-s', label=r'1e4$\cdot \Delta x^5$')
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$L_{\infty}$-norm of error')
plt.legend(loc='upper left')
plt.show()
# plt.savefig(bs.params['images_path'] + '/advection_convergence_weno3.eps')

for i in range(0, len(errorsList)):
    print(N[i], errorsList[i])

