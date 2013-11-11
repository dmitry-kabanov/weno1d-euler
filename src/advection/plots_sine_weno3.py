import advection as advec
import advection_util as au
import bootstrap as bs
import numpy as np

bs.bootstrap()

N = 320
T = 1.0
WENO_ORDER = 3

w = au.createWeno3Solver(N)
x_center = w.get_x_center()
x_boundary = w.get_x_boundary()
u0 = advec.initial_condition_sine(x_center, x_boundary)

solution = w.integrate(u0, T)
exact_solution = advec.exact_solution_sine(x_center - advec.CHAR_SPEED * T, T, N, x_boundary - advec.CHAR_SPEED * T)
error = np.abs(solution - exact_solution)
print(np.argmax(error))
au.plot_advection_solution(x_center, solution, exact_solution, T, N, WENO_ORDER)


