import advection as advec
import advection_util as au
import bootstrap as bs

bs.bootstrap()

N = 640
T = 0.5
WENO_ORDER = 3

w = au.createWeno3Solver(N)
x_center = w.get_x_center()
u0 = advec.initial_condition_square_wave(x_center)

solution = w.integrate(u0, T)
exact_solution = advec.exact_solution_square_wave(x_center - advec.CHAR_SPEED * T, T, N)
au.plot_advection_solution(x_center, solution, exact_solution, T, N, WENO_ORDER)


