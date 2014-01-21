import numpy as np
import matplotlib.pyplot as plt
import burgers as b
import weno3
import bootstrap as bs

def plot_burgers_solution():
    plt.plot(x_center, solution, 'o', label='WENO, $k = 3$')
    if T <= time_breakage:
        plt.plot(x_center, b.exact_solution(x_center, T, ncells), label='Exact')
    # plt.legend(loc='best')
    plt.ylim([-0.55, 1.55])
    plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
    plt.title('Time = ' + str(T) + ', ncells = ' + str(ncells) + r', $\nu$ = ' + str(CFL_NUMBER))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    # plt.show()
    plt.savefig(bs.params['images_path'] + '/images/weno3_burgers_N=' + str(ncells) + '_T=' + str(T) + '.eps')

bs.bootstrap()

ncells = 640
left_boundary = -1.0
right_boundary = 1.0
CHAR_SPEED = 1.0
CFL_NUMBER = 0.6
time_breakage = 0.3183
T = 0.3183

w = weno3.Weno3(left_boundary, right_boundary, ncells, b.flux, b.flux_deriv, b.max_flux_deriv, CHAR_SPEED, CFL_NUMBER)
x_center = w.get_x_center()
u0 = b.initial_condition(x_center)

solution = w.integrate(u0, T)
plot_burgers_solution()
