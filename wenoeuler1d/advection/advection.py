import numpy as np

a = -1.0
b = 1.0
CHAR_SPEED = 1.0
CFL_NUMBER = 0.8

def flux(val):
    return val

def flux_deriv(val):
    return CHAR_SPEED

def max_flux_deriv(a, b):
    return CHAR_SPEED

def initial_condition_square_wave(x_center):
    u0 = np.zeros(x_center.size)

    for i in range(0, x_center.size):
        if -0.2 <= x_center[i] <= 0.2:
            u0[i] = 1.0
        else:
            u0[i] = 0.0

    return u0

def exact_solution_square_wave(x, time, mesh_size):
    x = np.remainder(x, 2.0)
    for i in range(mesh_size):
        if x[i] >= 1.0:
            x[i] -= 2.0

    u_exact = np.zeros(mesh_size)

    for i in (range(mesh_size)):
        if -0.2 <= x[i] <= 0.2:
            u_exact[i] = 1.0

    return u_exact

def initial_condition_sine(x_center, x_boundary):
    u0 = np.zeros(x_center.size)

    for i in range(x_center.size):
        x_width = x_boundary[i+1] - x_boundary[i]
        u0[i] = -(np.cos(np.pi * x_boundary[i+1]) - np.cos(np.pi * x_boundary[i])) \
            / (np.pi * x_width)
    return u0


def exact_solution_sine(x, time, mesh_size, x_boundary):
    return initial_condition_sine(x, x_boundary)

