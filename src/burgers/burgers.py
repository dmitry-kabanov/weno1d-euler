import numpy as np
import scipy.optimize as sco


def flux(val):
    return (val ** 2) / 2.0

def flux_deriv(val):
    return val

def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

def initial_condition(x_center):
    u0 = 0.5 + np.sin(np.pi * x_center)

    return u0

def exact_solution(x, time, mesh_size):
    u_exact = np.zeros(mesh_size)

    for i in (range(mesh_size)):
        u_exact[i] = sco.newton(exact_func, 0.5, fprime=exact_func_prime, args=(x[i], time), maxiter=5000)

    return u_exact

def exact_func(u, x, t):
    return 0.5 + np.sin(np.pi * (x - u * t)) - u

def exact_func_prime(u, x, t):
    return -np.pi * t * np.cos(np.pi * (x - u * t)) - 1
