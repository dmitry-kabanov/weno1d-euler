import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.optimize as sco
import weno2
import bootstrap as bs

bs.bootstrap()

def flux(val):
    return (val ** 2) / 2.0

def flux_deriv(val):
    return val

def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

def initial_condition(x_center):
    u0 = 0.5 + np.sin(np.pi * x_center)

    return u0

def exact_solution(x, time):
    x = np.remainder(x, 2.0)
    for i in range(N):
        if x[i] >= 1.0:
            x[i] -= 2.0

    u_exact = np.zeros(N)

    for i in (range(N)):
        u_exact[i] = sco.newton(exact_func, 0.5, args=(x[i], time))

    return u_exact

def exact_func(u, x, t):
    return 0.5 + np.sin(np.pi * (x - u * t)) - u


# Number of cells.
N = 160
a = -1.0
b = 1.0
CHAR_SPEED = 1.0

w = weno2.Weno2(a, b, N, flux, flux_deriv, max_flux_deriv, CHAR_SPEED, cfl_number=0.6)
x_center = w.get_x_center()
u0 = initial_condition(x_center)
solution = u0

t = 0.0
nFrames = 800
t_final = 20.0
dt = t_final / nFrames

fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    global x_center, solution, w
    solution = w.integrate(solution, dt * i)
    line.set_data(x_center, solution)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nFrames, interval=20, blit=True)

anim.save(bs.params['videos_path'] + 'burgers.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
