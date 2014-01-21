import matplotlib.pyplot as plt
import matplotlib.animation as animation
from burgers import burgers
import weno3
import bootstrap as bs

bs.bootstrap()

N = 320
a = -1.0
b = 1.0
CHAR_SPEED = 1.0

w = weno3.Weno3(a, b, N, burgers.flux, burgers.flux_deriv, burgers.max_flux_deriv, CHAR_SPEED, cfl_number=0.6)
x_center = w.get_x_center()
u0 = burgers.initial_condition(x_center)
solution = u0

t = 0.0
nFrames = 200
t_final = 3.0
dt = t_final / nFrames

fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-0.55, 1.55))
line, = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# initialization function: plot the background of each frame
def init():
    line.set_data(x_center, u0)
    time_text.set_text('time = %s' % 0.0)
    return line, time_text

# animation function.  This is called sequentially
def animate(i):
    global x_center, solution, w
    solution = w.integrate(solution, dt * i)
    line.set_data(x_center, solution)
    time_text.set_text('time = %s' % (dt * i))
    return line, time_text

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nFrames, interval=20, blit=True)

anim.save(bs.params['videos_path'] + '/burgers_weno3.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
