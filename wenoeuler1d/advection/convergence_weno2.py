import numpy as np
import matplotlib.pyplot as plt
import advection as advec
import weno2
import bootstrap as bs

bs.bootstrap()

N = [80, 160, 320, 640, 1280, 2560]
T = 1.0

errorsList = []
dxList = []
for n in N:
    w = weno2.Weno2(advec.a, advec.b, n, advec.flux, advec.flux_deriv, advec.max_flux_deriv, advec.CHAR_SPEED)
    x_center = w.get_x_center()
    u0 = advec.initial_condition_square_wave(x_center)
    solution = w.integrate(u0, T)
    exact = advec.exact_solution_square_wave(x_center - advec.CHAR_SPEED * T, T, n)
    points = {'left1': 0, 'right1': 0, 'left2': 0, 'right2': 0}
    for i in range(0, n):
        if x_center[i] >= -0.82:
            points['left1'] = i
            break

    for i in range(0, n):
        if x_center[i] >= -0.78:
            points['right1'] = i
            break

    for i in range(0, n):
        if x_center[i] >= 0.78:
            points['left2'] = i
            break

    for i in range(0, n):
        if x_center[i] >= 0.82:
            points['right2'] = i
            break

    solution_smooth = []
    exact_smooth = []
    x_center_smooth = []
    for i in range(0, n):
        if points['left1'] < i < points['right1'] or points['left2'] < i < points['right2']:
            continue
        solution_smooth.append(solution[i])
        exact_smooth.append(exact[i])
        x_center_smooth.append(x_center[i])

    solution_smooth = np.asarray(solution_smooth)
    exact_smooth = np.asarray(exact_smooth)
    x_center_smooth = np.asarray(x_center_smooth)
    dx = (advec.b - advec.a) / n
    errorsList.append(dx * np.linalg.norm(exact_smooth - solution_smooth, 1))
    dxList.append(dx)


plt.loglog(dxList, errorsList, '-o')
plt.loglog(dxList, [1e6* dx**4 for dx in dxList])
plt.show()

for i in range(0, len(errorsList)):
    print N[i], errorsList[i]

