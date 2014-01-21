import numpy as np

class Weno2:
    def __init__(self, left_boundary, right_boundary, ncells, flux_callback, flux_deriv_callback, max_flux_deriv_callback, char_speed, cfl_number=0.8, eps=1.0e-6):
        """

        :rtype : Weno2
        """
        a = left_boundary
        b = right_boundary
        self.N = ncells
        self.dx = (b - a) / (self.N + 0.0)
        self.CFL_NUMBER = cfl_number
        self.CHAR_SPEED = char_speed
        self.t = 0.0
        ORDER_OF_SCHEME = 2
        self.EPS = eps
        self.D0 = 2.0 / 3.0
        self.D1 = 1.0 / 3.0
        self.flux = flux_callback
        self.flux_deriv = flux_deriv_callback
        self.max_flux_deriv = max_flux_deriv_callback
        self.x_boundary = np.linspace(a, b, self.N + 1)
        self.x_center = np.zeros(self.N)

        for i in range(0, self.N):
            self.x_center[i] = (self.x_boundary[i] + self.x_boundary[i + 1]) / 2.0

        self.u_right_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.u_left_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.u_right_boundary = np.zeros(self.N)
        self.u_left_boundary = np.zeros(self.N)
        self.beta0 = np.zeros(self.N)
        self.beta1 = np.zeros(self.N)
        self.alpha0 = np.zeros(self.N)
        self.alpha1 = np.zeros(self.N)
        self.sum_alpha = np.zeros(self.N)
        self.omega0 = np.zeros(self.N)
        self.omega1 = np.zeros(self.N)
        self.fFlux = np.zeros(self.N + 1)
        self.rhsValues = np.zeros(self.N)
        self.u_multistage = np.zeros((3, self.N))


    def integrate(self, u0, time_final):
        self.dt = self.CFL_NUMBER * self.dx / self.CHAR_SPEED
        self.T = time_final
        self.u_multistage[0] = u0

        while self.t < self.T:
            if self.t + self.dt > self.T:
                self.dt = self.T - self.t

            self.t += self.dt
            self.u_multistage[1] = self.u_multistage[0] + self.dt * self.rhs(self.u_multistage[0])
            self.u_multistage[2] = (3 * self.u_multistage[0] + self.u_multistage[1] + self.dt * self.rhs(self.u_multistage[1])) / 4.0
            self.u_multistage[0] = (self.u_multistage[0] + 2.0 * self.u_multistage[2] + 2.0 * self.dt * self.rhs(self.u_multistage[2])) / 3.0

        return self.u_multistage[0]


    def rhs(self, u):
        # WENO Reconstruction
        # Approximations for inner cells 0 < i < N-1.
        self.u_right_boundary_approx[0][1:-1] = 0.5 * u[1:-1] + 0.5 * u[2:]
        self.u_right_boundary_approx[1][1:-1] = -0.5 * u[0:-2] + 1.5 * u[1:-1]
        self.u_left_boundary_approx[0][1:-1] = 1.5 * u[1:-1] - 0.5 * u[2:]
        self.u_left_boundary_approx[1][1:-1] = 0.5 * u[0:-2] + 0.5 * u[1:-1]

        # Approximations for cell i = 0 (the leftmost cell).
        self.u_right_boundary_approx[0][0] = 0.5 * u[0] + 0.5 * u[1]
        self.u_right_boundary_approx[1][0]= -0.5 * u[self.N - 1] + 1.5 * u[0]
        self.u_left_boundary_approx[0][0] = 1.5 * u[0] - 0.5 * u[1]
        self.u_left_boundary_approx[1][0] = 0.5 * u[self.N - 1] + 0.5 * u[0]

        # Approximations for cell i = N-1 (the rightmost cell).
        self.u_right_boundary_approx[0][self.N - 1] = 0.5 * u[self.N - 1] + 0.5 * u[0]
        self.u_right_boundary_approx[1][self.N - 1] = -0.5 * u[self.N - 2] + 1.5 * u[self.N - 1]
        self.u_left_boundary_approx[0][self.N - 1] = 1.5 * u[self.N - 1] - 0.5 * u[0]
        self.u_left_boundary_approx[1][self.N - 1] = 0.5 * u[self.N - 2] + 0.5 * u[self.N - 1]

        self.beta0[1:-2] = (u[2:-1] - u[1:-2]) ** 2
        self.beta1[1:-2] = (u[1:-2] - u[0:-3]) ** 2
        self.beta0[0] = (u[1] - u[0]) ** 2
        self.beta1[0] = (u[0] - u[self.N - 1]) ** 2
        self.beta0[-1] = (u[0] - u[self.N - 1]) ** 2
        self.beta1[-1] = (u[self.N - 1] - u[self.N - 2]) ** 2
        self.alpha0 = self.D0 / ((self.EPS + self.beta0) ** 2)
        self.alpha1 = self.D1 / ((self.EPS + self.beta1) ** 2)
        self.sum_alpha = self.alpha0 + self.alpha1
        self.omega0 = self.alpha0 / self.sum_alpha
        self.omega1 = self.alpha1 / self.sum_alpha
        self.u_right_boundary = self.omega0 * self.u_right_boundary_approx[0][:] + \
                           self.omega1 * self.u_right_boundary_approx[1][:]
        self.u_left_boundary = self.omega0 * self.u_left_boundary_approx[0][:] + \
                          self.omega1 * self.u_left_boundary_approx[1][:]

        # Numerical flux calculation.
        self.fFlux[1:-1] = self.numflux(self.u_right_boundary[0:-1], self.u_left_boundary[1:])
        self.fFlux[0] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
        self.fFlux[self.N] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])

        # Right hand side calculation.
        rhsValues = self.fFlux[1:] - self.fFlux[0:-1]
        rhsValues = -rhsValues / self.dx

        return rhsValues


    def numflux(self, a, b):
        """
        Return Lax-Friedrichs numerical flux.
        """
        maxval = self.max_flux_deriv(a, b)

        return 0.5 * (self.flux(a) + self.flux(b) - maxval * (b - a))


    def get_x_center(self):
        return self.x_center
