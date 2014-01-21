import numpy as np

import riemannSolver


class Weno3:
    def __init__(self, left_boundary, right_boundary, ncells,
                 gamma, cfl_number=0.8, eps=1.0e-6):
        """

        :rtype : Weno3
        """
        a = left_boundary
        b = right_boundary
        self.N = ncells
        self.gamma = gamma
        self.dx = (b - a) / (self.N + 0.0)
        self.CFL_NUMBER = cfl_number
        self.CHAR_SPEED = 1.0
        self.t = 0.0
        ORDER_OF_SCHEME = 3
        self.EPS = eps
        # Ideal weights for the right boundary.
        self.iw_right = np.array([[3.0 / 10.0], [6.0 / 10.0], [1.0 / 10.0]])
        self.iw_left = np.array([[1.0 / 10.0], [6.0 / 10.0], [3.0 / 10.0]])
        self.x_boundary = np.linspace(a, b, self.N + 1)
        self.x_center = np.zeros(self.N)

        for i in range(0, self.N):
            self.x_center[i] = (self.x_boundary[i] +
                                self.x_boundary[i + 1]) / 2.0

        self.v_right_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.v_left_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.v_right_boundary = np.zeros(self.N)
        self.v_left_boundary = np.zeros(self.N)
        self.beta = np.zeros((ORDER_OF_SCHEME, self.N))
        self.alpha_right = np.zeros((ORDER_OF_SCHEME, self.N))
        self.alpha_left = np.zeros((ORDER_OF_SCHEME, self.N))
        self.sum_alpha_right = np.zeros(self.N)
        self.sum_alpha_left = np.zeros(self.N)
        self.omega_right = np.zeros((ORDER_OF_SCHEME, self.N))
        self.omega_left = np.zeros((ORDER_OF_SCHEME, self.N))
        self.fFlux = np.zeros(self.N + 1)
        self.rhsValues = np.zeros(3 * self.N)
        self.u_multistage = np.zeros((3, 3 * self.N))

        # Physical variables
        self.rho = np.zeros(self.N)
        self.rho_u = np.zeros(self.N)
        self.rho_etotal = np.zeros(self.N)
        self.u = np.zeros(self.N)
        self.etotal = np.zeros(self.N)
        self.p = np.zeros(self.N)

        self.rho_left_boundary = np.zeros(self.N)
        self.rho_right_boundary = np.zeros(self.N)
        self.rho_u_left_boundary = np.zeros(self.N)
        self.rho_u_right_boundary = np.zeros(self.N)
        self.rho_etotal_left_boundary = np.zeros(self.N)
        self.rho_etotal_right_boundary = np.zeros(self.N)
        self.u_left_boundary = np.zeros(self.N)
        self.u_right_boundary = np.zeros(self.N)
        self.etotal_left_boundary = np.zeros(self.N)
        self.etotal_right_boundary = np.zeros(self.N)
        self.p_left_boundary = np.zeros(self.N)
        self.p_right_boundary = np.zeros(self.N)

        # Auxilariary variables.
        self.v_left_boundary = np.zeros(self.N)
        self.v_right_boundary = np.zeros(self.N)
        self.v_left_boundary = np.zeros(self.N)
        self.v_right_boundary = np.zeros(self.N)
        self.v_left_boundary = np.zeros(self.N)

        # Solutions of Riemann problem.
        self.rho_star = np.zeros(self.N + 1)
        self.u_star = np.zeros(self.N + 1)
        self.p_star = np.zeros(self.N + 1)
        self.etotal_star = np.zeros(self.N + 1)

        self.rs = riemannSolver.riemannSolver()

    def integrate(self, u0, time_final):
        self.dt = self.CFL_NUMBER * self.dx / self.CHAR_SPEED
        self.T = time_final
        u0 = self.prepare_initial_conditions(u0)
        self.u_multistage[0] = u0

        while self.t < self.T:
            if self.t + self.dt > self.T:
                self.dt = self.T - self.t

            self.t += self.dt
            self.u_multistage[1] = self.u_multistage[0] + \
                self.dt * self.rhs(self.u_multistage[0])
            self.u_multistage[2] = (
                3 * self.u_multistage[0] + self.u_multistage[1] +
                self.dt * self.rhs(self.u_multistage[1])) / 4.0
            self.u_multistage[0] = (
                self.u_multistage[0] + 2.0 * self.u_multistage[2] +
                2.0 * self.dt * self.rhs(self.u_multistage[2])) / 3.0

        return self.u_multistage[0]

    def prepare_initial_conditions(self, u0):
        """Converts initial conditions from primitive variables
        to conservative variables.

        :u0: ndarray of primitive variables - density0, velocity0, pressure0
        :returns: ndarray of conservative variables

        """
        n = self.N
        result = np.zeros(3*n)
        result[0:n] = u0[0:n]
        result[n:2*n] = u0[0:n] * u0[n:2*n]
        result[2*n:3*n] = u0[0:n] * u0[2*n:3*n]

        return result

    def rhs(self, u):
        self.rho = u[0:self.N]
        self.rho_u = u[self.N:2*self.N]
        self.rho_etotal = u[2*self.N:3*self.N]

        self.reconstruct_primitive_variables()
        self.solve_riemann_problem()
        self.compute_fluxes()

        # Compute flux change in every cell.
        self.rhsValues[0:self.N] = self.flux_mass[1:] - self.flux_mass[0:-1]
        self.rhsValues[self.N:2 * self.N] = \
            self.flux_momentum[1:] - self.flux_momentum[0:-1]
        self.rhsValues[2 * self.N:3 * self.N] = \
            self.flux_energy[1:] - self.flux_energy[0:-1]
        self.rhsValues = -self.rhsValues / self.dx

        return self.rhsValues

    def reconstruct_primitive_variables(self):
        self.rho_left_boundary, self.rho_right_boundary = \
            self.reconstruct(self.rho)
        self.rho_u_left_boundary, self.rho_u_right_boundary = \
            self.reconstruct(self.rho_u)
        self.rho_etotal_left_boundary, self.rho_etotal_right_boundary = \
            self.reconstruct(self.rho_etotal)
        self.u_left_boundary = \
            self.rho_u_left_boundary / self.rho_left_boundary
        self.u_right_boundary = \
            self.rho_u_right_boundary / self.rho_right_boundary
        self.etotal_left_boundary = \
            self.rho_etotal_left_boundary / self.rho_left_boundary
        self.etotal_right_boundary = \
            self.rho_etotal_right_boundary / self.rho_right_boundary
        self.p_left_boundary = \
            (self.etotal_left_boundary - self.u_left_boundary ** 2 / 2.0) / \
            (self.gamma - 1.0) / self.rho_left_boundary
        self.p_right_boundary = \
            (self.etotal_right_boundary - self.u_right_boundary ** 2 / 2.0) / \
            (self.gamma - 1.0) / self.rho_right_boundary

    def reconstruct(self, u):
        # WENO Reconstruction
        # Approximations for inner cells 0 < i < N-1.
        self.v_right_boundary_approx[0][2:-2] = 1.0 / 3.0 * u[2:-2] + \
            5.0 / 6.0 * u[3:-1] - 1.0 / 6.0 * u[4:]
        self.v_right_boundary_approx[1][2:-2] = -1.0 / 6.0 * u[1:-3] + \
            5.0 / 6.0 * u[2:-2] + 1.0 / 3.0 * u[3:-1]
        self.v_right_boundary_approx[2][2:-2] = 1.0 / 3.0 * u[0:-4] - \
            7.0 / 6.0 * u[1:-3] + 11.0 / 6.0 * u[2:-2]
        self.v_left_boundary_approx[0][2:-2] = 11.0 / 6.0 * u[2:-2] - \
            7.0 / 6.0 * u[3:-1] + 1.0 / 3.0 * u[4:]
        self.v_left_boundary_approx[1][2:-2] = 1.0 / 3.0 * u[1:-3] + \
            5.0 / 6.0 * u[2:-2] - 1.0 / 6.0 * u[3:-1]
        self.v_left_boundary_approx[2][2:-2] = -1.0 / 6.0 * u[0:-4] + \
            5.0 / 6.0 * u[1:-3] + 1.0 / 3.0 * u[2:-2]

        # Approximations for cell i = 0 (the leftmost cell).
        self.v_right_boundary_approx[0][0] = 1.0 / 3.0 * u[0] + \
            5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
        self.v_right_boundary_approx[1][0] = -1.0 / 6.0 * u[-1] + \
            5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
        self.v_right_boundary_approx[2][0] = 1.0 / 3.0 * u[-2] - \
            7.0 / 6.0 * u[-1] + 11.0 / 6.0 * u[0]
        self.v_left_boundary_approx[0][0] = 11.0 / 6.0 * u[0] - \
            7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
        self.v_left_boundary_approx[1][0] = 1.0 / 3.0 * u[-1] + \
            5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
        self.v_left_boundary_approx[2][0] = -1.0 / 6.0 * u[-2] + \
            5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]

        # Approximations for cell i = 1.
        self.v_right_boundary_approx[0][1] = 1.0 / 3.0 * u[1] + \
            5.0 / 6.0 * u[2] - 1.0 / 6.0 * u[3]
        self.v_right_boundary_approx[1][1] = -1.0 / 6.0 * u[0] + \
            5.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
        self.v_right_boundary_approx[2][1] = 1.0 / 3.0 * u[-1] - \
            7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[1]
        self.v_left_boundary_approx[0][1] = 11.0 / 6.0 * u[1] - \
            7.0 / 6.0 * u[2] + 1.0 / 3.0 * u[3]
        self.v_left_boundary_approx[1][1] = 1.0 / 3.0 * u[0] + \
            5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
        self.v_left_boundary_approx[2][1] = -1.0 / 6.0 * u[-1] + \
            5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]

        # Approximations for cell i = N-2.
        self.v_right_boundary_approx[0][-2] = 1.0 / 3.0 * u[-2] + \
            5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
        self.v_right_boundary_approx[1][-2] = -1.0 / 6.0 * u[-3] + \
            5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
        self.v_right_boundary_approx[2][-2] = 1.0 / 3.0 * u[-4] - \
            7.0 / 6.0 * u[-3] + 11.0 / 6.0 * u[-2]
        self.v_left_boundary_approx[0][-2] = 11.0 / 6.0 * u[-2] - \
            7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
        self.v_left_boundary_approx[1][-2] = 1.0 / 3.0 * u[-3] + \
            5.0 / 6.0 * u[-2] - 1.0 / 6.0 * u[-1]
        self.v_left_boundary_approx[2][-2] = -1.0 / 6.0 * u[-4] + \
            5.0 / 6.0 * u[-3] + 1.0 / 3.0 * u[-2]

        # Approximations for cell i = N-1 (the rightmost cell).
        self.v_right_boundary_approx[0][-1] = 1.0 / 3.0 * u[-1] + \
            5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
        self.v_right_boundary_approx[1][-1] = -1.0 / 6.0 * u[-2] +\
            5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
        self.v_right_boundary_approx[2][-1] = 1.0 / 3.0 * u[-3] - \
            7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[-1]
        self.v_left_boundary_approx[0][-1] = 11.0 / 6.0 * u[-1] - \
            7.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
        self.v_left_boundary_approx[1][-1] = 1.0 / 3.0 * u[-2] + \
            5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
        self.v_left_boundary_approx[2][-1] = -1.0 / 6.0 * u[-3] + \
            5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]

        self.beta[0][2:-2] = 13.0 / 12.0 * \
            (u[2:-2] - 2 * u[3:-1] + u[4:]) ** 2 + \
            1.0 / 4.0 * (3*u[2:-2] - 4.0 * u[3:-1] + u[4:]) ** 2
        self.beta[1][2:-2] = 13.0 / 12.0 * \
            (u[1:-3] - 2 * u[2:-2] + u[3:-1]) ** 2 + \
            1.0 / 4.0 * (u[1:-3] - u[3:-1]) ** 2
        self.beta[2][2:-2] = 13.0 / 12.0 * \
            (u[0:-4] - 2 * u[1:-3] + u[2:-2]) ** 2 + \
            1.0 / 4.0 * (u[0:-4] - 4.0 * u[1:-3] + 3 * u[2:-2]) ** 2

        self.beta[0][0] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
            1.0 / 4.0 * (3*u[0] - 4.0 * u[1] + u[2]) ** 2
        self.beta[1][0] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
            1.0 / 4.0 * (u[-1] - u[1]) ** 2
        self.beta[2][0] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
            1.0 / 4.0 * (u[-2] - 4.0 * u[-1] + 3 * u[0]) ** 2

        self.beta[0][1] = 13.0 / 12.0 * (u[1] - 2 * u[2] + u[3]) ** 2 + \
            1.0 / 4.0 * (3*u[1] - 4.0 * u[2] + u[3]) ** 2
        self.beta[1][1] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
            1.0 / 4.0 * (u[0] - u[2]) ** 2
        self.beta[2][1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
            1.0 / 4.0 * (u[-1] - 4.0 * u[0] + 3 * u[1]) ** 2

        self.beta[0][-2] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
            1.0 / 4.0 * (3*u[-2] - 4.0 * u[-1] + u[0]) ** 2
        self.beta[1][-2] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
            1.0 / 4.0 * (u[-3] - u[-1]) ** 2
        self.beta[2][-2] = 13.0 / 12.0 * (u[-4] - 2 * u[-3] + u[-2]) ** 2 + \
            1.0 / 4.0 * (u[-4] - 4.0 * u[-3] + 3 * u[-2]) ** 2

        self.beta[0][-1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
            1.0 / 4.0 * (3*u[-1] - 4.0 * u[0] + u[1]) ** 2
        self.beta[1][-1] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
            1.0 / 4.0 * (u[-2] - u[0]) ** 2
        self.beta[2][-1] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
            1.0 / 4.0 * (u[-3] - 4.0 * u[-2] + 3 * u[-1]) ** 2

        self.alpha_right = self.iw_right / ((self.EPS + self.beta) ** 2)
        self.alpha_left = self.iw_left / ((self.EPS + self.beta) ** 2)
        self.sum_alpha_right = \
            self.alpha_right[0] + self.alpha_right[1] + self.alpha_right[2]
        self.sum_alpha_left = \
            self.alpha_left[0] + self.alpha_left[1] + self.alpha_left[2]
        self.omega_right = self.alpha_right / self.sum_alpha_right
        self.omega_left = self.alpha_left / self.sum_alpha_left
        self.v_right_boundary = \
            self.omega_right[0] * self.v_right_boundary_approx[0] + \
            self.omega_right[1] * self.v_right_boundary_approx[1] + \
            self.omega_right[2] * self.v_right_boundary_approx[2]
        self.v_left_boundary = \
            self.omega_left[0] * self.v_left_boundary_approx[0] + \
            self.omega_left[1] * self.v_left_boundary_approx[1] + \
            self.omega_left[2] * self.v_left_boundary_approx[2]

        return self.v_left_boundary, self.v_right_boundary

    def solve_riemann_problem(self):
        for i in range(0, self.N - 1):
            leftSide = {"rho": self.rho_right_boundary[i],
                        "u": self.u_right_boundary[i],
                        "p": self.p_right_boundary[i]}
            rightSide = {"rho": self.rho_left_boundary[i + 1],
                         "u": self.u_left_boundary[i + 1],
                         "p": self.p_left_boundary[i + 1]}
            self.rho_star[i + 1], self.u_star[i + 1], self.p_star[i + 1] = \
                self.rs.solve(leftSide, rightSide, self.gamma)

        self.etotal_star[:] = self.p_star[:] / (self.gamma - 1.0) / \
            self.rho_star[:] + self.u_star[:] * self.u_star[:] / 2.0

    def compute_fluxes(self):
        # Compute fluxes on the cells boundaries.
        self.flux_mass[:] = self.rho_star[:] * self.u_star[:]
        self.flux_momentum[:] = self.rho_star[:] * \
            self.u_star[:] * self.u_star[:] + self.p_star[:]
        self.flux_energy[:] = self.rho_star * self.u_star[:] * \
            self.etotal_star[:] + self.p_star[:] * self.u_star[:]

    def get_x_center(self):
        return self.x_center

    def get_x_boundary(self):
        return self.x_boundary

    def get_dx(self):
        return self.dx
