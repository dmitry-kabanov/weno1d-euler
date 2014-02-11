import numpy as np

import wenoeuler1d.weno5solver.weno3interpolator as wi
import wenoeuler1d.weno5solver.riemannsolver as rs


class Weno3:
    def __init__(self, left_boundary, right_boundary, ncells,
                 gamma, cfl_number=0.5, eps=1.0e-6):
        """
        Instantiate class.

        :param left_boundary:
        :param right_boundary:
        :param ncells:
        :param gamma:
        :param cfl_number:
        :param eps:
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
        self.x_boundary = np.linspace(a, b, self.N + 1)
        self.x_center = np.zeros(self.N)

        for i in range(0, self.N):
            self.x_center[i] = (self.x_boundary[i] +
                                self.x_boundary[i + 1]) / 2.0

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

        self.rs = rs.riemannSolver()
        self.flux_mass = np.zeros(self.N + 1)
        self.flux_momentum = np.zeros(self.N + 1)
        self.flux_energy = np.zeros(self.N + 1)

        self.interpolator = wi.Weno3Interpolator(self.N, eps)

    def integrate(self, u0, time_final):
        self.dt = self.CFL_NUMBER * self.dx / self.CHAR_SPEED
        self.T = time_final
        u0 = self.prepare_initial_conditions(u0)
        self.u_multistage[0] = u0

        while self.t < self.T:
            if self.t + self.dt > self.T:
                self.dt = self.T - self.t

            self.t += self.dt
            # self.u_multistage[1] = self.u_multistage[0] + \
            #     self.dt * self.rhs(self.u_multistage[0])
            # self.u_multistage[2] = (
            #     3 * self.u_multistage[0] + self.u_multistage[1] +
            #     self.dt * self.rhs(self.u_multistage[1])) / 4.0
            # self.u_multistage[0] = (
            #     self.u_multistage[0] + 2.0 * self.u_multistage[2] +
            #     2.0 * self.dt * self.rhs(self.u_multistage[2])) / 3.0
            self.u_multistage[0] += self.dt * self.rhs(
                self.u_multistage[0])

            n = self.N
            rho = self.u_multistage[0][0:n]
            u = self.u_multistage[0][n:2*n] / rho
            p = (self.u_multistage[0][2*n:3*n] / rho - u**2 / 2.0) * (self.gamma - 1.0) * rho

        return self.u_multistage[0]

    def prepare_initial_conditions(self, u0):
        """Converts initial conditions from primitive variables
        to conservative variables.

        :u0: ndarray of primitive variables - density0, velocity0, pressure0
        :returns: ndarray of conservative variables

        """
        n = self.N
        result = np.zeros(3*n)
        # First conservative variable is density.
        result[0:n] = u0[0:n]
        # Second conservative variable is density multiplied by velocity.
        result[n:2*n] = u0[0:n] * u0[n:2*n]
        # Third conservative variable is density multiplied by total energy.
        result[2*n:3*n] = u0[2*n:3*n] / (self.gamma - 1.0) + (
            result[0:n] * result[n:2*n] * result[n:2*n] / 2.0)

        return result

    def rhs(self, u):
        n = self.N
        self.rho = u[0:n]
        self.rho_u = u[n:2*n]
        self.rho_etotal = u[2*n:3*n]

        self.reconstruct_primitive_variables()
        self.solve_riemann_problem()
        self.compute_fluxes()

        # Compute flux change in every cell.
        self.rhsValues[0:n] = self.flux_mass[1:] - self.flux_mass[0:-1]
        self.rhsValues[n:2*n] = \
            self.flux_momentum[1:] - self.flux_momentum[0:-1]
        self.rhsValues[2*n:3*n] = \
            self.flux_energy[1:] - self.flux_energy[0:-1]

        # TODO: this is a hack to avoid dealing with boundary conditions.
        # The problem should be resolved by dealing with reconstruction
        # and implementing boundary conditions correctly.
        self.rhsValues[0:2] = 0.0
        self.rhsValues[n-2:n] = 0.0
        self.rhsValues[n:n+2] = 0.0
        self.rhsValues[2*n-2:2*n] = 0.0
        self.rhsValues[2*n:2*n+2] = 0.0
        self.rhsValues[3*n-2:3*n] = 0.0
        self.rhsValues = -self.rhsValues / self.dx

        return self.rhsValues

    def reconstruct_primitive_variables(self):
        self.rho_left_boundary, self.rho_right_boundary = \
            self.interpolator.reconstruct(self.rho)
        self.rho_u_left_boundary, self.rho_u_right_boundary = \
            self.interpolator.reconstruct(self.rho_u)
        self.rho_etotal_left_boundary, self.rho_etotal_right_boundary = \
            self.interpolator.reconstruct(self.rho_etotal)
        self.u_left_boundary = \
            self.rho_u_left_boundary / self.rho_left_boundary
        self.u_right_boundary = \
            self.rho_u_right_boundary / self.rho_right_boundary
        self.etotal_left_boundary = \
            self.rho_etotal_left_boundary / self.rho_left_boundary
        self.etotal_right_boundary = \
            self.rho_etotal_right_boundary / self.rho_right_boundary
        self.p_left_boundary = \
            (self.etotal_left_boundary - self.u_left_boundary ** 2 / 2.0) * \
            (self.gamma - 1.0) * self.rho_left_boundary
        self.p_right_boundary = \
            (self.etotal_right_boundary - self.u_right_boundary ** 2 / 2.0) * \
            (self.gamma - 1.0) * self.rho_right_boundary

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

        self.etotal_star[1:-1] = self.p_star[1:-1] / (self.gamma - 1.0) / \
            self.rho_star[1:-1] + self.u_star[1:-1] * self.u_star[1:-1] / 2.0

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
