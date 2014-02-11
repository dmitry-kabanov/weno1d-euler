import numpy as np


class Weno3Interpolator:
    def __init__(self, gridSize, eps):
        self.gridSize = gridSize
        self.EPS = eps
        # Ideal weights for the right boundary.
        self.iw_right = np.array([[3.0 / 10.0], [6.0 / 10.0], [1.0 / 10.0]])
        # Ideal weights for the left boundary.
        self.iw_left = np.array([[1.0 / 10.0], [6.0 / 10.0], [3.0 / 10.0]])

        self.ORDER_OF_SCHEME = 3
        self.v_right_boundary_approx = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))
        self.v_left_boundary_approx = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))
        self.v_right_boundary = np.zeros(self.gridSize)
        self.v_left_boundary = np.zeros(self.gridSize)
        # WENO weights.
        self.beta = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))
        self.alpha_right = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))
        self.alpha_left = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))
        self.sum_alpha_right = np.zeros(self.gridSize)
        self.sum_alpha_left = np.zeros(self.gridSize)
        self.omega_right = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))
        self.omega_left = np.zeros((self.ORDER_OF_SCHEME, self.gridSize))

    # def reconstruct(self, u):
    #     # WENO Reconstruction
    #     # Approximations for inner cells 0 < i < N-1.
    #     self.v_right_boundary_approx[0][2:-2] = 1.0 / 3.0 * u[2:-2] + \
    #                                                 5.0 / 6.0 * u[3:-1] - 1.0 / 6.0 * u[4:]
    #     self.v_right_boundary_approx[1][2:-2] = -1.0 / 6.0 * u[1:-3] + \
    #                                             5.0 / 6.0 * u[2:-2] + 1.0 / 3.0 * u[3:-1]
    #     self.v_right_boundary_approx[2][2:-2] = 1.0 / 3.0 * u[0:-4] - \
    #                                             7.0 / 6.0 * u[1:-3] + 11.0 / 6.0 * u[2:-2]
    #     self.v_left_boundary_approx[0][2:-2] = 11.0 / 6.0 * u[2:-2] - \
    #                                            7.0 / 6.0 * u[3:-1] + 1.0 / 3.0 * u[4:]
    #     self.v_left_boundary_approx[1][2:-2] = 1.0 / 3.0 * u[1:-3] + \
    #                                            5.0 / 6.0 * u[2:-2] - 1.0 / 6.0 * u[3:-1]
    #     self.v_left_boundary_approx[2][2:-2] = -1.0 / 6.0 * u[0:-4] + \
    #                                            5.0 / 6.0 * u[1:-3] + 1.0 / 3.0 * u[2:-2]
    #
    #     # Approximations for cell i = 0 (the leftmost cell).
    #     self.v_right_boundary_approx[0][0] = 1.0 / 3.0 * u[0] + \
    #                                          5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
    #     self.v_right_boundary_approx[1][0] = -1.0 / 6.0 * u[-1] + \
    #                                          5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
    #     self.v_right_boundary_approx[2][0] = 1.0 / 3.0 * u[-2] - \
    #                                          7.0 / 6.0 * u[-1] + 11.0 / 6.0 * u[0]
    #     self.v_left_boundary_approx[0][0] = 11.0 / 6.0 * u[0] - \
    #                                         7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
    #     self.v_left_boundary_approx[1][0] = 1.0 / 3.0 * u[-1] + \
    #                                         5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
    #     self.v_left_boundary_approx[2][0] = -1.0 / 6.0 * u[-2] + \
    #                                         5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
    #
    #     # Approximations for cell i = 1.
    #     self.v_right_boundary_approx[0][1] = 1.0 / 3.0 * u[1] + \
    #                                          5.0 / 6.0 * u[2] - 1.0 / 6.0 * u[3]
    #     self.v_right_boundary_approx[1][1] = -1.0 / 6.0 * u[0] + \
    #                                          5.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
    #     self.v_right_boundary_approx[2][1] = 1.0 / 3.0 * u[-1] - \
    #                                          7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[1]
    #     self.v_left_boundary_approx[0][1] = 11.0 / 6.0 * u[1] - \
    #                                         7.0 / 6.0 * u[2] + 1.0 / 3.0 * u[3]
    #     self.v_left_boundary_approx[1][1] = 1.0 / 3.0 * u[0] + \
    #                                         5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
    #     self.v_left_boundary_approx[2][1] = -1.0 / 6.0 * u[-1] + \
    #                                         5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
    #
    #     # Approximations for cell i = N-2.
    #     self.v_right_boundary_approx[0][-2] = 1.0 / 3.0 * u[-2] + \
    #                                           5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
    #     self.v_right_boundary_approx[1][-2] = -1.0 / 6.0 * u[-3] + \
    #                                           5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
    #     self.v_right_boundary_approx[2][-2] = 1.0 / 3.0 * u[-4] - \
    #                                           7.0 / 6.0 * u[-3] + 11.0 / 6.0 * u[-2]
    #     self.v_left_boundary_approx[0][-2] = 11.0 / 6.0 * u[-2] - \
    #                                          7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
    #     self.v_left_boundary_approx[1][-2] = 1.0 / 3.0 * u[-3] + \
    #                                          5.0 / 6.0 * u[-2] - 1.0 / 6.0 * u[-1]
    #     self.v_left_boundary_approx[2][-2] = -1.0 / 6.0 * u[-4] + \
    #                                          5.0 / 6.0 * u[-3] + 1.0 / 3.0 * u[-2]
    #
    #     # Approximations for cell i = N-1 (the rightmost cell).
    #     self.v_right_boundary_approx[0][-1] = 1.0 / 3.0 * u[-1] + \
    #                                           5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
    #     self.v_right_boundary_approx[1][-1] = -1.0 / 6.0 * u[-2] + \
    #                                           5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
    #     self.v_right_boundary_approx[2][-1] = 1.0 / 3.0 * u[-3] - \
    #                                           7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[-1]
    #     self.v_left_boundary_approx[0][-1] = 11.0 / 6.0 * u[-1] - \
    #                                          7.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
    #     self.v_left_boundary_approx[1][-1] = 1.0 / 3.0 * u[-2] + \
    #                                          5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
    #     self.v_left_boundary_approx[2][-1] = -1.0 / 6.0 * u[-3] + \
    #                                          5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
    #
    #     self.beta[0][2:-2] = 13.0 / 12.0 * \
    #                          (u[2:-2] - 2 * u[3:-1] + u[4:]) ** 2 + \
    #                          1.0 / 4.0 * (3*u[2:-2] - 4.0 * u[3:-1] + u[4:]) ** 2
    #     self.beta[1][2:-2] = 13.0 / 12.0 * \
    #                          (u[1:-3] - 2 * u[2:-2] + u[3:-1]) ** 2 + \
    #                          1.0 / 4.0 * (u[1:-3] - u[3:-1]) ** 2
    #     self.beta[2][2:-2] = 13.0 / 12.0 * \
    #                          (u[0:-4] - 2 * u[1:-3] + u[2:-2]) ** 2 + \
    #                          1.0 / 4.0 * (u[0:-4] - 4.0 * u[1:-3] + 3 * u[2:-2]) ** 2
    #
    #     self.beta[0][0] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
    #                       1.0 / 4.0 * (3*u[0] - 4.0 * u[1] + u[2]) ** 2
    #     self.beta[1][0] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
    #                       1.0 / 4.0 * (u[-1] - u[1]) ** 2
    #     self.beta[2][0] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
    #                       1.0 / 4.0 * (u[-2] - 4.0 * u[-1] + 3 * u[0]) ** 2
    #
    #     self.beta[0][1] = 13.0 / 12.0 * (u[1] - 2 * u[2] + u[3]) ** 2 + \
    #                       1.0 / 4.0 * (3*u[1] - 4.0 * u[2] + u[3]) ** 2
    #     self.beta[1][1] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
    #                       1.0 / 4.0 * (u[0] - u[2]) ** 2
    #     self.beta[2][1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
    #                       1.0 / 4.0 * (u[-1] - 4.0 * u[0] + 3 * u[1]) ** 2
    #
    #     self.beta[0][-2] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
    #                        1.0 / 4.0 * (3*u[-2] - 4.0 * u[-1] + u[0]) ** 2
    #     self.beta[1][-2] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
    #                        1.0 / 4.0 * (u[-3] - u[-1]) ** 2
    #     self.beta[2][-2] = 13.0 / 12.0 * (u[-4] - 2 * u[-3] + u[-2]) ** 2 + \
    #                        1.0 / 4.0 * (u[-4] - 4.0 * u[-3] + 3 * u[-2]) ** 2
    #
    #     self.beta[0][-1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
    #                        1.0 / 4.0 * (3*u[-1] - 4.0 * u[0] + u[1]) ** 2
    #     self.beta[1][-1] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
    #                        1.0 / 4.0 * (u[-2] - u[0]) ** 2
    #     self.beta[2][-1] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
    #                        1.0 / 4.0 * (u[-3] - 4.0 * u[-2] + 3 * u[-1]) ** 2
    #
    #     self.alpha_right = self.iw_right / ((self.EPS + self.beta) ** 2)
    #     self.alpha_left = self.iw_left / ((self.EPS + self.beta) ** 2)
    #     self.sum_alpha_right = \
    #         self.alpha_right[0] + self.alpha_right[1] + self.alpha_right[2]
    #     self.sum_alpha_left = \
    #         self.alpha_left[0] + self.alpha_left[1] + self.alpha_left[2]
    #     self.omega_right = self.alpha_right / self.sum_alpha_right
    #     self.omega_left = self.alpha_left / self.sum_alpha_left
    #     self.v_right_boundary = \
    #         self.omega_right[0] * self.v_right_boundary_approx[0] + \
    #         self.omega_right[1] * self.v_right_boundary_approx[1] + \
    #         self.omega_right[2] * self.v_right_boundary_approx[2]
    #     self.v_left_boundary = \
    #         self.omega_left[0] * self.v_left_boundary_approx[0] + \
    #         self.omega_left[1] * self.v_left_boundary_approx[1] + \
    #         self.omega_left[2] * self.v_left_boundary_approx[2]
    #
    #     return self.v_left_boundary, self.v_right_boundary

    def reconstruct(self, u):
        self.v_left_boundary = u
        self.v_right_boundary = u

        return self.v_left_boundary, self.v_right_boundary