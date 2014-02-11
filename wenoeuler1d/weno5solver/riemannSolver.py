import math


class riemannSolver:
    """
        Exact Riemann solver for calorically perfect gas from the book
        of Godunov S. K, et al. Numerical solution of multidimensional
        problems of gas dynamics. Nauka, 1976, pp. 105-117
    """

    def __init__(self):
        self.LEFT_SHOCK = False
        self.RIGHT_SHOCK = False
        self.LEFT_RAREFACTION = False
        self.RIGHT_RAREFACTION = False

        self.leftSide = {}
        self.rightSide = {}

        self.TOLERANCE = 1.0e-6
        self.MAX_ITERATIONS = 20

    def complete_func(self, p_iter):
        term1 = self.func(p_iter, self.leftSide["p"],
                          self.leftSide["rho"], self.leftSide["c"])
        term2 = self.func(p_iter, self.rightSide["p"],
                          self.rightSide["rho"], self.rightSide["c"])
        term3 = self.rightSide["u"] - self.leftSide["u"]
        return term1 + term2 + term3

    def func(self, p_iter, p, rho, c):
        assert p_iter > 0, 'Pressure must be positive'
        # Shock wave.
        if (p_iter > p):
            coeff = p_iter - p
            A = 2.0 / (self.gp1 * rho)
            B = self.gm1_over_gp1 * p
            coeff2 = math.sqrt(A / (p_iter + B))
            return coeff * coeff2
        # Rarefaction wave.
        else:
            p_ratio = p_iter / p
            coeff1 = (2.0 * c) / self.gm1
            coeff2 = math.pow(p_ratio, self.gm1_over2g) - 1.0
            return coeff1 * coeff2

    def complete_dfunc(self, p_iter):
        assert p_iter > 0, 'Pressure must be positive'
        term1 = self.dfunc(p_iter, self.leftSide["p"],
                           self.leftSide["rho"], self.leftSide["c"])
        term2 = self.dfunc(p_iter, self.rightSide["p"],
                           self.rightSide["rho"], self.rightSide["c"])
        return term1 + term2

    def dfunc(self, p_iter, p, rho, c):
        assert p_iter > 0, 'Pressure must be positive'
        # Shock wave.
        if p_iter > p:
            A = 2.0 / ((self.gamma + 1) * rho)
            B = (self.gamma - 1) / (self.gamma + 1) * p
            coeff1 = math.sqrt(A / (B + p_iter))
            coeff2 = 1 - (p_iter - p) / (2 * (B + p_iter))
            result = coeff1 * coeff2
            assert result > 0, "f'(p) must be positive (shock case)\n" \
                               "leftSide = (rho: %f, u: %f, p: %f)\n" \
                               "rightSide = (rho: %f, u: %f, p: %f)\n" % (
                                   self.leftSide["rho"], self.leftSide["u"],
                                   self.leftSide["p"], self.rightSide["rho"],
                                   self.rightSide["u"], self.rightSide["p"])
            return result
        # Rarefaction wave.
        else:
            p_ratio = p_iter / p
            power_expr = math.pow(p_ratio, -self.gp1_over2g)
            coeff = 1.0 / (rho * c)
            result = coeff * power_expr
            assert result > 0, "f'(p) must be positive (rarefaction case)\n" \
                               "leftSide = (rho: %f, u: %f, p: %f)\n" \
                               "rightSide = (rho: %f, u: %f, p: %f)\n" % (
                                   self.leftSide["rho"], self.leftSide["u"],
                                   self.leftSide["p"], self.rightSide["rho"],
                                   self.rightSide["u"], self.rightSide["p"])
            return result

    def solve(self, leftSide, rightSide, gamma):
        try:
            assert "rho" in leftSide
            assert "u" in leftSide
            assert "p" in leftSide
            assert "rho" in rightSide
            assert "u" in rightSide
            assert "p" in rightSide
            assert leftSide["rho"] > 0
            assert leftSide["p"] > 0
            assert rightSide["rho"] > 0
            assert rightSide["p"] > 0
        except:
            print(leftSide)
            print(rightSide)

        self.leftSide = leftSide
        self.rightSide = rightSide

        self.leftSide["c"] = math.sqrt(
            gamma * leftSide["p"] / leftSide["rho"])
        self.rightSide["c"] = math.sqrt(
            gamma * rightSide["p"] / rightSide["rho"])

        self.gamma = gamma
        self.gp1_over2 = (self.gamma + 1.0) / 2.0
        self.gm1_over2 = (self.gamma - 1.0) / 2.0
        self.gm1_over2g = (self.gamma - 1.0) / (2.0 * self.gamma)
        self.gp1_over2g = (self.gamma + 1.0) / (2.0 * self.gamma)
        self.gm1 = self.gamma - 1.0
        self.gp1 = self.gamma + 1.0
        self.two_over_gp1 = 2.0 / (self.gamma + 1.0)
        self.gm1_over_gp1 = (self.gamma - 1.0) / (self.gamma + 1.0)
        self.two_over_gm1 = 2.0 / (self.gamma - 1.0)
        self.two_g_over_gm1 = 2.0 * self.gamma / (self.gamma - 1.0)

        try:
            self.p_contact = self.compute_p_contact()
        except:
            raise Exception("Newton method failed to find "
                            "solution for Riemann problem")

        assert self.complete_func(self.p_contact) < self.TOLERANCE

        self.detect_wave_configuration()

        self.u_contact = self.compute_u_contact()

        self.compute_solution()

        return self.rho_solution, self.u_solution, self.p_solution

    def compute_p_contact(self):
        p0 = 0.5 * (self.leftSide["p"] + self.rightSide["p"])
        number_of_iterations = 0

        while number_of_iterations <= self.MAX_ITERATIONS:
            func_val = self.complete_func(p0)
            func_derivative = self.complete_dfunc(p0)
            p = p0 - func_val / func_derivative
            rel_change = 2.0 * abs((p - p0) / (p + p0))
            if rel_change < self.TOLERANCE:
                return p
            if p <= 0:
                p = self.TOLERANCE
            p0 = p
            number_of_iterations += 1

        raise Exception(
            "solve_for_pressure() exceeded max number of iterations.")


    def detect_wave_configuration(self):
        if self.p_contact > self.leftSide["p"]:
            self.LEFT_SHOCK = True
            self.LEFT_RAREFACTION = False
        else:
            self.LEFT_RAREFACTION = True
            self.LEFT_SHOCK = False

        if self.p_contact > self.rightSide["p"]:
            self.RIGHT_SHOCK = True
            self.RIGHT_RAREFACTION = False
        else:
            self.RIGHT_RAREFACTION = True
            self.RIGHT_SHOCK = False

    def compute_u_contact(self):
        """ Compute velocity on C0-characteristic."""

        term1 = self.leftSide["u"] + self.rightSide["u"]
        func_right = self.func(self.p_contact, self.rightSide["p"],
                               self.rightSide["rho"], self.rightSide["c"])
        func_left = self.func(self.p_contact, self.leftSide["p"],
                              self.leftSide["rho"], self.leftSide["c"])
        term2 = func_right - func_left

        return 0.5 * (term1 + term2)

    def compute_rho_left_shock(self):
        assert self.LEFT_SHOCK is True
        p_ratio = self.p_contact / self.leftSide["p"]
        numer = p_ratio + self.gm1_over_gp1
        denom = self.gm1_over_gp1 * p_ratio + 1.0
        rho_left = self.leftSide["rho"] * numer / denom
        self.p_solution = self.p_contact
        self.u_solution = self.u_contact
        self.rho_solution = rho_left

    def compute_rho_left_rarefaction(self):
        assert self.LEFT_RAREFACTION is True
        # We consider two cases:
        # 1. We calculate the density behind the rarefaction.
        # 2. We compute the density inside the rarefaction.
        # 3. We compute the density ahead of the rarefaction
        # (it's just initial data on the left).
        p_ratio = self.p_contact / self.leftSide["p"]
        c_star = self.leftSide["c"] * math.pow(p_ratio, self.gm1_over2g)
        speed_head = self.leftSide["u"] - self.leftSide["c"]
        speed_tail = self.u_contact - c_star

        # We are between the tail of left rarefaction and contact discontinuity.
        if speed_tail <= 0:
            p_ratio = self.p_contact / self.leftSide["p"]
            rho_left = self.leftSide["rho"] * math.pow(p_ratio, 1.0 / self.gamma)
            self.p_solution = self.p_contact
            self.u_solution = self.u_contact
            self.rho_solution = rho_left
        # We are inside of the left rarefaction.
        elif speed_head < 0 and speed_tail > 0:
            expr = self.two_over_gp1 + (
                self.gm1_over_gp1 * self.leftSide["u"] / self.leftSide["c"])
            self.rho_solution = self.leftSide["rho"] * math.pow(expr, self.two_over_gm1)
            self.u_solution = self.two_over_gp1 * (self.leftSide["c"] +
                self.gm1_over2 * self.leftSide["u"])
            self.p_solution = self.leftSide["p"] * math.pow(
                expr, self.two_g_over_gm1)
        # We are ahead of the left rarefaction head.
        else:
            self.p_solution = self.leftSide["p"]
            self.u_solution = self.leftSide["u"]
            self.rho_solution = self.leftSide["rho"]

    def compute_rho_right_shock(self):
        assert self.RIGHT_SHOCK is True
        p_ratio = self.p_contact / self.rightSide["p"]
        numer = p_ratio + self.gm1_over_gp1
        denom = self.gm1_over_gp1 * p_ratio + 1.0
        rho_right = self.rightSide["rho"] * numer / denom
        self.p_solution = self.p_contact
        self.u_solution = self.u_contact
        self.rho_solution = rho_right

    def compute_rho_right_rarefaction(self):
        assert self.RIGHT_RAREFACTION is True
        # We consider two cases:
        # 1. We calculate the density behind the rarefaction.
        # 2. We compute the density inside the rarefaction.
        # 3. We compute the density ahead of the rarefaction
        # (it's just initial data on the left).
        p_ratio = self.p_contact / self.rightSide["p"]
        c_star = self.rightSide["c"] * math.pow(p_ratio, self.gm1_over2g)
        speed_head = self.rightSide["u"] + self.rightSide["c"]
        speed_tail = self.u_contact + c_star

        # We are between the tail of right rarefaction and contact discontinuity.
        if speed_tail >= 0:
            p_ratio = self.p_contact / self.rightSide["p"]
            rho = self.rightSide["rho"] * math.pow(p_ratio, 1.0 / self.gamma)
            self.p_solution = self.p_contact
            self.u_solution = self.u_contact
            self.rho_solution = rho
        # We are inside of the right rarefaction.
        elif speed_head > 0 and speed_tail < 0:
            expr = self.two_over_gp1 - (
                self.gm1_over_gp1 * self.rightSide["u"] / self.rightSide["c"])
            self.rho_solution = self.rightSide["rho"] * math.pow(expr, self.two_over_gm1)
            self.u_solution = self.two_over_gp1 * (-self.rightSide["c"] +
                                                  self.gm1_over2 * self.rightSide["u"])
            self.p_solution = self.rightSide["p"] * math.pow(
                expr, self.two_g_over_gm1)
        # We are ahead of the right rarefaction head.
        else:
            self.rho_solution = self.rightSide["rho"]
            self.u_solution = self.rightSide["u"]
            self.p_solution = self.rightSide["p"]

    def compute_solution(self):
        if self.u_contact > 0:
            if self.LEFT_SHOCK:
                self.compute_rho_left_shock()

            if self.LEFT_RAREFACTION:
                self.compute_rho_left_rarefaction()
        else:
            if self.RIGHT_SHOCK:
                self.compute_rho_right_shock()

            if self.RIGHT_RAREFACTION:
                self.compute_rho_right_rarefaction()

    def sample_solution(self, xi):
        if xi < self.u_contact:
            if self.LEFT_SHOCK:
                left_shock_speed = self.leftSide["u"] - self.massvel_left / \
                                   self.leftSide["rho"]
                if xi < left_shock_speed:
                    rho = self.compute_rho_left_shock()
                    return rho, self.u_contact, self.p_contact
                else:
                    return self.leftSide["rho"], self.leftSide["u"], \
                           self.leftSide["p"]
            # LEFT_RAREFACTION
            else:
                p_ratio = self.p_contact / self.leftSide["p"]
                left_rarefaction_head = self.leftSide["u"] - self.leftSide["c"]
                c_left_star = self.leftSide["c"] * math.pow(
                    p_ratio, self.gm1_over2g)
                left_rarefaction_tail = self.u_contact - c_left_star
                if xi < left_rarefaction_head:
                    return self.leftSide["rho"],\
                        self.leftSide["u"], \
                        self.leftSide["p"]
                elif xi > left_rarefaction_tail:
                    p_ratio = self.p_contact / self.leftSide["p"]
                    rho = self.leftSide["rho"] * math.pow(
                        p_ratio, 1.0 / self.gamma)
                    return rho, self.u_contact, self.p_contact
                # Inside of left rarefaction wave
                else:
                    expr = self.two_over_gp1 + (
                        self.gm1_over_gp1 * (self.leftSide["u"] - xi) / self.leftSide["c"])
                    rho = self.leftSide["rho"] * math.pow(expr, self.two_over_gm1)
                    u = self.two_over_gp1 * (self.leftSide["c"] +
                                               self.gm1_over2 * self.leftSide["u"] + xi)
                    p = self.leftSide["p"] * math.pow(
                expr, self.two_g_over_gm1)

                    return rho, u, p
        else:
            if self.RIGHT_SHOCK:
                Ar = 2.0 / ((self.gamma + 1.0) * self.rightSide["rho"])
                Br = self.gm1_over_gp1 * self.rightSide["p"]
                qr = math.sqrt((self.p_contact + Br) / Ar)
                right_shock_speed = self.rightSide["u"] + qr / self.rightSide[
                    "rho"]
                if xi > right_shock_speed:
                    return self.rightSide["rho"], self.rightSide["u"], \
                           self.rightSide["p"]
                else:
                    rho = self.compute_rho_right_shock()
                    return rho, self.u_contact, self.p_contact
            # Right wave is rarefaction
            else:
                right_rarefaction_head_speed = self.rightSide["u"] + \
                                               self.rightSide["c"]
                c_right_star = self.rightSide["c"] - self.gm1_over2 * (
                    self.rightSide["u"] - self.u_contact)
                right_rarefaction_tail_speed = self.u_contact + c_right_star
                if xi > right_rarefaction_head_speed:
                    return self.rightSide["rho"], self.rightSide["u"], \
                           self.rightSide["p"]
                elif xi < right_rarefaction_tail_speed:
                    rho = self.compute_rho_right_rarefaction()
                    return rho, self.u_contact, self.p_contact
                # Inside of right rarefaction wave
                else:
                    rho = self.rightSide["rho"] * ( \
                        self.two_over_gp1 - self.gm1_over_gp1 * (
                            self.rightSide["u"] - xi)) ** self.two_over_gm1
                    u = self.two_over_gp1 * ( \
                        - self.rightSide["c"] + self.gm1_over2 * self.rightSide[
                            "u"] + xi)
                    p = self.rightSide["p"] * ( \
                        self.two_over_gp1 - self.gm1_over_gp1 * (
                            self.rightSide["u"] - xi)) ** self.two_g_over_gm1

                    return rho, u, p
