import math


class riemannSolver:
    """
        Exact Riemann solver for calorically perfect gas from the book
        Toro, E. F. (2009). Riemann Solvers and Numerical Methods
        for Fluid Dynamics. A Practical Introduction (3rd ed.).
        Springer Berlin Heidelberg. doi:10.1007/b79761
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

    def solve(self, leftSide, rightSide, gamma):
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

        self._compute_aux_quantities()

        self.p_contact = self._compute_p_contact()
        assert self._complete_func(self.p_contact) < self.TOLERANCE

        self.u_contact = self._compute_u_contact()

        return self.sample_solution(0)

    def sample_solution(self, xi):
        self._detect_wave_configuration()

        # We find the solution to the left of contact.
        if xi <= self.u_contact:
            if self.LEFT_SHOCK:
                return self._compute_left_shock(xi)
            # LEFT_RAREFACTION
            else:
                return self._compute_left_rarefaction(xi)

        else:
            if self.RIGHT_SHOCK:
                return self._compute_right_shock(xi)
            # Right wave is rarefaction
            else:
                return self._compute_right_rarefaction(xi)

    def _compute_aux_quantities(self):
        self.LEFT_A = 2.0 / (self.gp1 * self.leftSide["rho"])
        self.RIGHT_A = 2.0 / (self.gp1 * self.rightSide["rho"])
        self.LEFT_B = self.gm1_over_gp1 * self.leftSide["p"]
        self.RIGHT_B = self.gm1_over_gp1 * self.rightSide["p"]

    def _compute_p_contact(self):
        """ Find pressure in the region between waves.
        Use Newton method to find the pressure.
        """
        p0 = 0.5 * (self.leftSide["p"] + self.rightSide["p"])
        number_of_iterations = 0

        while number_of_iterations <= self.MAX_ITERATIONS:
            func_val = self._complete_func(p0)
            func_derivative = self._complete_dfunc(p0)
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

    def _compute_u_contact(self):
        """ Compute velocity in the region between waves."""

        term1 = self.leftSide["u"] + self.rightSide["u"]
        func_right = self._func(self.p_contact, self.rightSide["p"],
                                self.rightSide["rho"], self.rightSide["c"],
                                self.RIGHT_A, self.RIGHT_B)
        func_left = self._func(self.p_contact, self.leftSide["p"],
                               self.leftSide["rho"], self.leftSide["c"],
                               self.LEFT_A, self.LEFT_B)
        term2 = func_right - func_left

        return 0.5 * (term1 + term2)

    def _complete_func(self, p_iter):
        term1 = self._func(p_iter, self.leftSide["p"], self.leftSide["rho"],
                           self.leftSide["c"], self.LEFT_A, self.LEFT_B)
        term2 = self._func(p_iter, self.rightSide["p"], self.rightSide["rho"],
                           self.rightSide["c"], self.RIGHT_A, self.RIGHT_B)
        term3 = self.rightSide["u"] - self.leftSide["u"]
        return term1 + term2 + term3

    def _func(self, p_iter, p, rho, c, A, B):
        assert p_iter > 0, 'Pressure must be positive'
        # Shock wave.
        if p_iter > p:
            coeff = p_iter - p
            coeff2 = math.sqrt(A / (p_iter + B))
            return coeff * coeff2
        # Rarefaction wave.
        else:
            p_ratio = p_iter / p
            coeff1 = (2.0 * c) / self.gm1
            coeff2 = math.pow(p_ratio, self.gm1_over2g) - 1.0
            return coeff1 * coeff2

    def _complete_dfunc(self, p_iter):
        assert p_iter > 0, 'Pressure must be positive'
        term1 = self._dfunc(p_iter, self.leftSide["p"], self.leftSide["rho"],
                            self.leftSide["c"], self.LEFT_A, self.LEFT_B)
        term2 = self._dfunc(p_iter, self.rightSide["p"], self.rightSide["rho"],
                            self.rightSide["c"], self.RIGHT_A, self.RIGHT_B)
        return term1 + term2

    def _dfunc(self, p_iter, p, rho, c, A, B):
        assert p_iter > 0, 'Pressure must be positive'
        # Shock wave.
        if p_iter > p:
            coeff1 = math.sqrt(A / (B + p_iter))
            coeff2 = 1 - (p_iter - p) / (2 * (B + p_iter))
            result = coeff1 * coeff2
            assert result > 0, "f'(p) must be positive (shock case)"
            return result
        # Rarefaction wave.
        else:
            p_ratio = p_iter / p
            power_expr = math.pow(p_ratio, -self.gp1_over2g)
            coeff = 1.0 / (rho * c)
            result = coeff * power_expr
            assert result > 0, "f'(p) must be positive (rarefaction case)"
            return result

    def _detect_wave_configuration(self):
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

    def _compute_left_shock(self, xi):
        assert self.LEFT_SHOCK is True and self.LEFT_RAREFACTION is False
        mass_flux = math.sqrt((self.p_contact + self.LEFT_B) / self.LEFT_A)
        shock_speed = self.leftSide["u"] - mass_flux / self.leftSide["rho"]

        # We compute solution behind the shock.
        if xi >= shock_speed:
            p_ratio = self.p_contact / self.leftSide["p"]
            numer = p_ratio + self.gm1_over_gp1
            denom = self.gm1_over_gp1 * p_ratio + 1.0
            rho = self.leftSide["rho"] * numer / denom
            return rho, self.u_contact, self.p_contact
        else:
            return self.leftSide["rho"], self.leftSide["u"], self.leftSide["p"]

    def _compute_left_rarefaction(self, xi):
        assert self.LEFT_RAREFACTION is True and self.LEFT_SHOCK is False
        p_ratio = self.p_contact / self.leftSide["p"]
        c_star = self.leftSide["c"] * math.pow(p_ratio, self.gm1_over2g)
        speed_head = self.leftSide["u"] - self.leftSide["c"]
        speed_tail = self.u_contact - c_star

        # We are between the tail of left rarefaction and contact.
        if speed_tail <= xi:
            p_ratio = self.p_contact / self.leftSide["p"]
            rho = self.leftSide["rho"] * math.pow(p_ratio, 1.0 / self.gamma)
            return rho, self.u_contact, self.p_contact
        # We are inside of the left rarefaction.
        elif speed_head < xi < speed_tail:
            expr = self.two_over_gp1 + (
                self.gm1_over_gp1 * (self.leftSide["u"] - xi) /
                self.leftSide["c"])
            rho = self.leftSide["rho"] * math.pow(expr, self.two_over_gm1)
            u = self.two_over_gp1 * (self.leftSide["c"] +
                                     self.gm1_over2 * self.leftSide["u"] + xi)
            p = self.leftSide["p"] * math.pow(expr, self.two_g_over_gm1)
            return rho, u, p
        # We are ahead of the left rarefaction head.
        else:
            return self.leftSide["rho"], self.leftSide["u"], self.leftSide["p"]

    def _compute_right_shock(self, xi):
        assert self.RIGHT_SHOCK is True and self.RIGHT_RAREFACTION is False
        mass_flux = math.sqrt((self.p_contact + self.RIGHT_B) / self.RIGHT_A)
        shock_speed = self.rightSide["u"] + mass_flux / self.rightSide["rho"]

        # We are between contact discontinuity and right shock.
        if xi <= shock_speed:
            p_ratio = self.p_contact / self.rightSide["p"]
            numer = p_ratio + self.gm1_over_gp1
            denom = self.gm1_over_gp1 * p_ratio + 1.0
            rho = self.rightSide["rho"] * numer / denom
            return rho, self.u_contact, self.p_contact
        # We are ahead of right shock.
        else:
            return (self.rightSide["rho"], self.rightSide["u"],
                    self.rightSide["p"])

    def _compute_right_rarefaction(self, xi):
        assert self.RIGHT_RAREFACTION is True and self.RIGHT_SHOCK is False

        # Compute speed of rarefaction head and tail.
        p_ratio = self.p_contact / self.rightSide["p"]
        c_star = self.rightSide["c"] * math.pow(p_ratio, self.gm1_over2g)
        speed_head = self.rightSide["u"] + self.rightSide["c"]
        speed_tail = self.u_contact + c_star

        # We are between the tail of right rarefaction and contact.
        if speed_tail >= xi:
            p_ratio = self.p_contact / self.rightSide["p"]
            rho = self.rightSide["rho"] * math.pow(
                p_ratio, 1.0 / self.gamma)
            return rho, self.u_contact, self.p_contact
        # We are inside of the right rarefaction.
        elif speed_tail < xi < speed_head:
            expr = self.two_over_gp1 - (
                self.gm1_over_gp1 * ((self.rightSide["u"] - xi) /
                                     self.rightSide["c"]))
            rho = self.rightSide["rho"] * math.pow(
                expr, self.two_over_gm1)
            u = self.two_over_gp1 * (-self.rightSide["c"] +
                                     self.gm1_over2 * self.rightSide["u"] + xi)
            p = self.rightSide["p"] * math.pow(expr, self.two_g_over_gm1)
            return rho, u, p
        # We are ahead of the right rarefaction head.
        else:
            return (self.rightSide["rho"], self.rightSide["u"],
                    self.rightSide["p"])
