from math import sqrt
import scipy.optimize as sco


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


    def func(self, p_iter, p, rho, c):
        if (p_iter >= p):
            numer = p_iter - p
            p_ratio = p_iter / p
            sqrt_expr = (self.gamma + 1) / (2 * self.gamma) * p_ratio + \
                (self.gamma - 1) / (2 * self.gamma)
            denom = rho * c * sqrt(sqrt_expr)

            return numer / denom
        else:
            p_ratio = p_iter / p
            paren_expr = p_ratio ** ((self.gamma - 1) / (2 * self.gamma)) - 1
            return 2.0 / (self.gamma - 1) * c * paren_expr

    def dfunc(self, p_iter, p, rho, c):
        p_ratio = p_iter / p
        if p_iter >= p:
            numer = (self.gamma + 1) * p_ratio + (3 * self.gamma - 1)
            sqrt_expr = ((self.gamma + 1) * p_ratio / (2 * self.gamma) + \
                (self.gamma - 1) / (2 * self.gamma))**3
            denom = 4 * self.gamma * rho * c * sqrt(sqrt_expr)
            return numer / denom
        else:
            power_expr = p_ratio ** ((self.gamma - 1) / (2 * self.gamma))
            denom_expr = self.gamma * p_iter
            return c * power_expr / denom_expr


    def complete_func(self, p_iter):
        term1 = self.func(p_iter, self.leftSide["p"], self.leftSide["rho"], self.leftSide["c"])
        term2 = self.func(p_iter, self.rightSide["p"], self.rightSide["rho"], self.rightSide["c"])
        term3 = self.rightSide["u"] - self.leftSide["u"] 
        return term1 + term2 + term3

    
    def complete_dfunc(self, p_iter):
        term1 = self.dfunc(p_iter, self.leftSide["p"], self.leftSide["rho"], self.leftSide["c"])
        term2 = self.dfunc(p_iter, self.rightSide["p"], self.rightSide["p"], self.rightSide["c"])

        return term1 + term2

    def compute_mass_velocity_left_shock(self):
        return sqrt(self.leftSide["rho"] * ( \
            self.gp1_over2 * self.p_solution + self.gm1_over2 * self.leftSide["p"]
            ))


    def compute_mass_velocity_right_shock(self):
        return sqrt(self.rightSide["rho"] * ( \
            self.gp1_over2 * self.p_solution + self.gm1_over2 * self.rightSide["p"]
            ))

    def compute_mass_velocity_left_rarefaction(self):
        p_ratio = self.p_solution / self.leftSide["p"]
        return self.gm1_over2g * self.leftSide["rho"] * self.leftSide["c"] * \
            (1.0 - p_ratio) / (1.0 - p_ratio**self.gm1_over2g)


    def compute_mass_velocity_right_rarefaction(self):
        p_ratio = self.p_solution / self.rightSide["p"]
        return self.gm1_over2g * self.rightSide["rho"] * self.rightSide["c"] * \
            (1.0 - p_ratio) / (1.0 - p_ratio**self.gm1_over2g)


    def solve(self, leftSide, rightSide, gamma):
        self.leftSide = leftSide
        self.rightSide = rightSide

        self.leftSide["c"] = sqrt(gamma * leftSide["p"] / leftSide["rho"])
        self.rightSide["c"] = sqrt(gamma * rightSide["p"] / rightSide["rho"])
        
        self.gamma = gamma
        self.gp1_over2 = (self.gamma + 1.0) / 2.0
        self.gm1_over2 = (self.gamma - 1.0) / 2.0
        self.gm1_over2g = (self.gamma - 1.0) / (2.0 * self.gamma)
        self.gm1 = self.gamma - 1.0
        self.two_over_gp1 = 2.0 / (self.gamma + 1.0)
        self.gm1_over_gp1 =(self.gamma - 1.0) / (self.gamma + 1.0)
        self.two_over_gm1 = 2.0 / (self.gamma - 1.0)
        self.two_g_over_gm1 = 2.0 * self.gamma / (self.gamma - 1.0)

        numer = leftSide["p"] * rightSide["rho"] * rightSide["c"] + \
            rightSide["p"] * leftSide["rho"] * leftSide["c"] + \
            (leftSide["u"] - rightSide["u"]) * leftSide["rho"] * \
            leftSide["c"] * rightSide["rho"] * rightSide["c"]

        denom = leftSide["rho"] * leftSide["c"] + rightSide["rho"] * \
            rightSide["c"]

        p0 = 0.5 * (self.leftSide["p"] + self.rightSide["p"])

        self.p_solution = sco.newton(self.complete_func, p0, self.complete_dfunc)

        self.detect_wave_configuration()

        if self.LEFT_SHOCK:
            self.massvel_left = self.compute_mass_velocity_left_shock()
        if self.RIGHT_SHOCK:
            self.massvel_right = self.compute_mass_velocity_right_shock()
        if self.LEFT_RAREFACTION:
            self.massvel_left = self.compute_mass_velocity_left_rarefaction()
        if self.RIGHT_RAREFACTION:
            self.massvel_right = self.compute_mass_velocity_right_rarefaction()

        u_numer = self.massvel_left * self.leftSide["u"] + \
            self.massvel_right * self.rightSide["u"] + self.leftSide["p"] - \
            self.rightSide["p"]
        u_denom = self.massvel_left + self.massvel_right
        self.u_solution = u_numer / u_denom

        self.rho_solution = self.compute_rho_solution()

        return self.rho_solution, self.u_solution, self.p_solution


    def detect_wave_configuration(self):
        switch = False
        if self.leftSide["p"] > self.rightSide["p"]:
            switch = True
            tmp = self.leftSide
            self.leftSide = self.rightSide
            self.rightSide = tmp
            self.leftSide["u"] = -self.leftSide["u"]
            self.rightSide["u"] = -self.rightSide["u"]

        self.u_shock = self.compute_u_shock()
        self.u_rarefaction = self.compute_u_rarefaction()
        self.u_vacuum = self.compute_u_vacuum()

        u_diff = (self.leftSide["u"] - self.rightSide["u"])
        if u_diff > self.u_shock:
            self.LEFT_SHOCK = True
            self.RIGHT_SHOCK = True

        if self.u_rarefaction < u_diff < self.u_shock:
            self.LEFT_SHOCK = True
            self.RIGHT_RAREFACTION = True

        if self.u_vacuum < u_diff < self.u_rarefaction:
            self.LEFT_RAREFACTION = True
            self.RIGHT_RAREFACTION = True

        if u_diff < self.u_vacuum:
            raise Exception("riemannSolver - vacuum")

        # As we swapped parameters we need to return everything as it was.
        if switch:
            tmp = self.leftSide
            self.leftSide = self.rightSide
            self.rightSide = tmp
            self.leftSide["u"] = -self.leftSide["u"]
            self.rightSide["u"] = -self.rightSide["u"]

            # As we were considering situation when p_left <= p_right
            # we need to switch waves directions
            if self.LEFT_SHOCK and self.RIGHT_RAREFACTION:
                self.LEFT_SHOCK = False
                self.RIGHT_SHOCK = True
                self.RIGHT_RAREFACTION = False
                self.LEFT_RAREFACTION = True


    def compute_u_shock(self):
        u_shock_numer = self.rightSide["p"] - self.leftSide["p"]
        u_shock_denom = sqrt(self.leftSide["rho"] * ( \
            self.gp1_over2 * self.rightSide["p"] + self.gm1_over2 * self.leftSide["p"]))

        return u_shock_numer / u_shock_denom


    def compute_u_rarefaction(self):
        coeff = - 2.0 * self.rightSide["c"] / self.gm1
        frac = self.leftSide["p"] / self.rightSide["p"]
        power = frac ** self.gm1_over2g

        return coeff * (1 - power)


    def compute_u_vacuum(self):
        term1 = 2.0 * self.leftSide["c"] / self.gm1
        term2 = 2.0 * self.rightSide["c"] / self.gm1

        return - term1 - term2


    def compute_rho_left_shock(self):
        rho_left = self.leftSide["rho"] * self.massvel_left / \
            (self.massvel_left - self.leftSide["rho"] * \
            (self.leftSide["u"] - self.u_solution))

        return rho_left


    def compute_rho_left_rarefaction(self):
        c_left = self.leftSide["c"] + self.gm1_over2 * \
            (self.leftSide["u"] - self.u_solution)
        rho_left = self.gamma * self.p_solution / c_left ** 2
        return rho_left


    def compute_rho_right_shock(self):
        rho_right = self.rightSide["rho"] * self.massvel_right / \
            (self.massvel_right - self.rightSide["rho"] * \
            (self.rightSide["u"] - self.u_solution))
        return rho_right


    def compute_rho_right_rarefaction(self):
        c_right = self.rightSide["c"] - self.gm1_over2 * \
            (self.rightSide["u"] - self.u_solution)
        rho_right = self.gamma * self.p_solution / c_right ** 2
        return rho_right


    def compute_rho_solution(self):
        if self.LEFT_SHOCK:
            rho_left = self.compute_rho_left_shock()
            
        if self.LEFT_RAREFACTION:
            rho_left = self.compute_rho_left_rarefaction()

        if self.RIGHT_SHOCK:
            rho_right = self.compute_rho_right_shock()

        if self.RIGHT_RAREFACTION:
            rho_right = self.compute_rho_right_rarefaction()

        if self.u_solution > 0:
            return rho_left
        else:
            return rho_right

    def sample_solution(self, xi):
        if xi < self.u_solution:
            if self.LEFT_SHOCK:
                left_shock_speed = self.leftSide["u"] - self.massvel_left / self.leftSide["rho"]
                if xi < left_shock_speed:
                    rho = self.compute_rho_left_shock()
                    return rho, self.u_solution, self.p_solution
                else:
                    return self.leftSide["rho"], self.leftSide["u"], \
                        self.leftSide["p"]
            # LEFT_RAREFACTION
            else:
                left_rarefaction_head = self.leftSide["u"] - self.leftSide["c"]
                c_left_star = self.leftSide["c"] + \
                    self.gm1_over2 * (self.leftSide["u"] - self.u_solution)
                left_rarefaction_tail = self.u_solution - c_left_star
                if xi < left_rarefaction_head:
                    return self.leftSide["rho"], self.leftSide["u"], \
                        self.leftSide["p"]
                elif xi > left_rarefaction_tail:
                    rho = self.compute_rho_left_rarefaction()
                    return rho, self.u_solution, self.p_solution
                # Inside of left rarefaction wave
                else:
                    rho = self.leftSide["rho"] * ( \
                        self.two_over_gp1 + self.gm1_over_gp1 * \
                        (self.leftSide["u"] - xi) / self.leftSide["c"])**self.two_over_gm1
                    u = self.two_over_gp1 * ( \
                        self.leftSide["c"] + self.gm1_over2 * self.leftSide["u"] + xi)
                    p = self.leftSide["p"] * ( \
                        self.two_over_gp1 + self.gm1_over_gp1 * (self.leftSide["u"] - xi) / self.leftSide["c"])**self.two_g_over_gm1

                    return rho, u, p
        else:
            if self.RIGHT_SHOCK:
                right_shock_speed = self.rightSide["u"] + self.massvel_right / self.rightSide["rho"]
                if xi > right_shock_speed:
                    return self.rightSide["rho"], self.rightSide["u"], \
                        self.rightSide["p"]
                else:
                    rho = self.compute_rho_right_shock()
                    return rho, self.u_solution, self.p_solution
            # Right wave is rarefaction
            else:
                right_rarefaction_head_speed = self.rightSide["u"] + self.rightSide["c"]
                c_right_star = self.rightSide["c"] - self.gm1_over2 * (self.rightSide["u"] - self.u_solution)
                right_rarefaction_tail_speed = self.u_solution + c_right_star
                if xi > right_rarefaction_head_speed:
                    return self.rightSide["rho"], self.rightSide["u"], \
                        self.rightSide["p"]
                elif xi < right_rarefaction_tail_speed:
                    rho = self.compute_rho_right_rarefaction()
                    return rho, self.u_solution, self.p_solution
                # Inside of right rarefaction wave
                else:
                    rho = self.rightSide["rho"] * ( \
                        self.two_over_gp1 - self.gm1_over_gp1 * (self.rightSide["u"] - xi)) ** self.two_over_gm1
                    u = self.two_over_gp1 * ( \
                        - self.rightSide["c"] + self.gm1_over2 * self.rightSide["u"] + xi)
                    p = self.rightSide["p"] * ( \
                        self.two_over_gp1 - self.gm1_over_gp1 * (self.rightSide["u"] - xi)) ** self.two_g_over_gm1

                    return rho, u , p
