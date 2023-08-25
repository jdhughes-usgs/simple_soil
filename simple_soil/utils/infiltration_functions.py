import numpy as np

from .newton_raphson import newton_raphson


class Infiltration:
    def __init__(self):
        raise NotImplementedError("do not use base Infiltration class")

    def infiltration(self, rate: float, theta: float):
        raise NotImplementedError("do not use base Infiltration class")


class InfiltrationConstantLoss(Infiltration):
    def __init__(
        self,
        max_vertical_rate: float,
    ):
        self.max_vertical_rate = max_vertical_rate

    def infiltration(self, rate: float, theta: float):
        return min(rate, self.max_vertical_rate)


class GreenAmpt(Infiltration):
    def __init__(
        self,
        theta_sat: float,
        K_sat: float,
        length_units: str,
        delta_F: float = 1.0e-4,
        soil: str = "sand",
    ):
        self.theta_sat = theta_sat
        self.K_sat = K_sat
        self.length_units = length_units

        self._set_green_ampt_suction_head(soil)

        self.F_t = 0.0
        self.f_t = 0.0
        self.delta_F = delta_F

        self.theta = 0.0
        self.delta_theta = 0.0
        self.total_time = 0.0

        self.iterations = 0
        self.error = 0.0

    def _set_green_ampt_suction_head(self, soil: str) -> None:
        # wetting front suction head - Rawls, Brakensiek, and Miller (1983)
        # values are in inches
        # (https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/applying-loss-methods-within-hec-hms/applying-the-green-and-ampt-loss-method}
        suction_head_dict = {
            "sand": 1.9,
            "loamy sand": 2.4,
            "sandy loam": 4.3,
            "loam": 3.5,
            "silt loam": 6.6,
            "sandy clay loam": 8.6,
            "clay Loam": 8.2,
            "silty clay loam": 10.7,
            "sandy clay": 9.4,
            "silty clay": 11.5,
            "clay": 12.5,
        }
        if soil not in list(suction_head_dict.keys()):
            raise ValueError(
                f"Invalid soil type ({soil}). Valid soil types are "
                + f"'{', '.join(list(suction_head_dict.keys()))}'"
            )

        if self.length_units == "m":
            conversion_factor = 2.54 / 100.0
        elif self.length_units == "cm":
            conversion_factor = 2.54
        elif self.length_units == "ft":
            conversion_factor = 1.0 / 12.0
        elif self.length_units == "in":
            conversion_factor = 1.0

        self.soil = soil
        self.psi = suction_head_dict[soil] * conversion_factor

    def _green_ampt_residual(self, F: float):
        v = abs(self.psi) * self.delta_theta
        return F - v * np.log(1.0 + F / v) - self.K_sat * self.total_time

    def _green_ampt_derivative(self, F: float):
        return (
            self._green_ampt_residual(F + self.delta_F)
            - self._green_ampt_residual(F)
        ) / self.delta_F

    def _green_ampt_infiltration(self):
        v = abs(self.psi) * self.delta_theta
        if self.F_t == 0.0:
            f = 0.0
        else:
            f = self.K_sat * ((v / self.F_t) + 1)
        self.f_t = f
        return f

    def infiltration(
        self,
        rate: float,
        theta: float,
    ) -> float:
        self.theta = theta
        self.delta_theta = self.theta_sat - self.theta
        iterations, F, residual, converged = newton_raphson(
            self._green_ampt_residual,
            self._green_ampt_derivative,
            self.F_t,
        )
        self.F_t = F
        self.iterations = iterations
        self.error = residual
        self.f_t = self._green_ampt_infiltration()
        return min(self.f_t, rate)
