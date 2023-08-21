from ..utils.flow_functions import (
    aet_volumetric_rate,
    infiltration_volumetric_rate,
    recharge_volumetric_rate,
    rejected_infiltration_volumetric_rate,
    volume_change_rate,
)
from ..utils.fraction_functions import saturation_fraction
from ..utils.newton_raphson import newton_raphson


class ControlVolume:
    def __init__(
        self,
        area: float = 1.0,
        thickness: float = 1.0,
        discharge_thickness: float = 0.1,
        theta0: float = 0.01,
        theta_wp: float = 0.01,
        theta_fc: float = 0.1,
        theta_sat: float = 0.2,
        max_vertical_rate: float = 1e-3,
        horizontal_vertical_ratio: float = 10.0,
        pet_fraction: float = 0.15,
        smoothing_omega: float = 1.0e-6,
        delta_theta: float = 1.0e-4,
        max_iterations: int = 100,
    ) -> "ControlVolume":
        self.area = area
        self.thickness = thickness
        self.discharge_thickness = discharge_thickness
        self.theta0 = theta0
        self.theta_wp = theta_wp
        self.theta_fc = theta_fc
        self.theta_sat = theta_sat
        self.max_vertical_rate = max_vertical_rate
        self.horizontal_vertical_ratio = horizontal_vertical_ratio
        self.pet_fraction = pet_fraction
        self.theta_pet_max = (
            self.theta_wp
            + (self.theta_sat - self.theta_wp) * self.pet_fraction
        )
        self.smoothing_omega = smoothing_omega
        self.delta_theta = delta_theta
        self.max_iterations = max_iterations

        self._validate()

        self.volume_max = theta_sat * area * thickness
        self.volume0 = theta0 * area * thickness
        self.theta = theta0
        self.volume = theta0 * area * thickness
        self.theta_discharge = (
            theta_sat * (thickness - discharge_thickness) / thickness
        )
        self.max_horizontal_rate = (
            max_vertical_rate * horizontal_vertical_ratio
        )

        # time step data
        self.inflow_rate = None
        self.pet_rate = None
        self.delta_t = None

        # solver data
        self.iterations = None
        self.error = None
        self.converged = False

        # output data
        self.total_time = 0.0
        self.inflow_volume = None
        self.aet_volume = None
        self.recharge_volume = None
        self.storage_volume_change = None

        self.output_dict = {
            "total_time": [],
            "iterations": [],
            "theta": [],
            "volume": [],
            "rejected_inflow_volume": [],
            "inflow_volume": [],
            "aet_volume": [],
            "recharge_volume": [],
            "storage_volume_change": [],
            "residual": [],
        }

    def __repr__(self):
        values = ""
        for key, value in sorted(self.__dict__.items()):
            values += f"{key}={value}\n"
        return f"{values}"

    def _validate(self):
        if self.area < 0.0:
            raise ValueError(
                f"control volume area ({self.area}) "
                + "must be greater than zero"
            )
        if self.thickness < 0.0:
            raise ValueError(
                f"control volume thickness ({self.thickness}) "
                + "must be greater than zero"
            )
        if self.discharge_thickness < 0.0:
            raise ValueError(
                f"control volume discharge thickness"
                + f" ({self.discharge_thickness}) must be greater than zero"
            )
        if self.theta_wp < 0.0:
            raise ValueError(
                f"wilting point ({self.theta_wp}) must be greater than zero"
            )
        if self.theta_wp > self.theta_fc:
            raise ValueError(
                f"wilting point ({self.theta_wp}) must "
                + f"be less than field capacity ({self.theta_fc})"
            )
        if self.theta_fc > self.theta_sat:
            raise ValueError(
                f"field capacity ({self.theta_fc}) must "
                + f"be less than theta_sat ({self.theta_sat})"
            )
        if self.theta0 < 0.0:
            raise ValueError(
                f"initial moisture content ({self.theta0}) must "
                + "be greater than zero"
            )
        if self.theta0 > self.theta_sat:
            raise ValueError(
                f"initial moisture content ({self.theta_fc}) must "
                + f"be less than theta_sat ({self.theta_sat})"
            )
        if self.max_vertical_rate < 0.0:
            raise ValueError(
                f"maximum vertical rate ({self.max_vertical_rate}) "
                + "must be greater than zero"
            )
        if self.horizontal_vertical_ratio <= 0.0:
            raise ValueError(
                "horizontal to vertical ratio "
                + f"({self.horizontal_vertical_ratio}) "
                + "must be greater than zero"
            )
        if self.pet_fraction < 0.0:
            raise ValueError(
                f"pet_fraction ({self.pet_fraction}) must be greater than zero"
            )
        if self.pet_fraction > 1.0:
            raise ValueError(
                f"pet_fraction ({self.pet_fraction}) must be "
                + "less than or equal to one"
            )
        if self.smoothing_omega < 0.0:
            raise ValueError(
                f"smoothing_omega ({self.smoothing_omega}) must "
                + "be greater than zero"
            )
        if self.delta_theta < 0.0:
            raise ValueError(
                f"delta_theta ({self.delta_theta}) must "
                + "be greater than zero"
            )
        if self.max_iterations <= 1:
            raise ValueError(
                f"max_iterations ({self.max_iterations}) must "
                + "be greater than or equal to one"
            )

    def _calculate_volume(
        self,
        water_content: float,
    ):
        return (
            saturation_fraction(
                water_content,
                self.theta_sat,
                smoothing_omega=self.smoothing_omega,
            )
            * self.thickness
            * self.area
        )

    def update(
        self,
        inflow_rate: float = 0.0,
        pet_rate: float = 0.0,
        delta_t: float = 1.0,
    ):
        self.advance(
            inflow_rate=inflow_rate,
            pet_rate=pet_rate,
            delta_t=delta_t,
        )

        self.solve()

        self.output()

    def advance(
        self,
        inflow_rate: float = 0.0,
        pet_rate: float = 0.0,
        delta_t: float = 1.0,
    ) -> None:
        self.theta0 = self.theta
        self.volume0 = self._calculate_volume(self.theta)

        self.inflow_rate = inflow_rate
        self.pet_rate = pet_rate
        self.delta_t = delta_t
        self.total_time += delta_t

    def solve(
        self,
    ) -> bool:
        iterations, theta, residual, converged = newton_raphson(
            self.residual,
            self.derivative,
            self.theta,
            max_iter=self.max_iterations,
        )
        self.iterations = iterations
        self.theta = theta
        self.error = residual
        self.converged = converged
        self.volume = self._calculate_volume(theta)
        return

    def output(self):
        water_content = self.theta
        self.inflow_volume = infiltration_volumetric_rate(
            water_content,
            self.inflow_rate,
            self.theta_sat,
            self.theta_discharge,
            self.area,
            self.max_vertical_rate,
            smoothing_omega=self.smoothing_omega,
        )
        self.aet_volume = aet_volumetric_rate(
            water_content,
            self.pet_rate,
            self.theta_pet_max,
            self.theta_wp,
            self.area,
            smoothing_omega=self.smoothing_omega,
        )
        self.recharge_volume = recharge_volumetric_rate(
            water_content,
            self.theta_sat,
            self.theta_fc,
            self.area,
            self.max_vertical_rate,
            self.smoothing_omega,
        )
        self.storage_volume_change = volume_change_rate(
            water_content,
            self.theta0,
            self.theta_sat,
            self.area,
            self.thickness,
            self.delta_t,
            smoothing_omega=self.smoothing_omega,
        )

        self.output_dict["total_time"].append(self.total_time)
        self.output_dict["iterations"].append(self.iterations)

        self.output_dict["theta"].append(self.theta)
        self.output_dict["volume"].append(self.volume)

        self.output_dict["inflow_volume"].append(self.inflow_volume)
        self.output_dict["rejected_inflow_volume"].append(
            rejected_infiltration_volumetric_rate(
                water_content,
                self.inflow_rate,
                self.theta_sat,
                self.theta_discharge,
                self.area,
                self.max_vertical_rate,
                smoothing_omega=self.smoothing_omega,
            )
        )
        self.output_dict["aet_volume"].append(self.aet_volume)
        self.output_dict["recharge_volume"].append(self.recharge_volume)
        self.output_dict["storage_volume_change"].append(
            self.storage_volume_change
        )

        self.output_dict["residual"].append(self.error)

        return

    def residual(self, water_content: float) -> float:
        return (
            infiltration_volumetric_rate(
                water_content,
                self.inflow_rate,
                self.theta_sat,
                self.theta_discharge,
                self.area,
                self.max_vertical_rate,
                smoothing_omega=self.smoothing_omega,
            )
            + aet_volumetric_rate(
                water_content,
                self.pet_rate,
                self.theta_pet_max,
                self.theta_wp,
                self.area,
                smoothing_omega=self.smoothing_omega,
            )
            + recharge_volumetric_rate(
                water_content,
                self.theta_sat,
                self.theta_fc,
                self.area,
                self.max_vertical_rate,
                self.smoothing_omega,
            )
            + volume_change_rate(
                water_content,
                self.theta0,
                self.theta_sat,
                self.area,
                self.thickness,
                self.delta_t,
                smoothing_omega=self.smoothing_omega,
            )
        )

    def derivative(
        self,
        water_content: float,
    ) -> float:
        return (
            self.residual(
                water_content + self.delta_theta,
            )
            - self.residual(
                water_content,
            )
        ) / self.delta_theta
