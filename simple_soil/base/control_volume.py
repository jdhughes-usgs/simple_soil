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
