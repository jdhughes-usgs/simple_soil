class ControlVolume:
    def __init__(
        self,
        area: float = 1.0,
        thickness: float = 1.0,
        discharge_thickness: float = 0.1,
        theta_wp: float = 0.01,
        theta_fc: float = 0.1,
        theta: float = 0.2,
        Kv: float = 1e-3,
        KhKv_ratio: float = 10.0,
        pet_fraction: float = 1.1,
    ) -> "ControlVolume":
        self.area = area
        self.thickness = thickness
        self.discharge_thickness = discharge_thickness
        self.theta_wp = theta_wp
        self.theta_fc = theta_fc
        self.theta = theta
        self.Kv = Kv
        self.pet_fraction = pet_fraction
        self.theta_pet_max = self.theta_wp * self.pet_fraction

        self._validate()

        self.theta_discharge = (
            theta * (thickness - discharge_thickness) / thickness
        )
        self.Kh = Kv * KhKv_ratio

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
        if self.theta_fc > self.theta:
            raise ValueError(
                f"field capacity ({self.theta_fc}) must "
                + f"be less than theta ({self.theta})"
            )
        if self.Kv < 0.0:
            raise ValueError(
                f"vertical hydraulic conductivity ({self.Kv}) "
                + "must be greater than zero"
            )

        if self.theta_pet_max > self.theta:
            raise ValueError(
                f"pet_fraction ({self.pet_fraction}) "
                + f"exceeds the maximum value ({self.theta / self.theta_fc}), "
                + f"which is the ratio of theta ({self.theta})"
                + f"and theta_wp ({self.theta_wp})"
            )
