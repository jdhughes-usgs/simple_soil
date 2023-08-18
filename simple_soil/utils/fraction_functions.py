import numpy as np

from ..base.control_volume import ControlVolume
from .array_utils import array_check
from .smoothing import quadratic_smoother


def _maximum_water_content(
    water_content: float, control_volume: ControlVolume
) -> float:
    water_content = array_check(water_content)
    water_content = np.where(
        water_content > control_volume.theta,
        control_volume.theta,
        water_content,
    )
    return water_content


def _relative_fraction(
    water_content: float,
    control_volume: ControlVolume,
    theta0: float = 0.0,
    theta1: float = 1.0,
) -> float:
    water_content = _maximum_water_content(water_content, control_volume)
    fraction = np.where(
        water_content < theta0,
        0.0,
        (water_content - theta0) / (theta1 - theta0),
    )
    return np.where(
        water_content > theta1,
        1.0,
        fraction,
    )


def saturation_fraction(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            control_volume=control_volume,
            theta0=0.0,
            theta1=control_volume.theta,
        ),
        omega=omega,
    )


def groundwater_recharge_fraction(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            control_volume=control_volume,
            theta0=control_volume.theta_fc,
            theta1=control_volume.theta,
        ),
        omega=omega,
    )


def surface_discharge_fraction(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            control_volume=control_volume,
            theta0=control_volume.theta_discharge,
            theta1=control_volume.theta,
        ),
        omega=omega,
    )


def surface_infiltration_fraction(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return 1.0 - surface_discharge_fraction(
        water_content, control_volume, omega=omega
    )


def lateral_discharge_fraction(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            control_volume=control_volume,
            theta0=control_volume.theta_fc,
            theta1=control_volume.theta,
        ),
        omega=omega,
    )


def pet_fraction(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            control_volume=control_volume,
            theta0=control_volume.theta_wp,
            theta1=control_volume.theta_pet_max,
        ),
        omega=omega,
    )