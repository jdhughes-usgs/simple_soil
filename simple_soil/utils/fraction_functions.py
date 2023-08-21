from typing import Union

import numpy as np

from .array_utils import array_check
from .smoothing import quadratic_smoother


def _relative_fraction(
    water_content: Union[float, np.ndarray],
    theta0: float = 0.0,
    theta1: float = 1.0,
) -> np.ndarray:
    water_content = array_check(water_content)
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
    water_content: Union[float, np.ndarray],
    theta_sat: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            theta0=0.0,
            theta1=theta_sat,
        ),
        omega=smoothing_omega,
    )


def groundwater_recharge_fraction(
    water_content: Union[float, np.ndarray],
    theta_sat: float,
    theta_fc: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            theta0=theta_fc,
            theta1=theta_sat,
        ),
        omega=smoothing_omega,
    )


def surface_discharge_fraction(
    water_content: Union[float, np.ndarray],
    theta_sat: float,
    theta_discharge: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            theta0=theta_discharge,
            theta1=theta_sat,
        ),
        omega=smoothing_omega,
    )


def surface_infiltration_fraction(
    water_content: Union[float, np.ndarray],
    theta_sat: float,
    theta_discharge: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return 1.0 - surface_discharge_fraction(
        water_content,
        theta_sat,
        theta_discharge,
        smoothing_omega=smoothing_omega,
    )


def lateral_discharge_fraction(
    water_content: Union[float, np.ndarray],
    theta_sat: float,
    theta_fc: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            theta0=theta_fc,
            theta1=theta_sat,
        ),
        omega=smoothing_omega,
    )


def pet_fraction(
    water_content: Union[float, np.ndarray],
    theta_pet_max: float,
    theta_wp: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return quadratic_smoother(
        _relative_fraction(
            water_content,
            theta0=theta_wp,
            theta1=theta_pet_max,
        ),
        omega=smoothing_omega,
    )
