from typing import Union

import numpy as np

from .array_utils import array_check
from .fraction_functions import (
    groundwater_recharge_fraction,
    lateral_discharge_fraction,
    pet_fraction,
    saturation_fraction,
    surface_discharge_fraction,
    surface_infiltration_fraction,
)


def infiltration_depth(
    rate: Union[float, np.ndarray],
    max_vertical_rate: float,
) -> np.ndarray:
    rate = array_check(rate)
    return np.where(
        rate < max_vertical_rate,
        rate,
        max_vertical_rate,
    )


def aet_volumetric_rate(
    water_content: float,
    rate: float,
    theta_pet_max: float,
    theta_wp: float,
    area: float,
    smoothing_omega: float = 1e-6,
) -> float:
    fraction = float(
        pet_fraction(
            water_content,
            theta_pet_max,
            theta_wp,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return -area * fraction * rate


def infiltration_volumetric_rate(
    water_content: float,
    rate: float,
    theta_sat: float,
    theta_discharge: float,
    area: float,
    max_vertical_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    fraction = float(
        surface_infiltration_fraction(
            water_content,
            theta_sat,
            theta_discharge,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return (
        area * fraction * float(infiltration_depth(rate, max_vertical_rate)[0])
    )


def rejected_infiltration_volumetric_rate(
    water_content: float,
    rate: float,
    theta_sat: float,
    theta_discharge: float,
    area: float,
    max_vertical_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    return area * rate - infiltration_volumetric_rate(
        water_content,
        rate,
        theta_sat,
        theta_discharge,
        area,
        max_vertical_rate,
        smoothing_omega=smoothing_omega,
    )


def recharge_volumetric_rate(
    water_content: float,
    theta_sat: float,
    theta_fc: float,
    area: float,
    max_vertical_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    """

    Parameters
    ----------
    water_content
    theta_sat
    theta_fc
    area
    max_vertical_rate
    smoothing_omega

    Returns
    -------

    """
    water_content = array_check(water_content)
    fraction = float(
        groundwater_recharge_fraction(
            water_content,
            theta_sat,
            theta_fc,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return -area * fraction * max_vertical_rate


def surface_volumetric_rate(
    water_content: float,
    theta_sat: float,
    theta_discharge: float,
    area: float,
    max_vertical_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    water_content = array_check(water_content)
    fraction = float(
        surface_discharge_fraction(
            water_content,
            theta_sat,
            theta_discharge,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return -area * fraction * max_vertical_rate


def lateral_volumetric_rate(
    water_content: float,
    theta_sat: float,
    theta_fc: float,
    area: float,
    max_horizontal_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    water_content = array_check(water_content)
    fraction = float(
        lateral_discharge_fraction(
            water_content,
            theta_sat,
            theta_fc,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return -area * fraction * max_horizontal_rate


def volume_change_rate(
    water_content: float,
    water_content0: float,
    theta_sat: float,
    theta_wp: float,
    area: float,
    thickness: float,
    delta_t: float,
    smoothing_omega: float = 1e-6,
) -> np.ndarray:
    v0 = (
        float(
            saturation_fraction(
                water_content0,
                theta_sat,
                theta_wp,
                smoothing_omega=smoothing_omega,
            )[0]
        )
        * thickness
        * area
    )
    v1 = (
        float(
            saturation_fraction(
                water_content,
                theta_sat,
                theta_wp,
                smoothing_omega=smoothing_omega,
            )[0]
        )
        * thickness
        * area
    )
    return (v0 - v1) / delta_t
