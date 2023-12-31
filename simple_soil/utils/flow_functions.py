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
from .infiltration_functions import GreenAmpt, InfiltrationConstantLoss


def flow_factor(
    water_content: float,
    flow_rate: float,
    theta_sat: float,
    theta_wp: float,
    bc_epsilon: float = 3.5,
) -> float:
    return (
        flow_rate
        * ((water_content - theta_wp) / (theta_sat - theta_wp)) ** bc_epsilon
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
    water_content0: float,
    rate: float,
    theta_sat: float,
    theta_discharge: float,
    area: float,
    infiltration_method: Union["InfiltrationConstantLoss", "GreenAmpt"],
    smoothing_omega: float = 1e-6,
) -> float:
    area_fraction = float(
        surface_infiltration_fraction(
            water_content,
            theta_sat,
            theta_discharge,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    if rate == 0.0:
        infiltration_rate = 0.0
    else:
        infiltration_rate = infiltration_method.infiltration(
            rate,
            water_content,
            water_content0,
        )
    return area * area_fraction * infiltration_rate


def rejected_infiltration_volumetric_rate(
    water_content: float,
    water_content0: float,
    rate: float,
    theta_sat: float,
    theta_discharge: float,
    area: float,
    infiltration_method: str,
    smoothing_omega: float = 1e-6,
) -> float:
    return area * rate - infiltration_volumetric_rate(
        water_content,
        water_content0,
        rate,
        theta_sat,
        theta_discharge,
        area,
        infiltration_method,
        smoothing_omega=smoothing_omega,
    )


def recharge_volumetric_rate(
    water_content: float,
    theta_sat: float,
    theta_fc: float,
    theta_wp: float,
    area: float,
    thickness: float,
    max_vertical_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    gradient = thickness * float(
        saturation_fraction(
            water_content,
            theta_sat,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    rate = flow_factor(
        water_content,
        max_vertical_rate,
        theta_sat,
        theta_wp,
    )
    fraction = float(
        groundwater_recharge_fraction(
            water_content,
            theta_sat,
            theta_fc,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return -area * fraction * rate * gradient


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
    theta_wp: float,
    area: float,
    thickness: float,
    max_horizontal_rate: float,
    smoothing_omega: float = 1e-6,
) -> float:
    gradient = thickness * float(
        saturation_fraction(
            water_content,
            theta_sat,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    rate = flow_factor(
        water_content,
        max_horizontal_rate,
        theta_sat,
        theta_wp,
    )
    fraction = float(
        lateral_discharge_fraction(
            water_content,
            theta_sat,
            theta_fc,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    return -area * fraction * rate * gradient


def volume_change_rate(
    water_content: float,
    water_content0: float,
    theta_sat: float,
    area: float,
    thickness: float,
    delta_t: float,
    smoothing_omega: float = 1e-6,
) -> np.ndarray:
    sat0 = float(
        saturation_fraction(
            water_content0,
            theta_sat,
            smoothing_omega=smoothing_omega,
        )[0]
    )
    sat = float(
        saturation_fraction(
            water_content,
            theta_sat,
            smoothing_omega=smoothing_omega,
        )[0]
    )

    return area * thickness * theta_sat * (sat0 - sat) / delta_t

    # v0 = (
    #     float(
    #         saturation_fraction(
    #             water_content0,
    #             theta_sat,
    #             theta_wp,
    #             smoothing_omega=smoothing_omega,
    #         )[0]
    #     )
    #     * thickness
    #     * area
    # )
    # v1 = (
    #     float(
    #         saturation_fraction(
    #             water_content,
    #             theta_sat,
    #             theta_wp,
    #             smoothing_omega=smoothing_omega,
    #         )[0]
    #     )
    #     * thickness
    #     * area
    # )
    # return (v0 - v1) / delta_t
