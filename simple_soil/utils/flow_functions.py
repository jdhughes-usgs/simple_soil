from typing import Union

import numpy as np

from ..base.control_volume import ControlVolume
from .array_utils import array_check
from .fraction_functions import (
    surface_infiltration_fraction,
    groundwater_recharge_fraction,
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


def infiltration_volumetric_rate(
    water_content: float,
    rate: float,
    control_volume: ControlVolume,
) -> float:
    fraction = surface_infiltration_fraction(
        water_content,
        control_volume,
    )
    return (
        control_volume.area
        * fraction[0]
        * infiltration_depth(rate, control_volume.max_vertical_rate)
    )


def recharge_volumetric_rate(
    water_content: float,
    control_volume: ControlVolume,
) -> float:
    water_content = array_check(water_content)
    fraction = groundwater_recharge_fraction(
        water_content,
        control_volume,
    )
    return (
        -control_volume.area * fraction[0] * control_volume.max_vertical_rate
    )


def volume_change_rate(
    water_content: float,
    control_volume: ControlVolume,
    delta_t: float,
) -> np.ndarray:
    return None
