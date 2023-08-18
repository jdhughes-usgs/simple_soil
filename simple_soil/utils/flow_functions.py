from typing import Union

import numpy as np

from .array_utils import array_check


def infiltration(
    rate: Union[float, np.ndarray],
    max_vertical: float,
) -> np.ndarray:
    rate = array_check(rate)
    return np.where(
        rate < max_vertical,
        rate,
        max_vertical,
    )
