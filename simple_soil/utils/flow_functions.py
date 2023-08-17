from typing import Union

import numpy as np

from .array_utils import array_check


def infiltration(
    rate: Union[float, np.ndarray],
    Kv: float,
    bc_epsilon: float,
    theta: float,
    theta_wp: float,
) -> np.ndarray:
    rate = array_check(rate)
    return np.where(
        rate < Kv,
        (rate / Kv) ** (1.0 / bc_epsilon) * (theta - theta_wp) + theta_wp,
        Kv,
    )
