from typing import Union

import numpy as np


def array_check(arr: Union[float, np.ndarray]) -> np.ndarray:
    if isinstance(arr, float):
        arr = np.full(1, arr, dtype=float)
    return arr
