import numpy as np

from .array_utils import array_check


def quadratic_smoother(saturation: float, omega: float = None) -> float:
    if omega is None:
        omega = 1e-6
    else:
        if omega <= 0.0:
            raise ValueError(f"omega ({omega}) must be >= 0.")
        elif omega > 1.0:
            raise ValueError(f"omega ({omega}) must be <= 1.")
    saturation = array_check(saturation)

    a_omega = 1.0 / (1.0 - omega)
    factor = a_omega / (2.0 * omega)
    # saturation < 0
    smoothed_saturation = np.where(saturation < 0.0, 0.0, saturation)
    # saturation < omega
    smoothed_saturation = np.where(
        np.logical_and(saturation > 0.0, saturation < omega),
        factor * saturation**2.0,
        smoothed_saturation,
    )
    # omega <= saturation < 1 - omega
    smoothed_saturation = np.where(
        np.logical_and(saturation >= omega, saturation < 1.0 - omega),
        a_omega * saturation + 0.5 * (1.0 - a_omega),
        smoothed_saturation,
    )
    # 1 - omega <= saturation < 1
    smoothed_saturation = np.where(
        np.logical_and(saturation >= 1.0 - omega, saturation < 1.0),
        1.0 - factor * (1 - saturation) ** 2.0,
        smoothed_saturation,
    )
    # saturation > 1.
    smoothed_saturation = np.where(saturation > 1.0, 1.0, smoothed_saturation)

    return smoothed_saturation


def pet_smoother(saturation: float) -> float:
    saturation = array_check(saturation)
    # saturation < 0
    smoothed_saturation = np.where(saturation < 0.0, 0.0, saturation)
    # 0 <= saturation < 1
    smoothed_saturation = np.where(
        np.logical_and(saturation >= 0, saturation < 1.0),
        -2.0 * saturation**3.0 + 3.0 * saturation**2.0,
        smoothed_saturation,
    )
    # saturation > 1
    smoothed_saturation = np.where(
        saturation > 1.0,
        1.0,
        smoothed_saturation,
    )
    return smoothed_saturation
