def quadratic_smoother(saturation: float, omega: float = None) -> float:
    if omega is None:
        omega = 1e-6
    else:
        if omega <= 0.0:
            raise ValueError(f"omega ({omega}) must be >= 0.")
        elif omega > 1.0:
            raise ValueError(f"omega ({omega}) must be <= 1.")

    a_omega = 1.0 / (1.0 - omega)
    factor = a_omega / (2.0 * omega)
    if saturation < 0:
        smoothed_saturation = 0.0
    elif saturation < omega:
        smoothed_saturation = factor * saturation**2.0
    elif saturation >= omega and saturation < (1.0 - omega):
        smoothed_saturation = a_omega * saturation + 0.5 * (1.0 - a_omega)
    elif saturation >= (1.0 - omega) and saturation < 1.0:
        smoothed_saturation = 1.0 - factor * (1 - saturation) ** 2.0
    else:
        smoothed_saturation = 1.0

    return smoothed_saturation


def cubic_smoother(saturation: float) -> float:
    smoothed_saturation = 0.0
    return smoothed_saturation
