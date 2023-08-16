from ..base.control_volume import ControlVolume
from .smoothing import quadratic_smoother


def _maximum_water_content(
    water_content: float, control_volume: ControlVolume
) -> float:
    if water_content > control_volume.theta:
        water_content = control_volume.theta
    return water_content


def _saturation(
    water_content: float,
    control_volume: ControlVolume,
    theta0: float = 0.0,
    theta1: float = 1.0,
) -> float:
    water_content = _maximum_water_content(water_content, control_volume)
    return (water_content - theta0) / (theta1 - theta0)


def saturation(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    return quadratic_smoother(
        _saturation(
            water_content,
            control_volume=control_volume,
            theta1=control_volume.theta,
        ),
        omega=omega,
    )


def groundwater_recharge_saturation(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    theta0 = control_volume.theta_fc
    if water_content < theta0:
        saturation = 0.0
    else:
        saturation = _saturation(
            water_content,
            control_volume,
            theta0=theta0,
            theta1=control_volume.theta,
        )
    return quadratic_smoother(saturation, omega=omega)


def surface_discharge_saturation(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    theta0 = control_volume.theta_discharge
    if water_content < theta0:
        saturation = 0.0
    else:
        saturation = _saturation(
            water_content,
            control_volume,
            theta0=theta0,
            theta1=control_volume.theta,
        )
    return quadratic_smoother(saturation, omega=omega)


def lateral_discharge_saturation(
    water_content: float,
    control_volume: ControlVolume,
    omega: float = None,
) -> float:
    theta0 = control_volume.theta_fc
    if water_content < theta0:
        saturation = 0.0
    else:
        saturation = _saturation(
            water_content,
            control_volume,
            theta0=theta0,
            theta1=control_volume.theta,
        )
    return quadratic_smoother(saturation, omega=omega)
