from __future__ import annotations

import math

from .models import Channel, Objective


def recommended_z_step_um(
    objective: Objective,
    channels: tuple[Channel, ...],
) -> tuple[float, float]:
    """Return a conservative widefield axial Nyquist step and wavelength.

    Axial Rayleigh resolution is approximated as 2*n*lambda/NA^2. Sampling it
    twice gives a recommended Z interval of n*lambda/NA^2. The shortest enabled
    emission wavelength is used because it requires the finest sampling.
    """
    if objective.numerical_aperture <= 0:
        raise ValueError("Objective numerical aperture must be greater than zero")
    wavelength_nm = min(
        (channel.emission_wavelength_nm for channel in channels),
        default=550.0,
    )
    wavelength_um = wavelength_nm / 1000.0
    step_um = (
        objective.immersion_refractive_index
        * wavelength_um
        / objective.numerical_aperture**2
    )
    return step_um, wavelength_nm


def automatic_z_stack(
    begin_um: float,
    end_um: float,
    recommended_step_um_value: float,
) -> tuple[int, float]:
    """Return endpoint-inclusive plane count and resulting actual step."""
    if recommended_step_um_value <= 0:
        raise ValueError("Recommended Z step must be greater than zero")
    distance = abs(end_um - begin_um)
    if distance == 0:
        return 1, recommended_step_um_value
    planes = math.ceil(distance / recommended_step_um_value) + 1
    actual_step_um = distance / (planes - 1)
    return planes, actual_step_um
