from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import MicroscopeConfig, Objective


@dataclass(frozen=True)
class StagePosition:
    x_um: float
    y_um: float
    z_um: float


class MicroscopeBackend(Protocol):
    """Narrow hardware boundary used by the interface.

    A future pymmcore-plus adapter only needs to implement this protocol. The UI
    deliberately never talks directly to Micro-Manager device properties.
    """

    @property
    def position(self) -> StagePosition: ...

    @property
    def current_objective_id(self) -> str: ...

    def move_xy(self, x_um: float, y_um: float) -> StagePosition: ...

    def move_z(self, z_um: float) -> StagePosition: ...

    def change_objective(self, objective: Objective) -> None: ...

    def snap(self) -> None: ...


class SimulatedMicroscope:
    def __init__(self, config: MicroscopeConfig) -> None:
        self._config = config
        self._position = StagePosition(0.0, 0.0, 0.0)
        self._objective_id = config.objectives[0].id
        self.snap_count = 0

    @property
    def position(self) -> StagePosition:
        return self._position

    @property
    def current_objective_id(self) -> str:
        return self._objective_id

    def move_xy(self, x_um: float, y_um: float) -> StagePosition:
        limits = self._config.stage_limits
        x_um = min(limits.x_max_um, max(limits.x_min_um, x_um))
        y_um = min(limits.y_max_um, max(limits.y_min_um, y_um))
        self._position = StagePosition(x_um, y_um, self._position.z_um)
        return self._position

    def move_z(self, z_um: float) -> StagePosition:
        limits = self._config.stage_limits
        z_um = min(limits.z_max_um, max(limits.z_min_um, z_um))
        self._position = StagePosition(self._position.x_um, self._position.y_um, z_um)
        return self._position

    def change_objective(self, objective: Objective) -> None:
        # The real adapter will first move Z to a configured clearance height,
        # rotate the turret, restore focus conservatively, and then report ready.
        self._objective_id = objective.id

    def snap(self) -> None:
        self.snap_count += 1
