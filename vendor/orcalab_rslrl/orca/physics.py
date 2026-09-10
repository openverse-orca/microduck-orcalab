"""Stable Orca physics contract.

The concrete solver is deliberately private. Tasks and environments depend on
this contract so applications never need to import a backend implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class OrcaCapabilities:
    local_reset: bool = True
    contact_force: bool = False
    camera: bool = False
    body_mass_randomization: bool = False
    friction_randomization: bool = False
    push_force: bool = False
    native_render: bool = False


@dataclass(frozen=True, slots=True)
class OrcaState:
    """Zero-copy simulation state; every tensor starts with ``num_envs``."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    qacc: torch.Tensor
    ctrl: torch.Tensor
    actuator_force: torch.Tensor
    sensordata: torch.Tensor
    time: torch.Tensor
    xpos: torch.Tensor | None = None
    xquat: torch.Tensor | None = None
    cvel: torch.Tensor | None = None


class OrcaPhysics(ABC):
    """Stable tensor contract exposed to Orca task authors."""

    num_envs: int
    device: torch.device
    capabilities: OrcaCapabilities

    @property
    @abstractmethod
    def state(self) -> OrcaState: ...

    @abstractmethod
    def step(self, nstep: int = 1) -> None: ...

    @abstractmethod
    def forward(self) -> None: ...

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None: ...

    def reset_mask(self, reset_mask: torch.Tensor) -> None:
        raise NotImplementedError("Orca runtime does not support GPU mask reset")

    @abstractmethod
    def write_action(self, action: torch.Tensor) -> None: ...

    def joint_qpos(self, joint_names: tuple[str, ...] | None = None) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose joint positions")

    def joint_qvel(self, joint_names: tuple[str, ...] | None = None) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose joint velocities")

    def default_joint_qpos(self, joint_names: tuple[str, ...] | None = None) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose default joint positions")

    def default_actuated_qpos(self) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose default actuator targets")

    def actuated_qpos(self) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose actuator joint positions")

    def actuated_qvel(self) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose actuator joint velocities")

    def actuated_qacc(self) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose actuator joint accelerations")

    def actuated_joint_range(self) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose actuator joint limits")

    def sensor(self, name: str) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose named sensors")

    def has_sensor(self, name: str) -> bool:
        return False

    def body_id(self, name: str) -> int:
        raise NotImplementedError("Orca runtime does not expose body ids")

    def joint_qpos_addresses(self) -> dict[str, int]:
        """Return a copy of the name-to-qpos mapping used by Orca rendering."""
        raise NotImplementedError("Orca runtime does not expose joint addresses")

    def default_root_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Orca runtime does not expose default root state")

    def write_actuated_joint_state(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        raise NotImplementedError("Orca runtime does not support joint state writes")

    def set_param(self, name: str, value: Any, env_ids: torch.Tensor | None = None) -> None:
        raise NotImplementedError(f"Orca runtime parameter {name!r} is unsupported")
