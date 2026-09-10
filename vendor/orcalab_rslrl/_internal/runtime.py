from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ..orca.physics import OrcaCapabilities, OrcaPhysics, OrcaState


@contextmanager
def _suspend_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class OrcaPhysicsRuntime(OrcaPhysics):
    """Private batched GPU runtime used by Orca task factories."""

    capabilities = OrcaCapabilities(local_reset=True)

    def __init__(
        self,
        model_or_path: Any,
        *,
        num_envs: int,
        device: str = "cuda:0",
        model_options: dict[str, Any] | None = None,
        use_cuda_graph: bool | None = None,
        ls_parallel: bool = True,
        contact_sensor_maxmatch: int = 64,
        **put_data_kwargs: Any,
    ):
        import mujoco
        import mujoco_warp as mjwarp
        import warp as wp

        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.mjwarp, self.wp = mjwarp, wp
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.cpu_model = (
            mujoco.MjModel.from_xml_path(str(model_or_path))
            if isinstance(model_or_path, (str, Path)) else model_or_path
        )
        for name, value in (model_options or {}).items():
            if not hasattr(self.cpu_model.opt, name):
                raise ValueError(f"Unknown MuJoCo option {name!r}")
            setattr(self.cpu_model.opt, name, value)
        self.cpu_data = mujoco.MjData(self.cpu_model)
        if self.cpu_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.cpu_model, self.cpu_data, 0)
            mujoco.mj_forward(self.cpu_model, self.cpu_data)
        self._initialize_model_metadata(mujoco)
        with wp.ScopedDevice(device):
            self.wp_model = mjwarp.put_model(self.cpu_model)
            self.wp_model.opt.ls_parallel = ls_parallel
            self.wp_model.opt.contact_sensor_maxmatch = contact_sensor_maxmatch
            self.wp_data = mjwarp.put_data(
                self.cpu_model, self.cpu_data, nworld=self.num_envs, **put_data_kwargs
            )
        self._initialize_state_views(use_cuda_graph)

    def _initialize_model_metadata(self, mujoco: Any) -> None:
        self.timestep = float(self.cpu_model.opt.timestep)
        self.joint_names = tuple(
            name for joint_id in range(self.cpu_model.njnt)
            if (name := mujoco.mj_id2name(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        )
        self.actuator_names = tuple(
            name for actuator_id in range(self.cpu_model.nu)
            if (name := mujoco.mj_id2name(self.cpu_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
        )
        self._joint_qpos_addr = {
            name: int(self.cpu_model.jnt_qposadr[mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.joint_names
            if int(self.cpu_model.jnt_type[mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)]) != int(mujoco.mjtJoint.mjJNT_FREE)
        }
        self._joint_dof_addr = {
            name: int(self.cpu_model.jnt_dofadr[mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self._joint_qpos_addr
        }
        self._default_joint_names = tuple(self._joint_qpos_addr)
        self._default_joint_qpos_idx = torch.as_tensor(
            [self._joint_qpos_addr[name] for name in self._default_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._default_joint_dof_idx = torch.as_tensor(
            [self._joint_dof_addr[name] for name in self._default_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._joint_qpos_idx_cache: dict[tuple[str, ...], torch.Tensor] = {
            self._default_joint_names: self._default_joint_qpos_idx
        }
        self._joint_dof_idx_cache: dict[tuple[str, ...], torch.Tensor] = {
            self._default_joint_names: self._default_joint_dof_idx
        }
        self._sensor_slices = {}
        for sensor_id in range(self.cpu_model.nsensor):
            name = mujoco.mj_id2name(self.cpu_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if name:
                adr, dim = int(self.cpu_model.sensor_adr[sensor_id]), int(self.cpu_model.sensor_dim[sensor_id])
                self._sensor_slices[name] = (adr, adr + dim)
        self._body_ids = {
            name: body_id
            for body_id in range(self.cpu_model.nbody)
            if (name := mujoco.mj_id2name(self.cpu_model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        }
        self._actuated_qpos_idx = self._actuated_dof_idx = self._actuated_jnt_range = None
        if self.cpu_model.nu and (self.cpu_model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)).all():
            joint_ids = self.cpu_model.actuator_trnid[:, 0]
            self._actuated_qpos_idx = torch.as_tensor(
                self.cpu_model.jnt_qposadr[joint_ids], device=self.device, dtype=torch.long
            )
            self._actuated_dof_idx = torch.as_tensor(
                self.cpu_model.jnt_dofadr[joint_ids], device=self.device, dtype=torch.long
            )
            self._actuated_jnt_range = torch.as_tensor(
                self.cpu_model.jnt_range[joint_ids], device=self.device, dtype=torch.float32
            )
        if self.cpu_model.nu:
            self._ctrl_min = torch.as_tensor(self.cpu_model.actuator_ctrlrange[:, 0], device=self.device, dtype=torch.float32)
            self._ctrl_max = torch.as_tensor(self.cpu_model.actuator_ctrlrange[:, 1], device=self.device, dtype=torch.float32)
            limited = torch.as_tensor(self.cpu_model.actuator_ctrllimited, device=self.device, dtype=torch.bool)
            self._ctrl_min = torch.where(limited, self._ctrl_min, torch.full_like(self._ctrl_min, -torch.inf))
            self._ctrl_max = torch.where(limited, self._ctrl_max, torch.full_like(self._ctrl_max, torch.inf))
        else:
            self._ctrl_min = self._ctrl_max = None

    def _initialize_state_views(self, use_cuda_graph: bool | None) -> None:
        values = {}
        for name in OrcaState.__dataclass_fields__:
            values[name] = self.wp.to_torch(getattr(self.wp_data, name))
            if values[name].shape[0] != self.num_envs:
                raise RuntimeError(f"{name} shape {tuple(values[name].shape)} is not nworld={self.num_envs}")
        self._state = OrcaState(**values)
        self._qpos0, self._qvel0 = self.state.qpos.clone(), self.state.qvel.clone()
        self.use_cuda_graph = self._should_use_cuda_graph() if use_cuda_graph is None else bool(use_cuda_graph)
        self._step_graph = None
        self._forward_graph = None
        self._create_graphs()

    @property
    def state(self) -> OrcaState:
        return self._state

    def write_action(self, action: torch.Tensor) -> None:
        if action.shape != self.state.ctrl.shape:
            raise ValueError(f"actions {tuple(action.shape)} != ctrl {tuple(self.state.ctrl.shape)}")
        target = action.to(self.state.ctrl)
        if self._ctrl_min is not None:
            target = target.clamp(self._ctrl_min, self._ctrl_max)
        self.state.ctrl.copy_(target)

    def step(self, nstep: int = 1) -> None:
        if nstep < 1:
            raise ValueError("nstep must be positive")
        with self.wp.ScopedDevice(str(self.device)):
            for _ in range(nstep):
                if self._step_graph is not None:
                    self.wp.capture_launch(self._step_graph)
                else:
                    self.mjwarp.step(self.wp_model, self.wp_data)

    def forward(self) -> None:
        with self.wp.ScopedDevice(str(self.device)):
            if self._forward_graph is not None:
                self.wp.capture_launch(self._forward_graph)
            else:
                self.mjwarp.forward(self.wp_model, self.wp_data)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.state.qpos.device) if env_ids is None else env_ids.to(self.state.qpos.device, torch.long)
        self.state.qpos[ids] = self._qpos0[ids]
        self.state.qvel[ids] = self._qvel0[ids]
        self.state.ctrl[ids] = 0
        self.forward()

    def reset_mask(self, reset_mask: torch.Tensor) -> None:
        mask = reset_mask.to(device=self.state.qpos.device, dtype=torch.bool)
        q_mask = mask[:, None]
        self.state.qpos.copy_(torch.where(q_mask, self._qpos0, self.state.qpos))
        self.state.qvel.copy_(torch.where(q_mask, self._qvel0, self.state.qvel))
        self.state.ctrl.copy_(torch.where(mask[:, None], torch.zeros_like(self.state.ctrl), self.state.ctrl))
        self.forward()

    def joint_qpos(self, joint_names: tuple[str, ...] | None = None) -> torch.Tensor:
        return self.state.qpos.index_select(1, self._joint_qpos_indices(joint_names))

    def joint_qvel(self, joint_names: tuple[str, ...] | None = None) -> torch.Tensor:
        return self.state.qvel.index_select(1, self._joint_dof_indices(joint_names))

    def default_joint_qpos(self, joint_names: tuple[str, ...] | None = None) -> torch.Tensor:
        return self._qpos0.index_select(1, self._joint_qpos_indices(joint_names))

    def default_actuated_qpos(self) -> torch.Tensor:
        if self._actuated_qpos_idx is None:
            raise RuntimeError("Model actuators do not all target joints")
        return self._qpos0.index_select(1, self._actuated_qpos_idx)

    def actuated_qpos(self) -> torch.Tensor:
        if self._actuated_qpos_idx is None:
            raise RuntimeError("Model actuators do not all target joints")
        return self.state.qpos.index_select(1, self._actuated_qpos_idx)

    def actuated_qvel(self) -> torch.Tensor:
        if self._actuated_dof_idx is None:
            raise RuntimeError("Model actuators do not all target joints")
        return self.state.qvel.index_select(1, self._actuated_dof_idx)

    def actuated_qacc(self) -> torch.Tensor:
        if self._actuated_dof_idx is None:
            raise RuntimeError("Model actuators do not all target joints")
        return self.state.qacc.index_select(1, self._actuated_dof_idx)

    def sensor(self, name: str) -> torch.Tensor:
        start, stop = self._sensor_slices[name]
        return self.state.sensordata[:, start:stop]

    def has_sensor(self, name: str) -> bool:
        return name in self._sensor_slices

    def body_id(self, name: str) -> int:
        return self._body_ids[name]

    def joint_qpos_addresses(self) -> dict[str, int]:
        return dict(self._joint_qpos_addr)

    def default_root_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._qpos0[env_ids.to(self.device, torch.long), :7]

    def write_actuated_joint_state(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        if self._actuated_qpos_idx is None or self._actuated_dof_idx is None:
            raise RuntimeError("Model actuators do not all target joints")
        ids = env_ids.to(self.device, torch.long)
        if joint_pos.shape != (ids.numel(), self._actuated_qpos_idx.numel()):
            raise ValueError("joint_pos has an incompatible shape")
        if joint_vel.shape != (ids.numel(), self._actuated_dof_idx.numel()):
            raise ValueError("joint_vel has an incompatible shape")
        self.state.qpos[ids.unsqueeze(1), self._actuated_qpos_idx.unsqueeze(0)] = joint_pos
        self.state.qvel[ids.unsqueeze(1), self._actuated_dof_idx.unsqueeze(0)] = joint_vel

    def actuated_joint_range(self) -> torch.Tensor:
        """[nu, 2] joint limits of the actuated joints, in ctrl order."""
        if self._actuated_jnt_range is None:
            raise RuntimeError("Model actuators do not all target joints")
        return self._actuated_jnt_range

    def _joint_qpos_indices(self, joint_names: tuple[str, ...] | None) -> torch.Tensor:
        names = joint_names or self._default_joint_names
        cached = self._joint_qpos_idx_cache.get(names)
        if cached is None:
            cached = torch.as_tensor([self._joint_qpos_addr[name] for name in names], device=self.device, dtype=torch.long)
            self._joint_qpos_idx_cache[names] = cached
        return cached

    def _joint_dof_indices(self, joint_names: tuple[str, ...] | None) -> torch.Tensor:
        names = joint_names or self._default_joint_names
        cached = self._joint_dof_idx_cache.get(names)
        if cached is None:
            cached = torch.as_tensor([self._joint_dof_addr[name] for name in names], device=self.device, dtype=torch.long)
            self._joint_dof_idx_cache[names] = cached
        return cached

    def _create_graphs(self) -> None:
        if not self.use_cuda_graph:
            return
        with _suspend_gc(), self.wp.ScopedDevice(str(self.device)):
            with self.wp.ScopedCapture() as capture:
                self.mjwarp.step(self.wp_model, self.wp_data)
            self._step_graph = capture.graph
            with self.wp.ScopedCapture() as capture:
                self.mjwarp.forward(self.wp_model, self.wp_data)
            self._forward_graph = capture.graph

    def _should_use_cuda_graph(self) -> bool:
        wp_device = self.wp.get_device(str(self.device))
        if not wp_device.is_cuda:
            return False
        driver_ver = self.wp.get_cuda_driver_version()
        return bool(driver_ver and driver_ver >= (12, 4) and self.wp.is_mempool_enabled(wp_device))
