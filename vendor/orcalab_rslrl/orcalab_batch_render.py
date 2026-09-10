"""Batched OrcaLab playback for MJWarp nworld policies.

OrcaLocomotion's play path drives ONE OrcaGym scene; here the policy runs in
``num_envs`` independent MJWarp worlds, so rendering works differently:

1. publish ``num_envs`` robot actors into the OrcaLab scene on a grid
   (``OrcaGymScene`` + ``Actor``, same RPC as orca_rl's ``publish_g1_scene``),
2. download the combined scene MJCF that OrcaLab compiles
   (``OrcaGymLocal.load_model_xml``) and index each agent's free-joint +
   named-joint qpos addresses in that combined model,
3. every render tick scatter the per-world qpos of the MJWarp batch into one
   combined qpos vector (adding each agent's grid offset from the scene's
   qpos0) and stream it with the ``UpdateLocalEnv`` RPC.

OrcaLab only renders poses; the physics stays in MJWarp on GPU.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .scene_options import (
    apply_scene_options,
    apply_unitree_orca_scene_options,
    assert_flat_ground_options,
    assert_scene_options,
    patch_scene_xml_options,
    resolve_scene_options_profile,
    scene_options_snapshot,
    scene_xml_contract,
)


@dataclass(frozen=True)
class BatchRenderLayout:
    """Fancy-index map from an [num_envs, nq_local] batch into combined qpos."""

    src_index: np.ndarray  # [K] local qpos indices (0..6 root + joint addrs)
    dst_index: np.ndarray  # [num_envs, K] combined qpos indices
    root_offset: np.ndarray  # [num_envs, 3] grid offset added to root xyz
    root_anchor: np.ndarray  # [num_envs, 7] authored combined-scene root qpos0
    nq_combined: int


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("root quaternion must have a finite non-zero norm")
    return values / norms


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float64, copy=True)
    result[..., 1:] *= -1.0
    return result


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat)
    vector = np.asarray(vector, dtype=np.float64)
    pure = np.concatenate((np.zeros((*vector.shape[:-1], 1)), vector), axis=-1)
    return _quat_multiply_wxyz(
        _quat_multiply_wxyz(quat, pure), _quat_conjugate_wxyz(quat)
    )[..., 1:]


def discover_agent_names(
    model,
    joint_names: Sequence[str],
    *,
    expected_count: int = 1,
) -> list[str]:
    """Find complete actor prefixes by joint suffix, independent of XML order."""

    import mujoco

    required = tuple(str(name) for name in joint_names)
    if not required:
        raise ValueError("joint_names must not be empty when discovering scene actors")
    available = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        for joint_id in range(model.njnt)
    }
    candidates: set[str] | None = None
    for suffix in required:
        marker = f"_{suffix}"
        prefixes = {
            name[: -len(marker)]
            for name in available
            if name.endswith(marker) and len(name) > len(marker)
        }
        candidates = (
            prefixes if candidates is None else candidates.intersection(prefixes)
        )
    complete = sorted(candidates or ())
    if len(complete) != int(expected_count):
        raise RuntimeError(
            "Expected exactly "
            f"{expected_count} complete robot actor(s) in the current OrcaLab XML, "
            f"found {len(complete)}: {complete}. Keep one Go2 actor in the layout "
            "or pass its exact actor name."
        )
    return complete


def build_batch_layout(
    model, agent_names, joint_qpos_addr: dict[str, int]
) -> BatchRenderLayout:
    """Index each agent's free joint + prefixed named joints in a combined model.

    ``model``: the combined MjModel OrcaLab compiled from the published actors
    (every entity name is prefixed with the actor name). ``joint_qpos_addr``:
    local-model qpos address per named joint; the local root free joint is
    assumed at qpos[0:7].
    """
    import mujoco

    joint_names = list(joint_qpos_addr)
    src = np.array(
        list(range(7)) + [joint_qpos_addr[name] for name in joint_names], dtype=np.int64
    )
    dst = np.zeros((len(agent_names), src.size), dtype=np.int64)
    offsets = np.zeros((len(agent_names), 3))
    anchors = np.zeros((len(agent_names), 7))
    free_by_name = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == int(mujoco.mjtJoint.mjJNT_FREE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            free_by_name[name] = int(model.jnt_qposadr[joint_id])
    for env_index, agent in enumerate(agent_names):
        root_matches = [
            (name, adr)
            for name, adr in free_by_name.items()
            if name == agent or name.startswith(f"{agent}_")
        ]
        if not root_matches:
            raise RuntimeError(
                f"No free joint for agent {agent!r} in the OrcaLab scene "
                f"({sorted(free_by_name)[:4]}...); was the batch published?"
            )
        if len(root_matches) != 1:
            raise RuntimeError(
                f"Agent {agent!r} has ambiguous free joints in the OrcaLab scene: "
                f"{[name for name, _adr in root_matches]}"
            )
        root_adr = root_matches[0][1]
        dst[env_index, :7] = np.arange(root_adr, root_adr + 7)
        # Actor grid placement lands in qpos0 of the free joint at compile.
        offsets[env_index] = model.qpos0[root_adr : root_adr + 3]
        offsets[env_index, 2] = 0.0
        anchors[env_index] = model.qpos0[root_adr : root_adr + 7]
        anchors[env_index, 3:7] = _normalize_quat_wxyz(anchors[env_index, 3:7])
        for column, joint_name in enumerate(joint_names, start=7):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{agent}_{joint_name}"
            )
            if joint_id < 0:
                raise RuntimeError(f"Scene is missing joint {agent}_{joint_name}")
            dst[env_index, column] = int(model.jnt_qposadr[joint_id])
    return BatchRenderLayout(src, dst, offsets, anchors, int(model.nq))


def apply_orcalab_day_options(
    model, *, timestep: float | None, disable_air_resistance: bool
) -> None:
    """Align an OrcaLab Day export with unitree-orca's runtime scene options.

    OrcaLab Day currently exports ``density=1.225`` and
    ``viscosity=1.8e-05`` which enables MuJoCo's fluid drag model.  The play
    path renders MJWarp qpos and should not inherit those scene-level fluid
    parameters.
    """
    apply_unitree_orca_scene_options(
        model,
        timestep=timestep,
        align_air_resistance=disable_air_resistance,
    )


def resolve_terrain_position(
    terrain_asset_path: str | None,
    terrain_position: tuple[float, float, float] | list[float] | None,
) -> np.ndarray:
    if terrain_position is not None:
        if len(terrain_position) != 3:
            raise ValueError("--terrain-position expects exactly 3 values: X Y Z")
        return np.asarray(terrain_position, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def resolve_spawn_center(
    terrain_asset_path: str | None,
    spawn_center: tuple[float, float, float] | list[float] | None,
) -> np.ndarray:
    del terrain_asset_path
    if spawn_center is not None:
        if len(spawn_center) != 3:
            raise ValueError("--spawn-center expects exactly 3 values: X Y Z")
        return np.asarray(spawn_center, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def collect_scene_geoms(
    model, agent_names, *, collisions_only: bool = True, include_robot: bool = False
) -> list[dict]:
    """Collect compiled OrcaLab geoms from the downloaded MuJoCo scene."""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    agent_prefixes = tuple(f"{name}_" for name in agent_names)
    geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            or f"geom_{geom_id}"
        )
        is_robot = body_name.startswith(agent_prefixes) or geom_name.startswith(
            agent_prefixes
        )
        if is_robot and not include_robot:
            continue

        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        if collisions_only and contype == 0 and conaffinity == 0:
            continue

        geom_type = (
            mujoco.mjtGeom(int(model.geom_type[geom_id]))
            .name.removeprefix("mjGEOM_")
            .lower()
        )
        geoms.append(
            {
                "id": geom_id,
                "name": geom_name,
                "body": body_name,
                "type": geom_type,
                "world_pos": data.geom_xpos[geom_id].tolist(),
                "local_pos": model.geom_pos[geom_id].tolist(),
                "size": model.geom_size[geom_id].tolist(),
                "contype": contype,
                "conaffinity": conaffinity,
                "condim": int(model.geom_condim[geom_id]),
                "friction": model.geom_friction[geom_id].tolist(),
                "solref": model.geom_solref[geom_id].tolist(),
                "solimp": model.geom_solimp[geom_id].tolist(),
                "margin": float(model.geom_margin[geom_id]),
                "gap": float(model.geom_gap[geom_id]),
                "priority": int(model.geom_priority[geom_id]),
                "group": int(model.geom_group[geom_id]),
            }
        )
    return geoms


class OrcaLabBatchRenderer:
    """Streams a batched MJWarp state into an OrcaLab scene with N actors."""

    def __init__(
        self,
        *,
        orcagym_addr: str,
        num_envs: int,
        joint_qpos_addr: dict[str, int],
        agent_prefix: str = "g1",
        agent_names: Sequence[str] | None = None,
        discover_agents: bool = False,
        asset_path: str = "assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda",
        terrain_asset_path: str | None = None,
        terrain_position: tuple[float, float, float] | list[float] | None = None,
        spawn_center: tuple[float, float, float] | list[float] | None = None,
        spacing: float = 2.5,
        spawn_range: float | None = None,
        root_xy_scale: float = 1.0,
        render_root_offset: tuple[float, float, float] | list[float] | None = None,
        anchor_to_scene: bool = False,
        scene_timestep: float | None = None,
        scene_profile: str = "orca-runtime",
        disable_air_resistance: bool = True,
        strict_scene_options: bool = False,
        manual_xml_override: bool = False,
        aligned_xml_output: str | Path | None = None,
        spawn_height: float = 0.0,
        publish: bool = True,
        load_timeout_s: float = 600.0,
    ):
        """``joint_qpos_addr``: local-model qpos address per named (hinge) joint,
        e.g. ``OrcaPhysicsRuntime.joint_qpos_addresses()``. The local root free
        joint is assumed at qpos[0:7]."""
        self.orcagym_addr = orcagym_addr
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.discover_agents = bool(discover_agents)
        if agent_names is not None:
            self.agent_names = [str(name) for name in agent_names]
            if len(self.agent_names) != self.num_envs:
                raise ValueError(
                    f"agent_names must contain {self.num_envs} entries, got {self.agent_names}"
                )
        elif self.discover_agents:
            self.agent_names = []
        else:
            self.agent_names = [
                f"{agent_prefix}_{index:03d}" for index in range(self.num_envs)
            ]
        if publish and self.discover_agents:
            raise ValueError("discover_agents is only valid when publish=False")
        self.asset_path = asset_path
        self.terrain_asset_path = terrain_asset_path
        self.terrain_position = resolve_terrain_position(
            terrain_asset_path, terrain_position
        )
        self.spawn_center = resolve_spawn_center(terrain_asset_path, spawn_center)
        if render_root_offset is not None and len(render_root_offset) != 3:
            raise ValueError("--render-root-offset expects exactly 3 values: X Y Z")
        self.render_root_offset = np.asarray(
            render_root_offset or [0.0, 0.0, 0.0], dtype=np.float64
        )
        self.spacing = float(spacing)
        self.spawn_range = None if spawn_range is None else float(spawn_range)
        if root_xy_scale < 0.0:
            raise ValueError("--root-xy-scale must be non-negative")
        self.root_xy_scale = float(root_xy_scale)
        self.anchor_to_scene = bool(anchor_to_scene)
        self.scene_timestep = None if scene_timestep is None else float(scene_timestep)
        self.scene_profile = str(scene_profile)
        self.scene_options = resolve_scene_options_profile(self.scene_profile)
        self.disable_air_resistance = bool(disable_air_resistance)
        self.strict_scene_options = bool(strict_scene_options)
        self.manual_xml_override = bool(manual_xml_override)
        self.aligned_xml_output = (
            None
            if aligned_xml_output is None
            else Path(aligned_xml_output).expanduser().resolve()
        )
        if self.manual_xml_override and self.aligned_xml_output is None:
            raise ValueError("manual_xml_override requires aligned_xml_output")
        self.spawn_height = float(spawn_height)
        self._joint_qpos_addr = dict(joint_qpos_addr)
        self.load_timeout_s = float(load_timeout_s)
        # One shared event loop, installed as the current loop: grpc.aio
        # channels bind to the loop that is current at creation time, and
        # OrcaGymScene also grabs it via asyncio.get_event_loop(). Mixing
        # loops raises "Future attached to a different loop".
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            if publish:
                self._publish_actors()
            self._connect()
            if self.discover_agents:
                self.agent_names = discover_agent_names(
                    self.combined_model,
                    list(self._joint_qpos_addr),
                    expected_count=self.num_envs,
                )
            self.layout = self._build_layout()
            # Do not mutate remote options until the downloaded XML has one
            # complete, unambiguous robot mapping. Invalid/stale layouts fail
            # without touching the running OrcaLab scene.
            self._apply_remote_scene_options()
            self._qpos = np.array(self.combined_model.qpos0, copy=True).reshape(-1)
            self._source_root_reference: np.ndarray | None = None
            self.alignment_report = self._build_alignment_report()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    # -- scene publishing -------------------------------------------------

    def _grid_positions(self) -> np.ndarray:
        if self.spawn_range is not None:
            return self._bounded_center_positions()
        grid_width = int(np.ceil(np.sqrt(max(1, self.num_envs))))
        x0 = -0.5 * self.spacing * (grid_width - 1)
        positions = np.zeros((self.num_envs, 3))
        for index in range(self.num_envs):
            positions[index, 0] = x0 + self.spacing * (index % grid_width)
            positions[index, 1] = x0 + self.spacing * (index // grid_width)
            positions[index, 2] = self.spawn_height
        positions += self.spawn_center
        return positions

    def _bounded_center_positions(self) -> np.ndarray:
        """Dense center-first spawn inside a square half-range.

        Default grid behavior grows with ``num_envs``. When ``spawn_range`` is
        set, actor origins stay within ``[-spawn_range, spawn_range]`` in x/y.
        If there are more envs than unique grid cells, extra actors wrap onto
        the most central cells with a tiny deterministic jitter so OrcaLab still
        receives valid, distinct actor poses.
        """
        if self.spawn_range <= 0.0:
            raise ValueError("--spawn-range must be positive when provided")
        spacing = max(self.spacing, 1e-3)
        half_count = int(np.floor(self.spawn_range / spacing))
        coords_1d = np.arange(-half_count, half_count + 1, dtype=np.float64) * spacing
        cells = np.array(
            [(x, y) for y in coords_1d for x in coords_1d], dtype=np.float64
        )
        # Center-first order keeps small batches and overflow near the middle.
        order = np.lexsort(
            (np.abs(cells[:, 1]), np.abs(cells[:, 0]), np.linalg.norm(cells, axis=1))
        )
        cells = cells[order]

        positions = np.zeros((self.num_envs, 3), dtype=np.float64)
        capacity = max(1, cells.shape[0])
        jitter_radius = min(spacing * 0.18, self.spawn_range * 0.02)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(self.num_envs):
            cell = cells[index % capacity]
            wraps = index // capacity
            jitter = np.zeros(2)
            if wraps:
                angle = (index + 1) * golden_angle
                radius = jitter_radius * ((wraps - 1) % 3 + 1) / 3.0
                jitter = radius * np.array([np.cos(angle), np.sin(angle)])
            positions[index, :2] = np.clip(
                cell + jitter, -self.spawn_range, self.spawn_range
            )
            positions[index, 2] = self.spawn_height
        positions += self.spawn_center
        return positions

    def _publish_actors(self) -> None:
        from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene
        from orca_gym.utils.rotations import euler2quat

        # Publish an empty scene first to clear any previous batch.
        scene = OrcaGymScene(self.orcagym_addr)
        try:
            scene.publish_scene()
            time.sleep(1.0)
        finally:
            scene.close()
        time.sleep(1.0)

        scene = OrcaGymScene(self.orcagym_addr)
        try:
            if self.terrain_asset_path:
                scene.add_actor(
                    Actor(
                        name="terrain",
                        asset_path=self.terrain_asset_path.replace("//", "/"),
                        position=self.terrain_position.tolist(),
                        rotation=euler2quat([0.0, 0.0, 0.0]),
                        scale=1.0,
                    )
                )
            for name, position in zip(self.agent_names, self._grid_positions()):
                scene.add_actor(
                    Actor(
                        name=name,
                        asset_path=self.asset_path.replace("//", "/"),
                        position=position.tolist(),
                        rotation=euler2quat([0.0, 0.0, 0.0]),
                        scale=1.0,
                    )
                )
            print(
                f"[orcalab-play] publishing {self.num_envs} actors to the OrcaLab scene ..."
            )
            scene.publish_scene()
            time.sleep(3.0)
        finally:
            scene.close()
        time.sleep(1.0)

    # -- combined-model indexing ------------------------------------------

    def _connect(self) -> None:
        import grpc
        import mujoco
        from orca_gym import OrcaGymLocal
        from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

        self.channel = grpc.aio.insecure_channel(
            self.orcagym_addr,
            options=[
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ],
        )
        self.stub = GrpcServiceStub(self.channel)
        self.gym = OrcaGymLocal(self.stub)
        source_xml_path = self._load_scene_xml_with_retry()
        self.downloaded_xml_path = str(Path(source_xml_path).expanduser().resolve())
        self.downloaded_xml_contract = scene_xml_contract(self.downloaded_xml_path)
        model_xml_path = self.downloaded_xml_path
        if self.manual_xml_override:
            assert self.aligned_xml_output is not None
            model_xml_path = str(
                patch_scene_xml_options(
                    self.downloaded_xml_path,
                    self.aligned_xml_output,
                    profile=self.scene_profile,
                    timestep=self.scene_timestep,
                    align_air_resistance=self.disable_air_resistance,
                )
            )
        self.combined_model = mujoco.MjModel.from_xml_path(model_xml_path)
        apply_scene_options(
            self.combined_model,
            options=self.scene_options,
            timestep=self.scene_timestep,
            align_air_resistance=self.disable_air_resistance,
        )
        self.resolved_scene_options = assert_scene_options(
            self.combined_model,
            profile=self.scene_profile,
            timestep=self.scene_timestep,
            align_air_resistance=self.disable_air_resistance,
        )
        self.resolved_ground_options = assert_flat_ground_options(self.combined_model)
        self.combined_xml_path = model_xml_path
        self.effective_xml_contract = scene_xml_contract(self.combined_xml_path)

    def _apply_remote_scene_options(self) -> None:
        """Synchronize and optionally verify the remote MuJoCo scene options."""

        async def _apply():
            from orca_gym.protos import mjc_message_pb2

            options = self.scene_options
            current = await self.stub.QueryOptConfig(
                mjc_message_pb2.QueryOptConfigRequest()
            )
            timestep = (
                float(self.scene_timestep)
                if self.scene_timestep is not None
                else options.timestep
            )
            wind = (
                list(options.wind)
                if self.disable_air_resistance
                else list(current.wind)
            )
            density = (
                options.density if self.disable_air_resistance else current.density
            )
            viscosity = (
                options.viscosity if self.disable_air_resistance else current.viscosity
            )
            request = mjc_message_pb2.SetOptConfigRequest(
                timestep=timestep,
                impratio=current.impratio,
                tolerance=options.tolerance,
                ls_tolerance=options.ls_tolerance,
                noslip_tolerance=options.noslip_tolerance,
                ccd_tolerance=options.ccd_tolerance,
                gravity=list(options.gravity),
                wind=wind,
                magnetic=list(current.magnetic),
                density=density,
                viscosity=viscosity,
                o_margin=current.o_margin,
                o_solref=list(current.o_solref),
                o_solimp=list(current.o_solimp),
                o_friction=list(current.o_friction),
                integrator=options.integrator,
                cone=current.cone,
                jacobian=current.jacobian,
                solver=current.solver,
                iterations=options.iterations,
                ls_iterations=options.ls_iterations,
                noslip_iterations=options.noslip_iterations,
                ccd_iterations=options.ccd_iterations,
                disableflags=current.disableflags,
                enableflags=current.enableflags,
                disableactuator=current.disableactuator,
                sdf_initpoints=options.sdf_initpoints,
                sdf_iterations=options.sdf_iterations,
            )
            await self.stub.SetOptConfig(request)
            return await self.stub.QueryOptConfig(
                mjc_message_pb2.QueryOptConfigRequest()
            )

        try:
            actual = self.loop.run_until_complete(_apply())
            self.remote_scene_options = self._assert_remote_scene_options(actual)
            print(
                "[orcalab-play] OrcaLab scene options: "
                f"profile={self.scene_profile}, "
                f"timestep={self.scene_timestep if self.scene_timestep is not None else self.scene_options.timestep}, "
                f"air_resistance={'off' if self.disable_air_resistance else 'keep'}"
            )
        except Exception as exc:
            if self.strict_scene_options:
                raise RuntimeError(
                    "failed to align OrcaLab remote scene options to "
                    f"{self.scene_profile!r}: {type(exc).__name__}: {exc}"
                ) from exc
            print(f"[orcalab-play] warning: failed to sync remote scene options: {exc}")
            self.remote_scene_options = {"verified": False, "error": str(exc)}

    def _assert_remote_scene_options(self, actual) -> dict:
        options = self.scene_options
        timestep = (
            options.timestep if self.scene_timestep is None else self.scene_timestep
        )
        expected = {
            "timestep": float(timestep),
            "integrator": int(options.integrator),
            "gravity": list(options.gravity),
            "iterations": int(options.iterations),
            "ls_iterations": int(options.ls_iterations),
            "noslip_iterations": int(options.noslip_iterations),
            "ccd_iterations": int(options.ccd_iterations),
            "sdf_initpoints": int(options.sdf_initpoints),
            "sdf_iterations": int(options.sdf_iterations),
            "tolerance": float(options.tolerance),
            "ls_tolerance": float(options.ls_tolerance),
            "noslip_tolerance": float(options.noslip_tolerance),
            "ccd_tolerance": float(options.ccd_tolerance),
        }
        if self.disable_air_resistance:
            expected.update(
                {
                    "density": float(options.density),
                    "viscosity": float(options.viscosity),
                    "wind": list(options.wind),
                }
            )
        mismatches = []
        snapshot = {"verified": True, "profile": self.scene_profile}
        for name, wanted in expected.items():
            got = getattr(actual, name)
            if isinstance(wanted, list):
                got = [float(value) for value in got]
                matched = len(got) == len(wanted) and all(
                    abs(left - float(right)) <= 1.0e-6
                    for left, right in zip(got, wanted, strict=True)
                )
            elif isinstance(wanted, float):
                got = float(got)
                matched = abs(got - wanted) <= 1.0e-6
            else:
                got = int(got)
                matched = got == wanted
            snapshot[name] = got
            if not matched:
                mismatches.append(f"{name}={got!r} expected {wanted!r}")
        if mismatches:
            raise RuntimeError("remote option mismatch: " + "; ".join(mismatches))
        return snapshot

    def _load_scene_xml_with_retry(self) -> str:
        """Poll LoadLocalEnv until OrcaLab finishes compiling the batch scene.

        Publishing hundreds of actors makes OrcaLab rebuild its MuJoCo scene;
        until that finishes the RPC answers "MuJoCo has not been initialized
        while starting up. Try again later.", so keep retrying instead of dying.
        """
        deadline = time.perf_counter() + self.load_timeout_s
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.loop.run_until_complete(self.gym.load_model_xml())
            except Exception as exc:
                message = str(exc)
                retryable = (
                    "not been initialized" in message or "Try again later" in message
                )
                if not retryable or time.perf_counter() >= deadline:
                    raise
                wait = min(5.0, 1.0 + 0.5 * attempt)
                print(
                    f"[orcalab-play] OrcaLab is still compiling the scene "
                    f"(attempt {attempt}, retrying in {wait:.1f}s) ..."
                )
                time.sleep(wait)

    def _build_layout(self) -> BatchRenderLayout:
        return build_batch_layout(
            self.combined_model, self.agent_names, self._joint_qpos_addr
        )

    def _build_alignment_report(self) -> dict:
        collision_geoms = collect_scene_geoms(
            self.combined_model,
            self.agent_names,
            collisions_only=True,
            include_robot=False,
        )
        non_ground = [
            geom
            for geom in collision_geoms
            if geom["type"] != "plane"
            and "ActorManipulator" not in geom["name"]
            and "ActorManipulator" not in geom["body"]
        ]
        ground = [geom for geom in collision_geoms if geom["type"] == "plane"]
        return {
            "verified": bool(self.remote_scene_options.get("verified", False)),
            "downloaded_xml": self.downloaded_xml_path,
            "effective_xml": self.combined_xml_path,
            "downloaded_xml_contract": self.downloaded_xml_contract,
            "effective_xml_contract": self.effective_xml_contract,
            "manual_xml_override": self.manual_xml_override,
            "scene_options": scene_options_snapshot(
                self.combined_model,
                profile=self.scene_profile,
            ),
            "ground_options": self.resolved_ground_options,
            "remote_scene_options": self.remote_scene_options,
            "agent_names": list(self.agent_names),
            "joint_qpos_addr": dict(self._joint_qpos_addr),
            "nq_combined": int(self.layout.nq_combined),
            "anchor_to_scene": self.anchor_to_scene,
            "scene_root_anchor": self.layout.root_anchor.tolist(),
            "source_root_reference": None,
            "ground_collision_geoms": ground,
            "non_robot_collision_geom_count": len(collision_geoms),
            "non_ground_collision_geom_count": len(non_ground),
            "non_ground_collision_geoms": [geom["name"] for geom in non_ground[:64]],
        }

    def dump_scene_geoms(
        self,
        path: str | Path,
        *,
        collisions_only: bool = True,
        include_robot: bool = False,
    ) -> list[dict]:
        geoms = collect_scene_geoms(
            self.combined_model,
            self.agent_names,
            collisions_only=collisions_only,
            include_robot=include_robot,
        )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(geoms, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return geoms

    # -- streaming ---------------------------------------------------------

    def map_root_pose(self, qpos_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map local MJWarp roots into the authored OrcaLab actor frames.

        In scene-anchor mode the first local root is treated as the source
        reference.  Subsequent motion is applied relative to the actor's
        compiled qpos0, preserving a hand-authored scene XY/yaw while keeping
        the local physics Z used by the trained Go2 policy.
        """

        qpos = np.asarray(qpos_batch, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[0] != self.num_envs or qpos.shape[1] < 7:
            raise ValueError(
                f"qpos_batch must have shape [{self.num_envs}, >=7], got {qpos.shape}"
            )
        roots = np.array(qpos[:, :7], dtype=np.float64, copy=True)
        roots[:, 3:7] = _normalize_quat_wxyz(roots[:, 3:7])
        if not self.anchor_to_scene:
            positions = roots[:, :3]
            positions[:, :2] *= self.root_xy_scale
            positions += self.layout.root_offset
            positions += self.render_root_offset
            return positions, roots[:, 3:7]

        if self._source_root_reference is None:
            self._source_root_reference = roots.copy()
            if hasattr(self, "alignment_report"):
                self.alignment_report["source_root_reference"] = (
                    self._source_root_reference.tolist()
                )
        reference = self._source_root_reference
        alignment_quat = _quat_multiply_wxyz(
            self.layout.root_anchor[:, 3:7],
            _quat_conjugate_wxyz(reference[:, 3:7]),
        )
        delta = roots[:, :3] - reference[:, :3]
        delta[:, :2] *= self.root_xy_scale
        positions = self.layout.root_anchor[:, :3] + _quat_rotate_wxyz(
            alignment_quat, delta
        )
        # Orca prefab root qpos0 may differ from the local training root.
        # Preserve scene XY/yaw, but keep the actual local physics height.
        positions[:, 2] = roots[:, 2]
        positions += self.render_root_offset
        quaternions = _normalize_quat_wxyz(
            _quat_multiply_wxyz(alignment_quat, roots[:, 3:7])
        )
        return positions, quaternions

    def render(self, qpos_batch: np.ndarray, sim_time: float) -> None:
        """``qpos_batch``: [num_envs, nq_local] with root free joint at 0:7."""
        layout = self.layout
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        values = np.array(qpos_batch[:, layout.src_index], copy=True)  # [N, K]
        positions, quaternions = self.map_root_pose(qpos_batch)
        values[:, 0:3] = positions
        values[:, 3:7] = quaternions
        self._qpos[layout.dst_index.reshape(-1)] = values.reshape(-1)
        self.loop.run_until_complete(
            self.gym.update_local_env(self._qpos, float(sim_time))
        )

    def close(self) -> None:
        loop = getattr(self, "loop", None)
        channel = getattr(self, "channel", None)
        if loop is None or loop.is_closed():
            return
        try:
            if channel is not None:
                loop.run_until_complete(channel.close())
        finally:
            loop.close()
