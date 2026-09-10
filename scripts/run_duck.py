#!/usr/bin/env python
"""在 OrcaLab 里用 ONNX 策略控制 Microduck 鸭子机器人行走。

唯一控制动作 = 速度跟随行走：策略输入 61 维观测，输出 14 个关节位置偏移，
``ctrl = DEFAULT_POSE + action * 1.0``，50 Hz 控制。

用法：
    python run_duck.py                                    # headless 回放，打印位置/速度
    python run_duck.py --lin-vel-x 0.0 --ang-vel-z 1.0    # 原地左转
    python run_duck.py --orcalab --asset-path <prefab>    # 在 OrcaLab 里可视化

依赖：
    - 运行时代码已内置在 vendor/orcalab_rslrl/（OrcaLab 物理运行时 + 渲染器）。
    - Python 包依赖见 requirements.txt / environment.yml，用 setup_env.sh 一键建环境。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent

# 优先用本文件夹内 vendored 的运行时，避免依赖外部 orca_warp 路径
_VENDOR = _HERE / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# ── 鸭子机器人常量（与 ONNX 元数据 / 浏览器端 constants.js 严格一致）─────
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
DEFAULT_POSE = (
    0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
    0.3490658503988659, 0.3490658503988659, 0, 0,
    0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
)
NUM_JOINTS = 14
OBS_SIZE = 61
CMD_SIZE = 13
ACTION_SCALE = 1.0
TIMESTEP = 0.005          # 物理步长 s
DECIMATION = 4            # 每控制步物理子步数 → 控制频率 50 Hz
ROOT_POS = (0.0, 0.0, 0.12)
GYRO_SENSOR = "imu_ang_vel"


def build_model(xml_path: Path):
    """加载鸭子 MJCF，注入地面和 stand 关键帧，编译成 MjModel。"""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_path))
    # 地面（与浏览器端一致：默认摩擦/软度）
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05], pos=[0.0, 0.0, 0.0],
    )
    # stand 关键帧：自由关节 = 根位置 + 单位四元数，14 关节 = DEFAULT_POSE
    qpos = [*ROOT_POS, 1.0, 0.0, 0.0, 0.0] + list(DEFAULT_POSE)
    spec.add_key(name="stand", qpos=qpos, ctrl=list(DEFAULT_POSE))
    spec.option.timestep = float(TIMESTEP)
    return spec.compile()


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """世界向量 v 用四元数 q(wxyz) 的逆转到机体系。"""
    w, xyz = q[..., 0:1], q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


class DuckController:
    """跑 ONNX 策略的闭环：obs → ONNX → action → 物理步进。"""

    def __init__(self, orca, session, in_name, out_name, *, lin_vel_x, lin_vel_y, ang_vel_z):
        self.orca = orca
        self.session = session
        self.in_name, self.out_name = in_name, out_name
        self.device = orca.state.qpos.device
        self.n = orca.num_envs
        self.default_pose = torch.tensor(DEFAULT_POSE, dtype=torch.float32, device=self.device)
        # 13 维指令：只有 twist 三元素非零（每环境相同）
        self.command = torch.zeros(self.n, CMD_SIZE, dtype=torch.float32, device=self.device)
        self.command[:, 0] = lin_vel_x
        self.command[:, 1] = lin_vel_y
        self.command[:, 2] = ang_vel_z
        self.last_action = torch.zeros(self.n, NUM_JOINTS, dtype=torch.float32, device=self.device)


    def build_obs(self) -> torch.Tensor:
        orca = self.orca
        n = orca.num_envs
        gyro = orca.sensor(GYRO_SENSOR)                       # [N, 3]
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        gravity_vec = gravity_vec.expand(n, 3)
        gravity = quat_apply_inverse(orca.state.qpos[:, 3:7], gravity_vec)  # [N, 3]
        joint_pos = orca.actuated_qpos() - orca.default_actuated_qpos()      # [N, 14]
        joint_vel = orca.actuated_qvel()                                     # [N, 14]
        command = self.command
        return torch.cat(
            [gyro, gravity, joint_pos, joint_vel, self.last_action, command], dim=-1
        )  # [N, 61]

    def step(self) -> torch.Tensor:
        obs = self.build_obs()
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        # ONNX 输入固定 batch=1，逐环境推理后再合并
        actions = np.stack(
            [
                self.session.run([self.out_name], {self.in_name: obs_np[i : i + 1]})[0][0]
                for i in range(self.n)
            ],
            axis=0,
        )
        action = torch.as_tensor(actions, device=self.device)
        self.last_action = action
        self.orca.write_action(self.default_pose + action * ACTION_SCALE)  # 位置目标
        self.orca.step(DECIMATION)
        return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="控制 Microduck 鸭子机器人（ONNX 策略）")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lin-vel-x", type=float, default=0.25, help="前进速度 m/s")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="侧向速度 m/s")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="转向角速度 rad/s")
    parser.add_argument("--onnx", default=str(_HERE / "policy" / "BEST_alpha_walking.onnx"), help="ONNX 路径（默认用 policy/BEST_alpha_walking.onnx）")
    parser.add_argument("--orcalab", action="store_true", help="把批量状态流到 OrcaLab 渲染")
    parser.add_argument("--orca-addr", default="localhost:50051")
    parser.add_argument("--asset-path", default="", help="OrcaStudio 里鸭子的 prefab 路径（--orcalab 必填）")
    parser.add_argument("--agent-prefix", default="duck")
    parser.add_argument("--spacing", type=float, default=1.0, help="多环境网格间距 m")
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--realtime", action="store_true", help="按实时速率节流")
    args = parser.parse_args()

    if args.orcalab and not args.asset_path:
        parser.error("--orcalab 需要 --asset-path（OrcaStudio 里的鸭子 prefab 路径）")

    import onnxruntime as ort
    from orcalab_rslrl._internal.runtime import OrcaPhysicsRuntime

    onnx_path = args.onnx
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    model = build_model(_HERE / "robot" / "robot_allcollisions.xml")
    orca = OrcaPhysicsRuntime(model, num_envs=args.num_envs, device=args.device)
    orca.forward()  # 确保传感器就绪

    ctrl = DuckController(
        orca, session, in_name, out_name,
        lin_vel_x=args.lin_vel_x, lin_vel_y=args.lin_vel_y, ang_vel_z=args.ang_vel_z,
    )

    step_dt = TIMESTEP * DECIMATION
    renderer = None
    render_interval = 1.0 / args.render_fps
    if args.orcalab:
        from orcalab_rslrl.orcalab_batch_render import OrcaLabBatchRenderer

        renderer = OrcaLabBatchRenderer(
            orcagym_addr=args.orca_addr,
            num_envs=args.num_envs,
            joint_qpos_addr=orca.joint_qpos_addresses(),
            agent_prefix=args.agent_prefix,
            asset_path=args.asset_path,
            spacing=args.spacing,
            scene_timestep=TIMESTEP,
        )

    sim_time = 0.0
    last_render = 0.0
    try:
        with torch.inference_mode():
            for step in range(args.steps):
                start = time.perf_counter()
                ctrl.step()
                sim_time += step_dt

                if renderer is not None:
                    now = time.perf_counter()
                    if now - last_render >= render_interval:
                        renderer.render(orca.state.qpos.detach().cpu().numpy(), sim_time)
                        last_render = now

                if args.log_every and step % args.log_every == 0:
                    root = orca.state.qpos[:, 0:3].detach().cpu().numpy()
                    vel = orca.state.qvel[:, 0:3].detach().cpu().numpy()
                    print(
                        f"step {step:5d}  root_xyz=({root[0, 0]:+.3f}, {root[0, 1]:+.3f}, "
                        f"{root[0, 2]:+.3f})  lin_vel=({vel[0, 0]:+.3f}, {vel[0, 1]:+.3f}, {vel[0, 2]:+.3f})"
                    )

                if args.realtime:
                    remaining = step_dt - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
