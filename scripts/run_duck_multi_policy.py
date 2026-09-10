#!/usr/bin/env python
"""在 OrcaLab 里用键盘连续控制 Microduck 鸭子，随时切换多个 ONNX 策略（翻滚平滑修复版）。

在 run_duck_multi_policy_rollfix.py 基础上再改两处：
1. 修复多机器人翻滚时「先翻完的鸭子低头等所有鸭子都翻完才抬头」的问题（同 rollfix）——
   每只鸭子翻完并站直后，就地改喂行走策略原地站住（抬头、保持平衡），等其它鸭子也翻完再
   一起切回行走；不会因为一只先翻完就提前结束翻滚。
2. 修复「翻跟头较慢的鸭子直接一闪到新位置」的问题——翻滚结束（全部完成或超时）时，若有
   鸭子还没站直，不再用 reset_mask 把它瞬移复位到出生点（那就是「一闪」的来源），而是切到
   起立恢复（stand）策略，让它在原地平滑站起，随后再一起切回行走。

与 run_duck.py（单一行走策略）互补：本脚本把所有腿部策略一次性加载进来，用键盘
即时切换（行走 / 坐 / 站 / 翻滚 / 左脚踢 / 右脚踢 /
捡拾 / 起立恢复 / 复位），实现
「连续控制」——不需要重启进程，按一下键就换一个动作。

复用 run_duck.py 里已经验证过的模型构建与观测/动作契约（**不修改该文件的任何代码**）。

用法：
    python run_duck_multi_policy.py                  # 实时键盘控制（默认实时）
    python run_duck_multi_policy.py --no-realtime    # 不节流，跑得飞快
    python run_duck_multi_policy.py --orcalab --asset-path <prefab>   # OrcaLab 可视化

键盘映射见启动时打印的帮助横幅。默认只控制 1 只鸭子（一个键盘 → 一只鸭子），
``--num-envs N`` 可让 N 只鸭子执行同一指令。
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent

# 优先用本文件夹内 vendored 的运行时，避免依赖外部 orca_warp 路径
_VENDOR = _HERE / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# 复用 run_duck.py 的常量与模型构建（不修改该文件）
from run_duck import (  # noqa: E402
    build_model,
    quat_apply_inverse,
    DEFAULT_POSE,
    NUM_JOINTS,
    CMD_SIZE,
    ACTION_SCALE,
    TIMESTEP,
    DECIMATION,
    GYRO_SENSOR,
)

CTRL_DT = TIMESTEP * DECIMATION  # 0.02 s 控制周期（50 Hz）

# ── 策略注册表：7 个腿部策略共享同一套 61 维观测 / 14 维动作接口 ──────────
# 注意：drive / crouch 是「滚轮」变体策略（4 个被动轮、不同 MJCF），此处不支持。
POLICIES = {
    "walk":       ("BEST_alpha_walking.onnx",  "行走"),
    "sitstand":   ("BEST_alpha_sitstand.onnx", "坐 / 站"),
    "roll":       ("roulade.onnx",             "翻滚"),
    "kickL":      ("ball_kick_left.onnx",      "左脚踢"),
    "kickR":      ("ball_kick_right.onnx",     "右脚踢"),
    "groundpick": ("alpha_ground_pick.onnx",   "捡拾"),
    "stand":      ("BEST_alpha_stand.onnx",    "起立恢复"),
}

# 连续速度步态（都用 twist 指令 [vx, vy, wz] 驱动，方向键随时转向）
VELOCITY_MODES = {"walk"}

# ── 时序常数（控制步，50 Hz，与浏览器端 game.js 一致）─────────────────────
KICK_STEPS = 25               # 0.5 s 单次踢腿窗口
POST_KICK_LOCK_STEPS = 20     # 0.4 s 踢后锁（冻结指令，避免被行走策略掀翻）
ROLL_MIN_STEPS = 40           # 翻滚完成的最少步数
ROLL_EXPIRE_STEPS = 250       # 5 s 翻滚超时（给翻得慢的鸭子留足时间，超时才走起立恢复）
GROUND_PICK_PERIOD_S = 4.0    # 捡拾相位周期
GROUND_PICK_END_PHASE = 0.7   # 捡拾结束相位（~2.8 s）
SIT_HANDOVER_STEPS = 40       # 0.8 s 先站住再坐下（软交接）
STAND_HANDOVER_STEPS = 100    # 2.0 s 先站起再行走
RECOVER_UPRIGHT_STEPS = 50    # 1 s 连续直立判定为恢复完成
RECOVER_GIVEUP_STEPS = 300    # 6 s 起立超时


class KeyReader:
    """非阻塞读取单个按键（方向键解析成 UP/DOWN/LEFT/RIGHT）。"""

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self.fd = None
        self.old = None
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            # cbreak：关回显、关行缓冲，但保留 Ctrl-C 信号
            tty.setcbreak(self.fd)

    def _read_seq(self) -> bytes:
        seq = b""
        for _ in range(2):
            if select.select([sys.stdin], [], [], 0.02)[0]:
                seq += os.read(self.fd, 1)
        return seq

    def poll(self) -> list[str]:
        """返回本次可读的所有按键；非终端返回空列表。"""
        if not self.enabled:
            return []
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            b = os.read(self.fd, 1)
            if not b:
                break
            if b == b"\x1b":  # 方向键转义序列（CSI \x1b[A / SS3 \x1bOA）
                seq = self._read_seq()
                if seq in (b"[A", b"OA"):
                    keys.append("UP")
                elif seq in (b"[B", b"OB"):
                    keys.append("DOWN")
                elif seq in (b"[C", b"OC"):
                    keys.append("RIGHT")
                elif seq in (b"[D", b"OD"):
                    keys.append("LEFT")
                else:
                    keys.append("ESC")
            else:
                keys.append(b.decode("latin-1"))
        return keys

    def close(self) -> None:
        if self.enabled and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class MultiPolicyController:
    """按当前 mode 跑对应 ONNX 策略，维护各策略的指令约定与一次性动作状态机。"""

    def __init__(self, orca, sessions: dict, *, vx: float = 0.25, vy: float = 0.0, wz: float = 0.0):
        self.orca = orca
        self.sessions = sessions
        self.device = orca.state.qpos.device
        self.n = orca.num_envs
        self.default_pose = torch.tensor(DEFAULT_POSE, dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros(self.n, NUM_JOINTS, dtype=torch.float32, device=self.device)

        # 行走指令（方向键 / WASD 修改）
        self.vx, self.vy, self.wz = float(vx), float(vy), float(wz)

        # 当前模式与一次性动作状态
        self.mode = "walk"
        self.sit_flag = 0                      # sitstand 的 cmd[0]：0=站，1=坐
        self.handover = None                   # ("sit"|"stand", 剩余步数) 软交接
        self.kick_steps = 0
        self.post_kick_lock = 0
        self.roll_steps = 0
        self.roll_tipped = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.roll_done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.pick_phase = 0.0
        self.recover_steps = 0
        self.recover_upright = torch.zeros(self.n, dtype=torch.long, device=self.device)

    # ── 观测 / 动作 ────────────────────────────────────────────────────
    def build_obs(self) -> torch.Tensor:
        orca = self.orca
        n = orca.num_envs
        gyro = orca.sensor(GYRO_SENSOR)                                   # [N, 3]
        g = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).expand(n, 3)
        gravity = quat_apply_inverse(orca.state.qpos[:, 3:7], g)          # [N, 3]
        joint_pos = orca.actuated_qpos() - orca.default_actuated_qpos()   # [N, 14]
        joint_vel = orca.actuated_qvel()                                  # [N, 14]
        return torch.cat([gyro, gravity, joint_pos, joint_vel, self.last_action, self.command], dim=-1)

    @property
    def command(self) -> torch.Tensor:
        """13 维指令，按当前模式填充（每环境相同）。"""
        cmd = torch.zeros(self.n, CMD_SIZE, dtype=torch.float32, device=self.device)
        if self.mode in VELOCITY_MODES:
            cmd[:, 0], cmd[:, 1], cmd[:, 2] = self.vx, self.vy, self.wz
        elif self.mode == "sitstand":
            cmd[:, 0] = float(self.sit_flag)
        elif self.mode == "groundpick":
            ph = 2.0 * math.pi * self.pick_phase
            cmd[:, 0], cmd[:, 1] = math.cos(ph), math.sin(ph)
        # roll / kickL / kickR / stand（恢复）→ 全零指令
        return cmd

    def step(self) -> None:
        obs = self.build_obs()
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        sess = self.sessions[self.mode]
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        # ONNX 输入固定 batch=1，逐环境推理后再合并
        actions = np.stack(
            [sess.run([out_name], {in_name: obs_np[i : i + 1]})[0][0] for i in range(self.n)],
            axis=0,
        )
        # 翻滚期间：已翻完并站直的鸭子改喂行走策略（零指令→原地站住、抬头），未完成的继续滚
        if self.mode == "roll" and self.roll_done.any().item():
            walk_sess = self.sessions["walk"]
            w_in = walk_sess.get_inputs()[0].name
            w_out = walk_sess.get_outputs()[0].name
            done_np = self.roll_done.detach().cpu().numpy()
            for i in range(self.n):
                if done_np[i]:
                    actions[i] = walk_sess.run([w_out], {w_in: obs_np[i : i + 1]})[0][0]
        action = torch.as_tensor(actions, device=self.device)
        self.last_action = action
        self.orca.write_action(self.default_pose + action * ACTION_SCALE)  # 位置目标
        self.orca.step(DECIMATION)
        self._update_oneshots()

    def proj_grav_z(self) -> torch.Tensor:
        """躯干投影重力 z 分量，形状 [N]（< -0.85 直立，> -0.5 倒地）。"""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).expand(self.n, 3)
        return quat_apply_inverse(self.orca.state.qpos[:, 3:7], g)[:, 2]

    # ── 一次性动作状态机（step 末尾推进，为下一步准备好 mode/指令/last_action）──
    def _update_oneshots(self) -> None:
        if self.post_kick_lock > 0:
            self.post_kick_lock -= 1

        # 踢腿：固定 0.5 s 后回 walk（last_action 故意不清零，保持连续动作历史）
        if self.mode in ("kickL", "kickR"):
            self.kick_steps += 1
            if self.kick_steps >= KICK_STEPS:
                self.kick_steps = 0
                self.mode = "walk"
                self.post_kick_lock = POST_KICK_LOCK_STEPS
            return

        # 翻滚：每只鸭子独立判定「翻到一半再立起来」即完成；完成者原地站住等别人，全部完成或 5 s 超时结束
        if self.mode == "roll":
            self.roll_steps += 1
            gz = self.proj_grav_z()                            # [N]
            self.roll_tipped |= gz > -0.3                      # [N] 是否翻倒过
            upright = gz < -0.85                               # [N] 是否已直立
            # 锁存：一旦「翻倒过且已直立」就永久 done（step() 里会改喂行走策略让它抬头站住）
            self.roll_done |= self.roll_tipped & upright & (self.roll_steps >= ROLL_MIN_STEPS)
            expired = self.roll_steps >= ROLL_EXPIRE_STEPS
            if self.roll_done.all().item() or expired:
                self.roll_steps = 0
                self.roll_tipped = torch.zeros(self.n, dtype=torch.bool, device=self.device)
                self.roll_done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
                if upright.all().item():
                    # 全部站直 → 直接切回行走
                    self.mode = "walk"
                    self.last_action = torch.zeros_like(self.last_action)
                else:
                    # 还有没站起来的（翻得慢 / 卡住的）→ 交给起立恢复策略平滑站起；
                    # 不再 reset_mask 瞬移复位（那就是「一闪到新位置」的来源）
                    self.request_recover()
            return

        # 捡拾：相位钟 0 → 0.7（4 s 周期的 ~2.8 s）后回 walk（last_action 不清零）
        if self.mode == "groundpick":
            self.pick_phase += CTRL_DT / GROUND_PICK_PERIOD_S
            if self.pick_phase >= GROUND_PICK_END_PHASE:
                self.pick_phase = 0.0
                self.mode = "walk"
            return

        # 起立恢复：每只鸭子用 stand 策略连续 1 s 直立即完成；6 s 放弃
        if self.mode == "stand":
            self.recover_steps += 1
            upright = self.proj_grav_z() < -0.85            # [N] 是否已直立
            self.recover_upright = torch.where(
                upright, self.recover_upright + 1, torch.zeros_like(self.recover_upright)
            )
            recovered = self.recover_upright >= RECOVER_UPRIGHT_STEPS
            giveup = self.recover_steps >= RECOVER_GIVEUP_STEPS
            if recovered.all().item() or giveup:
                self.recover_steps = 0
                self.recover_upright = torch.zeros(self.n, dtype=torch.long, device=self.device)
                # 仍然倒地的鸭子单独复位，别把倒地鸭子交给行走策略
                self.reset_envs(~upright)
                self.mode = "walk"
                self.last_action = torch.zeros_like(self.last_action)
            return

        # 坐 / 站软交接
        if self.handover is not None:
            kind, steps = self.handover
            steps -= 1
            if steps <= 0:
                if kind == "sit":
                    self.sit_flag = 1
                else:  # "stand"
                    self.mode = "walk"
                    self.sit_flag = 0
                    self.last_action = torch.zeros_like(self.last_action)
                self.handover = None
            else:
                self.handover = (kind, steps)

    # ── 键盘请求（返回 True 表示动作已触发）────────────────────────────
    def set_velocity(self, vx=None, vy=None, wz=None) -> None:
        if vx is not None:
            self.vx = vx
        if vy is not None:
            self.vy = vy
        if wz is not None:
            self.wz = wz
        # 坐姿时按方向键 → 先站起再走
        if self.mode == "sitstand" and self.sit_flag == 1:
            self.request_stand()

    def request_velocity_mode(self, mode: str) -> bool:
        """切换到某个连续速度步态（walk / hop / flamingo）。"""
        if mode not in VELOCITY_MODES or mode == self.mode:
            return False
        if self.handover is not None or self.mode not in VELOCITY_MODES:
            return False
        self.mode = mode
        self.sit_flag = 0
        self.last_action = torch.zeros_like(self.last_action)
        return True

    def request_sit(self) -> bool:
        if self.mode != "walk" or self.handover is not None:
            return False
        self.mode = "sitstand"
        self.sit_flag = 0
        self.handover = ("sit", SIT_HANDOVER_STEPS)
        self.last_action = torch.zeros_like(self.last_action)
        return True

    def request_stand(self) -> bool:
        if not (self.mode == "sitstand" and self.sit_flag == 1):
            return False
        self.sit_flag = 0
        self.handover = ("stand", STAND_HANDOVER_STEPS)
        return True

    def request_roll(self) -> bool:
        if self.mode != "walk" or self.handover is not None:
            return False
        self.mode = "roll"
        self.sit_flag = 0
        self.roll_steps = 0
        self.roll_tipped = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.roll_done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        return True

    def request_kick(self, foot: str) -> bool:
        if self.mode != "walk" or self.handover is not None:
            return False
        self.mode = "kickL" if foot == "left" else "kickR"
        self.sit_flag = 0
        self.kick_steps = 0
        return True

    def request_groundpick(self) -> bool:
        if self.mode != "walk" or self.handover is not None:
            return False
        self.mode = "groundpick"
        self.sit_flag = 0
        self.pick_phase = 0.0
        return True

    def request_recover(self) -> bool:
        if self.mode == "stand":
            return False
        self.mode = "stand"
        self.sit_flag = 0
        self.handover = None
        self.recover_steps = 0
        self.recover_upright = torch.zeros(self.n, dtype=torch.long, device=self.device)
        return True

    def reset(self) -> None:
        """复位到 stand 关键帧（原点站姿），回到行走模式。"""
        self.orca.reset()
        self.last_action = torch.zeros_like(self.last_action)
        self.mode = "walk"
        self.sit_flag = 0
        self.handover = None
        self.kick_steps = 0
        self.post_kick_lock = 0
        self.roll_steps = 0
        self.roll_tipped = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.roll_done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.pick_phase = 0.0
        self.recover_steps = 0
        self.recover_upright = torch.zeros(self.n, dtype=torch.long, device=self.device)

    def reset_envs(self, mask: torch.Tensor) -> None:
        """只复位被 mask 选中的环境到 stand 关键帧（个别鸭子翻车/倒地时用）。"""
        self.orca.reset_mask(mask)

    @property
    def mode_label(self) -> str:
        if self.mode == "sitstand":
            return "坐" if self.sit_flag == 1 else "站"
        return {
            "walk": "行走", "roll": "翻滚", "kickL": "左脚踢", "kickR": "右脚踢",
            "groundpick": "捡拾", "stand": "恢复",
        }.get(self.mode, self.mode)


# 用于触发退出的哨兵值
QUIT = "__QUIT__"


def handle_key(ctrl: MultiPolicyController, key: str):
    """把单个按键映射成控制器动作，返回人类可读标签；QUIT 表示退出。"""
    k = key.lower() if len(key) == 1 else key

    if k in ("q", "\x03", "\x04"):          # q / Ctrl-C / Ctrl-D
        return QUIT
    if k in ("up", "w"):
        ctrl.set_velocity(vx=0.25, vy=0.0, wz=0.0)
        return "前进 (vx=+0.25)"
    if k in ("down", "s"):
        ctrl.set_velocity(vx=-0.15, vy=0.0, wz=0.0)
        return "后退 (vx=-0.15)"
    if k in ("left", "a"):
        ctrl.set_velocity(vx=0.0, vy=0.0, wz=1.0)
        return "左转 (wz=+1.0)"
    if k in ("right", "d"):
        ctrl.set_velocity(vx=0.0, vy=0.0, wz=-1.0)
        return "右转 (wz=-1.0)"
    if k == "0":
        ctrl.set_velocity(vx=0.0, vy=0.0, wz=0.0)
        return "停止"
    if k == "1":
        return "坐下" if ctrl.request_sit() else None
    if k == "2":
        return "站起" if ctrl.request_stand() else None
    if k == "3":
        return "翻滚" if ctrl.request_roll() else None
    if k == "4":
        return "左脚踢" if ctrl.request_kick("left") else None
    if k == "5":
        return "右脚踢" if ctrl.request_kick("right") else None
    if k == "6":
        return "捡拾" if ctrl.request_groundpick() else None
    if k == "7":
        return "起立恢复" if ctrl.request_recover() else None
    if k == "8":
        ctrl.reset()
        return "复位到站姿"
    return None


BANNER = """\
════════════════════════════════════════════════════════════════
  Microduck 多策略键盘控制（50 Hz，随时切换）
════════════════════════════════════════════════════════════════
  方向（转向当前步态；坐姿时按方向键会先站起）：
    ↑ / w  前进      ↓ / s  后退
    ← / a  左转      → / d  右转
    0       停止（速度归零）

  快速切换策略（一次性动作仅在「行走」时触发）：
    1  坐下          2  站起
    3  翻滚          4  左脚踢      5  右脚踢
    6  捡拾          7  起立恢复
    8  复位到站姿

  q / Ctrl-C  退出
════════════════════════════════════════════════════════════════
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="键盘连续控制 Microduck（多策略切换）")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=0, help="最大控制步数，0 = 无限循环")
    parser.add_argument("--log-every", type=int, default=50, help="状态打印间隔（控制步）")
    parser.add_argument("--vx", type=float, default=0.25, help="初始前进速度 m/s")
    parser.add_argument("--policy-dir", default=None, help="ONNX 策略目录（默认 policy/）")
    parser.add_argument("--orcalab", action="store_true", help="把批量状态流到 OrcaLab 渲染")
    parser.add_argument("--orca-addr", default="localhost:50051")
    parser.add_argument("--asset-path", default="", help="OrcaStudio 里鸭子的 prefab 路径（--orcalab 必填）")
    parser.add_argument("--agent-prefix", default="duck")
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument(
        "--realtime", action=argparse.BooleanOptionalAction, default=True,
        help="按实时速率运行（默认开；--no-realtime 关闭，跑得飞快）",
    )
    args = parser.parse_args()

    if args.orcalab and not args.asset_path:
        parser.error("--orcalab 需要 --asset-path（OrcaStudio 里的鸭子 prefab 路径）")

    import onnxruntime as ort
    from orcalab_rslrl._internal.runtime import OrcaPhysicsRuntime

    # ── 加载所有可用策略 ──────────────────────────────────────────────
    policy_dir = Path(args.policy_dir) if args.policy_dir else _HERE / "policy"
    sessions: dict = {}
    for mode, (fname, _) in POLICIES.items():
        path = policy_dir / fname
        if not path.exists():
            print(f"[warn] 找不到策略 {fname}，跳过「{mode}」", file=sys.stderr)
            continue
        sessions[mode] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if "walk" not in sessions:
        sys.exit("至少需要行走策略 BEST_alpha_walking.onnx 才能运行")

    model = build_model(_HERE / "robot" / "robot_allcollisions.xml")
    orca = OrcaPhysicsRuntime(model, num_envs=args.num_envs, device=args.device, ls_parallel=False)
    orca.forward()  # 确保传感器就绪

    ctrl = MultiPolicyController(orca, sessions, vx=args.vx)

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

    print(BANNER)
    print(f"已加载策略：{'、'.join(POLICIES[m][1] for m in sessions)}")
    if not sys.stdin.isatty():
        print("[warn] 非交互终端：键盘控制不可用，将按初始速度持续行走（Ctrl-C 退出）")

    keys = KeyReader()
    sim_time = 0.0
    last_render = 0.0

    def status_line(step: int) -> str:
        root = orca.state.qpos[0, :3].detach().cpu().numpy()
        gz = ctrl.proj_grav_z()                       # [N]
        gz0 = float(gz[0].item())
        n_down = int((gz > -0.5).sum().item())
        flag = f"  [倒地 {n_down}/{ctrl.n}! 按 7 恢复]" if n_down else ""
        return (
            f"[{step:6d}] mode={ctrl.mode_label:<6s} "
            f"root=({root[0]:+.2f},{root[1]:+.2f},{root[2]:+.2f}) "
            f"vx={ctrl.vx:+.2f} wz={ctrl.wz:+.2f} gz0={gz0:+.2f}{flag}"
        )

    try:
        with torch.inference_mode():
            step = 0
            while args.steps <= 0 or step < args.steps:
                start = time.perf_counter()

                # 处理键盘输入
                for key in keys.poll():
                    label = handle_key(ctrl, key)
                    if label == QUIT:
                        print("\n退出。")
                        return
                    if label:
                        print(f"[动作] {label}", flush=True)

                ctrl.step()
                sim_time += CTRL_DT

                if renderer is not None:
                    now = time.perf_counter()
                    if now - last_render >= render_interval:
                        renderer.render(orca.state.qpos.detach().cpu().numpy(), sim_time)
                        last_render = now

                if args.log_every and step % args.log_every == 0:
                    print(status_line(step), flush=True)

                step += 1

                if args.realtime:
                    remaining = CTRL_DT - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n中断。")
    finally:
        keys.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
