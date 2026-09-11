# Microduck · OrcaLab 完整教程：从数学到代码
> **作者：Claude Code**

> 一份**自包含**的完整教程：不假设你懂机器人、物理仿真或强化学习，只要求你会一点 Python。
> 我们从直觉出发，把每一条数学公式严格推导出来，再逐行对上仓库里的真实代码，最后落到「怎么跑起来」。
>
> **读完这份文档，你应该能回答三个问题：**
> 1. 这只鸭子机器人**是什么**、它的身体是怎么被描述的？
> 2. 一个神经网络策略（ONNX）**如何**一步步把 61 个数字变成 14 个数字，又怎么让鸭子动起来？
> 3. 我**如何**跑起它、切换动作、看懂每一行代码在干嘛？

---

## 目录

- [0. 全景：一分钟看懂整个系统](#0-全景一分钟看懂整个系统)
- [第一部分 · 基础数学（深入浅出）](#第一部分--基础数学深入浅出)
  - [1.1 机器人 = 连杆 + 关节](#11-机器人--连杆--关节)
  - [1.2 广义坐标 qpos 与广义速度 qvel](#12-广义坐标-qpos-与广义速度-qvel)
  - [1.3 旋转与四元数（新手友好版）](#13-旋转与四元数新手友好版)
  - [1.4 投影重力：项目里最重要的一个观测](#14-投影重力项目里最重要的一个观测)
  - [1.5 执行器与 PD 控制](#15-执行器与-pd-控制)
- [第二部分 · 机器人模型（MJCF 逐行拆解）](#第二部分--机器人模型mjcf-逐行拆解)
- [第三部分 · 策略与 ONNX（数学 → 代码）](#第三部分--策略与-onnx数学--代码)
- [第四部分 · 代码导读](#第四部分--代码导读)
- [第五部分 · 动手跑起来](#第五部分--动手跑起来)
- [附录 A · 术语速查表](#附录-a--术语速查表)
- [附录 B · 文件职责清单](#附录-b--文件职责清单)
- [附录 C · 关键数字速查](#附录-c--关键数字速查)

---

## 0. 全景：一分钟看懂整个系统

### 0.1 一句话

这是一个**自包含**的文件夹，在 **OrcaLab**（机器人仿真 + 可视化平台）里，用 **ONNX 强化学习
策略**控制一只叫 **Microduck（鸭子）** 的 14 关节机器人行走、跳跃、踢球、翻滚。

### 0.2 三件东西

| 组件 | 是什么 | 在哪 |
|---|---|---|
| **机器人** | 14 关节的鸭子（两腿 + 脖子 + 头），模型在 MJCF 文件里 | `robot/robot_allcollisions.xml` + `robot/assets/*.stl` |
| **策略** | 已训练好的神经网络，输入 61 数字 → 输出 14 数字 | `policy/*.onnx` |

你的代码（`scripts/run_duck.py` / `scripts/run_duck_multi_policy.py`）只是中间那层**胶水**，每 0.02 s 循环一圈：
读观测 → 喂 ONNX → 拿动作 → 写回物理 → 步进。

### 0.3 数据流图

```
                 命令行参数 / 键盘
                        │
                        ▼
      ┌─────────────────────────────────────────┐
      │  DuckController / MultiPolicyController │   Python，CPU 上的「胶水」
      │  （build_obs → 调 ONNX → write_action）  │
      └───────────┬──────────────┬──────────────┘
                  │ obs (61 维)   │ action (14 维)
                  ▼               ▼
      ┌──────────────────────┐   ┌──────────────────────────────┐
      │  ONNX 策略            │   │  OrcaPhysicsRuntime           │
      │  onnxruntime (CPU)   │   │  MjModel(CPU 元数据) + MJWarp  │
      │  obs → action        │   │  nworld = num_envs (GPU)      │
      └──────────────────────┘   │  step(decimation=4)           │
                                  └───────────────┬──────────────┘
                                                  │ qpos (姿态)
                                                  ▼
                                  ┌──────────────────────────────┐
                                  │  OrcaLabBatchRenderer（可选） │
                                  │  gRPC → OrcaLab 渲染          │
                                  └──────────────────────────────┘
```

### 0.4 一个关键设计决定

**物理在 GPU、策略推理在 CPU。** 每个控制步，程序把 GPU 上的观测搬到 CPU（`.detach().cpu().numpy()`），
跑 ONNX，再把动作搬回 GPU（`torch.as_tensor`）。对鸭子这种只有 20 万参数的小网络，这个搬运开销
可忽略，却换来了「策略推理」和「GPU 物理」互不干扰的清晰边界。

---

## 第一部分 · 基础数学（深入浅出）

> 这一部分建立全部的地基。每个概念都按「**直觉 → 数学 → 代码**」的顺序展开。
> 别被「数学」两个字吓到：这里只用到了**向量**和**三角函数**，而且每一步都会先用大白话讲一遍直觉，
> 再给你公式，最后对上一行真实代码。**如果公式一时看不懂，抓住每节的「直觉」和「一句话记住」就够用了。**

### 1.1 机器人 = 连杆 + 关节

#### 直觉：硬棍子 + 合页

任何机器人（机械臂、双足、这只鸭子）都由两种零件搭成：

- **连杆（link / body）**：坚硬的部件，如「大腿」「小腿」「脚」「头」。它自己不变形。
- **关节（joint）**：把两个连杆连起来的「铰链」，让它们能相对运动。

想象一根链条：**连杆 = 链节，关节 = 连接处**。关节只允许特定方向的相对运动。

鸭子身上出现两种关节：

| 关节类型 | 说明 | 鸭子里的例子 |
|---|---|---|
| **铰链（hinge）** | 像门合页，只能绕**一根固定轴**转，1 个自由度 | 14 个：膝盖、髋、踝、脖子…… |
| **自由（free）** | 与地面不固定连接，可任意位置 + 任意姿态，6 个自由度 | 躯干 `trunk_base` |

**自由度（DoF）** = 系统「能独立变化」的维数。一个铰链 = 1，一个自由关节 = 6。
鸭子的自由度 = **6（躯干自由）+ 14（铰链）= 20**。

> 但等一下——下面你会看到「姿势」坐标是 **21** 个数，因为「姿态的 3 个旋转」用四元数表示要
> **4 个数**。先记住这个「20 ≠ 21」的小悬念，答案在 1.3 节。

**代码对照**（`robot/robot_allcollisions.xml`）：

```xml
<body name="trunk_base" pos="0 0 0.12" quat="1 0 0 0">
  <freejoint name="trunk_base_freejoint"/>          <!-- 自由关节：6 DoF -->
  ...
  <joint axis="0 0 1" name="left_hip_yaw" type="hinge" .../>   <!-- 铰链：1 DoF -->
```

> **一句话记住**：机器人 = 一串「会转的硬棍子」。自由度 = 能独立变化的维数，鸭子共 20 个。

### 1.2 广义坐标 qpos 与广义速度 qvel

#### 直觉：把「全身姿势」打包成一个数串

要知道机器人「现在是什么姿势」，只需知道**根的位置 + 朝向**，以及**每个关节转了多少度**。
把这些数排成一列，就是**广义坐标 `qpos`**：

$$
\underbrace{[x,\ y,\ z]}_{\text{trunk position}}
\ \underbrace{[w,\ x,\ y,\ z]}_{\text{trunk orientation (quaternion)}}
\ \underbrace{[q_1,\ q_2,\ \dots,\ q_{14}]}_{\text{14 joint angles}}
\quad\Rightarrow\quad 3 + 4 + 14 = 21\ \text{numbers}
$$

把它摊开写就是：

```
qpos = [根位置x, 根位置y, 根位置z,     ← 躯干在世界里的位置（3 个）
        根姿态qw, qx, qy, qz,          ← 躯干朝向（四元数，4 个）
        关节1, 关节2, ..., 关节14]      ← 14 个关节角
     = 7 + 14 = 21 个数
```

**代码对照**（`scripts/run_duck.py:55-69` 的 `build_model`，注入关键帧时）：

```python
ROOT_POS = (0.0, 0.0, 0.12)                       # 躯干初始位置
qpos = [*ROOT_POS, 1.0, 0.0, 0.0, 0.0] + list(DEFAULT_POSE)   # 7 + 14 = 21
```

`1.0, 0, 0, 0` 是「不旋转」的单位四元数（ $w=1$，1.3 节详解）。

#### qvel：qpos 的变化率

`qvel` 是 `qpos` 对时间的导数：根线速度（3）、躯干角速度（3）、14 个关节角速度（14）。
维数 = **20**（与自由度一致）。

注意一个「不对称」：姿态在 `qpos` 里是 **4 个数**（四元数），到了 `qvel` 里却只有 **3 个数**
（角速度向量）——因为「四元数的变化率」不写成 4 个数，而是用一个 3 维的**角速度**表示。
**这正是「自由度 20、但姿势坐标 21」的原因。**

#### 关节角为什么有正负、有范围

每个铰链都定义了 `range`（允许转的角度范围）：

```xml
<joint name="left_hip_yaw" range="-0.4363323129985824 0.5235987755982988" .../>
<!-- ≈ -25° 到 +30° -->
```

`DEFAULT_POSE` 里的数就是**站立时每个关节的角度**（弧度， $1\ \text{rad} \approx 57.3^{\circ}$）：

```
left_hip_pitch = -0.457924 rad ≈ -26°    （左髋前倾）
neck_pitch     =  0.3490658 rad ≈ 20°    （脖子抬起）
```

> **一句话记住**：`qpos`（21 个数）= 身体此刻「长什么样」；`qvel`（20 个数）= 它正在「怎么动」。
> 位置 21、速度 20，多出来的那个 1 就是四元数多占的一个数。

### 1.3 旋转与四元数（新手友好版）

#### 直觉：为什么「朝向」特别难

「位置」用 3 个数（ $x,y,z$）就能描述，但「朝向」不行。描述空间旋转有三种主流方法：

| 方法 | 形式 | 优缺点 |
|---|---|---|
| 旋转矩阵 | 3×3（9 个数） | 直观、易算，但冗余、占地方 |
| 欧拉角 | 3 个角（roll/pitch/yaw） | 直观，但有「万向锁」奇点 |
| **四元数** | 4 个数 | 无奇点、好插值、运算快，但**不直观** |

机器人/游戏引擎几乎都用四元数。鸭子也是：`qpos[3:7]` 就是躯干的四元数。

#### 核心直觉：一个四元数 = 「绕某根轴转某个角」

这是理解四元数最关键的一句话：

> **一个单位四元数，就是「绕某根轴转某个角度」的完整描述。**

单位四元数 $q = (w,\ x,\ y,\ z)$ 可以写成：

$$
q = \cos\frac{\theta}{2} \;+\; \sin\frac{\theta}{2}\,\big(x\,i + y\,j + z\,k\big)
$$

它的含义：**「绕轴 $(x,y,z)$ 旋转角度 $\theta$」**。其中 $w=\cos\frac{\theta}{2}$ 是「标量部分」，
$(x,y,z)$ 是「向量部分」（表示旋转轴，已经被 $\sin\frac{\theta}{2}$ 缩放过）。

用两个特例立刻验证这个直觉：

- **不旋转**（ $\theta=0$）→ $q=(1,0,0,0)$。这正是代码里 `1.0, 0, 0, 0` 的含义。
- **绕 z 轴转 90°**（ $\theta=90^{\circ}$）→ $q=(\cos45^{\circ},\ 0,\ 0,\ \sin45^{\circ}) = (0.707,\ 0,\ 0,\ 0.707)$。

**⚠️ 顺序是 wxyz**：MuJoCo 和本项目用 `wxyz` 顺序（很多库是 `xyzw`，混用会得到完全错误的结果）：

```python
q = [w, x, y, z]        # w 是标量部分，x,y,z 是向量部分
```

#### 单位四元数

四元数 $q = w + x\,i + y\,j + z\,k$，写作 $(w,\ \mathbf{u})$，其中 $\mathbf{u}=(x,y,z)$ 是「向量部分」。
**单位四元数**满足 $\lVert q \rVert^2 = w^2 + x^2 + y^2 + z^2 = 1$，表示一个纯旋转（没有缩放、没有镜像）。

#### 两个四元数相乘 = 两次旋转叠加（Hamilton 乘积）

四元数乘法记作 $\otimes$，**不满足交换律**（先绕 A 再绕 B ≠ 先绕 B 再绕 A）：

$$
q_1 \otimes q_2 = \Big(\ w_1 w_2 - \mathbf{u}_1 \cdot \mathbf{u}_2\ ,\quad w_1 \mathbf{u}_2 + w_2 \mathbf{u}_1 + \mathbf{u}_1 \times \mathbf{u}_2\ \Big)
$$

其中 $\cdot$ 是点积， $\times$ 是叉积。直觉：**$q_1 \otimes q_2$ = 先做 $q_2$ 的旋转，再做 $q_1$ 的旋转**。

#### 用四元数旋转一个向量

把向量 $\mathbf{v}$ 当「纯四元数」 $(0,\ \mathbf{v})$，单位四元数 $q$ 旋转它的公式是：

$$
\mathbf{v}' = q \otimes \mathbf{v} \otimes q^{-1}
$$

共轭（单位四元数下即逆） $q^{-1}=(w,\ -\mathbf{u})$。展开成向量形式（罗德里格斯式）：

**正向**（ $q$ 作用）：

$$
\mathbf{v}' = \mathbf{v} + 2w\,(\mathbf{u}\times\mathbf{v}) + 2\,\mathbf{u}\times(\mathbf{u}\times\mathbf{v})
$$

**反向**（ $q^{-1}$ 作用）：

$$
\mathbf{v}' = \mathbf{v} - 2w\,(\mathbf{u}\times\mathbf{v}) + 2\,\mathbf{u}\times(\mathbf{u}\times\mathbf{v})
$$

（看到没？正向和反向只差中间那项的**一个正负号**。）

#### 对照代码：`quat_apply_inverse`

`scripts/run_duck.py:72-76`：

```python
def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """世界向量 v 用四元数 q(wxyz) 的逆转到机体系。"""
    w, xyz = q[..., 0:1], q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)            # t = 2(u × v)
    return v - w * t + torch.cross(xyz, t, dim=-1)   # v − 2w(u×v) + 2u×(u×v)
```

把 $t = 2(\mathbf{u}\times\mathbf{v})$ 代回去，正是上面「反向」公式：

$$
\begin{aligned}
\text{return} &= \mathbf{v} - w\,t + \mathbf{u}\times t \\
&= \mathbf{v} - 2w(\mathbf{u}\times\mathbf{v}) + 2\,\mathbf{u}\times(\mathbf{u}\times\mathbf{v})
\end{aligned}
$$

**语义**：`quat_apply_inverse(q, v)` 把**世界系**向量 $\mathbf{v}$ 用 $q$ 的**逆**旋转，得到它在
「 $q$ 所表示的机体坐标系」里的坐标。为什么叫 inverse？因为 $q$ 描述的是「机体系 → 世界系」的旋转，
要把世界向量转到机体系，就得用它的逆 $q^{-1}$。

#### 世界系 vs 机体系

- **世界系（world frame）**：固定、绝对的地面坐标系。重力永远朝世界系 $-z$。
- **机体系（body frame）**：跟着机器人走，固定在躯干上。

「把世界向量转到机体系」靠 `quat_apply_inverse`——这正是下一节投影重力要做的事。

#### 新手作弊小抄

> - 你**只需要记住**：`qpos[3:7]` = 躯干朝向，顺序 `wxyz`；「不旋转」就是 `(1,0,0,0)`。
> - 四元数 = 「绕某根轴转某个角」；`quat_apply_inverse(q, v)` = 用 $q$ 的逆把世界向量转到机体系。
> - 上面的乘法、罗德里格斯展开，属于「想知道为什么」才需要看；**平时写代码根本不会手算它们。**

### 1.4 投影重力：项目里最重要的一个观测

#### 一个绝妙的技巧

机器人在**没有 GPS、没有视觉**时，怎么知道自己站没站直？答案藏在重力里：

> **重力永远朝下。** 如果你知道「重力方向在我的身体坐标系里指向哪」，你就能推断身体相对地面
> 的倾斜程度。这正是 IMU（惯性测量单元，手机里也有）的原理。

#### 数学：把「朝下的重力」转到身体坐标系

世界系重力方向（归一化到单位向量）固定为：

$$
\mathbf{g}_{\text{world}} = [0,\ 0,\ -1]
$$

投影重力 = 用躯干四元数 $q$ 的**逆**把它转到机体系：

$$
\begin{aligned}
\mathbf{g}_{\text{body}} &= q^{-1} \otimes \mathbf{g}_{\text{world}} \otimes q \\
&= \texttt{quat\_apply\_inverse}(q,\ \mathbf{g}_{\text{world}})
\end{aligned}
$$

**代码对照**（`scripts/run_duck.py:100-103` 的 `build_obs`）：

```python
gravity_vec = torch.tensor([0.0, 0.0, -1.0], ...)
gravity = quat_apply_inverse(orca.state.qpos[:, 3:7], gravity_vec)   # [N, 3]
```

#### 结果的含义：看 z 分量

| 躯干状态 | $\mathbf{g}_{\text{body}} = [g_x,\ g_y,\ g_z]$ | 含义 |
|---|---|---|
| 完全直立 | $[0,\ 0,\ -1]$ | 重力正好沿机体系 $-z$，身体没歪 |
| 侧躺 | $[\pm1,\ 0,\ 0]$ 附近 | 重力落在机体系 $\pm x$，躺平了 |
| 倒立 | $[0,\ 0,\ +1]$ | 头朝下 |

所以 **$g_z$ 越接近 $-1$ 越直立，越接近 $0$ 越接近躺平**。

#### 几何推导： $g_z = -\cos\theta$

设躯干绕 $x$ 轴倾斜（roll）了角度 $\theta$，机体系 $z$ 轴在世界系变为
$\hat{\mathbf{z}}_{\text{body}} = [0,\ \sin\theta,\ \cos\theta]$。
世界重力 $[0,0,-1]$ 在机体系 $z$ 轴上的分量由点积给出：

$$
\begin{aligned}
g_z &= \mathbf{g}_{\text{world}} \cdot \hat{\mathbf{z}}_{\text{body}} \\
    &= [0,\ 0,\ -1] \cdot [0,\ \sin\theta,\ \cos\theta] = -\cos\theta
\end{aligned}
$$

- 直立（ $\theta=0$）→ $g_z = -1$
- 倾斜 30° → $-\cos30^{\circ} = -0.866$
- 躺平（ $\theta=90^{\circ}$）→ $-\cos90^{\circ} = 0$

**这正是代码里所有判据的数学来源**（`scripts/run_duck_multi_policy.py:221-226` 的 `proj_grav_z`）：

```python
gz < -0.85    → 直立      # −0.85 ≈ −cos(31.8°)，留了走路自然晃动的余量
gz > -0.3     → 翻倒/明显倾斜
```

> 为什么用 $-0.85$ 而不是 $-1$？因为走路时身体会自然轻微晃动， $-0.85$ 留了余量，避免把正常晃动
> 误判成摔倒。真机的 $g_z$ 也不严格等于 $-\cos\theta$（还受 pitch/yaw 影响），但作为「身体相对重力
> 有多正」的一阶度量，这个标量足够鲁棒。

> **一句话记住**：把「朝下的重力」转到身体坐标系，看它的 $z$ 分量——越接近 $-1$ 越正，越接近 $0$ 越歪。

### 1.5 执行器与 PD 控制

#### 执行器（actuator）= 电机

「让关节转起来」的部件是执行器（通常是电机/舵机）。MJCF 文件末尾的 `<actuator>` 块声明了
14 个执行器，**每个控制一个关节**：

```xml
<position class="chosen_actuator" name="left_hip_yaw" joint="left_hip_yaw"/>
```

#### 三种控制模式

| 模式 | 你给的是 | 电机自己管的是 |
|---|---|---|
| 力矩控制 | 力/力矩 | 什么都不管（最底层） |
| **位置控制** | 目标角度 | 自己产生力矩去「追」这个角度 |
| 速度控制 | 目标角速度 | 自己产生力矩去「追」这个速度 |

**鸭子用位置控制**：ONNX 输出 14 个「目标关节角」，底下的电机自己去追。

#### PD 控制器：电机怎么「追」目标角（弹簧 + 阻尼器）

`<position kp kv>` 执行器内部是 **PD（比例-微分）控制器**，按此公式产生力矩 $\tau$：

$$
\tau = k_p\,(q^* - q) \;-\; k_v\,\dot{q}
$$

其中 $q^*$ 是目标角， $q$ 是当前角， $\dot{q}$ 是当前角速度。

- **P（比例）项** $k_p\,(q^* - q)$：误差越大（离目标越远），力越大——**像弹簧**，拉得越开弹力越大。
- **D（微分）项** $-k_v\,\dot{q}$：速度越快，反向阻力越大——**像阻尼器**，防止震荡过头。

鸭子的参数（`robot_allcollisions.xml` 里 `class="chosen_actuator"`）：

```xml
<position kp="0.55" kv="0.0" forcerange="-0.96 0.96" ctrlrange="-10.0 10.0"/>
```

- $k_p = 0.55$：**低刚度**（软）。力矩 $= 0.55\times$ 角度差，很「软」，鸭子软软地被拉到目标位姿。
- $k_v = 0.0$：没有微分阻尼 → 是**纯比例（P）控制器**。
- `forcerange = ±0.96`：电机力矩上限（饱和）。
- `ctrlrange = ±10`：允许的控制输入范围（`write_action` 会 clamp 到这里）。

代入鸭子参数：

$$
\tau = 0.55\,(q^* - q),\qquad |\tau| \le 0.96\ \text{N·m}
$$

> 关节本身还带 `damping=0.053`（关节阻尼）、`frictionloss=0.0048`（摩擦损耗）、`armature=0.0018`
> （转子惯性）。这些一起决定「电机发出 0.55 的力矩，关节实际转多快」。它们是训练时就固定下来的
> 真实电机特性，运行时**不改**。

> **一句话记住**：电机是个「软弹簧」——看到目标角在哪，就朝那儿拉；差得越远拉得越用力（ $k_p$ 决定软硬）。

---

## 第二部分 · 机器人模型（MJCF 逐行拆解）

MJCF（MuJoCo 的 XML 格式）完整定义了鸭子。`robot/robot_allcollisions.xml` 由 Onshape（CAD）自动导出，
共 434 行。理解它就理解了「鸭子本体」。

### 2.1 自由度：1 个自由根 + 14 个铰链

14 个铰链关节的**顺序严格固定**（这是跟 ONNX 对齐的生命线）：

```
左腿:    left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
脖子/头: neck_pitch, head_pitch, head_yaw, head_roll
右腿:    right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

规律：左腿 5 个 → 脖子头 4 个 → 右腿 5 个。左右腿是镜像对称（右腿的 `range` 和惯性符号跟左腿相反）。

这 14 个名字精确对应 `scripts/run_duck.py:35-39` 的 `JOINT_NAMES`，也是 ONNX 元数据里 `joint_names` 的顺序。
**顺序一旦错位，策略就失效**——它会把这个关节的动作用到另一个关节上。

所以 `qpos` 一共 **7 + 14 = 21 维**：`[根x,根y,根z, qw,qx,qy,qz, 关节1..14]`。

### 2.2 关节物理参数

每个被驱动关节挂在 `class="chosen_actuator"` 里，继承三个数：

```xml
<default class="chosen_actuator">
  <joint damping="0.053" frictionloss="0.0048" armature="0.0018"/>
  <position kp="0.55" kv="0.0" forcerange="-0.96 0.96" ctrlrange="-10.0 10.0"/>
</default>
```

- `damping`：关节黏性阻尼（速度阻力）
- `frictionloss`：库仑摩擦损耗
- `armature`：电枢（转子）惯性，反映「电机转子自身也有质量、转起来有惯性」

> 文件里还留着 `chosen_actuator_old` / `_new` / `_antoine` 等历史版本（`kp` 从 0.386 到 0.52 不等），
> 是不同批次的实测参数。**当前生效的是 `chosen_actuator`（注释 "marc / 200 kp"）：`kp=0.55`。**

### 2.3 执行器：PD 位置伺服

文件末尾 `<actuator>` 块声明 14 个执行器（`robot_allcollisions.xml:417-432`），每个对应一个关节：

```xml
<position class="chosen_actuator" name="left_hip_yaw" joint="left_hip_yaw"/>
<!-- ... 共 14 个 ... -->
```

`<position>` 是 MuJoCo 内置的 **PD 位置伺服**，力律：

```
τ = kp·(ctrl − q) − kv·q̇   =   0.55·(目标角 − 当前角) − 0·角速度
```

其中 `ctrl`（控制信号）就是**目标关节角**。这正是 `ctrl = DEFAULT_POSE + action` 公式的来源：
ONNX 输出的 `action` 是**相对默认站姿的偏移量**，加上 `DEFAULT_POSE` 得到绝对目标角。

### 2.4 传感器

```xml
<sensor>
  <framequat name="orientation" objtype="site" noise="0.001" objname="imu"/>
  <gyro name="angular-velocity" site="imu" noise="0.005"/>
  <gyro name="imu_ang_vel" site="imu"/>                      <!-- ← 实际用的 -->
  <velocimeter name="imu_lin_vel" site="imu"/>
  <accelerometer name="imu_accel" site="imu"/>
  <subtreeangmom name="root_angmom" body="trunk_base"/>
</sensor>
```

注意有**两个陀螺仪**：`angular-velocity`（带噪声 0.005，模拟真实 IMU）和 `imu_ang_vel`（无噪声，理想值）。
代码用 **`imu_ang_vel`**（`scripts/run_duck.py:52` 的 `GYRO_SENSOR = "imu_ang_vel"`），因为回放不需要加噪声。

传感器挂在 `site="imu"`（躯干里的一个刚体点）。传感器本质是从状态 `(q, q̇)` **计算**出来的函数：
读传感器 = 对 `qpos/qvel` 做一次运动学计算——这也正是 `orca.forward()` 干的事（先算好传感器值，
`orca.sensor(name)` 才能取到数）。

### 2.5 关键帧 `stand`

`build_model`（`scripts/run_duck.py:55-69`）会注入一个**关键帧**，作为「默认站姿 / 复位状态」：

```python
qpos = [*ROOT_POS, 1.0, 0.0, 0.0, 0.0] + list(DEFAULT_POSE)   # 7 + 14 = 21
spec.add_key(name="stand", qpos=qpos, ctrl=list(DEFAULT_POSE))
```

`DEFAULT_POSE`（`scripts/run_duck.py:40-44`）：

```
左腿:   0, -0.0872665, -0.457924, -0.004940,  0.452984
脖子头:  0.3490659,  0.3490659,  0,        0
右腿:   0,  0.0872665,  0.457924,  0.004940, -0.452984
```

左右腿对称（符号相反）：`±0.0873 rad = ±5°`（髋 roll），`±0.4579 rad ≈ ±26°`（髋 pitch），
`±0.4530 rad ≈ ±26°`（踝），脖子/头 pitch `0.3491 rad = 20°`。

这个关键帧干了两件关键事：

1. 设定「默认关节角」= `DEFAULT_POSE`，于是运行时 `default_actuated_qpos() == DEFAULT_POSE`。
2. 让 `ctrl`（目标角）的初始值也等于 `DEFAULT_POSE`，保证 `ctrl = DEFAULT_POSE + action` 成立。

### 2.6 地面与时间步

`build_model` 还注入了一个平面地面，并把物理步长设为 0.005 s：

```python
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0,0,0.05], pos=[0,0,0])
spec.option.timestep = float(TIMESTEP)   # 0.005 s → 200 Hz 物理
```

**其它 solver / integrator 参数全部保持 MuJoCo 默认**——因为策略就是在默认求解器下训出来的，
改了会破坏训练时依赖的物理一致性。

---

## 第三部分 · 策略与 ONNX（数学 → 代码）

这一部分是「策略接口」的精确定义 + ONNX 神经网络内部的完整数学。

### 3.1 观测 / 动作 / 指令契约

**任何 ONNX 要能在这里跑，就必须遵守这个契约。** 两个脚本都在 `build_obs` / `step` 里实现它。

#### 观测（61 维）

`build_obs()` 拼出 61 维向量，顺序严格如下：

| 下标 | 含义 | 来源 |
|---|---|---|
| `[0:3]` | 躯干角速度（陀螺仪） | `orca.sensor("imu_ang_vel")` |
| `[3:6]` | 投影重力（机体系中的重力方向） | `quat_apply_inverse(qpos[:,3:7], [0,0,-1])` |
| `[6:20]` | 14 关节角（相对默认站姿） | `actuated_qpos() − default_actuated_qpos()` |
| `[20:34]` | 14 关节角速度 | `actuated_qvel()` |
| `[34:48]` | 上一拍动作（14 维） | `self.last_action` |
| `[48:61]` | 指令（13 维） | `self.command` |

**「关节角相对站姿」** 这段减法很关键：它把观测中心移到 `DEFAULT_POSE`，与动作的「相对偏移」约定对称——
**策略看到的是「偏离站姿多少」，输出的也是「偏离站姿多少」**，整个学习问题被中心化，数值条件更好。

**「上一拍动作」**（action history）让策略「记得」自己上一拍发了什么，有助于产生平滑连贯的步态
（相当于给了一个低通/平滑先验）。

#### 动作（14 维）

ONNX 输出 14 个数，是**关节位置偏移**。应用公式：

```
ctrl[j] = DEFAULT_POSE[j] + action[j] * ACTION_SCALE     # ACTION_SCALE = 1.0
```

`ACTION_SCALE = 1.0`，所以就是 `DEFAULT_POSE + action`。这是**位置控制**（不是力矩/速度），
因为底层是 `<position>` 执行器。

#### 指令（13 维）

`self.command` 是 `[N, 13]` 的张量。对行走类策略，只有前 3 个非零（twist）：

```
command = [vx, vy, wz, 0,0,0,0,0,0,0,0,0,0]   # 前向速度、侧向速度、转向角速度
```

其余 10 维是给「头姿(4) + 身体姿态(6)」预留的，行走时恒为 0。

#### 时间：decimation（抽取）

```
TIMESTEP   = 0.005 s   # 物理步长（200 Hz）
DECIMATION = 4         # 每个控制步走 4 个物理子步
→ 控制频率 = 1 / (0.005 × 4) = 50 Hz（0.02 s）
```

策略每 0.02 s 才决策一次，中间 4 个物理子步沿用同一个目标角。

### 3.2 神经网络结构（实测）

用 `onnx` 库 dump `policy/BEST_alpha_walking.onnx` 的真实计算图：

| 层 | 算子 | 形状 | 激活 |
|---|---|---|---|
| 输入 | — | `[1, 61]` | — |
| 归一化 | `Sub` → `Div` | `[1, 61]` | 见 3.3 |
| 隐藏层 1 | `Gemm`（全连接） | `61 → 512` | `ELU` |
| 隐藏层 2 | `Gemm` | `512 → 256` | `ELU` |
| 隐藏层 3 | `Gemm` | `256 → 128` | `ELU` |
| 输出层 | `Gemm` | `128 → 14` | 无 |
| 输出 | — | `[1, 14]` | — |

即一个 **3 隐藏层 MLP**：`61 → 512 → 256 → 128 → 14`。总参数量 = **197,774**。

### 3.3 归一化：减去均值、除以标准差

神经网络对输入的**数值范围**很敏感：如果某个输入经常是几百、另一个只有零点几，网络很难同时用好它们。
所以先把每个输入维度标准化（61 个维度各自独立做）：

$$
\hat{x}_i = \frac{o_i - \mu_i}{\sigma_i}
$$

其中 $\mu_i$（`obs_normalizer._mean`）和 $\sigma_i$（`onnx::Div_24`）都是 `[1, 61]` 常量，是**推理前离线算好、
固化在模型里的统计量**。实测各段：

| 观测段 | 均值 $\mu$（典型范围） | 标准差 $\sigma$（典型范围） |
|---|---|---|
| 陀螺仪角速度 [0:3] | ≈ 0 | 0.80 ~ 1.19 |
| 投影重力 [3:6] | 0.005 ~ **−0.995** | 0.03 ~ 0.067 |
| 关节角 [6:20] | −0.057 ~ +0.046 | 0.14 ~ 0.16 |
| 关节速度 [20:34] | ≈ 0 | 1.68 ~ 2.05 |
| 上拍动作 [34:48] | −0.053 ~ +0.007 | 0.40 ~ 0.44 |
| 指令 [48:61] | 0.0445 → 0 | 0.21 → 0.039 |

注意投影重力的 $z$ 分量均值是 **−0.995**（几乎恒 $-1$），标准差很小（0.03）——正好印证 1.4 节：
直立时 $g_z = -1$。

> **数学要点**：标准化把每个输入维度都拉到「均值 0、方差 1」附近，否则 `joint_vel`（量级 ~2）会
> 在数值上压过 `projected_gravity`（量级 ~0.03），让网络难以同时利用两者。

### 3.4 全连接层（Gemm）：矩阵乘 + 偏置

ONNX 里全连接层叫 `Gemm`（GEneral Matrix multiply），`transB=1, alpha=1, beta=1` 时等价于
PyTorch 的 `nn.Linear`：

$$
\mathbf{y} = \mathbf{W}^{\top} \mathbf{x} + \mathbf{b}
$$

具体地，第 $l$ 层输入 $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$，输出 $\mathbf{y} \in \mathbb{R}^{d_{\text{out}}}$：

$$
y_j = \sum_{i=1}^{d_{\text{in}}} W[j,\ i]\cdot x_i + b_j,\qquad j = 1 \dots d_{\text{out}}
$$

四层维度变化（ $\mathbf{W}$ 的形状是 `[d_out, d_in]`）：

| 层 | $\mathbf{W}$ 形状 | 计算 | 偏置 $\mathbf{b}$ |
|---|---|---|---|
| L1 | `[512, 61]` | 512×61 矩阵乘 | `[512]` |
| L2 | `[256, 512]` | 256×512 | `[256]` |
| L3 | `[128, 256]` | 128×256 | `[128]` |
| L4 | `[14, 128]` | 14×128 | `[14]` |

参数量核对：

$$
\begin{aligned}
& (512\times61 + 512) + (256\times512 + 256) + (128\times256 + 128) + (14\times128 + 14) \\
&= 31{,}744 + 131{,}328 + 32{,}896 + 1{,}806 = 197{,}774 \quad\checkmark
\end{aligned}
$$

> 一句大白话：每层就是「把输入的数乘上一大张权重表，再加一组偏置」。512 个神经元 = 512 个不同的
> 「加权和」，每个都带着自己对 61 个输入的不同「关注程度」。

### 3.5 ELU 激活函数

三个隐藏层之间夹着 `ELU`（Exponential Linear Unit），`alpha=1.0`：

$$
\text{ELU}(z) =
\begin{cases}
z, & z \ge 0 \\
\alpha\,(e^{z} - 1), & z < 0
\end{cases}
\qquad (\alpha = 1)
$$

为什么用它而不是 ReLU：

- 负半轴**光滑**（处处可导），梯度不会在 0 处「断崖」；负输入也有非零输出和梯度，缓解
  「死神经元」（ReLU 在负区间梯度恒为 0 的问题）。
- 输出均值更接近 0（不像 ReLU 恒非负），配合标准化让深层训练更稳。

> 一句大白话：激活函数给网络「加一点非线性的弯」——否则三层线性层叠在一起，数学上等价于一层，
> 就学不出复杂动作了。

### 3.6 完整前向传播（写成复合函数）

把上面几节串起来，整个策略就是一个**函数** $f_\theta$（符号 $\circ$ 读作「然后」，先做右边、再做左边）：

$$
f_\theta = L_4 \circ \text{ELU} \circ L_3 \circ \text{ELU} \circ L_2 \circ \text{ELU} \circ L_1,
\qquad L_i(\mathbf{x}) = \mathbf{W}_i^{\top} \mathbf{x} + \mathbf{b}_i
$$

给定观测 $o_t$，先归一化再送入 $f_\theta$，得到动作：

$$
a_t = f_\theta\big(\hat{x}\big),\qquad \hat{x} = (o_t - \mu) \oslash \sigma
$$

展开成一句话：**归一化 → 线性 → ELU → 线性 → ELU → 线性 → ELU → 线性**，
维度 $61 \to 512 \to 256 \to 128 \to 14$。

> 输出层 $L_4$ **没有**激活函数，所以动作可以取任意实数（正负皆可）——这正是「位置偏移」应有的
> 样子：关节角要能在默认站姿上下双向调整。

### 3.7 从动作到目标角

ONNX 输出 14 个数 `a_t`，但 XML 执行器要的是**绝对目标关节角** `c_t`：

```
c_t = q_default + s ⊙ a_t
```

- `q_default` = `DEFAULT_POSE`（14 维）
- `s` = `action_scale` = 1.0，所以就是 `c_t = q_default + a_t`

**代码对照**（`scripts/run_duck.py:124`）：

```python
self.orca.write_action(self.default_pose + action * ACTION_SCALE)   # ACTION_SCALE = 1.0
```

**为什么要有这个偏移？** 神经网络的输出如果直接当绝对角度，训练时要在「−π 到 π」的巨大范围里找
平衡点，难学；让它只输出**相对站姿的小偏移**（典型 ±0.5 rad），问题就变成「在 0 附近微调」，
学习更容易也更稳定。`q_default` 就是「学习中心」。

---

## 第四部分 · 代码导读

> 这一部分不「逐函数、逐行号」地念代码，而是先讲清楚**每段代码在干什么、为什么这么写**。
> 想知道某个函数在哪一行，按文件名去搜就行。

### 4.0 先记住一句话

整个项目本质上就是一个**循环**，每 **0.02 秒**转一圈：

```
看 → 想 → 动 →（画）
```

1. **看**：读传感器（陀螺仪、关节角度、关节速度……），整理成 61 个数字，叫「观测」。
2. **想**：把这 61 个数字喂给一个神经网络（ONNX 文件），它吐出 14 个数字，叫「动作」。
3. **动**：把这 14 个数字当成「电机要转到的目标角度」，交给物理引擎；物理引擎算出鸭子下一瞬间的姿势。
4. **画**（可选）：把新姿势发给 OrcaLab 显示在屏幕上。

一圈一圈转下去，鸭子就「走」起来了。下面所有代码，都是为了让这四步顺畅地转起来。

### 4.1 `scripts/run_duck.py` —— 一个策略的「回放器」

这是最简单、最该先看的文件。它只干一件事：**加载一个 ONNX 策略，让它控制鸭子**。

把它想成一个「播放器」：你给它一盘磁带（一个 ONNX 策略），它就不停地播放（控制鸭子动）。

#### 文件开头的「参数表」

文件顶部有一堆大写名字。它们不是逻辑，只是**把数字起了个好记的名字**，方便后面引用：

- `JOINT_NAMES`：鸭子的 14 个关节叫什么名字（顺序不能乱，要和神经网络对齐）。
- `DEFAULT_POSE`：鸭子的「标准站姿」——14 个关节各是多少度。
- `TIMESTEP = 0.005`：物理引擎每算一步是 0.005 秒（一秒算 200 次）。
- `DECIMATION = 4`：每「想」一次，物理引擎要先算 4 步，所以控制频率是 200 ÷ 4 = 50 Hz，
  也就是前面说的「每 0.02 秒想一次」。

> 新手只要记住一句：**物理跑得快（200 Hz），大脑想得慢（50 Hz），中间靠 DECIMATION 对齐**。

#### `build_model()`：把机器人「说明书」读进来

鸭子长什么样、关节在哪、电机有多大力，都写在 XML 说明书里（`robot/robot_allcollisions.xml`）。
`build_model()` 做三件事：

1. 读这份 XML 说明书；
2. 往里面**补一块地面**（不然鸭子会掉下去）；
3. 往里面**存一个「标准站姿」**（后面复位、起立都用它）。

#### `DuckController.step()`：整个项目的心脏

`step()` 就是前面那个循环「看 → 想 → 动」，写出来只有几行：

```
看：读传感器 → 拼成 61 个数字
想：61 个数字交给 ONNX → 得到 14 个数字
动：14 个数字 + 标准站姿 = 目标角度 → 交给电机 → 物理引擎推一步
```

有一个小坑值得知道：这个 ONNX 是「一次只能看一只鸭子」的（batch=1）。跑多只鸭子时，就得一只一只地问，
再把答案拼起来。演示无所谓，真正的大规模训练走的是另一条路（PyTorch 批量推理）。

#### `main()`：把零件组装起来

`main()` 是入口，按顺序做：读命令行参数 → 加载 ONNX → 建模型 → 建物理引擎 → 建控制器 → 进入循环。

> 到这里，你已经能让**一只鸭子走起来**了。剩下两个文件，是为了「让鸭子会**切换动作**」和「让它在**屏幕上**动」。

### 4.2 `scripts/run_duck_multi_policy.py` —— 让鸭子「会多项技能」

一个 ONNX 只会一件事：有的会走，有的会坐，有的会翻滚。怎么让鸭子「又会走、又会坐、又会翻滚」？

**答案：同时加载 7 个 ONNX，用键盘在它们之间切换。**

#### `POLICIES`：一张「菜单」

```python
POLICIES = {
    "walk":       "行走",
    "sitstand":   "坐 / 站",
    "roll":       "翻滚",
    "kickL":      "左脚踢",
    "kickR":      "右脚踢",
    "groundpick": "捡拾",
    "stand":      "起立恢复",
}
```

这就是一张菜单：每个「模式」对应一个 ONNX 文件。按哪个键，就切到哪个模式、用哪个神经网络。

#### `VELOCITY_MODES`：谁吃「油门」

走路时，方向键的意思是「往前 / 往后 / 转弯」；但翻滚、踢腿时，方向键没有意义。
所以代码标记了哪些模式吃「速度指令」：**只有 `walk` 吃**。其它模式喂给神经网络的指令是固定的：
「坐/站」看一个 0 还是 1 的标志，「捡拾」喂一个循环的相位，翻滚/踢腿/起立喂全零。

> 一句话：**走路是「油门」，其它动作是「一键触发」**。

#### 状态机：管理「一次性动作」的排队

有些动作不是「一直做」，而是「做一下、做完自动切回」——比如踢腿、翻滚、捡拾。这些靠一个**状态机**管理：
它记住「现在在干什么、干了多久、干完没有」。

几个关键机制，用大白话讲：

- **踢腿**：固定 0.5 秒，时间到自动切回走路。踢完还「锁」0.4 秒，防止走路策略立刻纠正踢腿姿势、把鸭子掀翻。
- **翻滚**：怎么知道翻过去了？看「投影重力」——身体倒过来时，重力的方向会反过来。先「倒过去」、再「立起来」、
  步数也够，才算翻完；翻完的鸭子原地站住，等其它鸭子一起切回。
- **坐 / 站切换**：不是硬切，而是「先站稳再坐下、先站起再走路」，中间有 0.8~2 秒的「软交接」，避免摔倒。
- **起立恢复**：翻滚结束还有鸭子没站直，就切到「起立恢复」策略慢慢站；超时还站不起来的，才单独复位。

> 别被「状态机」三个字吓到。它就是一个 `如果 现在在干什么 / 干了多久 / 干完没有 → 下一步怎么办` 的分支逻辑。

#### `KeyReader`：不用敲回车就能读键盘

普通输入要敲回车，程序才收得到。这里用了个小技巧，把终端调成「按一下立刻收到」的模式，
这样方向键一按，鸭子立刻有反应。方向键在底层是几个特殊字节，代码把它们翻译成「上 / 下 / 左 / 右」。

### 4.3 谁在真正「算物理」？—— `vendor/` 里的 mujoco-warp

神经网络只负责「想」，真正让鸭子能站稳、能走路、能摔倒的，是**物理引擎**。

这里的物理引擎叫 **mujoco-warp（MJWarp）**：一个跑在**显卡（GPU）**上的物理引擎，一次能同时算好多只鸭子
（这就是为什么能 `--num-envs 9` 九只一起跑）。

#### 它跟你的代码之间是什么关系？

你的代码只跟一个「抽象接口」打交道，不关心底层是 CPU 还是 GPU：

- `orca/physics.py`：定义「物理引擎应该能干什么」（一份接口/合同）。
- `_internal/runtime.py`：真正的实现（在 GPU 上干这些事）。

> 类比：你只跟「服务员」点菜，不关心后厨用煤气还是电磁炉。

#### 你只需要认识这几个「动词」

- `write_action(目标角度)`：告诉电机「转到哪」。
- `step(4)`：推物理 4 步。
- `forward()`：刷新传感器读数（读之前必须先刷一次，就像先拍照才能看照片）。
- `reset(...)`：把某只鸭子复位回标准站姿。
- `sensor("名字")`：按名字取某个传感器（比如陀螺仪）的读数。

### 4.4 画面从哪来？—— OrcaLab 批量渲染

物理在 GPU 上同时算 N 只鸭子，但 OrcaLab 的显示工具一次只认一个场景。怎么把 N 只鸭子一起画出来？

**办法：把 N 只鸭子「拼」进一个组合场景，再一起发给 OrcaLab。**

- 先把 N 只鸭子摆成一个网格（间距可调）；
- 每只鸭子的「姿势」从 GPU 里取出来，填进组合场景里对应的位置；
- 一次发送，屏幕上 N 只鸭子一起动。

> 核心思想一句话：**物理在 GPU 算，画面在 OrcaLab 画，两者只同步「姿势」这一件事**。

### 4.5 一张图总结

```
你的键盘
   │  (KeyReader 读出按键)
   ▼
模式 / 状态机  ──决定──▶  用哪个 ONNX 策略
   │
   ▼
读传感器(看) ─▶ 神经网络(想) ─▶ 写目标角(动) ─▶ mujoco-warp 推物理(GPU)
                                                   │
                                                   ▼
                                            OrcaLab 渲染(画)
```

### 4.6 想看代码时，按这个顺序读

1. 先读 `scripts/run_duck.py` 的 `main()`，知道「程序从哪开始」；
2. 再读 `DuckController.step()`，看懂那个循环；
3. 然后读 `scripts/run_duck_multi_policy.py` 的 `POLICIES` 和 `_update_oneshots`，看懂怎么切换；
4. 最后扫一眼 `vendor/.../runtime.py` 的 `write_action` / `step`，知道物理引擎长什么样。

其它都是细节，用到再查。

---


## 第五部分 · 动手跑起来

### 5.1 环境准备

```bash
cd /home/hpb/auto_research/microduck-orcalab

# 需要 Python ≥ 3.12；conda / venv 均可，这里以 conda 为例
conda create -n microduck python=3.12 -y && conda activate microduck

# 安装核心依赖（依赖清单在 pyproject.toml）
pip install -e .
```

`pyproject.toml` 声明的依赖（`pip install -e .` 自动装好）：

| 包 | 用途 |
|---|---|
| `torch` | 张量计算 |
| `mujoco` | 解析 MJCF、编译模型、CPU 元数据 |
| `mujoco-warp` / `warp-lang` | GPU 批量物理（核心） |
| `onnxruntime` | 加载运行 `.onnx` 策略 |
| `numpy` | 数组工具 |
| `orca-lab` | OrcaLab 可视化（gRPC 桥接） |

**资产下载**：登录资产库，搜索并订阅机器人模型资产 `robot_allcollisions_colour`。

**启动引擎**：命令行输入 `orcalab` 启动引擎，在 OrcaLab 中打开场景（点击左上角
**开启仿真 → 其他选项 → 手动启动**）。仅 headless（不加 `--orcalab`）无需这一步。

### 5.2 第一次运行（headless）

```bash
# headless：默认就是行走策略（policy/BEST_alpha_walking.onnx），无需 --onnx
python scripts/run_duck.py --steps 1000 --log-every 100
```

判断正常看两件事：`root_xyz` 的 **x 越来越大**（朝 +x 前进），**z 保持 ≈ 0.12**（躯干没趴下）。

### 5.3 参数表

| 参数 | 含义 | 默认 |
|---|---|---|
| `--steps` | 总控制步数（每步 0.02 s） | `2000` |
| `--log-every` | 每多少步打印状态 | `100` |
| `--num-envs` | 并行跑几只鸭子（GPU 批量） | `1` |
| `--device` | 推理 / 物理设备 | `cuda:0` |
| `--lin-vel-x / -y` | 指令前向/侧向速度（m/s） | `0.25 / 0` |
| `--ang-vel-z` | 指令转向角速度（rad/s） | `0` |
| `--onnx` | 策略文件 | `policy/BEST_alpha_walking.onnx` |
| `--realtime` | 按真实时间节流（否则跑得飞快） | 关 |

`--orcalab` 可视化相关参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `--orca-addr` | OrcaLab gRPC 服务地址 | `localhost:50051` |
| `--asset-path` | OrcaStudio 里鸭子的 prefab 路径（`--orcalab` 必填） | 空 |
| `--agent-prefix` | 代理名前缀 | `duck` |
| `--spacing` | 多环境网格间距（m） | `1.0` |
| `--render-fps` | 渲染帧率 | `30` |

### 5.4 多策略键盘控制

```bash
python scripts/run_duck_multi_policy.py
```

| 按键 | 动作 | 按键 | 动作 |
|---|---|---|---|
| `↑` / `w` | 前进 | `1` | 坐下 |
| `↓` / `s` | 后退 | `2` | 站起 |
| `←` / `a` | 左转 | `3` | 翻滚 |
| `→` / `d` | 右转 | `4` / `5` | 左脚踢 / 右脚踢 |
| `0` | 停止 | `6` | 捡拾 |
| `7` | 起立恢复 | `8` | 复位到站姿 |
| `q` / `Ctrl-C` | 退出 |  |  |

### 5.5 OrcaLab 可视化

```bash
# 直接跑仓库自带的启动脚本（已配好 --orcalab / --orca-addr / --asset-path / --render-fps / --num-envs）
bash scripts/run_duck.sh                # 单策略
bash scripts/run_duck_multi_policy.sh   # 多策略键盘控制

# 或手动指定：
python scripts/run_duck.py --orcalab \
    --orca-addr localhost:50051 \
    --asset-path <OrcaStudio 里鸭子的 prefab 路径> \
    --render-fps 30
```

`--asset-path` 必填；完整路径示例见 `scripts/run_duck.sh`。

### 5.6 常见问题

| 问题 | 解决 |
|---|---|
| 找不到 `mujoco_warp` / `warp` | 环境没建好/没激活，回 5.1 |
| 没 GPU | 加 `--device cpu`（慢但能跑） |
| `--orcalab` 卡住/一直 retry | 桥接服务没起，或 `--asset-path` 错 |
| 想走慢一点 | **调不了**（见下） |

**⚠️ 关于速度（重要）**：这套行走策略是「满速走 / 停」**两态开关，不是线性油门**。实测
`--lin-vel-x 0.25` → 约 0.11 m/s；`0.22` 及以下 → 基本原地站立。**没法靠调小 vx 让它走慢**，
真实步速固定约 0.11 m/s。若觉得「太快」，通常是**回放速率**问题（不加 `--realtime` 仿真跑得比实时
快几十倍），加 `--realtime` 回到真实时间。

---

## 附录 A · 术语速查表

| 术语 | 中文 | 一句话 | 项目里在哪 |
|---|---|---|---|
| DoF | 自由度 | 系统能独立变化的维数 | 鸭子 20 DoF |
| body / link | 连杆/刚体 | 不变形的部件 | `trunk_base`、`upper_leg_left`… |
| joint | 关节 | 连杆间的铰链 | 14 hinge + 1 free |
| hinge | 铰链关节 | 只能绕一根轴转 | 膝盖、髋、踝 |
| freejoint | 自由关节 | 空间任意运动 | `trunk_base_freejoint` |
| qpos / qvel | 广义坐标/速度 | 全身姿势/其变化率 | 21 / 20 维 |
| actuator | 执行器/电机 | 让关节动的部件 | `<actuator>` 14 个 |
| PD control | PD 控制 | 按误差+速度产生力 | `kp=0.55 kv=0` |
| kp / kv | 比例/微分增益 | 刚度/阻尼系数 | `chosen_actuator` |
| quaternion | 四元数 | 描述旋转的 4 个数 | `qpos[3:7]`（wxyz） |
| world/body frame | 世界/机体系 | 绝对 vs 随动 | `quat_apply_inverse` |
| projected gravity | 投影重力 | 重力在机体系的投影 | 观测 `[3:6]` |
| timestep / decimation | 时间步/抽取 | 物理步长 / 控制=几个物理步 | `0.005` / `4`→50 Hz |
| MuJoCo | 物理引擎 | 刚体接触仿真 | 整个物理层 |
| mujoco-warp / MJWarp | GPU 版 MuJoCo | GPU 批量物理 | `OrcaPhysicsRuntime` |
| observation / action | 观测/动作 | 策略输入/输出 | 61 / 14 维 |
| command | 指令 | 想让机器人怎么走 | 13 维 `[vx,vy,wz,0×10]` |
| reward / policy | 奖励/策略 | 好坏反馈 / 观测→动作函数 | 训练用 / `*.onnx` |
| inference / training | 推理/训练 | 用训好的策略跑 / 调权重 | 本项目 / `third_party/microduck_rl` |
| PPO | 近端策略优化 | 主流 RL 训练算法 | `third_party/microduck_rl` |
| ONNX | 模型交换格式 | 跨框架存网络 | `policy/*.onnx` |

## 附录 B · 文件职责清单

| 文件 | 职责 |
|---|---|
| `scripts/run_duck.py` | 单策略回放入口（`build_model` / `quat_apply_inverse` / `DuckController`） |
| `scripts/run_duck_multi_policy.py` | 多策略键盘控制（`KeyReader` + `MultiPolicyController`） |
| `scripts/run_duck.sh` | 单策略 OrcaLab 启动脚本（已配好可视化参数） |
| `scripts/run_duck_multi_policy.sh` | 多策略 OrcaLab 启动脚本 |
| `pyproject.toml` | 依赖声明（Python ≥ 3.12）+ ruff 配置 |
| `robot/robot_allcollisions.xml` | 鸭子 MJCF（关节/执行器/传感器/碰撞体） |
| `robot/assets/*.stl` | 43 个网格 |
| `policy/*.onnx` | 11 个策略（README「策略一览」登记 7 个常用） |
| `vendor/orcalab_rslrl/orca/physics.py` | 物理契约 `OrcaState` / `OrcaPhysics` |
| `vendor/orcalab_rslrl/_internal/runtime.py` | `OrcaPhysicsRuntime`（MJWarp 实现） |
| `vendor/orcalab_rslrl/orcalab_batch_render.py` | OrcaLab 批量渲染 |
| `third_party/microduck_rl` | 上游训练仓库（git 子模块，训练/任务/奖励代码都在这里） |
| `asset/` | README 用到的图片素材 |

## 附录 C · 关键数字速查

| 项目 | 值 |
|---|---|
| 关节数 / 自由度 | 14 铰链 + 1 自由根 = 20 DoF |
| qpos / qvel 维度 | 21 / 20 |
| 观测 / 动作 / 指令维度 | 61 / 14 / 13 |
| 神经网络结构 | `61 → 512 → 256 → 128 → 14`（ELU） |
| 参数量 | 197,774 |
| PD 增益 | `kp=0.55, kv=0`，力矩上限 ±0.96 N·m |
| 物理步长 / 控制频率 | 0.005 s（200 Hz）/ 50 Hz（decimation=4） |
| 默认站姿根位置 | `(0, 0, 0.12)` |
| 实际步速 | ~0.11 m/s（满速两态，非油门） |
| 直立 / 翻倒判据 | `gz < -0.85` / `gz > -0.3` |

---

*本文档基于对 `microduck-orcalab` 全部源码（`scripts/run_duck.py`、`scripts/run_duck_multi_policy.py`、
`robot/robot_allcollisions.xml`、`vendor/orcalab_rslrl/`）的逐行阅读，以及仓库内既有
《新手教程》《背景知识》《原理详解》《数学原理》四份文档的整理与合并，全部数字对照真实代码与
实测模型参数。*
