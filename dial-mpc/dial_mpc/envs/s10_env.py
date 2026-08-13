"""S10 轮足 dial-mpc 环境 v4：裸 mjx + 自定义 MjxLikeState（绕开 brax 兼容问题）。

关键修复历程：
- brax PipelineEnv 编译 OOM（31GB）→ 原生 mjx（~2GB）
- mjx 接触求解迭代不足 → PD 站姿发散 → iterations=30 + Newton
- S10 关节交织排列 → act2tau 用 LEG_IDX/WHEEL_IDX 映射
- wheel range 限位关节严重拖累滚动 → 移除
- brax pipeline.step 的 act 参数被 act2ctrl 误解 → 裸 mjx.step
- brax MjxState contact 结构不兼容 mjx 3.11 → 自定义 MjxLikeState
- 轮驱动：静摩擦锁定 ~1.6Nm，力矩 3Nm 最佳（wheel_tau_scale=3.0）
"""
from dataclasses import dataclass
import os
USE_ELEV = int(os.environ.get("S10_USE_ELEV", "1"))
LEAN_K = float(os.environ.get("S10_LEAN_K", "0.0"))   # v187: lean target scaled by vx*vyaw (0=legacy)
LEAN_LEG_W = float(os.environ.get("S10_LEAN_LEG_W", "0.0"))
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import mjx

from brax.base import Transform, Motion

# 感知-voxel 世界对齐高程瓦片查图（纯 jnp，固定形状，零 retrace）。
# dial_mpc 可独立运行（无 perception 包时降级为无地形代价）。
try:
    from perception.elevation_lookup import terrain_cost, sample_grid
except Exception:
    terrain_cost = None
    sample_grid = None

S10_MPC_XML = os.environ.get(
    "S10_MPC_XML",
    "/home/wfx/DR_competition/0810new/deeprobot_competition"
    "/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/s10_mpc.xml",
)

def _stand_joint_from_env():
    """竞速驾驶姿态（参考 go2-w 蹲伏驾驶）：base z≈0.204m。
    实测 z=0.238 高站姿 4m/s 前翻，z=0.20 稳定 4.04m/s。
    S10_STAND_HIPX 控制胯（hipx）外翻角：正值越大，四轮向外张开、
    轮距越宽 → 差速转向力臂与抗侧翻能力越强（竞速转向稳定性关键参数）。
    """
    hx = float(os.environ.get("S10_STAND_HIPX", "0.05"))
    return np.array([
        -hx, -1.16, 2.30, 0.0,
         hx, -1.16, 2.30, 0.0,
        -hx,  1.16, -2.30, 0.0,
         hx,  1.16, -2.30, 0.0,
    ], dtype=np.float32)


S10_STAND_JOINT = _stand_joint_from_env()

LEG_IDX = jnp.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
WHEEL_IDX = jnp.array([3, 7, 11, 15])
LEG_IDX_NP = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
WHEEL_IDX_NP = np.array([3, 7, 11, 15])
WHEEL_BODY_IDS = jnp.array([5, 9, 13, 17])   # s10_mpc.xml 轮体 id（fl/fr/hl/hr）


def quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotate(q, v):
    qv = jnp.concatenate([jnp.zeros(1), v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_inv_rotate(q, v):
    return quat_rotate(quat_conj(q), v)


def quat_to_euler_z(q):
    w, x, y, z = q
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@struct.dataclass
class MjxLikeState:
    """mjx Data 包装：提供 MBDPI 需要的 .q/.qd/.x/.xd 别名 + mjx 字段透传。"""
    data: mjx.Data
    x: Transform
    xd: Motion

    @property
    def q(self):
        return self.data.qpos

    @property
    def qd(self):
        return self.data.qvel

    def __getattr__(self, name):
        # 透传 mjx Data 字段（qpos/qvel/ctrl/xpos/xquat/xmat/cvel/ncon...）
        return getattr(self.data, name)


@struct.dataclass
class MpcState:
    pipeline_state: MjxLikeState
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


@dataclass
class S10WheeledEnvConfig:
    task_name: str = "default"
    dt: float = 0.02
    timestep: float = 0.02
    solver_iterations: int = 6      # 实测 it=6 是稳定下限（2/4 失稳）；越小越快
    solver_ls_iterations: int = 6
    leg_control: str = "torque"
    action_scale: float = 1.0
    kp: float = 80.0
    kd: float = 2.0
    leg_damping: float = 0.5
    wheel_damping: float = 0.05
    leg_action_scale: float = 0.25   # 腿动作范围（原值：模式 B 转向需腿压俯仰）
    wheel_control: str = "velocity"   # velocity | torque
    vel_scale: float = 50.0           # 轮速上限 = 50*0.081 ≈ 4.05 m/s（竞速目标）
    kd_wheel: float = 0.3             # >0.5 刹死轮子 -> 锁轮倒立摆失稳；0.3 稳定
    wheel_tau_scale: float = 3.0      # torque-mode wheel scaling (legacy)
    default_vx: float = 0.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ang_vel_weight: float = 10.0   # yaw 跟踪权重（go2-w 参考 tracking_ang_vel=2.0 相对 lin=4.0）
    vel_weight: float = 25.0       # 线速度跟踪权重（go2-w 参考 tracking_lin_vel=4.0，提速用）
    height_tar: float = 0.20
    base_z_init: float = 0.20
    height_weight: float = 0.1     # r_height 权重（地形跟随高度项；实验 3.0，见 0804 §3.6）
    height_lookahead: float = 0.35  # 地形跟随前瞻 (m)：目标高度 = 前方地形+站姿+抬升
    height_lift_cap: float = 0.15   # 地形跟随抬升上限 (m)（避免过早抬升失稳）
    terrain_w_slope: float = 8.0     # 地形代价权重（感知瓦片 gather）
    terrain_w_rough: float = 3.0
    terrain_w_step: float = 20.0
    terrain_w_ground: float = 120.0  # 轮-地形贴合权重（抬腿引导，m² 误差）
    terrain_w_overlift: float = 200.0  # 过抬惩罚：轮子高于目标+0.05m 重罚（防翘头）
    overlift_band: float = 0.05     # 过抬惩罚带（m）：高于目标+此带才罚
    w_pitch_rate_cap: float = 0.0   # 俯仰角速度上限软罚（0=关）
    pitch_rate_cap: float = 0.35    # 俯仰角速度上限阈值（rad/s）
    terrain_w_leg: float = 2.0       # 腿默认姿态正则（保持蹲姿）
    terrain_w_upright: float = 25.0  # 地形相对直立权重（允许顺坡倾斜，罚侧倾/翘头）
    terrain_w_attdamp: float = 0.8   # 俯仰/侧倾角速度阻尼（杀爬坡俯仰振荡）
    # ---- 前瞻抬轮 + 撞阶（MARG 参考，见 doc/0804.md §2.4/§3.2）----
    lift_lookahead: float = 0.4      # 前轮采样前方距离 (m)
    lift_max: float = 0.15           # 最大抬轮量 (m)（轮足机构稳健翻越上限）
    lift_clear_margin: float = 0.05  # 抬轮净空 margin (m)：抬轮目标在
                                       # 台阶面+轮半径之上再加此量，确保轮底
                                       # 越过 riser 棱边（实测 12mm 缺口即钩死）
    stair_sym_w: float = 0.0          # 爬梯左右轮高对称罚（v99）：罚前对/后对
                                       # 轮心高度差²，仅抬轮时激活。爬梯侧翻
                                       # 主因 = fl/fr 交替高差 0.15~0.25m。
    stair_air_w: float = 0.0          # 爬梯轮悬空时间罚（v128）：台阶区轮子
                                       # 悬空 >0.2s 重罚——前轮悬空钩 riser
                                       # 不落台是滑回根因；罚悬空逼轮子落台面
                                       # 获得支撑（后轮才能卸载跟抬）。
    stair_wheel_brake_w: float = 0.0  # 台阶区单向后滑制动（v112）：只罚轮子
                                       # 反向旋转（qd>0=向后打滑），正向滚动
                                       # 不受影响——棘轮式阻止"到顶后整机
                                       # 向后滑回"（v105/v111 z=0.96 滑回复现）。
    leg_hipy_scale: float = 1.0      # hipy 动作尺度倍率（v96 运动学修复）：
                                       # 前 hipy 需正摆 ~+0.8~1.5 才能抬前轮，
                                       # 但默认 -1.16 + las=1.0 最大只能到 -0.16，
                                       # 永远够不到；单独放大 hipy 让抬腿可达，
                                       # 膝保持原尺度避免 v95 的膝失控。
    lift_threshold: float = 0.05     # 前方高差超过该值才触发抬轮 (m)
    rear_follow_thresh: float = 0.10  # 后轮跟抬延迟阈值：前轮须比后轮高足
                                       # 一级台阶（v140：0.05 太早，前后轮
                                       # 同时抬→四轮失牵引→机身前栽翻车）
    lift_step_gate: float = 0.3      # 抬轮 step_flag 门控（2026-08-06 用户：
                                     # "门控越少越好"，0.5→0.3，采样空间更大）
    lift_steep_gate: float = 0.6     # 陡升梯度门控（链 44）：探针间局部上升
                                     # 梯度 > 该值即视为离散台阶（与 step_flag
                                     # 门控 OR 并联）。step_flag 依赖网格对齐，
                                     # 楼梯立面空洞被 min 填补后边界只在单个
                                     # 网格上，0.15~0.4m 探针常错过 → 抬轮不触发
                                     # （wp7 卡死复现）；梯度门对连续坡道天然
                                     # 免疫（20% 坡 ≈0.32 < 0.6，0.125m riser
                                     # ≈0.96 > 0.6）。
    contact_lift_ratio: float = 2.0  # 接触触发抬轮：轮体水平/法向力比阈值（CTBC reward 化）
    stumble_ratio: float = 4.0       # 撞阶判据：轮体水平/垂直接触力比（MARG）
    terrain_w_stumble: float = 0.5   # 撞阶惩罚权重（平滑斜坡，超出阈值部分）
    leg_relax_on_step: float = 0.2   # 台阶区 r_leg 权重缩放（放松蹲姿正则）
    lean_leg_w: float = 0.0            # 非对称压弯腿姿态权重（0=关，v185 实验）
    terrain_w_wheel_air: float = 0.0   # 牵引感知轮速惩罚权重（**默认关闭**）：
                                       # 实测 S10 行驶中轮子间歇离地（弹跳）导致
                                       # 误罚轮速 → riser 前侧翻（seg1/3/5 复现），
                                       # 需先解决弹跳（软接触/阻尼）再启用；
                                       # 实验开关：S10_TERRAIN_W_WHEEL_AIR
    wheel_ref_force: float = 20.0      # 单轮参考法向力 (N)：8kg/4≈20N，牵引系数归一化基准
    wheel_radius: float = 0.081
    # ---- 地形自适应姿态目标（E3，用户提议，默认关闭）----
    # 上坡：整体 pitch 仰头（目标 = 前方地形坡度）；下坡：低头；
    # 过弯：roll 向弯内倾斜（摩托车压弯，roll_tar = gain * vyaw * |vx|）。
    # 与 r_upright（地形法线直立）互补：upright 只罚"比地形更陡"，姿态目标
    # 主动把机身拉向预期坡度/压弯角，让 MPC 提前抬头/压弯而不是撞上才反应。
    pose_w_pitch: float = 0.0          # pitch 目标跟踪权重（0=关闭；实验 20~60）
    pose_w_roll: float = 0.0           # roll 目标跟踪权重（0=关闭；实验 10~30）
    pose_lookahead: float = 0.4        # 坡度采样前瞻 (±m，沿水平前向)
    pose_roll_gain: float = 0.06       # 压弯角 = gain * vyaw * |vx| (rad)
    pose_roll_max: float = 0.25        # 压弯角饱和 (rad)
    lift_rear: bool = False            # 后轮前瞻抬升总开关（链 7~12 实测后轮
                                       # 抬升方案均未超过前轮-only，默认关；
                                       # 用户"后腿爬楼梯"的诉求保留在
                                       # S10_LIFT_REAR=1 供后续改进机制）
    rear_lift_scale: float = 1.0       # 后轮跟抬幅度系数（0=关，1=全幅；
                                       # v10 实测全幅跟抬牺牲爬梯高度
                                       # 第6级→第3-4级，实验 0.3/0.6 折中）
    rear_lift_zgate: float = 0.0       # 后轮跟抬 body-z 门控（0=关闭；>0 时
                                       # 仅当机身高于该值才触发跟抬——"顶缘
                                       # 专项"：前段不跟抬保前轮冲高，顶缘
                                       # 后轮跟上防滑落）
    # ---- 参考路径跟踪（E4，用户提议，默认关闭）----
    # info["ref_path"] = (REF_N,2) 世界系路径点（固定形状，零 retrace）；
    # r_path 罚"距路径最近点距离"（横向偏离），r_path_head 罚"航向与路径
    # 切线不一致"——让 MPC 沿轨道规划，而不是只追瞬时速度（解决"歪歪扭扭"）。
    ref_n: int = 10                    # 参考路径点数量（固定）
    w_path: float = 0.0                # 路径横向偏离权重（0=关闭；实验 2~6）
    w_prog: float = 0.0                # MPCC 进度权重：沿路径切线推进速度
    w_path_head: float = 0.0           # 路径航向跟踪权重（0=关闭；实验 8~20）
    w_path_z: float = 0.0              # 路径 z（高程图轨迹）跟踪权重（0=关闭；
                                       # 爬坡/台阶时引导机身沿地形上下，实验 20~60）
    w_clear: float = 0.0               # 机身离地净高参考权重（0=关闭；链 43：
                                       # 目标 = 四轮下地形 max + 站姿高，随爬升
                                       # 逐级抬高机身，避免 k+2 前视一步拉到
                                       # 楼梯顶导致失稳；与 w_path_z 互补）
    leg_ext_w: float = 0.0             # 抬腿延伸引导权重（0=关闭；链 45）：
    stair_pitch_w: float = 0.0         # 爬梯仰头 reward（用户 2026-08-07）：
                                       # 目标 pitch（负=仰头），仅 STAIR 模式生效。
                                       # 仰头把重心后移、前轮卸载 → 前腿更易
                                       # 缩回抬轮（Chamorro/Ascento 爬梯姿态参考）
    stair_pitch_tar: float = -0.45     # 目标俯仰（rad，负=仰头 ~26°）
    sync_front_ext: float = 1.0        # 前轮抬升目标左右同步（链 55）：1=fl/fr
                                       # 取 max 防侧倾；0=左右独立（允许交替爬升，
                                       # 2026-08-07 实验开关）
                                       # lift_on 时把该腿 knee/hipy 引向
                                       # "轮心抬到 lift_need"所需伸展量
                                       # （0.175 m/rad 灵敏度，前后腿符号相反）。
                                       # 软性 shaping：不改动作空间/不加硬指令，
                                       # 只让采样器更快找到抬腿轨迹（实测
                                       # 0.45rad 动作尺度 + 独立采样难同时命中
                                       # hipy+knee 最大伸展 → 腿部锁死卡台阶）。
    lockpush_w: float = 0.0            # 锁轮推身软约束（0=关闭；用户 B 方案）：
                                       # 台阶区前轮"轮速≈0"= 挂台阶顶的好状态
                                       # （后轮蹬地滚动推、前轮顶住棱角不空转），
                                       # 罚前轮空转转速，MPC 自然发现
                                       # "前锁后蹬"上楼梯样本；后轮不罚（保持
                                       # 滚动推力）。平地/横脊不触发（靠动量滚过）。
    w_crouch: float = 0.0              # CRUISE 单向防趴低（0=关闭；用户方案 4）：
                                       # 机身相对脚下地形高度低于 nominal_z 时
                                       # 罚（relu 下界），高于不罚——防"姿态太低"
                                       # 但不强制高站姿（实测高站姿 4m/s 前翻）。
                                       # STAIR_SEQUENCE 模式 w_crouch=0 放开屈膝。
    w_push: float = 0.0                # 做功正奖（0=关闭；用户方案 2.4）：
                                       # 轮地法向力×前移速度 + 髋伸矩×前移速度
                                       # = "后腿蹬"直接激励（纯物理、无感知依赖）。
    w_swing_ok: float = 0.0            # 抬轮到位微奖（0=关闭）：lift_on 轮轮心
                                       # 接近目标（地形+R+lift）时微奖——打破
                                       # "不抬腿"局部极小（用户方案 swing 微奖）。
    w_z_smooth: float = 0.0            # 机身 z 加速度罚（0=关闭）：抑制爬梯
                                       # 阶间砸落/弹跳（用户方案连续阶平滑）。
    left_boost: float = 1.0            # v207 左侧抬轮补偿（默认 1.0）：M10 后轴
                                       # 左右差 9mm（hl +0.0549 vs hr -0.0459），
                                       # 实测左轮从不抬到位（HL 恒 <0.65 vs HR 0.78）；
                                       # >1 给左轮 ground/foothold 更高权重，
                                       # 软 cost 补偿几何劣势（非门控）。
    swing_thresh: float = 0.04         # v206 摆动相阈值（m）：轮心距 ref 高差
                                       # 超过该值判摆动相（要抬）；卡点左轮
                                       # 2.7-3.3cm 被 0.04 静音 → 可降 0.01。
    w_roll_level: float = 0.0          # v211 台阶区横向回平（0=关闭）：只罚 roll²
                                       # 不罚 pitch——卡点 body roll -0.33 使左轮
                                       # 低 0.13m（roll×轴距≈高差，同腿角实测），
                                       # 强制横向回平让左轮随身体升起。
    w_support: float = 0.0             # v214 支撑稳定性软罚（0=关闭；Takahashi
                                       # 2023 支撑多边形思想软版）：CoM 对接地轮
                                       # 支撑区越界惩罚，防单轮抬升侧翻。
    support_margin: float = 0.06       # 支撑区缩边安全余量 (m)
    support_fz_min: float = 20.0       # 判定接地的最小法向力 (N)
    support_exclude_lift: float = 0.0  # v215k 规划支撑：欠抬>0.05m 的轮不算支撑
    w_pitch_cap: float = 0.0           # v215 前高俯仰上限软罚（0=关闭）：防
                                       # 后仰翘头（pitch<−0.5rad 即罚）——旧配置
                                       # 爬升时 pitch 1s 内冲到 -1.20（69°）翻车。
    pitch_cap_rad: float = 0.50        # 俯仰上限阈值（rad，负=前高）
    lift_pose_fl_hipy: float = 1.00    # v215h 前轮抬升姿态 hipy（可调）
    lift_pose_fl_knee: float = 1.50    # v215h 前轮抬升姿态 knee（FK 0.90-0.92）
    lift_pose_hl_hipy: float = 1.80    # v215c HL 抬升姿态 hipy（可调）
    lift_pose_hl_knee: float = -1.40   # v215c HL 抬升姿态 knee
    lift_pose_hr_hipy: float = 1.50    # v215c HR 抬升姿态 hipy
    lift_pose_hr_knee: float = -1.80   # v215c HR 抬升姿态 knee
    ext_hl_boost: float = 1.0          # v215i HL 专属 r_ext 放大（左后不对称补偿）
    stair_wheel_lock_w: float = 0.0    # v216 轮锁（腿足狗爬梯，0=关）：STAIR
                                       # 区罚轮速²——轮子锁死，只能用腿抬放走楼梯
    swing_prox: float = 1e9            # v215d 摆动邻近门控（m）：轮距下一
                                       # riser 距离小于该值才进摆动相（豁免
                                       # 悬空罚）；远处悬空被 stance 压回
    w_lift_prog: float = 0.0           # v210 抬升进度正奖（0=关闭）：摆动相轮子
                                       # 每抬离地面 1cm 都给正奖（0→1 连续），
                                       # 制造"开始抬轮立即有回报"的梯度——解决
                                       # 精英 DEBUG 发现的 cost 表面无梯度。
    w_foothold: float = 0.0            # v206 落脚点前拉（0=关闭）：摆动相轮子
                                       # 向下一级踏面落脚点 y 前拉，激励 hipy
                                       # 前摆把轮放到下一级（卡点运动学根因）。
    w_wheel_ref: float = 0.0           # 轮速参考保持（0=关闭；用户"加零偏"）：
                                       # 罚轮速偏离"无滑移参考"(-vx/r)——MPPI
                                       # 覆盖 Y 前馈后轮速自由偏离（爬梯顶缘
                                       # 正反转 -55~+15 rad/s 实测），此项把
                                       # 轮速拉回无滑移前进值；接触顶死时参考
                                       # 降 0（锁轮防打滑），与 lockpush 互补。
    nominal_z: float = 0.205           # CRUISE 名义站姿高度（w_crouch 基准）
    # ---- 顶缘阶段（2026-08-07，"到顶停滞"诊断）----
    # body z 高于 top_z（前轮挂顶）时：
    # r_ext/r_clear/r_path_z 按比例软化（<1 放开伸腿与抬机身），
    # r_push 按比例强化（>1 强调前推）。连续乘数，无硬门控。
    top_z: float = 1.05
    top_ext_scale: float = 1.0
    top_clear_scale: float = 1.0
    top_pathz_scale: float = 1.0
    top_push_scale: float = 1.0
    top_upright_scale: float = 1.0   # 顶缘时 terrain_w_upright 乘数（>1 防侧倾）
    top_attdamp_scale: float = 1.0   # 顶缘时 terrain_w_attdamp 乘数（>1 抑振荡）
    top_lockpush_scale: float = 1.0  # 顶缘时 lockpush_w 乘数（>1：前轮挂顶锁轮
                                       # 防打滑空转，后轮蹬推——"前挂后蹬"）




"""s10_env 参数化纯函数（预热缓存重构，2026-08-06）：
把 rollout 热路径从"jit 闭包捕获 env"改为"ctx 作为 pytree 参数"，
使 JAX persistent cache 真正生效（30s 冷编译 -> ~7s 热启动）。
"""
from types import SimpleNamespace as _NS


def _act2tau_pure(ctx, act, d):
    cfg = ctx["cfg"]
    per_j = jnp.asarray(
        [1.0, cfg["leg_hipy_scale"], 1.0] * 4, dtype=jnp.float32)
    leg_target = ctx["default_leg"] + act[:12] * per_j * cfg["leg_action_scale"]
    q_leg = d.qpos[7:][LEG_IDX]
    qd_leg = d.qvel[6:][LEG_IDX]
    tau_leg = cfg["kp"] * (leg_target - q_leg) - cfg["kd"] * qd_leg
    qd_wheel = d.qvel[6:][WHEEL_IDX]
    vel_ref = -act[12:] * cfg["vel_scale"]
    tau_wheel_vel = cfg["kd_wheel"] * (vel_ref - qd_wheel)
    tau_wheel_torque = -act[12:] * cfg["wheel_tau_scale"]
    tau_wheel = jnp.where(cfg["wheel_mode"], tau_wheel_torque, tau_wheel_vel)
    tau = jnp.zeros(16).at[LEG_IDX].set(tau_leg).at[WHEEL_IDX].set(tau_wheel)
    return jnp.clip(tau, ctx["torque_range"][:, 0], ctx["torque_range"][:, 1])


def _make_state_pure(ctx, d):
    x = Transform(pos=d.xpos[1:], rot=d.xquat[1:])
    cvel = Motion(vel=d.cvel[1:, 3:], ang=d.cvel[1:, :3])
    offset = d.xpos[1:, :] - d.subtree_com[ctx["body_rootid"][1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    return MjxLikeState(data=d, x=x, xd=xd)


def s10_step_rollout_pure(ctx, state, action):
    """MBDPI 采样 rollout 专用 step（纯函数版，ctx 参数化）。"""
    d = state.pipeline_state
    ctrl = _act2tau_pure(ctx, action, d)
    dx = mjx.step(ctx["mx"], d.data.replace(ctrl=ctrl))
    # 关节角 clip（2026-08-07 修复）：σ 大时采样极端动作 → 关节转飞
    # （wp7 台阶区 hl 后轮 knee 实测 -3836 rad）→ 数值爆炸卡死/侧翻。
    # 把 qpos[7:]（16 关节）clip 到 jnt_range，防 rollout 内关节漂移。
    _q_clip = jnp.clip(
        dx.qpos[7:],
        ctx["joint_range"][:, 0], ctx["joint_range"][:, 1])
    dx = dx.replace(qpos=jnp.concatenate([dx.qpos[:7], _q_clip]))
    ps = _make_state_pure(ctx, dx)
    info = dict(state.info)
    info["step"] = info["step"] + 1
    reward = _reward_pure(ctx["cfg"], ctx, ps, info, ctrl)
    return MpcState(ps, state.obs, reward, jnp.zeros(()), info)


def s10_rollout_us(ctx, state, us):
    def _step(state, u):
        state = s10_step_rollout_pure(ctx, state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(_step, state, us)
    return rews, pipline_states


def _reward_pure(cfg, ctx, d, info, ctrl):

    xquat = d.xquat[ctx["torso"]]
    cvel = d.cvel[ctx["torso"]]
    vb = quat_inv_rotate(xquat, cvel[3:])
    ab = quat_inv_rotate(xquat, cvel[:3])
    r_vel = (-jnp.sum((vb[:2] - info["vel_tar"][:2]) ** 2)
             * cfg["vel_weight"])
    r_ang = (-jnp.square(ab[2] - info["ang_vel_tar"][2])
             * cfg["ang_vel_weight"])
    vec = quat_rotate(xquat, jnp.array([0.0, 0.0, 1.0]))
    # 地形相对直立：用高程图高度场梯度求地形法线，允许机身顺坡倾斜，
    # 惩罚"翘头"（姿态比地形更陡）——爬坡稳定的关键（世界垂直惩罚会
    # 逼 MPC 保持水平 → 前轮离地 → 打滑卡死）。
    n_ref = jnp.array([0.0, 0.0, 1.0])
    elev_up = info.get("elevation_map")
    if elev_up is not None and sample_grid is not None:
        base_xy = d.xpos[ctx["torso"]][:2]
        fwd = quat_rotate(xquat, jnp.array([1.0, 0.0, 0.0]))
        lat = quat_rotate(xquat, jnp.array([0.0, 1.0, 0.0]))
        # 水平归一化：机体俯仰/侧倾时 3D 轴向的水平投影会缩短，
        # 直接乘 0.3 会让探测点比预期近（0804 §3.6 姿态敏感点 2）。
        fwd2 = fwd[:2] / (jnp.linalg.norm(fwd[:2]) + 1e-6)
        lat2 = lat[:2] / (jnp.linalg.norm(lat[:2]) + 1e-6)
        probe = jnp.stack([
            base_xy + fwd2 * 0.3, base_xy - fwd2 * 0.3,
            base_xy + lat2 * 0.3, base_xy - lat2 * 0.3])
        hp, okp = sample_grid(
            elev_up["heightmap"], elev_up["features"]["valid"],
            elev_up["origin"], elev_up["resolution"], probe, fill=0.0)
        dhf = (hp[0] - hp[1]) / 0.6
        dhl = (hp[2] - hp[3]) / 0.6
        n_terr = jnp.array([-dhf, -dhl, 1.0])
        n_terr = n_terr / (jnp.linalg.norm(n_terr) + 1e-6)
        n_ref = jnp.where(okp.all(), n_terr, n_ref)
    cos_a = jnp.clip(jnp.dot(vec, n_ref), -1.0, 1.0)
    r_upright = -cfg["terrain_w_upright"] * jnp.square(1.0 - cos_a)
    # 姿态角速度阻尼：抑制爬坡时的俯仰/侧倾振荡（轮矩反作用激起）
    r_attdamp = -cfg["terrain_w_attdamp"] * jnp.sum(
        jnp.square(ab[:2]))
    yaw = quat_to_euler_z(xquat)
    yaw_tar = info["yaw_tar"] + info["ang_vel_tar"][2] * cfg["dt"] * info["step"]
    d_yaw = yaw - yaw_tar
    r_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
    r_height = -jnp.square(d.xpos[ctx["torso"], 2] - info["pos_tar"][2])
    r_energy = -jnp.sum(jnp.square(ctrl))
    # 机身 z 加速度罚（用户方案 2.4"连续阶平滑"）：**只罚向下砸落**
    # （relu(-ab[2])，body 竖直向下加速）——向上爬升是爬梯必要动作，
    # 双向罚会压制机身起伏 → 卡台阶底部（chain 74 复现，同 attdamp=2.0
    # 教训）。ab 是 body 局部系加速度，ab[2] 竖直分量。
    r_z_smooth = -cfg["w_z_smooth"] * jnp.square(
        jnp.clip(-ab[2], 0.0, 10.0))
    # 地形代价（感知-voxel 世界对齐瓦片）：按预测轮落点 gather。
    # 坡度/粗糙度/台阶特征在感知侧预计算，越界/空洞按"未知不惩罚"。
    # 数据来自 info["elevation_map"]（仿真节点 get_local_map() 经
    # MPCController.set_elevation_map 注入，固定 (60,60) 形状）。
    r_terrain = 0.0
    r_ground = 0.0
    r_overlift = 0.0
    r_stumble = 0.0
    r_ext = 0.0
    r_lockpush = 0.0
    r_wheel_ref = 0.0
    r_crouch = 0.0
    r_pitch_cap = 0.0
    r_push = 0.0
    r_swing_ok = 0.0
    r_pitch = 0.0
    r_sym = 0.0
    r_airpen = 0.0
    leg_scale = 1.0
    elev = info.get("elevation_map")
    if (elev is not None and terrain_cost is not None
            and USE_ELEV > 0):
        wheel_xy = d.xpos[WHEEL_BODY_IDS][:, :2]
        wheel_z = d.xpos[WHEEL_BODY_IDS][:, 2]
        fwd2 = quat_rotate(xquat, jnp.array([1.0, 0.0, 0.0]))[:2]
        fwd2 = fwd2 / (jnp.linalg.norm(fwd2) + 1e-6)  # 水平归一化（抗俯仰缩短）
        cost, _ok = terrain_cost(
            elev["features"], elev["origin"], elev["resolution"],
            wheel_xy,
            w_slope=cfg["terrain_w_slope"],
            w_rough=cfg["terrain_w_rough"],
            w_step=cfg["terrain_w_step"])
        r_terrain = -jnp.sum(cost)
        # 轮-地形贴合（感知高程图引导抬腿）：预测轮心应落在
        # 地形高度 + 轮半径处；前方地形升高 → 该误差增大 →
        # MPC 被激励伸腿让轮子骑上台阶（而不是顶死打滑）。
        h_terrain, ok_h = sample_grid(
            elev["heightmap"], elev["features"]["valid"],
            elev["origin"], elev["resolution"], wheel_xy,
            # 轮下地图格无效（riser 立面遮挡）时用"轮心真实高度 − 轮半径"
            # 作为地形参考（= 轮子实际压着的地面），而不是轮心高度本身：
            # fill=wheel_z 会让 target_z=wheel_z+r 把轮子往下压 0.081m，
            # 且 lift_need 用轮心做基准会误算台阶净高（wp7 卡点实测）。
            fill=wheel_z - cfg["wheel_radius"])
        # 2026-08-05 曾加"轮周 ±0.2m min-窗口"回退（修爬升中高差趋 0），
        # 实测造成左右轮不对称 → riser 前侧翻（链 20 3/3 复现），已还原
        # 链 5 原样 fill=wheel_z−r（对称、可靠）。
        # 前瞻抬轮（MARG feet-air-time 的轮式化，见 0804 §2.4）：
        # 四轮（含后轮！2026-08-05 用户指出：前轮上台阶后，后轮也要
        # 爬同一级 riser，wp7 卡点即"前轮上、后轮卡"）在 0.15~0.4m
        # 三点窗口采样高程与台阶标志；窗口内任一处出现离散台阶
        # （step_flag）且高度明显高于当前轮下地形 → 抬高该轮目标高度。
        # 3 点窗口为实测最优（climb3：wp0→wp7 34s）；密集带会造成
        # 过早抬轮（坡前翻）、过近/过高阈值会漏检（卡台阶）。
        offs = jnp.array(
            [cfg["lift_lookahead"] * 0.375,
             cfg["lift_lookahead"] * 0.7,
             cfg["lift_lookahead"]])
        probe = (wheel_xy[:, None, :]
                 + fwd2[None, None, :] * offs[None, :, None])
        h_win, ok_w = sample_grid(
            elev["heightmap"], elev["features"]["valid"],
            elev["origin"], elev["resolution"], probe,
            fill=jnp.broadcast_to(h_terrain[:, None], probe.shape[:2]))
        # 台阶标志（离散 riser）门控：连续坡道上 h_ahead-h_now 也 > 阈值
        # （坡度×前瞻），若只看高差会把抬轮误触发在坡道上（实测坡顶侧翻）。
        # step_flag 来自感知侧（local_map 默认 >0.08m 的相邻高差）。
        step_win, ok_s = sample_grid(
            elev["features"]["step_flag"], elev["features"]["valid"],
            elev["origin"], elev["resolution"], probe,
            fill=jnp.zeros(probe.shape[:2]))
        # lift_need = 台阶净高（轮半径在目标式中抵消）：
        #   目标 = h_now + r + lift_need = h_ahead + r（轮心骑上台阶顶）
        h_ahead = jnp.max(jnp.where(ok_w, h_win, -1e6), axis=1)
        step_ahead = jnp.max(step_win, axis=1)
        # 陡升梯度门控（链 44）：探针间局部上升梯度 = 台阶特征，对网格
        # 对齐不敏感（step_flag 只在 riser 边界单格上，0.15~0.4m 探针
        # 常整段落在台面上 → 永远触发不了）。梯度 = 相邻探针高差/间距：
        #   0.125m riser ≈ 0.96；20% 坡 ≈ 0.32 → 0.6 阈值天然区分。
        rise12 = jnp.where(
            ok_w[:, 0] & ok_w[:, 1],
            (h_win[:, 1] - h_win[:, 0]) / 0.13, 0.0)
        rise23 = jnp.where(
            ok_w[:, 1] & ok_w[:, 2],
            (h_win[:, 2] - h_win[:, 1]) / 0.12, 0.0)
        steep = jnp.maximum(rise12, rise23) > cfg["lift_steep_gate"]
        lift_on = ((h_ahead - h_terrain) > cfg["lift_threshold"]) \
            & ((step_ahead > cfg["lift_step_gate"]) | steep)
        # 多级台阶区检测（stair_ahead，链 46）：单级 0.13m 台阶靠动量可
        # 滚过（wp5→6 横脊实测），连续台阶才需要抬腿序列。判据 = 机身
        # 前方 0.2m 与 1.0m 处地形仍持续上升（>0.15m）——楼梯在 1m 前瞻
        # 内叠 2+ 级，横脊/单台阶 1m 后已平。带 stair_t 记忆（衰减）覆盖
        # 最后一级：前轮上顶后 ahead 变平，后轮仍需跟抬。
        base_xy = d.xpos[ctx["torso"]][:2]
        probe_s = base_xy[None, :] + fwd2[None, :] * jnp.array(
            [0.2, 0.6, 1.0])[:, None]
        h_s, ok_s2 = sample_grid(
            elev["heightmap"], elev["features"]["valid"],
            elev["origin"], elev["resolution"], probe_s,
            fill=-1e6)
        h_near = jnp.where(ok_s2[0], h_s[0], -1e6)
        h_far = jnp.where(ok_s2[2], h_s[2], -1e6)
        stair_ahead = (h_far - h_near > 0.15) \
            & (h_near > -1e5) & (h_far > -1e5)
        stair_t = info.get("stair_t")
        if stair_t is None:
            stair_t = jnp.zeros(())
        stair_t = jnp.where(
            stair_ahead, jnp.minimum(stair_t + cfg["dt"], 3.0),
            jnp.maximum(stair_t - 2.0 * cfg["dt"], 0.0))
        info["stair_t"] = stair_t
        in_stairs = stair_ahead | (stair_t > 0.5)
        # 后轮抬升总开关（S10_LIFT_REAR，默认 1）：链 8 实测后轮抬升使
        # 第一/二级 riser 成功率下降（r2/r3 翻、r1 卡第二级），对比
        # 前轮-only（链 5：2/2 双 riser 通过）。默认开，实验可关。
        # 链 53：S10_LIFT_REAR 只控制"跟抬"（前轮上顶后后轮抬到前轮高度），
        # **前视抬轮（steep 门控）始终保留**——单级横脊也需要后轮适度抬
        # （0.12m）才能过（chain 52 复现：后轮完全不抬 → 顶死卡横脊）。
        # 用户 A"后腿蹬"= 后轮适度抬 + 滚动推（不锁定、不跟抬到前轮高度）。
        # 后轮抬升（2026-08-05，用户指出"后腿也要爬楼梯"）：
        # 探测证实后轮 0.15~0.3m 前视窗口位于 riser 阴影（LiDAR 被立面
        # 遮挡，地图恒空洞），"后轮前视采样"永远触发不了。改为"跟抬"：
        # 同侧前轮已上台阶顶（前轮下地形 h_terrain[0/1] 高于后轮下地形
        # h_terrain[2/3]，且前轮正在抬）→ 后轮目标直接抬到前轮高度。
        # 前轮下地形用地图真值（在台阶顶上有效），后轮下地形用轮心−轮
        # 半径回退（不受空洞影响）——两者差 = 后轮需爬的台阶净高。
        # v139b：后轮跟抬量改用物理轮高差（前轮 z - 后轮 z）——高程图
        # 在 riser 阴影处常空洞，h_terrain 前后差不可靠；轮心高是本体感知、
        # 恒有效（v135-v137 后轮不跟抬/过迟的根因之一）。
        rear_need = jnp.clip(
            wheel_z[:2] - wheel_z[2:], 0.0,
            cfg["lift_max"])
        # 前轮后方 0.15m 的 step_flag：前轮刚爬过的 riser 边界（在轮后），
        # 用于区分"离散台阶"与"连续坡道"（坡道上前/后轮下地形差也会
        # >0.05，不能触发跟抬，否则长坡上后轮乱抬失去抓地）。
        probe_behind = wheel_xy[:2] - fwd2[None, :] * 0.15
        step_behind, _okb = sample_grid(
            elev["features"]["step_flag"], elev["features"]["valid"],
            elev["origin"], elev["resolution"], probe_behind,
            fill=0.0)
        # 链 46：后轮跟抬增加 stair 门控——单级横脊上后轮抬腿会失去
        # 抓地卡死（链 45 sigma 2.0 复现）；只有多级台阶区（含最后一级
        # 的记忆窗口）才允许后轮跟抬。链 47 修正：**不能覆盖前视抬轮**
        # （steep 门控，横脊处也触发）——横脊需要"适度抬腿"（0.12m，
        # r_ground 目标抬高，r_ext 关闭防过度伸展）；跟抬只在楼梯区叠加。
        _ms = info.get("mode_stair", jnp.array(0.0, dtype=jnp.float32))
        # v135：楼梯区（全局已知 mode_stair）可绕过感知 step_behind 门控
        # ——riser 立面在 LiDAR 阴影下 step_flag 常缺失，后轮永远不跟抬
        # （v134 r2 前轮上顶、后轮卡第二级根因）。坡道由 in_stairs 排除。
        rear_follow = (rear_need > cfg["rear_follow_thresh"]) \
            & ((step_behind > cfg["lift_step_gate"]) | (_ms > 0.5)) \
            & in_stairs \
            & (cfg["lift_rear"] > 0.0) \
            & (d.xpos[ctx["torso"], 2] > cfg["rear_lift_zgate"])
        lift_on = lift_on.at[2:].set(
            lift_on[2:] | rear_follow)
        lift_need = jnp.where(
            lift_on,
            jnp.clip(h_ahead - h_terrain, 0.0, cfg["lift_max"]),
            0.0)
        lift_need = lift_need.at[2:].set(jnp.minimum(
            jnp.where(lift_on[2:],
                      rear_need * cfg["rear_lift_scale"], 0.0),
            cfg["lift_max"]))
        # v150：lift_need 左右同步（fl/fr、hl/hr 各取 max）——与 class 版
        # 链 64 一致，rollout 此前漏掉：左右轮抬升目标不一致（地图格对齐/
        # 接触时机差）→ MPC 优化出的动作左右不对称 → 车身侧倾翻车
        # （v144-v149 第一级 riser 侧翻主因，实测右 0.88/左 0.55）。
        lift_need = lift_need.at[0:2].set(jnp.max(lift_need[:2]))
        lift_need = lift_need.at[2:4].set(jnp.max(lift_need[2:]))
        # v160：STAIR 模式确定性每轮 z 参考（Perceptive Stair Climbing 式）——
        # 前轮用**无条件前瞻**（前方 0.375~1.0×lift_lookahead 窗口最高地形，
        # 无 step_flag/steep 门控，riser 阴影不失效），后轮按前轮抬升量的
        # 0.5 倍跟抬（部分抬保持抓地，避免四轮齐抬死锁）。左右几何对称。
        # 坡道（CRUISE）保持旧门控；楼梯（mode_stair=1）用确定性公式。
        _lf_det = jnp.clip(h_ahead[:2] - h_terrain[:2], 0.0, cfg["lift_max"])
        _lf_det = jnp.max(_lf_det)          # 左右对称（fl/fr 同目标）
        _lr_det = jnp.minimum(0.5 * _lf_det, cfg["lift_max"])
        _lift_det = jnp.concatenate([
            jnp.array([_lf_det, _lf_det], dtype=jnp.float32),
            jnp.array([_lr_det, _lr_det], dtype=jnp.float32)])
        lift_need = jnp.where(_ms > 0.5, _lift_det, lift_need)
        # 接触触发抬轮（CTBC reward 化，0804 §3.7）：轮子顶在台阶面时
        # 水平接触力远大于法向（力比爆表）——物理可靠、不受感知遮挡/
        # 格对齐影响，且与弹跳可区分（弹跳水平力也小，比值不高）。
        # 触发时该轮目标高度抬升 → MPC 自己抬轮离开立面 → 法向力恢复 →
        # 目标回落 → 轮子落到台阶顶。与感知前瞻抬轮互补（感知提前、接触兜底）。
        f = d.cfrc_ext[WHEEL_BODY_IDS][:, :3]
        f_xy = jnp.sqrt(jnp.sum(jnp.square(f[:, :2]), axis=1) + 1e-6)
        f_z = jnp.abs(f[:, 2]) + 1e-3
        ratio = f_xy / f_z
        # 持续门控：弹跳是单帧尖峰（实测平地力比>2 占 5~10%），顶死是持续
        # 数百 ms 的信号；累积 0.1s 才触发，避免弹跳误抬（0804 §3.7）。
        cl_cond = (ratio > cfg["contact_lift_ratio"]) \
            & (f_z < cfg["wheel_ref_force"] * 1.5) \
            & (f_xy > 10.0)
        cl_t = info.get("contact_lift_t")
        if cl_t is None:
            cl_t = jnp.zeros(4)
        cl_t = jnp.where(cl_cond, cl_t + cfg["dt"],
                         jnp.clip(cl_t - 4.0 * cfg["dt"], 0.0, 0.5))
        info["contact_lift_t"] = cl_t
        contact_lift = cl_t > 0.10
        target_z = h_terrain + cfg["wheel_radius"]
        # v135：感知前瞻抬轮与接触触发抬轮取 max 不叠加（v134 r1 前轮
        # 抬到 1.13m：lift_need 0.25 + contact 0.2 + margin 0.05 叠到 0.5m，
        # 机身 pitch -2.0 前翻）。互补机制：感知先抬、接触兜底。
        target_z = h_terrain + cfg["wheel_radius"]
        # v162b：已知地图轮心 z 参考场（STAIR，世界系）优先——riser 前提前
        # 抬轮目标（含净空 margin），前后轮按各自 y 采样（相位差=轴距天然
        # 表达）。场有效时**取代**感知/接触 lift（防双重计数过抬 1.0m，v162
        # r1 翻车根因）；区外/无效回退感知-接触机制。
        _wr_ok = jnp.zeros(4, dtype=jnp.bool_)
        _wr_feat = elev["features"].get("wheel_ref")
        if _wr_feat is not None:
            _wr, _wr_ok = sample_grid(
                _wr_feat, elev["features"]["wheel_ref_valid"],
                elev["origin"], elev["resolution"], wheel_xy,
                fill=target_z)
            target_z = jnp.where((_ms > 0.5) & _wr_ok, _wr, target_z)
        lift_total = jnp.maximum(
            lift_need, jnp.where(contact_lift, cfg["lift_max"] * 0.8, 0.0))
        lift_total = jnp.where((_ms > 0.5) & _wr_ok, 0.0, lift_total)
        target_z = target_z + lift_total
        # 2026-08-07 v94：棱边净空 margin——lift_on 轮目标再抬
        # lift_clear_margin，让轮底越过 riser 棱边（钩死根因）。
        target_z = target_z + jnp.where(
            lift_on, cfg["lift_clear_margin"], 0.0)
        dev = wheel_z - target_z
        # v169：分相 ground（ground_phase=1，推荐）——摆动相（场要抬的轮）
        # 只罚"没抬到位"（relu(target-z)），支撑相只罚"悬空"（relu(z-target)），
        # 天然表达"该抬的抬、该落地的落地"，避免全局对称/单向的 bang-bang。
        # v164 单向（ground_oneway=1）与旧对称（0）保留作对照。
        # v206: 摆动相阈值参数化（S10_SWING_THRESH，默认 0.04）。卡点实测
        # 左轮距 ref 仅 2.7~3.3cm（<0.04）→ 被误判为支撑相，抬轮信号静音；
        # 降到 0.01 让"临界欠抬"的左轮进入摆动相（配合 foothold 前拉）。
        _sw_th = cfg.get("swing_thresh", 0.04)
        # v213: 顺序步态调度摆动标志（info 注入，0=关/纯 lift-need 启发式）。
        # 调度器只决定"哪条腿现在摆动"（软 cost 相位），MPC 采样执行。
        _gsw = info.get("gait_swing")
        # v215d: 摆动邻近门控（轮距下一 riser 距离 < S10_SWING_PROX 才豁免
        # 悬空罚）——前轮远处悬空（无牵引）应被 stance 压回地面滚动。
        _prox = info.get(
            "stair_prox", jnp.full(4, 1e9, dtype=jnp.float32))
        # v215e: prox 门控只作用于前轮——后轮按跟抬机制提前抬（距 riser
        # 0.375m 就需抬，若也被 prox 锁 stance 则死锁（前轮卡面、后轮禁抬）。
        _prox_ok = (_prox < cfg.get("swing_prox", 1e9)) | (
            jnp.arange(4) >= 2)
        if cfg.get("ground_phase", 0.0) > 0.0:
            # v214: gait_swing 可作连续 swing 权重（0~1，utility 选腿）；
            # 摆动相误差 × 权重 + 支撑相误差 × (1-权重)——软相位无门控。
            if _gsw is not None:
                # v215l: utility 摆动权重与 prox 门控 AND——前轮远处悬停仍被
                # stance 压回，后轮（prox 豁免）保持完整摆动窗口。
                _sww = _gsw * _prox_ok.astype(jnp.float32)
            else:
                _sww = ((_ms > 0.5) & _wr_ok & _prox_ok & (
                    _wr - (h_terrain + cfg["wheel_radius"]) > _sw_th)
                ).astype(jnp.float32)
            # v215g: stance 相悬空罚用"地面接触高度"（h_terrain+R）而非抬升
            # 目标——前轮悬停在 0.84（低于抬升目标 0.87）曾逃过惩罚；正确
            # 物理：stance 轮应贴地，swing 轮应到抬升目标。
            _stance_tar = h_terrain + cfg["wheel_radius"]
            _ph_err = (
                _sww * jnp.square(jnp.clip(target_z - wheel_z, 0.0, 1.0))
                + (1.0 - _sww) * jnp.square(
                    jnp.clip(wheel_z - _stance_tar, 0.0, 1.0)))
            _lw = cfg.get("left_boost", 1.0)
            _w4 = jnp.array([_lw, 1.0, _lw, 1.0], dtype=jnp.float32)
            r_ground = -cfg["terrain_w_ground"] * jnp.sum(
                jnp.where(ok_h, _ph_err * _w4, 0.0))
        elif cfg.get("ground_oneway", 0.0) > 0.0:
            r_ground = -cfg["terrain_w_ground"] * jnp.sum(
                jnp.where(ok_h,
                          jnp.square(jnp.clip(target_z - wheel_z, 0.0, 1.0)),
                          0.0))
        else:
            r_ground = -cfg["terrain_w_ground"] * jnp.sum(
                jnp.where(ok_h, jnp.square(dev), 0.0))
        # 抬轮到位微奖（用户方案 swing 微奖）：lift_on 轮轮心接近目标时
        # 微奖（dev≈0），打破"不抬腿"局部极小——MPC 探索抬腿后得到正反馈。
        r_swing_ok = cfg["w_swing_ok"] * jnp.sum(
            jnp.where(lift_on, 1.0 - jnp.minimum(jnp.abs(dev) / 0.1, 1.0), 0.0))
        # v210: 抬升进度正奖——摆动相轮子轮心高于"地面+半径"的部分按比例给奖
        #（clip 0~0.15m → 0~1），任何抬离地面都立即有回报（连续梯度），
        # 而非只在到位时给奖。仅楼梯带+摆动相（与 ground_phase 同判据）。
        _wlp = cfg.get("w_lift_prog", 0.0)
        # 注意：_wlp 是 tracer，不能用 Python if；权重 0 时该项自动为 0。
        if _gsw is not None:
            _sw3 = _gsw
        else:
            _sw3 = ((_ms > 0.5) & _wr_ok & _prox_ok & (
                _wr - (h_terrain + cfg["wheel_radius"]) > _sw_th)
            ).astype(jnp.float32)
        _prog = jnp.clip(
            (wheel_z - (h_terrain + cfg["wheel_radius"])) / 0.15,
            0.0, 1.0)
        r_lift_prog = _wlp * jnp.sum(_sw3 * _prog)
        # 做功正奖（用户方案 2.4"后腿蹬"）：轮地法向力×前移速度 +
        # 髋伸矩×前移速度。纯物理、无感知依赖；前移速度 vb[0]（body 局部
        # x），法向力 f_z（轮），髋伸矩 = ctrl 腿力矩的 hipy 分量。
        v_fwd = jnp.clip(vb[0], 0.0, 5.0)
        hipy_tau = ctrl[LEG_IDX][1::3]
        r_push = cfg["w_push"] * (
            jnp.sum(f_z) * v_fwd + jnp.sum(hipy_tau) * v_fwd)
        # 过抬惩罚（0804 §3.7 诊断）：抬轮目标 0.71m 时轮子冲到 1.0m →
        # 机身后仰翘头翻车；对"高于目标+0.05m"的部分重罚，让 MPC 停在
        # 目标附近而不是过冲。
        overlift = jnp.clip(wheel_z - (target_z + 0.05), 0.0, 1.0)
        r_overlift = -cfg["terrain_w_overlift"] * jnp.sum(
            jnp.square(overlift))
        # 撞阶惩罚（MARG feet-stumble：水平接触力 > 4× 垂直 = 顶死台阶面）：
        # 平滑斜坡惩罚（超阈值部分），直接对抗"轮矩反作用掀翻车身"。
        r_stumble = -cfg["terrain_w_stumble"] * jnp.sum(
            jnp.clip(ratio - cfg["stumble_ratio"], 0.0, 3.0))
        # 台阶区放松蹲姿正则：感知前瞻或接触触发抬轮时 leg_scale 缩小，
        # 避免 r_leg 与抬腿打架（0804 §3.2 缺口 2）。
        # v167：场激活（wheel_ref 高于地形+轮半径）也放宽蹲姿正则
        _field_lift = (_ms > 0.5) & _wr_ok & (
            _wr - (h_terrain + cfg["wheel_radius"]) > 0.04)
        leg_scale = jnp.where(
            jnp.any(lift_on) | jnp.any(contact_lift) | jnp.any(_field_lift),
            cfg["leg_relax_on_step"], 1.0)
        # 抬腿延伸引导（链 45，软 shaping）：lift_on 的腿把 knee/hipy 引向
        # "轮心抬升 lift_need"所需的伸展角（灵敏度 0.175 m/rad，前后腿
        # 符号相反：fl/fr knee+、hl/hr knee-，hipy 同理）。
        # 只改 reward 梯度，不改动作空间/不加硬指令；与 overlift 惩罚
        # （高于目标+0.05m 重罚）配合，伸展到位即停。
        # 链 47：延伸引导只在多级台阶区（含记忆）生效——**包括
        # contact_lift 分支**（链 46 复现：后轮顶在单级横脊面上触发
        # 接触抬轮 → σ 大时 MPC 真抬腿 → 失去抓地卡死）。单级横脊
        # 一律靠动量滚过，不引导任何抬腿。权重 0 时 r_ext 自然为 0。
        ext_need = lift_total
        # 前轮左右同步（链 55）：fl/fr 取 max——地图格对齐导致左右伸展
        # 不对称 → 爬梯时车身侧倾（chain 54 r2 底部 roll -1.96 侧翻复现）。
        ext_need = jnp.where(
            cfg["sync_front_ext"] > 0.0,
            ext_need.at[0:2].set(jnp.max(ext_need[:2])),
            ext_need)
        # v97 运动学修正（2026-08-07 实测）：原 0.175/0.35 m/rad 近似把
        # 前 hipy 目标只给到 -0.38 且符号方向与实测相反（默认 -1.16 时
        # 需 hipy 正摆 ~+0.85 才抬前轮）。统一换显式抬腿姿态：
        #   前轮：hipy +0.85（抬前腿）+ 膝 1.8（中段）
        #   后轮：膝 -2.6（tuck 收后腿），后 hipy 保持默认 1.16
        # 由 lift_on（感知前瞻，0.15~0.4m）或 contact_lift 门控——只在
        # 台阶前才引导，平地上不抬（v96 恒定 hipy 偏置进台阶前塌腿复现）。
        q_leg = d.qpos[7:][LEG_IDX]
        q_leg4 = q_leg.reshape(4, 3)
        # v214 运动学反解（卡死姿态 FK）：抬后轮需 hipy 大幅前摆 +2.3（近
        # 限位 2.53）+ 膝伸直 -1.2；旧后轮目标（hipy 1.16/knee -2.6）≈默认
        # 姿态，无抬升引导（后轮恒 0.61 卡死根因之一）。前轮维持 v97 实证。
        # v214e 精确抬升姿态（卡死姿态 FK 反解，pitch 0.25/body 0.81）：
        # 前轮 front_z=0.88 需 hipy+0.5~0.9/knee+1.5~2.6；后轮 rear_z=0.76
        # 需 hipy+1.5/knee-1.8（旧 2.3/-1.2 过抬到 0.96→翻车）。
        # v214h: 后轮左右侧分离姿态（FK 反解：HL 需 hipy+1.8/knee-1.4，
        # HR 只需 hipy+1.5/knee-1.8——9mm 不对称）；前轮过抬自适应——轮心
        # 超过目标+0.03m 时前腿目标下调，防俯仰时 r_ext 把前轮拉到
        # 0.97-1.03（overlift 权重压不住，v214-X/Y/AD/AJ 侧翻主因）。
        _fl_hy = cfg.get("lift_pose_fl_hipy", 1.00)
        _fl_kn = cfg.get("lift_pose_fl_knee", 1.50)
        _hl_hy = cfg.get("lift_pose_hl_hipy", 1.80)
        _hl_kn = cfg.get("lift_pose_hl_knee", -1.40)
        _hr_hy = cfg.get("lift_pose_hr_hipy", 1.50)
        _hr_kn = cfg.get("lift_pose_hr_knee", -1.80)
        lift_pose = jnp.array([
            0.0, _fl_hy, _fl_kn, 0.0, _fl_hy, _fl_kn,
            0.0, _hl_hy, _hl_kn, 0.0, _hr_hy, _hr_kn], dtype=jnp.float32)
        _front_ov = jnp.clip(
            (wheel_z[:2] - (target_z[:2] + 0.03)) / 0.15, 0.0, 1.0)
        _fscale = 1.0 - 0.55 * jnp.max(_front_ov)
        _fscale = jnp.maximum(_fscale, 0.45)
        lift_pose = lift_pose.at[1].set(lift_pose[1] * _fscale)
        lift_pose = lift_pose.at[2].set(lift_pose[2] * _fscale)
        lift_pose = lift_pose.at[4].set(lift_pose[4] * _fscale)
        lift_pose = lift_pose.at[5].set(lift_pose[5] * _fscale)
        lift_pose4 = lift_pose.reshape(4, 3)
        # v167：抬腿姿态引导按**场时序**逐腿门控（wheel_ref 高于地形+R 的腿
        # 才引导）——感知 lift_on 在 riser 阴影/网格对齐下时序不可靠（v163
        # 后腿过早 tuck 翻车）；场给出"该抬哪条腿、什么时候抬"。
        # v214: utility 选腿的轮子也进抬腿姿态引导——卡点后轮 _field_lift
        # 门控常关（欠抬 2.7cm < 0.04 阈值），r_ext 收腿引导静音；swing 权重
        # > 0.3 的轮子强制开引导（软先验，非门控：MPC 仍可偏离）。
        _util_on = jnp.zeros(4, dtype=jnp.bool_)
        if _gsw is not None:
            _util_on = _gsw > 0.3
        # v215f: r_ext 抬腿引导也受 prox 门控（前轮近 riser 才引导）——
        # 否则前轮离 riser 0.15m 就被拉到抬升姿态悬停（r_ext 60 压过
        # stance 悬空罚，prox 对 ground 的门控被 r_ext 绕过）。
        on = (lift_on | contact_lift | _field_lift | _util_on) & _prox_ok
        # v215i: HL 专属 r_ext 放大（S10_EXT_HL_BOOST）——左后轮 9mm 不对称
        # 下抬升姿态被负载压住（hipy 只到 1.4 而非 1.8），放大 HL 姿态拉力。
        _ext_hl = cfg.get("ext_hl_boost", 1.0)
        _ext_w4 = jnp.array([1.0, 1.0, _ext_hl, 1.0], dtype=jnp.float32)
        r_ext = -cfg["leg_ext_w"] * jnp.sum(
            jnp.where(on[:, None], jnp.square(q_leg4 - lift_pose4), 0.0)
            * _ext_w4[:, None])
        # v99 左右轮高对称罚：爬梯时 fl/fr（及 rl/rr）轮心高度差²，
        # 强制左右同步抬升（软 cost，非硬指令）。仅在抬轮激活时生效。
        r_sym = -cfg.get("stair_sym_w", 0.0) * jnp.any(on).astype(
            jnp.float32) * (
            jnp.square(wheel_z[0] - wheel_z[1])
            + jnp.square(wheel_z[2] - wheel_z[3]))
        # v128 轮悬空时间罚：台阶区轮子悬空超 0.2s 的部分重罚（air_t
        # 已在 wheel_air 计算中累积，>0.3s 视为持续悬空）。逼前轮从
        # riser 悬空状态落回台面获得支撑（v105-v127 滑回根因）。
        # v131 前轮悬空罚：只罚前两轮（fl/fr）悬空——v128 四轮全罚会压住
        # 后轮跟抬；前轮悬空钩 riser 才是滑回主因，单独逼前轮落台面。
        _air_t = info.get("wheel_air_t", jnp.zeros(4))
        r_airpen = -cfg.get("stair_air_w", 0.0) * jnp.sum(
            jnp.clip(_air_t[:2] - 0.2, 0.0, 1.0)) * (
            in_stairs > 0).astype(jnp.float32)

        # 锁轮推身软约束（用户 B，2026-08-06）：前轮"轮速≈0"= 挂台阶顶的
        # 好状态——罚前轮空转转速。链 63 修正只在接触顶死（contact_lift）
        # 激活。2026-08-07 改**单向**：对称罚 qd² 让 MPC 采样到反转侧
        # （顶死时正反转振荡 -58~+30 rad/s，run2 实测）→ 改只罚"前进打滑"
        # （S10 轮轴 0,-1,0，前进时 qd<0），反转（qd>0）不罚——MPC 收轮速
        # 到 0（锁住）但不反转。后轮不罚（保持滚动推力）。
        qd_wheel = d.qvel[6:][WHEEL_IDX]
        lock_on = contact_lift[:2]
        spin_fwd = jnp.clip(-qd_wheel, 0.0, 60.0)   # 前进打滑量（≥0）
        r_lockpush = -cfg["lockpush_w"] * jnp.sum(
            jnp.where(lock_on, jnp.square(spin_fwd[:2]), 0.0))
        # 轮速参考保持（2026-08-07 用户"加零偏"落地 reward 层）：MPPI
        # 覆盖 Y 前馈后轮速反转（-55~+15 rad/s 实测）→ 罚轮速偏离无滑移
        # 参考（-vx/r，S10 轮轴 0,-1,0 前进为负）；顶死（lock_on）时参考
        # 降 0（锁轮），正常时拉向前进。四轮都管（前轮防打滑+反转，
        # 后轮保持推力方向）。
        vx_ref = info["vel_tar"][0]
        ref_wheel = -vx_ref / jnp.maximum(cfg["wheel_radius"], 1e-3)
        ref_wheel = jnp.where(
            jnp.concatenate([lock_on, jnp.zeros(2, dtype=bool)]),
            0.0, ref_wheel)
        r_wheel_ref = -cfg["w_wheel_ref"] * jnp.sum(
            jnp.square(qd_wheel - ref_wheel))
        # v206：落脚点前拉（foothold planning 软落地）——摆动相轮子向下一级
        # 踏面落脚点 y 前拉（clip ±0.15m 防拉爆），激励 hipy 前摆把轮放到
        # 下一级；支撑相/区外不罚。与 ground_phase 摆动判定一致。
        _wf = cfg.get("w_foothold", 0.0)
        # 注意：_wf 是动态输入（tracer），不能用 Python if 判零；权重 0 时
        # 该项自动为 0（jnp.where 乘法），开关只改数值不 retrace。
        _fy_feat = elev["features"].get("foothold_y")
        if _fy_feat is not None:
            _fy, _fy_ok = sample_grid(
                _fy_feat, elev["features"].get("foothold_valid"),
                elev["origin"], elev["resolution"], wheel_xy,
                fill=wheel_xy[:, 1])
            if _gsw is not None:
                _sw2 = _gsw
            else:
                _sw2 = ((_ms > 0.5) & _wr_ok & _prox_ok & (
                    _wr - (h_terrain + cfg["wheel_radius"]) > _sw_th)
                ).astype(jnp.float32)
            _dfy = jnp.clip((_fy - wheel_xy[:, 1]) / 0.15, -1.0, 1.0)
            _lw = cfg.get("left_boost", 1.0)
            _w4 = jnp.array([_lw, 1.0, _lw, 1.0], dtype=jnp.float32)
            r_foothold = -_wf * jnp.sum(
                _sw2 * _fy_ok.astype(jnp.float32)
                * jnp.square(_dfy) * _w4)
        else:
            r_foothold = 0.0
        # v211: 台阶区横向回平（只罚 roll²，不罚 pitch——upright 同时罚
        # pitch 会杀爬升；卡点 roll -0.33 使左轮低 0.13m，同腿角实测）。
        _wrl = cfg.get("w_roll_level", 0.0)
        _qx = d.xquat[ctx["torso"]]
        _w, _x, _y, _z = _qx
        _roll = jnp.arctan2(2.0 * (_w * _x + _y * _z),
                            1.0 - 2.0 * (_x * _x + _y * _y))
        r_roll_level = -_wrl * jnp.square(_roll) * (
            in_stairs > 0).astype(jnp.float32)
        # v203 P1.2：台阶区单向后滑制动（棘轮式，v112 设计落地）——只罚
        # 轮子反向旋转（qd>0=向后打滑，S10 轮轴 0,-1,0 前进为负），正向
        # 滚动不受影响；阻止"到顶后整机向后滑回"（g3 实测 t=33s 滑回根因）。
        back_spin = jnp.clip(qd_wheel, 0.0, 60.0)
        r_brake = -cfg["stair_wheel_brake_w"] * jnp.sum(
            jnp.square(back_spin) * (in_stairs > 0).astype(jnp.float32))
        # v216: 轮锁（腿足狗爬梯）——仅当轮距下一 riser < S10_WHEEL_LOCK_PROX
        # （默认 0.25m）才锁轮（四轮锁死，只能腿抬放走楼梯）；接近段保持
        # 滚动（否则机器人滚不到楼梯）。软 cost 无门控。
        _lock_prox = float(os.environ.get("S10_WHEEL_LOCK_PROX", "0.25"))
        _lock_on = (in_stairs > 0) & (_prox < _lock_prox)
        r_wheel_lock = -cfg["stair_wheel_lock_w"] * jnp.sum(
            jnp.square(qd_wheel) * _lock_on.astype(jnp.float32))
        # v214: 支撑稳定性软罚（Takahashi 2023 支撑多边形思想软版）——
        # 单轮抬升时 CoM 必须留在接地轮支撑区内：1) 全轮四边形（循环序）
        # 缩边检查；2) 空载轮角点检查——CoM 越过"空载轮两邻轮连线"
        # （支撑三角形边界）即罚。权重 0 时恒 0，纯软 cost 无门控。
        # 注意：_wsp 是 tracer（dataclass 字段经 ctx 进入 jit），不能用
        # Python if；权重 0 时整项自动为 0（jnp 乘零）。
        _wsp = cfg.get("w_support", 0.0)
        _fz2 = jnp.abs(f[:, 2])
        _gnd = _fz2 > cfg.get("support_fz_min", 20.0)
        # v215k: 规划支撑多边形（S10_SUPPORT_EXCLUDE_LIFT=1，默认关）——
        # lift-need 高的轮子（欠抬 >0.05m）不算支撑轮，CoM 被推离该角→
        # 卸载→抬升（HL 专项：3 轮上台后 HL 欠抬 0.13m，仍被算作支撑轮
        # 使 CoM 可压在其角上，卸载-抬序列被采样卡死）。
        # 注意：support_exclude_lift 是 tracer（dataclass 字段），不能用
        # Python if；用广播乘零实现开关（0=不排除）。
        _need_hi = (_wr - wheel_z) > 0.05
        _gnd = _gnd & ~(jnp.broadcast_to(
            cfg.get("support_exclude_lift", 0.0) > 0.0, (4,)) & _need_hi)
        _n_gnd = jnp.sum(_gnd)
        _com = d.xpos[ctx["torso"]][:2]
        _cen = (jnp.sum(jnp.where(_gnd[:, None], wheel_xy, 0.0), axis=0)
                / (jnp.sum(_gnd) + 1e-6))
        _ang = jnp.arctan2(wheel_xy[:, 1] - _cen[1],
                           wheel_xy[:, 0] - _cen[0])
        _order = jnp.argsort(_ang)
        _v = wheel_xy[_order]
        _e = jnp.roll(_v, -1, axis=0) - _v
        _cr = (_e[:, 0] * (_com[1] - _v[:, 1])
               - _e[:, 1] * (_com[0] - _v[:, 0]))
        _cr2 = (_e[:, 0] * (jnp.roll(_v, -1, axis=0)[:, 1] - _v[:, 1])
                - _e[:, 1] * (jnp.roll(_v, -1, axis=0)[:, 0] - _v[:, 0]))
        _orient = jnp.sign(jnp.sum(_cr2))
        _orient = jnp.where(_orient == 0.0, 1.0, _orient)
        _len = jnp.linalg.norm(_e, axis=1) + 1e-6
        _sd = _cr / _len * _orient
        _mgn = cfg.get("support_margin", 0.06)
        _pen_q = jnp.square(jnp.clip(_mgn - _sd, 0.0, 0.5))
        # 空载轮角点：循环序下该轮两邻轮连线 = 支撑三角形边界；
        # CoM 越过边界（向空载角方向）即罚（_out>0 = 外侧）。
        _pa = _v[(jnp.arange(4) + 1) % 4]
        _pb = _v[(jnp.arange(4) - 1) % 4]
        _ea = _pa - _pb
        _la = jnp.linalg.norm(_ea, axis=1) + 1e-6
        _sda = ((_pb[:, 0] - _com[0]) * (_pa[:, 1] - _pb[:, 1])
                - (_pb[:, 1] - _com[1]) * (_pa[:, 0] - _pb[:, 0])) / _la
        _sdv = ((_pb[:, 0] - _v[:, 0]) * (_pa[:, 1] - _pb[:, 1])
                - (_pb[:, 1] - _v[:, 1]) * (_pa[:, 0] - _pb[:, 0])) / _la
        _sgn2 = jnp.sign(_sdv)
        _sgn2 = jnp.where(_sgn2 == 0.0, 1.0, _sgn2)
        _out = _sda * _sgn2
        _air = (~_gnd)[_order].astype(jnp.float32)  # 与 _v 排序对齐
        _pen_c = jnp.square(
            jnp.clip(_mgn + _out, 0.0, 0.5)) * _air
        # v214e: <3 轮接地 = 坠落风险（双轮齐抬），额外按 (3-n_gnd)² 强罚；
        # ≥2 轮时角点/四边形检查生效（2 轮时把 CoM 拉向接地区段）。
        _fall_risk = jnp.square(
            jnp.clip(3.0 - _n_gnd, 0.0, 2.0)) * 0.1  # v214i: 降权——2 轮
        # 接地（双前轮齐抬前进）是爬梯必要动作，0.5 权重压死（位移冻结）
        r_support = -_wsp * (
            jnp.sum(_pen_q) + jnp.sum(_pen_c) + _fall_risk
        ) * (_n_gnd >= 2).astype(jnp.float32)
    else:
        # use_elev=0：elevation 块被跳过，这里补下游依赖的默认值
        lift_on = jnp.zeros(4, dtype=jnp.bool_)
        contact_lift = jnp.zeros(4, dtype=jnp.bool_)
        in_stairs = jnp.zeros((), dtype=jnp.bool_)
        _field_lift = jnp.zeros(4, dtype=jnp.bool_)
        r_brake = 0.0
        r_wheel_lock = 0.0
        r_foothold = 0.0
        r_lift_prog = 0.0
        r_roll_level = 0.0
        r_support = 0.0
    # 牵引感知轮速惩罚（0804 §2.4，纯物理、无台阶检测）：
    # 轮体法向力小 = 悬空 / 主要压在台阶立面上 → 空转轮速被罚；
    # 有正常支撑 → 轮速不罚、正常驱动。MPC 在预测视界内自然学会
    # "接近台阶收轮速 → 抬轮 → 落地获得支撑后恢复加速"，
    # 并降低顶死时 kd_w*(vel_ref-qd) 的满力矩（反力矩掀翻源头）。
    # 用"失压时间"累积（MARG air-time 式）抗弹跳：平地行驶轮子会短暂
    # 离地（实测 <10N 占多数），瞬时 traction 判定会满负荷误罚轮速导致
    # 失稳（seg1/3 复现）；只有持续失压（≥0.2s 的抬轮/顶死）才激活。
    f_all = d.cfrc_ext[WHEEL_BODY_IDS][:, :3]
    f_z = jnp.abs(f_all[:, 2])
    traction = jnp.clip(f_z / cfg["wheel_ref_force"], 0.0, 1.0)
    air_t = info.get("wheel_air_t")
    if air_t is None:
        air_t = jnp.zeros(4)
    air_t = jnp.where(
        traction < 0.30, air_t + cfg["dt"],
        jnp.clip(air_t - 4.0 * cfg["dt"], 0.0, 0.6))
    info["wheel_air_t"] = air_t
    air_w = jnp.clip(air_t / 0.20, 0.0, 1.0)
    r_wheel_air = -cfg["terrain_w_wheel_air"] * jnp.sum(
        air_w * jnp.square(ctrl[12:]))
    # 权重平衡：让"动起来追速度"优于"不动"（能量/姿态惩罚须小于速度收益）
    # 弹跳惩罚降权：轮驱动时狗轻微弹跳（z 0.12→0.22），姿态/高度权重过高
    # 会让 MPC 判"动=弹跳=差"→ 恒 0。只罚摔倒(done)，允许弹跳。
    # 腿默认姿态正则：保持蹲姿驾驶，除非地形贴合奖励要求伸腿（爬台阶）。
    q_leg = d.qpos[7:][LEG_IDX]
    r_leg = -cfg["terrain_w_leg"] * leg_scale * jnp.sum(
        jnp.square(q_leg - ctx["default_leg"]))
    # ---- E3：地形自适应姿态目标（上坡仰头/下坡低头/过弯压弯）----
    r_pose = 0.0
    w_pitch = cfg["pose_w_pitch"]
    w_roll = cfg["pose_w_roll"]
    # 权重 0 时 r_pose 自然为 0（无 Python if，参数化后 cfg 为动态值）
    xm = d.xmat[ctx["torso"]]          # mjx: (3,3)，行 = 体轴在世界系
    pitch = jnp.arcsin(jnp.clip(-xm[2, 0], -1.0, 1.0))
    roll = jnp.arctan2(xm[2, 1], xm[2, 2])
    pitch_tar = info.get("pitch_tar", jnp.array(0.0))
    roll_tar = info.get("roll_tar", jnp.array(0.0))
    r_pose = (-w_pitch * jnp.square(pitch - pitch_tar)
              - w_roll * jnp.square(roll - roll_tar))
    # v215: 前高俯仰上限（防后仰翘头翻车）——pitch 超过上限（前高）即
    # 重罚；软 cost 无门控。爬升允许前高到 0.5rad（29°），更高视为失控。
    _wpc = cfg.get("w_pitch_cap", 0.0)
    _pc_rad = cfg.get("pitch_cap_rad", 0.50)
    r_pitch_cap = -_wpc * jnp.square(
        jnp.clip(-pitch - _pc_rad, 0.0, 1.0))
    # 爬梯仰头 reward（v88 实装）：目标 pitch 为负=仰头（本工程约定，
    # 实测绕侧轴抬前轮 => pitch 读数变负）。仰头把重心后移、前轮卸载，
    # 前腿更易缩回抬轮（Chamorro/Ascento 爬梯姿态参考）。仅 STAIR 模式
    # cfg 注入 stair_pitch_w；CRUISE cfg 无该键 -> 0，不干扰巡航。
    # v89 钳制：只罚"仰头不足"（偏差 +0.5 内随偏差增大），
    # 过度仰头（偏差 < -0.2）轻罚防后仰；误差钳制避免 v88 的
    # -1.5rad 失控过冲（w=30 时平方项无上限，越仰罚越大反而
    # 把机身顶到后翻）。
    # v122 动态爬梯 pitch（完全在 rollout 内计算，不碰 info 结构——
    # info 加键会破坏 JAX persistent cache 键导致频率 16Hz->2.5Hz，
    # v121 实测复现）。前后轮高差反馈：前高后低>0.12m 时低头 +0.15
    # 卸载后轮让后轮跟抬；前低贴 riser 时仰头 -0.45 抬前轮。
    _wg = jnp.mean(d.xpos[[5, 9], 2]) - jnp.mean(d.xpos[[13, 17], 2])
    _pt_dyn = jnp.where(
        _wg > 0.12, 0.15,
        jnp.where(_wg > 0.05, -0.15, -0.45))
    # v162：已知地图 pitch 剖面优先（info["pitch_tar"] 由 stair_pitch_ref
    # 注入，负=仰头），动态反馈作后备（无规划覆盖时）。
    _ms_p = info.get("mode_stair", jnp.array(0.0, dtype=jnp.float32))
    _pt_plan = info.get("pitch_tar", jnp.array(0.0, dtype=jnp.float32))
    _pt_ref = jnp.where(_ms_p > 0.5, _pt_plan, _pt_dyn)
    # v162b：仰头误差钳制放宽到 [-0.6, 0.5]——场目标下 MPC 曾过仰到 -0.94
    # （翻车），-0.2 上界使过仰惩罚饱和失效；放宽后强权重可压住过仰。
    r_pitch = -cfg.get("stair_pitch_w", 0.0) * jnp.square(
        jnp.clip(pitch - _pt_ref, -0.6, 0.5))
    # ---- E4：参考路径跟踪（解决高速跑歪/弯道走线差）----
    # 用 rollout 当前机身位置找路径最近点（横向偏离）+ 最近点切线航向误差；
    # ref 固定 (REF_N,2)，valid 标志控制启用，jnp 全向量化零 retrace。
    r_path = 0.0
    r_path_head = 0.0
    ref = info.get("ref_path")
    ref_valid = info.get("ref_valid")
    if ref is not None and ref_valid is not None:
        pos = d.xpos[ctx["torso"]][:2]
        d2 = jnp.sum(jnp.square(ref[:, :2] - pos[None, :]), axis=1)
        d_min = jnp.sqrt(jnp.min(d2))
        r_path = -cfg["w_path"] * jnp.minimum(d_min, 3.0)
        k = jnp.argmin(d2)
        k2 = jnp.minimum(k + 1, cfg["ref_n"] - 1)
        dir_ref = ref[k2, :2] - pos
        ang_ref = jnp.arctan2(dir_ref[1], dir_ref[0])
        yaw = quat_to_euler_z(d.xquat[ctx["torso"]])
        d_yaw = jnp.arctan2(jnp.sin(ang_ref - yaw),
                            jnp.cos(ang_ref - yaw))
        r_path_head = -cfg["w_path_head"] * jnp.square(d_yaw)
        # ---- MPCC 进度项（2026-08-08）：沿路径切线方向推进速度最大化。
        # 用现有 ref_path（不新增 info 键，避免破坏 JAX 缓存导致频率暴跌）。
        # 世界系速度投影到最近点切线；正=前进奖励，负=倒车惩罚。
        v_world = quat_rotate(xquat, vb)
        tang = ref[k2, :2] - ref[k, :2]
        tang = tang / (jnp.linalg.norm(tang) + 1e-6)
        r_progress = cfg["w_prog"] * (
            v_world[0] * tang[0] + v_world[1] * tang[1])
        r_progress = jnp.where(ref_valid, r_progress, 0.0)
        # z 轨迹跟踪（高程图决策的参考高度）——用**近点+1** ref_z
        # （+0.3m @0.3m 间距）：链 39 的 k+2（0.6m 前视）在楼梯底就把
        # 机身拉到楼梯顶（z 目标 1.3），机身抬起失稳（链 42 侧翻复现）。
        # 机身净高改由 r_clear（四轮下地形 max）逐级引导；这里只做
        # 轻度前视（k+1），配合 ref-z 兜底。
        k_la = jnp.minimum(k + 1, cfg["ref_n"] - 1)
        r_path_z = -cfg["w_path_z"] * jnp.square(
            d.xpos[ctx["torso"], 2] - ref[k_la, 2])
        # ref_valid 标志：无效（未注入路径）时两项强制为 0
        r_path = jnp.where(ref_valid, r_path, 0.0)
        r_path_head = jnp.where(ref_valid, r_path_head, 0.0)
        r_path_z = jnp.where(ref_valid, r_path_z, 0.0)
    else:
        r_path_z = 0.0
    # ---- 机身离地净高（r_clear，链 43）----
    # 目标 = 四轮下地形最大值 + 站姿高：前轮骑上台阶顶时地形 max 升到
    # 台阶顶 → 机身逐级抬起（不用看 0.6m 前）→ 后轮有空间跟抬；
    # 后轮还在下一级时 max 保持当前级 → 机身不提前抽高、不失去重心。
    # 与 r_ground（轮-地形贴合）互补：r_ground 管"轮子抬到哪"，
    # r_clear 管"机身跟着抬到哪"。空洞回退 = 当前机身高度（不惩罚）。
    r_clear = 0.0
    if (elev is not None and terrain_cost is not None
            and USE_ELEV > 0):
        h_wheel, ok_wc = sample_grid(
            elev["heightmap"], elev["features"]["valid"],
            elev["origin"], elev["resolution"], wheel_xy,
            fill=-1e6)
        h_max_body = jnp.max(
            jnp.where(ok_wc, h_wheel, -1e6), axis=0)
        z_ref_body = h_max_body + cfg["height_tar"]
        body_z = d.xpos[ctx["torso"], 2]
        # 2026-08-07：恢复双向（单向化削弱爬梯时逐级抬机身，v56 复现
        # 爬升不足）。双向跟踪 = 纯几何自然项：机身 = 四轮下地形 max +
        # 站姿高，随前轮上台阶逐级抬高，让后轮有空间跟抬。
        r_clear = -cfg["w_clear"] * jnp.where(
            ok_wc.any(), jnp.square(body_z - z_ref_body), 0.0)
        # CRUISE 单向防趴低（用户方案 4）：机身相对脚下地形高度低于
        # nominal_z 才罚（relu 下界），高于不罚——防"姿态太低"但不强制
        # 高站姿。链 71 修正：**台阶/横脊检测（lift_on/contact/in_stairs）
        # 时自动关闭**——横脊处前轮上脊后"机身相对地形 max"偏低，
        # 若罚会逼 MPC 在横脊上抬机身 → 侧翻（chain 70 r1 复现）。
        h_rel = body_z - h_max_body
        _no_step = 1.0 - jnp.maximum(
            jnp.maximum(jnp.any(lift_on), jnp.any(contact_lift)),
            in_stairs).astype(jnp.float32)
        r_crouch = -cfg["w_crouch"] * jnp.square(
            jnp.clip(cfg["nominal_z"] - h_rel, 0.0, 0.5)) * _no_step
    # v107 顶缘阶段接线（原实现只在 class _reward，rollout 从未生效）：
    # body z > top_z 且在楼梯区时，软化抬身/抬腿（<1）并强化前推、
    # 防侧倾、锁轮（>1）——解决"到顶后前轮下探、整机向后滑回"
    # （v105 r1 z=0.99 处滑回复现；top_z 由 S10_STAIR_TOP_Z 配置）。
    _top_on = (d.xpos[ctx["torso"], 2] > cfg["top_z"]) & (in_stairs > 0)
    _top_f = lambda v: jnp.where(_top_on, v, 1.0)
    r_upright = r_upright * _top_f(cfg["top_upright_scale"])
    r_attdamp = r_attdamp * _top_f(cfg["top_attdamp_scale"])
    r_path_z = r_path_z * _top_f(cfg["top_pathz_scale"])
    r_clear = r_clear * _top_f(cfg["top_clear_scale"])
    r_ext = r_ext * _top_f(cfg["top_ext_scale"])
    r_push = r_push * _top_f(cfg["top_push_scale"])
    r_lockpush = r_lockpush * _top_f(cfg["top_lockpush_scale"])
    # ---- v185：非对称压弯腿姿态（摩托压弯式）----
    # 左右轮足高度差产生机身 roll：左转（vyaw>0）时左轮抬（knee 差>0）→
    # 车向左倾，用几何压弯对抗离心侧翻。目标按命令 yaw 率生成，
    # 只罚"没做到位"，允许 MPC 在稳定性需要时偏离（软 reward）。
    r_lean = 0.0
    if LEAN_LEG_W > 0.0:
        _vyaw_c = info["ang_vel_tar"][2]
        _vx_c = jnp.abs(info["vel_tar"][0])
        if LEAN_K > 0.0:
            _lean_tar = jnp.clip(LEAN_K * _vx_c * _vyaw_c, -3.0, 3.0)
        else:
            _lean_tar = jnp.clip(_vyaw_c, -3.0, 3.0)
        _knee = d.qpos[7:][LEG_IDX][2::3]      # 4 腿 knee（fl/fr/hl/hr）
        _lminr = (jnp.mean(_knee[0:1]) - jnp.mean(_knee[1:2])
                  + jnp.mean(_knee[2:3]) - jnp.mean(_knee[3:4])) * 0.5
        r_lean = -LEAN_LEG_W * jnp.square(_lminr - _lean_tar)
    return (r_lean + r_vel + r_ang + r_upright + r_attdamp + r_yaw * 0.5
            + r_height * cfg["height_weight"] + r_energy * 0.0001
            + r_terrain + r_ground + r_overlift + r_stumble
            + r_wheel_air + r_leg + r_pose + r_path + r_path_head
            + r_path_z + r_clear + r_ext + r_lockpush + r_crouch
            + r_push + r_swing_ok + r_z_smooth + r_wheel_ref
            + r_pitch + r_sym + r_airpen + r_progress + r_brake
            + r_foothold + r_lift_prog + r_roll_level + r_support
            + r_pitch_cap + r_wheel_lock)
class S10WheeledEnv:
    def __init__(self, config: S10WheeledEnvConfig = S10WheeledEnvConfig()):
        self._config = config
        self.dt = config.dt
        self.timestep = config.timestep
        self.action_size = 16
        m = mujoco.MjModel.from_xml_path(S10_MPC_XML)
        m.opt.timestep = config.timestep
        m.opt.iterations = config.solver_iterations
        m.opt.ls_iterations = config.solver_ls_iterations
        m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        m.dof_damping[6:] = np.array(
            ([config.leg_damping]*3 + [config.wheel_damping]) * 4)
        self.mj_model = m
        self.mx = mjx.put_model(m)
        self._nq = m.nq
        self._nv = m.nv
        self.joint_torque_range = jnp.array(m.actuator_ctrlrange)
        self.joint_range = jnp.array(m.jnt_range[1:])
        self._stand_joint = jnp.array(S10_STAND_JOINT)
        self._default_leg = jnp.array(S10_STAND_JOINT[LEG_IDX_NP])
        self._torso = 1
        self._ctx = self.build_ctx()

    def build_ctx(self):
        """把 rollout 热路径依赖打包成 pytree 上下文（预热缓存重构）。

        全部叶子为 jnp/numpy 数组或标量（JAX pytree），不含字符串/函数：
        wheel_control 字符串编码为 wheel_mode float（0=velocity,1=torque），
        Python if 分支改为 jnp.where。ctx 作为 jit 参数传入后，persistent
        cache 的键由叶子内容决定（跨进程稳定）→ 第二次启动直接命中。
        """
        import dataclasses
        cfg = dataclasses.asdict(self._config)
        # 字符串字段不能作为 JAX pytree 叶子（jit 参数），统一排除：
        # wheel_control → wheel_mode float；leg_control/task_name 未参与
        # rollout 数值计算（S10 腿恒 PD 位置控制）。
        cfg.pop("task_name", None)
        cfg.pop("wheel_control", None)
        cfg.pop("leg_control", None)
        cfg["wheel_mode"] = (
            0.0 if self._config.wheel_control == "velocity" else 1.0)
        return {
            "mx": self.mx,
            "cfg": cfg,
            "default_leg": self._default_leg,
            "torso": self._torso,
            "torque_range": self.joint_torque_range,
            "joint_range": self.joint_range,
            "body_rootid": np.asarray(
                self.mj_model.body_rootid, dtype=np.int32),
            "nq": self._nq,
            "nv": self._nv,
        }

    def _make_state(self, d: mjx.Data) -> MjxLikeState:
        x = Transform(pos=d.xpos[1:], rot=d.xquat[1:])
        cvel = Motion(vel=d.cvel[1:, 3:], ang=d.cvel[1:, :3])
        offset = d.xpos[1:, :] - d.subtree_com[self.mj_model.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        return MjxLikeState(data=d, x=x, xd=xd)

    def act2joint(self, act_leg):
        hipy_s = float(getattr(self._config, "leg_hipy_scale", 1.0))
        per_j = jnp.array([1.0, hipy_s, 1.0, 1.0, hipy_s, 1.0,
                           1.0, hipy_s, 1.0, 1.0, hipy_s, 1.0],
                          dtype=jnp.float32)
        return self._default_leg + act_leg * per_j * self._config.leg_action_scale

    def act2tau(self, act, d):
        leg_target = self.act2joint(act[:12])
        q_leg = d.qpos[7:][LEG_IDX]
        qd_leg = d.qvel[6:][LEG_IDX]
        tau_leg = self._config.kp * (leg_target - q_leg) - self._config.kd * qd_leg
        qd_wheel = d.qvel[6:][WHEEL_IDX]
        if self._config.wheel_control == "velocity":
            # 速度伺服：S10 轮 axis=0,-1,0，正 qvel→后退，故目标速度取反。
            # act=1.0 → 目标 -10 rad/s（前进）；tau=kd_w*(vel_ref - qvel)
            vel_ref = -act[12:] * self._config.vel_scale
            tau_wheel = self._config.kd_wheel * (vel_ref - qd_wheel)
        else:
            # 力矩模式（旧）：+力矩→qvel+→后退；负力矩前进
            tau_wheel = -act[12:] * self._config.wheel_tau_scale
        tau = jnp.zeros(16).at[LEG_IDX].set(tau_leg).at[WHEEL_IDX].set(tau_wheel)
        return jnp.clip(tau, self.joint_torque_range[:, 0],
                        self.joint_torque_range[:, 1])

    def _get_obs(self, d, info):
        xquat = d.xquat[self._torso]
        cvel = d.cvel[self._torso]
        vb = quat_inv_rotate(xquat, cvel[3:])
        ab = quat_inv_rotate(xquat, cvel[:3])
        return jnp.concatenate([
            info["vel_tar"], info["ang_vel_tar"],
            d.ctrl, d.qpos, vb, ab, d.qvel[6:],
        ])

    def _reward(self, d, info, ctrl):
        xquat = d.xquat[self._torso]
        cvel = d.cvel[self._torso]
        vb = quat_inv_rotate(xquat, cvel[3:])
        ab = quat_inv_rotate(xquat, cvel[:3])
        r_vel = (-jnp.sum((vb[:2] - info["vel_tar"][:2]) ** 2)
                 * self._config.vel_weight)
        r_ang = (-jnp.square(ab[2] - info["ang_vel_tar"][2])
                 * self._config.ang_vel_weight)
        vec = quat_rotate(xquat, jnp.array([0.0, 0.0, 1.0]))
        # 地形相对直立：用高程图高度场梯度求地形法线，允许机身顺坡倾斜，
        # 惩罚"翘头"（姿态比地形更陡）——爬坡稳定的关键（世界垂直惩罚会
        # 逼 MPC 保持水平 → 前轮离地 → 打滑卡死）。
        n_ref = jnp.array([0.0, 0.0, 1.0])
        elev_up = info.get("elevation_map")
        if elev_up is not None and sample_grid is not None:
            base_xy = d.xpos[self._torso][:2]
            fwd = quat_rotate(xquat, jnp.array([1.0, 0.0, 0.0]))
            lat = quat_rotate(xquat, jnp.array([0.0, 1.0, 0.0]))
            # 水平归一化：机体俯仰/侧倾时 3D 轴向的水平投影会缩短，
            # 直接乘 0.3 会让探测点比预期近（0804 §3.6 姿态敏感点 2）。
            fwd2 = fwd[:2] / (jnp.linalg.norm(fwd[:2]) + 1e-6)
            lat2 = lat[:2] / (jnp.linalg.norm(lat[:2]) + 1e-6)
            probe = jnp.stack([
                base_xy + fwd2 * 0.3, base_xy - fwd2 * 0.3,
                base_xy + lat2 * 0.3, base_xy - lat2 * 0.3])
            hp, okp = sample_grid(
                elev_up["heightmap"], elev_up["features"]["valid"],
                elev_up["origin"], elev_up["resolution"], probe, fill=0.0)
            dhf = (hp[0] - hp[1]) / 0.6
            dhl = (hp[2] - hp[3]) / 0.6
            n_terr = jnp.array([-dhf, -dhl, 1.0])
            n_terr = n_terr / (jnp.linalg.norm(n_terr) + 1e-6)
            n_ref = jnp.where(okp.all(), n_terr, n_ref)
        cos_a = jnp.clip(jnp.dot(vec, n_ref), -1.0, 1.0)
        r_upright = -self._config.terrain_w_upright * jnp.square(1.0 - cos_a)
        # 姿态角速度阻尼：抑制爬坡时的俯仰/侧倾振荡（轮矩反作用激起）
        r_attdamp = -self._config.terrain_w_attdamp * jnp.sum(
            jnp.square(ab[:2]))
        yaw = quat_to_euler_z(xquat)
        yaw_tar = info["yaw_tar"] + info["ang_vel_tar"][2] * self.dt * info["step"]
        d_yaw = yaw - yaw_tar
        r_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        r_height = -jnp.square(d.xpos[self._torso, 2] - info["pos_tar"][2])
        r_energy = -jnp.sum(jnp.square(ctrl))
        # 地形代价（感知-voxel 世界对齐瓦片）：按预测轮落点 gather。
        # 坡度/粗糙度/台阶特征在感知侧预计算，越界/空洞按"未知不惩罚"。
        # 数据来自 info["elevation_map"]（仿真节点 get_local_map() 经
        # MPCController.set_elevation_map 注入，固定 (60,60) 形状）。
        r_terrain = 0.0
        r_ground = 0.0
        r_overlift = 0.0
        r_stumble = 0.0
        r_ext = 0.0
        leg_scale = 1.0
        elev = info.get("elevation_map")
        if elev is not None and terrain_cost is not None:
            wheel_xy = d.xpos[WHEEL_BODY_IDS][:, :2]
            wheel_z = d.xpos[WHEEL_BODY_IDS][:, 2]
            fwd2 = quat_rotate(xquat, jnp.array([1.0, 0.0, 0.0]))[:2]
            fwd2 = fwd2 / (jnp.linalg.norm(fwd2) + 1e-6)  # 水平归一化（抗俯仰缩短）
            cost, _ok = terrain_cost(
                elev["features"], elev["origin"], elev["resolution"],
                wheel_xy,
                w_slope=self._config.terrain_w_slope,
                w_rough=self._config.terrain_w_rough,
                w_step=self._config.terrain_w_step)
            r_terrain = -jnp.sum(cost)
            # 轮-地形贴合（感知高程图引导抬腿）：预测轮心应落在
            # 地形高度 + 轮半径处；前方地形升高 → 该误差增大 →
            # MPC 被激励伸腿让轮子骑上台阶（而不是顶死打滑）。
            h_terrain, ok_h = sample_grid(
                elev["heightmap"], elev["features"]["valid"],
                elev["origin"], elev["resolution"], wheel_xy,
                # 轮下地图格无效（riser 立面遮挡）时用"轮心真实高度 − 轮半径"
                # 作为地形参考（= 轮子实际压着的地面），而不是轮心高度本身：
                # fill=wheel_z 会让 target_z=wheel_z+r 把轮子往下压 0.081m，
                # 且 lift_need 用轮心做基准会误算台阶净高（wp7 卡点实测）。
                fill=wheel_z - self._config.wheel_radius)
            # 2026-08-05 曾加"轮周 ±0.2m min-窗口"回退（修爬升中高差趋 0），
            # 实测造成左右轮不对称 → riser 前侧翻（链 20 3/3 复现），已还原
            # 链 5 原样 fill=wheel_z−r（对称、可靠）。
            # 前瞻抬轮（MARG feet-air-time 的轮式化，见 0804 §2.4）：
            # 四轮（含后轮！2026-08-05 用户指出：前轮上台阶后，后轮也要
            # 爬同一级 riser，wp7 卡点即"前轮上、后轮卡"）在 0.15~0.4m
            # 三点窗口采样高程与台阶标志；窗口内任一处出现离散台阶
            # （step_flag）且高度明显高于当前轮下地形 → 抬高该轮目标高度。
            # 3 点窗口为实测最优（climb3：wp0→wp7 34s）；密集带会造成
            # 过早抬轮（坡前翻）、过近/过高阈值会漏检（卡台阶）。
            offs = jnp.array(
                [self._config.lift_lookahead * 0.375,
                 self._config.lift_lookahead * 0.7,
                 self._config.lift_lookahead])
            probe = (wheel_xy[:, None, :]
                     + fwd2[None, None, :] * offs[None, :, None])
            h_win, ok_w = sample_grid(
                elev["heightmap"], elev["features"]["valid"],
                elev["origin"], elev["resolution"], probe,
                fill=jnp.broadcast_to(h_terrain[:, None], probe.shape[:2]))
            # 台阶标志（离散 riser）门控：连续坡道上 h_ahead-h_now 也 > 阈值
            # （坡度×前瞻），若只看高差会把抬轮误触发在坡道上（实测坡顶侧翻）。
            # step_flag 来自感知侧（local_map 默认 >0.08m 的相邻高差）。
            step_win, ok_s = sample_grid(
                elev["features"]["step_flag"], elev["features"]["valid"],
                elev["origin"], elev["resolution"], probe,
                fill=jnp.zeros(probe.shape[:2]))
            # lift_need = 台阶净高（轮半径在目标式中抵消）：
            #   目标 = h_now + r + lift_need = h_ahead + r（轮心骑上台阶顶）
            h_ahead = jnp.max(jnp.where(ok_w, h_win, -1e6), axis=1)
            step_ahead = jnp.max(step_win, axis=1)
            # 陡升梯度门控（链 44）：探针间局部上升梯度 = 台阶特征，对网格
            # 对齐不敏感（step_flag 只在 riser 边界单格上，0.15~0.4m 探针
            # 常整段落在台面上 → 永远触发不了）。梯度 = 相邻探针高差/间距：
            #   0.125m riser ≈ 0.96；20% 坡 ≈ 0.32 → 0.6 阈值天然区分。
            rise12 = jnp.where(
                ok_w[:, 0] & ok_w[:, 1],
                (h_win[:, 1] - h_win[:, 0]) / 0.13, 0.0)
            rise23 = jnp.where(
                ok_w[:, 1] & ok_w[:, 2],
                (h_win[:, 2] - h_win[:, 1]) / 0.12, 0.0)
            steep = jnp.maximum(rise12, rise23) > self._config.lift_steep_gate
            lift_on = ((h_ahead - h_terrain) > self._config.lift_threshold) \
                & ((step_ahead > self._config.lift_step_gate) | steep)
            # 多级台阶区检测（stair_ahead，链 46）：单级 0.13m 台阶靠动量可
            # 滚过（wp5→6 横脊实测），连续台阶才需要抬腿序列。判据 = 机身
            # 前方 0.2m 与 1.0m 处地形仍持续上升（>0.15m）——楼梯在 1m 前瞻
            # 内叠 2+ 级，横脊/单台阶 1m 后已平。带 stair_t 记忆（衰减）覆盖
            # 最后一级：前轮上顶后 ahead 变平，后轮仍需跟抬。
            base_xy = d.xpos[self._torso][:2]
            probe_s = base_xy[None, :] + fwd2[None, :] * jnp.array(
                [0.2, 0.6, 1.0])[:, None]
            h_s, ok_s2 = sample_grid(
                elev["heightmap"], elev["features"]["valid"],
                elev["origin"], elev["resolution"], probe_s,
                fill=-1e6)
            h_near = jnp.where(ok_s2[0], h_s[0], -1e6)
            h_far = jnp.where(ok_s2[2], h_s[2], -1e6)
            stair_ahead = (h_far - h_near > 0.15) \
                & (h_near > -1e5) & (h_far > -1e5)
            stair_t = info.get("stair_t")
            if stair_t is None:
                stair_t = jnp.zeros(())
            stair_t = jnp.where(
                stair_ahead, jnp.minimum(stair_t + self.dt, 3.0),
                jnp.maximum(stair_t - 2.0 * self.dt, 0.0))
            info["stair_t"] = stair_t
            in_stairs = stair_ahead | (stair_t > 0.5)
            # 后轮抬升总开关（S10_LIFT_REAR，默认 1）：链 8 实测后轮抬升使
            # 第一/二级 riser 成功率下降（r2/r3 翻、r1 卡第二级），对比
            # 前轮-only（链 5：2/2 双 riser 通过）。默认开，实验可关。
            if not self._config.lift_rear:
                lift_on = lift_on.at[2:].set(False)
            # 后轮抬升（2026-08-05，用户指出"后腿也要爬楼梯"）：
            # 探测证实后轮 0.15~0.3m 前视窗口位于 riser 阴影（LiDAR 被立面
            # 遮挡，地图恒空洞），"后轮前视采样"永远触发不了。改为"跟抬"：
            # 同侧前轮已上台阶顶（前轮下地形 h_terrain[0/1] 高于后轮下地形
            # h_terrain[2/3]，且前轮正在抬）→ 后轮目标直接抬到前轮高度。
            # 前轮下地形用地图真值（在台阶顶上有效），后轮下地形用轮心−轮
            # 半径回退（不受空洞影响）——两者差 = 后轮需爬的台阶净高。
            # v139b：同 pure 版，用物理轮高差。
            rear_need = jnp.clip(
                wheel_z[:2] - wheel_z[2:], 0.0,
                self._config.lift_max)
            # 前轮后方 0.15m 的 step_flag：前轮刚爬过的 riser 边界（在轮后），
            # 用于区分"离散台阶"与"连续坡道"（坡道上前/后轮下地形差也会
            # >0.05，不能触发跟抬，否则长坡上后轮乱抬失去抓地）。
            probe_behind = wheel_xy[:2] - fwd2[None, :] * 0.15
            step_behind, _okb = sample_grid(
                elev["features"]["step_flag"], elev["features"]["valid"],
                elev["origin"], elev["resolution"], probe_behind,
                fill=0.0)
            # 链 46：后轮跟抬增加 stair 门控——单级横脊上后轮抬腿会失去
            # 抓地卡死（链 45 sigma 2.0 复现）；只有多级台阶区（含最后一级
            # 的记忆窗口）才允许后轮跟抬。链 47 修正：**不能覆盖前视抬轮**
            # （steep 门控，横脊处也触发）——横脊需要"适度抬腿"（0.12m，
            # r_ground 目标抬高，r_ext 关闭防过度伸展）；跟抬只在楼梯区叠加。
            _ms = info.get("mode_stair", jnp.array(0.0, dtype=jnp.float32))
            # v135：与 pure 版一致，楼梯区用全局知识绕过感知阴影门控。
            rear_follow = (rear_need > self._config.rear_follow_thresh) \
                & ((step_behind > self._config.lift_step_gate)
                   | (_ms > 0.5)) \
                & in_stairs \
                & (d.xpos[self._torso, 2] > self._config.rear_lift_zgate)
            lift_on = lift_on.at[2:].set(
                lift_on[2:] | rear_follow)
            lift_need = jnp.where(
                lift_on,
                jnp.clip(h_ahead - h_terrain, 0.0, self._config.lift_max),
                0.0)
            lift_need = lift_need.at[2:].set(jnp.minimum(
                jnp.where(lift_on[2:],
                          rear_need * self._config.rear_lift_scale, 0.0),
                self._config.lift_max))
            # 链 64：lift_need 左右同步（fl/fr、hl/hr 各取 max）——爬升时
            # 左右轮抬升不一致（地图格对齐/接触时机差）→ 车身侧倾累积 →
            # 侧翻（chain 63 roll -0.14→-2.09 复现）。同步后爬升近似
            # "整体抬升"，与 r_clear（机身逐级抬）协同。
            lift_need = lift_need.at[0:2].set(jnp.max(lift_need[:2]))
            lift_need = lift_need.at[2:4].set(jnp.max(lift_need[2:]))
            # 接触触发抬轮（CTBC reward 化，0804 §3.7）：轮子顶在台阶面时
            # 水平接触力远大于法向（力比爆表）——物理可靠、不受感知遮挡/
            # 格对齐影响，且与弹跳可区分（弹跳水平力也小，比值不高）。
            # 触发时该轮目标高度抬升 → MPC 自己抬轮离开立面 → 法向力恢复 →
            # 目标回落 → 轮子落到台阶顶。与感知前瞻抬轮互补（感知提前、接触兜底）。
            f = d.cfrc_ext[WHEEL_BODY_IDS][:, :3]
            f_xy = jnp.sqrt(jnp.sum(jnp.square(f[:, :2]), axis=1) + 1e-6)
            f_z = jnp.abs(f[:, 2]) + 1e-3
            ratio = f_xy / f_z
            # 持续门控：弹跳是单帧尖峰（实测平地力比>2 占 5~10%），顶死是持续
            # 数百 ms 的信号；累积 0.1s 才触发，避免弹跳误抬（0804 §3.7）。
            cl_cond = (ratio > self._config.contact_lift_ratio) \
                & (f_z < self._config.wheel_ref_force * 1.5) \
                & (f_xy > 10.0)
            cl_t = info.get("contact_lift_t")
            if cl_t is None:
                cl_t = jnp.zeros(4)
            cl_t = jnp.where(cl_cond, cl_t + self.dt,
                             jnp.clip(cl_t - 4.0 * self.dt, 0.0, 0.5))
            info["contact_lift_t"] = cl_t
            contact_lift = cl_t > 0.10
            target_z = h_terrain + self._config.wheel_radius
            # v135：与 pure 版一致，抬轮机制取 max 不叠加。
            lift_total = jnp.maximum(
                lift_need, jnp.where(
                    contact_lift, self._config.lift_max * 0.8, 0.0))
            target_z = target_z + lift_total
            dev = wheel_z - target_z
            r_ground = -self._config.terrain_w_ground * jnp.sum(
                jnp.where(ok_h, jnp.square(dev), 0.0))
            # 过抬惩罚（0804 §3.7 诊断）：抬轮目标 0.71m 时轮子冲到 1.0m →
            # 机身后仰翘头翻车；对"高于目标+0.05m"的部分重罚，让 MPC 停在
            # 目标附近而不是过冲。
            overlift = jnp.clip(wheel_z - (target_z + 0.05), 0.0, 1.0)
            r_overlift = -self._config.terrain_w_overlift * jnp.sum(
                jnp.square(overlift))
            # 撞阶惩罚（MARG feet-stumble：水平接触力 > 4× 垂直 = 顶死台阶面）：
            # 平滑斜坡惩罚（超阈值部分），直接对抗"轮矩反作用掀翻车身"。
            r_stumble = -self._config.terrain_w_stumble * jnp.sum(
                jnp.clip(ratio - self._config.stumble_ratio, 0.0, 3.0))
            # 台阶区放松蹲姿正则：感知前瞻或接触触发抬轮时 leg_scale 缩小，
            # 避免 r_leg 与抬腿打架（0804 §3.2 缺口 2）。
            leg_scale = jnp.where(
                jnp.any(lift_on) | jnp.any(contact_lift),
                self._config.leg_relax_on_step, 1.0)
            # 抬腿延伸引导（链 45，软 shaping）：lift_on 的腿把 knee/hipy 引向
            # "轮心抬升 lift_need"所需的伸展角（灵敏度 0.175 m/rad，前后腿
            # 符号相反：fl/fr knee+、hl/hr knee-，hipy 同理）。
            # 只改 reward 梯度，不改动作空间/不加硬指令；与 overlift 惩罚
            # （高于目标+0.05m 重罚）配合，伸展到位即停。
            if self._config.leg_ext_w > 0.0:
                ext_need = lift_total
                lift_sign = jnp.array([1.0, 1.0, -1.0, -1.0])
                q_leg = d.qpos[7:][LEG_IDX]
                q_knee = q_leg[2::3]
                q_hipy = q_leg[1::3]
                dk_tar = lift_sign * ext_need / 0.175
                dh_tar = lift_sign * ext_need / 0.35   # hipy 分担一半伸展
                knee_tar = self._default_leg[2::3] + dk_tar
                hipy_tar = self._default_leg[1::3] + dh_tar
                # 链 47：延伸引导只在多级台阶区（含记忆）生效——**包括
                # contact_lift 分支**（链 46 复现：后轮顶在单级横脊面上触发
                # 接触抬轮 → σ 大时 MPC 真抬腿 → 失去抓地卡死）。单级横脊
                # 一律靠动量滚过，不引导任何抬腿。
                on = (lift_on | contact_lift) & in_stairs
                _top2 = (d.xpos[self._torso, 2]
                         > self._config.top_z) & in_stairs
                r_ext = -self._config.leg_ext_w * jnp.where(
                    _top2, self._config.top_ext_scale, 1.0) * (
                    jnp.sum(jnp.where(on, jnp.square(q_knee - knee_tar), 0.0))
                    + jnp.sum(jnp.where(on, jnp.square(q_hipy - hipy_tar), 0.0)))
            else:
                r_ext = 0.0
        # 牵引感知轮速惩罚（0804 §2.4，纯物理、无台阶检测）：
        # 轮体法向力小 = 悬空 / 主要压在台阶立面上 → 空转轮速被罚；
        # 有正常支撑 → 轮速不罚、正常驱动。MPC 在预测视界内自然学会
        # "接近台阶收轮速 → 抬轮 → 落地获得支撑后恢复加速"，
        # 并降低顶死时 kd_w*(vel_ref-qd) 的满力矩（反力矩掀翻源头）。
        # 用"失压时间"累积（MARG air-time 式）抗弹跳：平地行驶轮子会短暂
        # 离地（实测 <10N 占多数），瞬时 traction 判定会满负荷误罚轮速导致
        # 失稳（seg1/3 复现）；只有持续失压（≥0.2s 的抬轮/顶死）才激活。
        f_all = d.cfrc_ext[WHEEL_BODY_IDS][:, :3]
        f_z = jnp.abs(f_all[:, 2])
        traction = jnp.clip(f_z / self._config.wheel_ref_force, 0.0, 1.0)
        air_t = info.get("wheel_air_t")
        if air_t is None:
            air_t = jnp.zeros(4)
        air_t = jnp.where(
            traction < 0.30, air_t + self.dt,
            jnp.clip(air_t - 4.0 * self.dt, 0.0, 0.6))
        info["wheel_air_t"] = air_t
        air_w = jnp.clip(air_t / 0.20, 0.0, 1.0)
        r_wheel_air = -self._config.terrain_w_wheel_air * jnp.sum(
            air_w * jnp.square(ctrl[12:]))
        # 权重平衡：让"动起来追速度"优于"不动"（能量/姿态惩罚须小于速度收益）
        # 弹跳惩罚降权：轮驱动时狗轻微弹跳（z 0.12→0.22），姿态/高度权重过高
        # 会让 MPC 判"动=弹跳=差"→ 恒 0。只罚摔倒(done)，允许弹跳。
        # 腿默认姿态正则：保持蹲姿驾驶，除非地形贴合奖励要求伸腿（爬台阶）。
        q_leg = d.qpos[7:][LEG_IDX]
        r_leg = -self._config.terrain_w_leg * leg_scale * jnp.sum(
            jnp.square(q_leg - self._default_leg))
        # ---- E3：地形自适应姿态目标（上坡仰头/下坡低头/过弯压弯）----
        r_pose = 0.0
        w_pitch = self._config.pose_w_pitch
        w_roll = self._config.pose_w_roll
        if w_pitch > 0.0 or w_roll > 0.0:
            xm = d.xmat[self._torso]          # mjx: (3,3)，行 = 体轴在世界系
            pitch = jnp.arcsin(jnp.clip(-xm[2, 0], -1.0, 1.0))
            roll = jnp.arctan2(xm[2, 1], xm[2, 2])
            pitch_tar = info.get("pitch_tar", jnp.array(0.0))
            roll_tar = info.get("roll_tar", jnp.array(0.0))
            r_pose = (-w_pitch * jnp.square(pitch - pitch_tar)
                      - w_roll * jnp.square(roll - roll_tar))
        # ---- E4：参考路径跟踪（解决高速跑歪/弯道走线差）----
        # 用 rollout 当前机身位置找路径最近点（横向偏离）+ 最近点切线航向误差；
        # ref 固定 (REF_N,2)，valid 标志控制启用，jnp 全向量化零 retrace。
        r_path = 0.0
        r_path_head = 0.0
        r_progress = 0.0
        ref = info.get("ref_path")
        ref_valid = info.get("ref_valid")
        if ref is not None and ref_valid is not None:
            pos = d.xpos[self._torso][:2]
            d2 = jnp.sum(jnp.square(ref[:, :2] - pos[None, :]), axis=1)
            d_min = jnp.sqrt(jnp.min(d2))
            r_path = -self._config.w_path * jnp.minimum(d_min, 3.0)
            k = jnp.argmin(d2)
            k2 = jnp.minimum(k + 1, self._config.ref_n - 1)
            dir_ref = ref[k2, :2] - pos
            ang_ref = jnp.arctan2(dir_ref[1], dir_ref[0])
            yaw = quat_to_euler_z(d.xquat[self._torso])
            d_yaw = jnp.arctan2(jnp.sin(ang_ref - yaw),
                                jnp.cos(ang_ref - yaw))
            r_path_head = -self._config.w_path_head * jnp.square(d_yaw)
            # MPCC 进度项（与 _reward_pure 一致）
            v_world = quat_rotate(xquat, vb)
            tang = ref[k2, :2] - ref[k, :2]
            tang = tang / (jnp.linalg.norm(tang) + 1e-6)
            r_progress = self._config.w_prog * (
                v_world[0] * tang[0] + v_world[1] * tang[1])
            # z 轨迹跟踪（高程图决策的参考高度）——用**近点+1** ref_z
            # （+0.3m @0.3m 间距）：链 39 的 k+2（0.6m 前视）在楼梯底就把
            # 机身拉到楼梯顶（z 目标 1.3），机身抬起失稳（链 42 侧翻复现）。
            # 机身净高改由 r_clear（四轮下地形 max）逐级引导；这里只做
            # 轻度前视（k+1），配合 ref-z 兜底。
            k_la = jnp.minimum(k + 1, self._config.ref_n - 1)
            r_path_z = -self._config.w_path_z * jnp.square(
                d.xpos[self._torso, 2] - ref[k_la, 2])
            # ref_valid 标志：无效（未注入路径）时两项强制为 0
            r_path = jnp.where(ref_valid, r_path, 0.0)
            r_path_head = jnp.where(ref_valid, r_path_head, 0.0)
            r_path_z = jnp.where(ref_valid, r_path_z, 0.0)
        else:
            r_path_z = 0.0
        # ---- 机身离地净高（r_clear，链 43）----
        # 目标 = 四轮下地形最大值 + 站姿高：前轮骑上台阶顶时地形 max 升到
        # 台阶顶 → 机身逐级抬起（不用看 0.6m 前）→ 后轮有空间跟抬；
        # 后轮还在下一级时 max 保持当前级 → 机身不提前抽高、不失去重心。
        # 与 r_ground（轮-地形贴合）互补：r_ground 管"轮子抬到哪"，
        # r_clear 管"机身跟着抬到哪"。空洞回退 = 当前机身高度（不惩罚）。
        r_clear = 0.0
        if self._config.w_clear > 0.0 and elev is not None:
            h_wheel, ok_wc = sample_grid(
                elev["heightmap"], elev["features"]["valid"],
                elev["origin"], elev["resolution"], wheel_xy,
                fill=-1e6)
            h_max_body = jnp.max(
                jnp.where(ok_wc, h_wheel, -1e6), axis=0)
            z_ref_body = h_max_body + self._config.height_tar
            body_z = d.xpos[self._torso, 2]
            r_clear = -self._config.w_clear * jnp.where(
                ok_wc.any(), jnp.square(body_z - z_ref_body), 0.0)
        return (r_vel + r_ang + r_upright + r_attdamp + r_yaw * 0.5
                + r_height * self._config.height_weight + r_energy * 0.0001
                + r_terrain + r_ground + r_overlift + r_stumble
                + r_wheel_air + r_leg + r_pose + r_path + r_path_head
                + r_progress
                + r_path_z + r_clear + r_ext)

    def reset(self, rng):
        q = jnp.concatenate([
            jnp.array([0.0, 0.0, self._config.base_z_init, 1.0, 0.0, 0.0, 0.0]),
            self._stand_joint,
        ])
        dx = mjx.make_data(self.mx).replace(
            qpos=q, qvel=jnp.zeros(self._nv), ctrl=jnp.zeros(self.action_size))
        info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, self._config.height_tar]),
            "vel_tar": jnp.array([self._config.default_vx,
                                  self._config.default_vy, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, self._config.default_vyaw]),
            "yaw_tar": 0.0,
            "step": 0,
            "wheel_air_t": jnp.zeros(4),
            "contact_lift_t": jnp.zeros(4),
            "stair_t": jnp.zeros(()),
            "pitch_tar": jnp.array(0.0),
            "roll_tar": jnp.array(0.0),
            "ref_path": jnp.zeros((self._config.ref_n, 3)),
            "ref_valid": jnp.array(False),
        }
        ps = self._make_state(dx)
        obs = self._get_obs(ps, info)
        return MpcState(ps, obs, jnp.zeros(()), jnp.zeros(()), info)

    def step(self, state, action):
        d = state.pipeline_state
        ctrl = self.act2tau(action, d)
        dx = mjx.step(self.mx, d.data.replace(ctrl=ctrl))
        ps = self._make_state(dx)
        info = dict(state.info)
        info["step"] = info["step"] + 1
        obs = self._get_obs(ps, info)
        reward = self._reward(ps, info, ctrl)
        up = jnp.array([0.0, 0.0, 1.0])
        rot = ps.xquat[self._torso]
        done = jnp.dot(quat_rotate(rot, up), up) < 0.3
        done |= ps.xpos[self._torso, 2] < 0.06
        done |= jnp.any(ps.qpos[7:] < self.joint_range[:, 0])
        done |= jnp.any(ps.qpos[7:] > self.joint_range[:, 1])
        return MpcState(ps, obs, reward, done.astype(jnp.float32), info)

    def step_rollout(self, state, action):
        """MBDPI 采样 rollout 专用 step：跳过 obs/done 计算（每步省 ~2ms）。"""
        # 预热缓存重构：委托给参数化纯函数（ctx 为 pytree 参数）
        return s10_step_rollout_pure(self._ctx, state, action)
