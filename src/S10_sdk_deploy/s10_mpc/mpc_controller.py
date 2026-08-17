"""S10 轮足 dial-mpc 控制器（嵌入仿真进程）。

职责：
- 读部署 yaml（hardware_profile 决定 Nsample 等、mode 决定目标来源）
- 构建 S10WheeledEnv + MBDPI（dial-mpc 采样 MPC 核心）
- plan_once(): 用真仿真 qpos/qvel 初始化 MPC 状态 → 注入目标指令 → 扩散采样 → 返回当前 action
- action 经 act2tau 转 16 维力矩，供仿真侧施加

用法（仿真节点内）：
    ctrl = MPCController(yaml_path)
    ctrl.init_state(qpos, qvel)
    # 每 dt（50Hz）：
    action = ctrl.plan_once(qpos, qvel, sim_time)
    tau = ctrl.env.act2tau(action, ctrl.state.pipeline_state)  # 或 ctrl.get_tau(action)
"""
import time
import os
from dataclasses import dataclass
from typing import Optional

import yaml
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from dial_mpc.core.dial_core import DialConfig, MBDPI
from dial_mpc.utils.io_utils import load_dataclass_from_dict
from dial_mpc.envs.s10_env import S10WheeledEnv, S10WheeledEnvConfig

# ---- JAX 持久化编译缓存：首次 plan_once 的 JIT（~16s）离线预编译一次，
#      之后每次启动直接从磁盘加载（实测 16.7s → 4.3s，2026-08-06）。
#      根因：mpc_controller import 时 update 太晚（dial_core 的 import 链
#      先初始化 compilation_cache，update 无效）→ 必须用**环境变量**
#      JAX_COMPILATION_CACHE_DIR（jax import 时读取）在启动脚本里设置。
#      这里只在环境变量未设置时兜底 update（可能无效，但无害）。
#      代码改动后首次运行会自动重建缓存。----
_JAX_CACHE_DIR = (
    os.environ.get("JAX_COMPILATION_CACHE_DIR")
    or os.environ.get("S10_JAX_CACHE_DIR")
    or os.path.expanduser("~/.cache/s10_dial_mpc"))
_JAX_CACHE_ENABLED = False
if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    try:
        jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", 0.0)
        _JAX_CACHE_ENABLED = True
    except Exception:
        _JAX_CACHE_ENABLED = False

# hardware_profile → dial-mpc 参数覆盖
HARDWARE_PROFILES = {
    "desktop_4090": dict(Nsample=512, Hsample=14, Hnode=4, Ndiffuse=1,  # P1-5b: 1024->512 提频余量；需质量可 S10_MPC_NSAMPLE=1024/2048
                         dt=0.02, jax_platform="cuda"),
    "orin_agx": dict(Nsample=1024, Hsample=10, Hnode=4, Ndiffuse=1,
                     dt=0.025, jax_platform="cpu"),
}


class MPCController:
    def __init__(self, yaml_path: str):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        hw = cfg.get("hardware_profile", "desktop_4090")
        hw_ov = HARDWARE_PROFILES.get(hw, HARDWARE_PROFILES["desktop_4090"])
        # yaml hardware 节优先（单源配置；代码 HARDWARE_PROFILES 为兜底默认）
        yaml_hw = (cfg.get("hardware") or {}).get(hw) or {}
        hw_ov = {**hw_ov, **yaml_hw}
        # 合并 dial-mpc 参数（yaml 顶层 + hardware 覆盖）
        dial_kw = dict(Nsample=1024, Hsample=25, Hnode=5, Ndiffuse=4,
                       Ndiffuse_init=10, temp_sample=0.05,
                       horizon_diffuse_factor=1.0, traj_diffuse_factor=0.5,
                       update_method="mppi", sigma_scale=1.0)
        for k in ("Nsample", "Hsample", "Hnode", "Ndiffuse", "Ndiffuse_init",
                  "temp_sample", "horizon_diffuse_factor",
                  "traj_diffuse_factor", "update_method", "sigma_scale"):
            if k in cfg:
                dial_kw[k] = cfg[k]
        dial_kw.update({k: v for k, v in hw_ov.items()
                        if k in dial_kw})  # hardware_profile 档位优先（yaml 顶层为示例默认）
        # 环境变量硬覆盖（最高优先级；不写 yaml 即可临时切换）
        if os.environ.get("S10_MPC_NDIFFUSE"):
            dial_kw["Ndiffuse"] = int(os.environ["S10_MPC_NDIFFUSE"])
        if os.environ.get("S10_MPC_NSAMPLE"):
            dial_kw["Nsample"] = int(os.environ["S10_MPC_NSAMPLE"])
        if os.environ.get("S10_MPC_HSAMPLE"):
            dial_kw["Hsample"] = int(os.environ["S10_MPC_HSAMPLE"])
        if os.environ.get("S10_MPC_TEMP"):
            dial_kw["temp_sample"] = float(os.environ["S10_MPC_TEMP"])
        if os.environ.get("S10_MPC_HORIZON_DF"):
            dial_kw["horizon_diffuse_factor"] = float(
                os.environ["S10_MPC_HORIZON_DF"])
        if os.environ.get("S10_MPC_TRAJ_DF"):
            dial_kw["traj_diffuse_factor"] = float(
                os.environ["S10_MPC_TRAJ_DF"])
        # 定向增大抬腿维度采样噪声（0806 §3.5）：腿 12 维 × leg_scale、轮 4 维 × wheel_scale。
        # S10 动作布局固定为 12 腿 + 4 轮；默认 1.0（各向同性），实验用
        # S10_LEG_SIGMA_SCALE=1.5~2.0 提高采样搜到抬腿轨迹的概率。
        leg_sigma = float(os.environ.get("S10_LEG_SIGMA_SCALE", "1.0"))
        wheel_sigma = float(os.environ.get("S10_WHEEL_SIGMA_SCALE", "1.0"))
        dial_kw["sigma_dim"] = [leg_sigma] * 12 + [wheel_sigma] * 4
        # 顶缘阶段 σ 缩放（2026-08-07）：STAIR 到顶时缩小腿采样方差，
        # 抑制顶缘"腿打直 + 轮速震荡 + 侧倾"（wp7 顶缘停滞/侧翻）。
        self._top_sigma_scale = float(os.environ.get(
            "S10_STAIR_TOP_SIGMA", "1.0"))
        self._base_sigma_dim = None
        # v200: DBaS 自适应采样方差（arXiv 2502.14387，默认关，A/B 测试）
        # Se = mu*ln(e + C_B(X*))：标称轨迹代价高（卡台阶）-> 放大探索；
        # 代价低（顺利通行）-> 收敛精细跟踪。每 plan 后用上轮样本奖励统计
        # 更新下一次 reverse_once 的 sigma_dim（host numpy，零 JAX 开销）。
        self._ada_enabled = os.environ.get("S10_ADA_VAR", "0") == "1"
        self._ada_mu = float(os.environ.get("S10_ADA_MU", "1.0"))
        self._ada_ref = float(os.environ.get("S10_ADA_REF", "3000.0"))
        self._ada_cscale = float(os.environ.get("S10_ADA_CSCALE", "1500.0"))
        self._ada_ema = float(os.environ.get("S10_ADA_EMA", "0.3"))
        self._ada_max = float(os.environ.get("S10_ADA_MAX", "3.0"))
        self._ada_leg_only = os.environ.get("S10_ADA_LEG_ONLY", "1") == "1"
        self._ada_stair_only = os.environ.get("S10_ADA_STAIR_ONLY", "1") == "1"
        self._ada_se = None
        self._sigma_dim_base = None
        self._mode = None
        # v200e: DBaS 信号驱动 bias 放大（均值位移而非噪声）：卡住打滑时
        # 把抬腿先验 blend 从 0.30 提高到 S10_STAIR_BIAS_BLEND_STUCK，
        # 让采样均值更贴近"抬腿-蹬"动作（cost 仍可覆盖，软先验非门控）。
        self._ada_slip = 0.0
        self._ada_slip_active = False
        self._ada_bias_on = os.environ.get("S10_ADA_BIAS", "0") == "1"
        self._ada_bias_stuck = float(os.environ.get(
            "S10_STAIR_BIAS_BLEND_STUCK", "0.65"))
        # v209: MPOPI 式精英协方差自适应（arXiv 2508.11917，默认关）：
        # 每 plan 取 top-K 精英样本动作，算每维均值/方差 → 更新 sigma_dim
        #（与基线混合防塌缩）+ 精英均值偏置注入 Y——数据驱动"学习好样本
        # 分布"，让抬左后轮等动作随迭代自然涌现。
        self._elite_ada = os.environ.get("S10_ELITE_ADA", "0") == "1"
        self._elite_frac = float(os.environ.get("S10_ELITE_FRAC", "0.15"))
        self._elite_alpha = float(os.environ.get("S10_ELITE_ALPHA", "0.5"))
        self._elite_bias_blend = float(os.environ.get(
            "S10_ELITE_BIAS_BLEND", "0.25"))
        self._elite_sigma = None
        self._elite_bias = None
        self.dt = float(os.environ.get(
            "S10_MPC_DT",
            str(hw_ov.get("dt", cfg.get("dt", 0.02)))))   # MPC 控制周期（dt 属 env，非 DialConfig）
        dial_kw["env_name"] = cfg.get("env_name", "s10_wheeled")
        dial_kw["n_steps"] = int(cfg.get("n_steps", 100000))
        self.dial_config = DialConfig(**dial_kw)

        # env 配置（yaml + hardware）
        env_kw = dict(kp=float(os.environ.get(
            "S10_MPC_KP", str(cfg.get("kp", 80.0)))),
            kd=float(os.environ.get(
                "S10_MPC_KD", str(cfg.get("kd", 2.0)))),
                      leg_action_scale=float(os.environ.get(
                          "S10_LEG_ACTION_SCALE",
                          str(cfg.get("leg_action_scale", 0.25)))),
                      leg_damping=cfg.get("leg_damping", 0.5),
                      wheel_damping=cfg.get("wheel_damping", 0.05),
                      vel_scale=float(os.environ.get(
                          "S10_MPC_VEL_SCALE",
                          str(cfg.get("vel_scale", 50.0)))),
                      kd_wheel=cfg.get("kd_wheel", 0.3),
                      wheel_tau_scale=cfg.get("wheel_tau_scale", 14.0),
                      ang_vel_weight=float(os.environ.get(
                          "S10_MPC_ANG_W",
                          str(cfg.get("ang_vel_weight", 10.0)))),
                      vel_weight=float(os.environ.get(
                          "S10_MPC_VEL_W",
                          str(cfg.get("vel_weight", 25.0)))),
                      height_tar=cfg.get("height_tar", 0.20),
                      base_z_init=cfg.get("base_z_init", 0.20),
                      height_weight=float(os.environ.get(
                          "S10_MPC_HEIGHT_WEIGHT",
                          str(cfg.get("height_weight", 0.1)))),
                      height_lookahead=float(os.environ.get(
                          "S10_HEIGHT_LOOKAHEAD",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("height_lookahead", 0.35)))),
                      height_lift_cap=float(os.environ.get(
                          "S10_HEIGHT_LIFT_CAP",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("height_lift_cap", 0.15)))),
                      terrain_w_slope=float(os.environ.get(
                          "S10_TERRAIN_W_SLOPE",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("slope", 2.0)))),
                      terrain_w_rough=float(os.environ.get(
                          "S10_TERRAIN_W_ROUGH",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("roughness", 1.0)))),
                      terrain_w_step=float(os.environ.get(
                          "S10_TERRAIN_W_STEP",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("step", 5.0)))),
                      terrain_w_ground=float(os.environ.get(
                          "S10_TERRAIN_W_GROUND",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("ground", 120.0)))),
                      terrain_w_overlift=float(os.environ.get(
                          "S10_TERRAIN_W_OVERLIFT",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("overlift", 200.0)))),
                      terrain_w_leg=float(os.environ.get(
                          "S10_TERRAIN_W_LEG",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("leg", 1.0)))),
                      terrain_w_upright=float(os.environ.get(
                          "S10_TERRAIN_W_UPRIGHT",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("upright", 25.0)))),
                      terrain_w_attdamp=float(os.environ.get(
                          "S10_TERRAIN_W_ATTDAMP",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("attdamp", 0.8)))),
                      # 前瞻抬轮 + 撞阶（0806 §2.4/§3.2）：yaml perception 段 + env 覆盖
                      terrain_w_stumble=float(os.environ.get(
                          "S10_TERRAIN_W_STUMBLE",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("stumble", 0.5)))),
                      leg_relax_on_step=float(os.environ.get(
                          "S10_LEG_RELAX_STEP",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("leg_relax_on_step", 0.2)))),
                      lift_lookahead=float(os.environ.get(
                          "S10_LIFT_LOOKAHEAD",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("lookahead", 0.4)))),
                      lift_max=float(os.environ.get(
                          "S10_LIFT_MAX",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("max_lift", 0.15)))),
                      lift_threshold=float(os.environ.get(
                          "S10_LIFT_THRESHOLD",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("threshold", 0.05)))),
                      rear_follow_thresh=float(os.environ.get(
                          "S10_REAR_FOLLOW_THRESH", "0.10")),
                      lift_step_gate=float(os.environ.get(
                          "S10_LIFT_STEP_GATE",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("step_gate", 0.3)))),
                      lift_steep_gate=float(os.environ.get(
                          "S10_LIFT_STEEP_GATE",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("steep_gate", 0.6)))),
                      contact_lift_ratio=float(os.environ.get(
                          "S10_CONTACT_LIFT_RATIO",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("contact_lift_ratio", 2.0)))),
                      stumble_ratio=float(os.environ.get(
                          "S10_STUMBLE_RATIO",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("stumble_ratio", 4.0)))),
                      terrain_w_wheel_air=float(os.environ.get(
                          "S10_TERRAIN_W_WHEEL_AIR",
                          str(((cfg.get("perception") or {})
                               .get("cost_weights") or {}).get("wheel_air", 15.0)))),
                      lean_leg_w=float(os.environ.get(
                          "S10_LEAN_LEG_W", "0.0")),
                      wheel_ref_force=float(os.environ.get(
                          "S10_WHEEL_REF_FORCE",
                          str(((cfg.get("perception") or {})
                               .get("lift") or {}).get("wheel_ref_force", 20.0)))),
                      # E3：地形自适应姿态目标（上坡仰头/下坡低头/过弯压弯）
                      pose_w_pitch=float(os.environ.get(
                          "S10_MPC_POSE_W_PITCH",
                          str(cfg.get("pose_w_pitch", 0.0)))),
                      pose_w_roll=float(os.environ.get(
                          "S10_MPC_POSE_W_ROLL",
                          str(cfg.get("pose_w_roll", 0.0)))),
                      pose_lookahead=float(os.environ.get(
                          "S10_MPC_POSE_LOOKAHEAD",
                          str(cfg.get("pose_lookahead", 0.4)))),
                      pose_roll_gain=float(os.environ.get(
                          "S10_MPC_POSE_ROLL_GAIN",
                          str(cfg.get("pose_roll_gain", 0.06)))),
                      pose_roll_max=float(os.environ.get(
                          "S10_MPC_POSE_ROLL_MAX",
                          str(cfg.get("pose_roll_max", 0.25)))),
                      lift_rear=os.environ.get(
                          "S10_LIFT_REAR", "0") == "1",
                      rear_lift_scale=float(os.environ.get(
                          "S10_REAR_LIFT_SCALE", "1.0")),
                      rear_lift_zgate=float(os.environ.get(
                          "S10_REAR_LIFT_ZGATE", "0.0")),
                      # 顶缘阶段软调制（2026-08-07，wp7 顶缘停滞修复）
                      top_z=float(os.environ.get(
                          "S10_STAIR_TOP_Z", "1.05")),
                      top_ext_scale=float(os.environ.get(
                          "S10_STAIR_TOP_EXT", "1.0")),
                      top_clear_scale=float(os.environ.get(
                          "S10_STAIR_TOP_CLEAR", "1.0")),
                      top_pathz_scale=float(os.environ.get(
                          "S10_STAIR_TOP_PATHZ", "1.0")),
                      top_push_scale=float(os.environ.get(
                          "S10_STAIR_TOP_PUSH", "1.0")),
                      top_upright_scale=float(os.environ.get(
                          "S10_STAIR_TOP_UPRIGHT", "1.0")),
                      top_attdamp_scale=float(os.environ.get(
                          "S10_STAIR_TOP_ATTDAMP", "1.0")),
                      top_lockpush_scale=float(os.environ.get(
                          "S10_STAIR_TOP_LOCKPUSH", "1.0")),
                      # E4：参考路径跟踪（world 系路径点，固定 (REF_N,2)）
                      ref_n=int(os.environ.get(
                          "S10_MPC_REF_N", str(cfg.get("ref_n", 10)))),
                      w_path=float(os.environ.get(
                          "S10_MPC_W_PATH",
                          str(cfg.get("w_path", 0.0)))),
                      w_path_head=float(os.environ.get(
                          "S10_MPC_W_PATH_HEAD",
                          str(cfg.get("w_path_head", 0.0)))),
                      w_path_z=float(os.environ.get(
                          "S10_MPC_W_PATH_Z",
                          str(cfg.get("w_path_z", 0.0)))),
                      w_prog=float(os.environ.get(
                          "S10_MPC_W_PROG",
                          str(cfg.get("w_prog", 0.0)))),
                      w_clear=float(os.environ.get(
                          "S10_MPC_W_CLEAR",
                          str(cfg.get("w_clear", 0.0)))),
                      leg_ext_w=float(os.environ.get(
                          "S10_LEG_EXT_W",
                          str(cfg.get("leg_ext_w", 0.0)))),
                      sync_front_ext=float(os.environ.get(
                          "S10_SYNC_FRONT_EXT", "1.0")),
                      lift_clear_margin=float(os.environ.get(
                          "S10_LIFT_CLEAR_MARGIN", "0.05")),
                      leg_hipy_scale=float(os.environ.get(
                          "S10_LEG_HIPY_SCALE", "1.0")),
                      stair_pitch_w=float(os.environ.get(
                          "S10_STAIR_PITCH_W", "0.0")),
                      stair_pitch_tar=float(os.environ.get(
                          "S10_STAIR_PITCH_TAR", "-0.45")),
                      lockpush_w=float(os.environ.get(
                          "S10_LOCKPUSH_W",
                          str(cfg.get("lockpush_w", 0.0)))),
                      stair_wheel_brake_w=float(os.environ.get(
                          "S10_STAIR_WHEEL_BRAKE_W", "0.0")),
                      w_foothold=float(os.environ.get(
                          "S10_STAIR_W_FOOTHOLD", "0.0")),
                      w_lift_prog=float(os.environ.get(
                          "S10_STAIR_W_LIFT_PROG", "0.0")),
                      w_roll_level=float(os.environ.get(
                          "S10_STAIR_W_ROLL_LEVEL", "0.0")),
                      w_pitch_cap=float(os.environ.get(
                          "S10_STAIR_W_PITCH_CAP", "0.0")),
                      pitch_cap_rad=float(os.environ.get(
                          "S10_PITCH_CAP_RAD", "0.50")),
                      lift_pose_fl_hipy=float(os.environ.get(
                          "S10_LIFT_POSE_FL_HIPY", "1.00")),
                      lift_pose_fl_knee=float(os.environ.get(
                          "S10_LIFT_POSE_FL_KNEE", "1.50")),
                      lift_pose_hl_hipy=float(os.environ.get(
                          "S10_LIFT_POSE_HL_HIPY", "1.80")),
                      lift_pose_hl_knee=float(os.environ.get(
                          "S10_LIFT_POSE_HL_KNEE", "-1.40")),
                      lift_pose_hr_hipy=float(os.environ.get(
                          "S10_LIFT_POSE_HR_HIPY", "1.50")),
                      lift_pose_hr_knee=float(os.environ.get(
                          "S10_LIFT_POSE_HR_KNEE", "-1.80")),
                      swing_prox=float(os.environ.get(
                          "S10_SWING_PROX", "1e9")),
                      ext_hl_boost=float(os.environ.get(
                          "S10_EXT_HL_BOOST", "1.0")),
                      overlift_band=float(os.environ.get(
                          "S10_OVERLIFT_BAND", "0.05")),
                      w_pitch_rate_cap=float(os.environ.get(
                          "S10_W_PITCH_RATE_CAP", "0.0")),
                      pitch_rate_cap=float(os.environ.get(
                          "S10_PITCH_RATE_CAP", "0.35")),
                      stair_wheel_lock_w=float(os.environ.get(
                          "S10_STAIR_WHEEL_LOCK_W", "0.0")),
                      w_support=float(os.environ.get(
                          "S10_STAIR_W_SUPPORT", "0.0")),
                      support_margin=float(os.environ.get(
                          "S10_SUPPORT_MARGIN", "0.06")),
                      support_fz_min=float(os.environ.get(
                          "S10_SUPPORT_FZ_MIN", "20.0")),
                      support_exclude_lift=float(os.environ.get(
                          "S10_SUPPORT_EXCLUDE_LIFT", "0.0")),
                      swing_thresh=float(os.environ.get(
                          "S10_SWING_THRESH", "0.04")),
                      left_boost=float(os.environ.get(
                          "S10_LEFT_BOOST", "1.0")),
                      w_wheel_ref=float(os.environ.get(
                          "S10_WHEEL_REF_W",
                          str(cfg.get("w_wheel_ref", 0.0)))),
                      solver_iterations=int(os.environ.get(
                          "S10_MPC_SOLVER_IT",
                          str(cfg.get("solver_iterations", 6)))),
                      solver_ls_iterations=int(os.environ.get(
                          "S10_MPC_SOLVER_IT",
                          str(cfg.get("solver_iterations", 6)))),
                      dt=self.dt, timestep=self.dt)
        self.env_config = S10WheeledEnvConfig(**env_kw)

        print(f"[MPC] JAX 编译缓存: {_JAX_CACHE_DIR} "
              f"(enabled={_JAX_CACHE_ENABLED})")
        print(f"[MPC] 构建 env (dt={self.env_config.dt}) ...")
        self.env = S10WheeledEnv(self.env_config)
        print(f"[MPC] 构建 MBDPI (Nsample={self.dial_config.Nsample}, "
              f"Hsample={self.dial_config.Hsample}) ...")
        # 双视界 MBDPI（用户方案模式化 H）：CRUISE H=14（0.28s 横脊动量、
        # chain 44 3/3 验证）、STAIR H=20（0.4s 长视界爬梯）。Hnode 相同
        # （4）→ Y 状态 (5,16) 可直接复用，切换无重映射。
        _h_cruise = int(os.environ.get("S10_MPC_H_CRUISE", "20"))
        _h_stair = int(os.environ.get("S10_MPC_H_STAIR", "20"))
        import dataclasses as _dc
        # v217: 巡航/台阶允许不同采样规模（S10_MPC_NSAMPLE_CRUISE/STAIR，
        # 默认沿用全局 S10_MPC_NSAMPLE）：巡航小 N 提频，台阶保留大 N 稳定。
        _n_cruise = int(os.environ.get(
            "S10_MPC_NSAMPLE_CRUISE", "0")) or self.dial_config.Nsample
        _n_stair = int(os.environ.get(
            "S10_MPC_NSAMPLE_STAIR", "0")) or self.dial_config.Nsample
        _cfg14 = _dc.replace(self.dial_config, Hsample=_h_cruise,
                             Nsample=_n_cruise)
        _cfg20 = _dc.replace(self.dial_config, Hsample=_h_stair,
                             Nsample=_n_stair)
        self.mbdpi_h14 = MBDPI(_cfg14, self.env, ctx=self.env._ctx)
        self.mbdpi_h20 = MBDPI(_cfg20, self.env, ctx=self.env._ctx)
        self.mbdpi = self.mbdpi_h14
        self.rng = jax.random.PRNGKey(seed=self.dial_config.seed)

        self.Y = jnp.zeros([self.dial_config.Hnode + 1, self.mbdpi.nu])
        self.state = None
        self._last_plan_t = -1.0
        self._first = True
        self._last_vx = 0.0
        self._last_vyaw = 0.0
        # 前进轮速前馈斜率限制状态：起步渐进防翘头；刹车快速回落；差速转向不斜坡
        self._ff_fwd = 0.0
        # yaw 前馈增益覆盖：自动导航用 1:1 增益（15）避免反馈过冲；
        # 遥控保留大增益（50）支持超快原地转。None = 用环境变量/默认。
        self._yaw_gain_lo_override = None
        # 感知-voxel 世界对齐高程瓦片（默认空瓦片：valid=False → 地形代价恒 0，
        # 保证 info 结构固定、首次 trace 后不 retrace；set_elevation_map 只换数值）
        _n = 60
        _zero = np.zeros((_n, _n), dtype=np.float32)
        _zero_z = np.zeros((_n, _n), dtype=np.float32)
        _invalid = np.zeros((_n, _n), dtype=np.bool_)
        self._elev_np = {
            "heightmap": _zero_z,
            "features": {
                "valid": _invalid,
                "slope": _zero,
                "roughness": _zero,
                "step": _zero,
                "step_flag": _zero,
                # v162：轮心 z 参考场（默认无效，结构固定防 retrace）
                "wheel_ref": _zero_z,
                "wheel_ref_valid": _invalid.copy(),
                # v206：落脚点前拉场（foothold planning 软落地）
                "foothold_y": _zero_z,
                "foothold_valid": _invalid.copy(),
            },
            "origin": np.zeros(2, dtype=np.float32),
            "resolution": 0.1,
        }
        # elevation jnp 缓存（v184）：瓦片 8Hz 更新但 plan 12Hz+，只有
        # set_elevation_map 后 _elev_np 才变，版本号不变则复用 jnp 转换。
        self._elev_jnp_cache = None
        self._elev_np_version = 0
        # Obstacle distance field for DIAL-MPC cost (wall avoidance, cruise only).
        # Fixed 80x80 at 0.2 m resolution -> 16 m local window; far = dmax means no penalty.
        self._obst_np = np.full((80, 80), 2.0, dtype=np.float32)
        self._obst_origin = np.zeros(2, dtype=np.float32)
        self._obst_res = 0.2
        self._obst_dmax = 2.0

        def _scan_body(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            mbdpi = self.mbdpi
            rng, Y0, info = mbdpi.reverse_once(
                state, rng, Y0, factor, mbdpi.sigma_dim)
            return (rng, Y0, state), info

        self._scan_body = _scan_body

        # 遥控/导航目标指令
        self.cmd_vel = jnp.array([0.0, 0.0, 0.0])
        self.cmd_ang = jnp.array([0.0, 0.0, 0.0])
        # E4：参考路径（世界系，固定 (REF_N,3) = x,y,z；set_ref_path 注入）
        self._ref_path = np.zeros((self.env_config.ref_n, 3), dtype=np.float32)
        # v162：STAIR 已知地图几何剖面覆盖（set_stair_ref 注入，update_state 使用）
        self._stair_ref_set = False
        self._stair_pitch = 0.0
        self._stair_base_z = 0.0
        # v168：场驱动抬腿动作偏置（soft prior, (Hnode+1,12) 或 (12,)）
        self._stair_action_bias = None
        self._ref_valid = False
        # v213: 顺序步态调度摆动标志（节点侧 gait_schedule 逐帧写入，
        # 经 info 注入 rollout cost，0=关/纯 lift-need 启发式）
        self._gait_swing = np.zeros(4, dtype=np.float32)
        # 主线程规划模式：latest_tau 由主循环每步更新；初始化防首帧崩溃
        self.latest_tau = np.zeros(16, dtype=np.float32)
        self.latest_action = np.zeros(16, dtype=np.float32)

    # ---- 状态注入 ----
    def _set_state(self, st, q, qd, t=None):
        """mjx Data 更新 qpos/qvel 并重新包装 MjxLikeState。"""
        d = st.pipeline_state.data.replace(
            qpos=jnp.asarray(q, dtype=jnp.float32),
            qvel=jnp.asarray(qd, dtype=jnp.float32))
        info = dict(st.info)
        if t is not None:
            info["step"] = int(t / self.env_config.dt)
        # v203: x/xd 只依赖 d.xpos/xquat/cvel（注入 qpos/qvel 不改这些字段），
        # 且主控制/奖励路径不使用输入状态的 x/xd（卷动内部重算）。
        # 缓存一次后复用，去掉每轮 _make_state 的 ~11ms JAX 组合开销
        # （升频率 13.5Hz -> 15Hz+）。
        kin = getattr(self, "_state_kin_cache", None)
        if kin is None:
            made = self.env._make_state(d)
            self._state_kin_cache = (made.x, made.xd)
            return st.replace(pipeline_state=made, info=info)
        from dial_mpc.envs.s10_env import MjxLikeState
        x, xd = kin
        return st.replace(
            pipeline_state=MjxLikeState(data=d, x=x, xd=xd), info=info)


    def init_state(self, q: np.ndarray, qd: np.ndarray):
        """用真仿真初始状态初始化 MPC state。"""
        st = self.env.reset(jax.random.PRNGKey(0))
        self.state = self._set_state(st, q, qd, t=0.0)
        self.Y = jnp.zeros([self.dial_config.Hnode + 1, self.mbdpi.nu])
        self._ff_fwd = 0.0

    def set_elevation_map(self, elev: dict):
        """注入感知层世界对齐高程瓦片（get_local_map() 输出，8Hz 更新）。
        仅存 numpy（线程安全，无 JAX dispatch）；update_state 在主线程转 jnp。
        固定形状 (60,60) + origin(2,) + res，仅替换数值，不触发 retrace。"""
        if elev is None:
            return
        # 契约桥接：local_map.get_tile() 的 valid 在瓦片顶层、features 内无 valid，
        # 而 elevation_lookup.terrain_cost 期望 features["valid"]。
        f = dict(elev["features"])
        f["valid"] = elev["valid"]
        self._elev_np = {
            "heightmap": np.asarray(elev["heightmap"], dtype=np.float32),
            "features": {
                "valid": np.asarray(f["valid"], dtype=np.bool_),
                "slope": np.asarray(f["slope"], dtype=np.float32),
                "roughness": np.asarray(f["roughness"], dtype=np.float32),
                "step": np.asarray(f["step"], dtype=np.float32),
                "step_flag": np.asarray(f["step_flag"], dtype=np.float32),
                # v162：已知地图轮心 z 参考场（楼梯段，0=区外，配合 wheel_ref_valid）
                "wheel_ref": (np.asarray(f["wheel_ref"], dtype=np.float32)
                              if f.get("wheel_ref") is not None
                              else np.zeros_like(f["slope"], dtype=np.float32)),
                "wheel_ref_valid": (np.asarray(f["wheel_ref_valid"], dtype=np.bool_)
                                    if f.get("wheel_ref_valid") is not None
                                    else np.zeros_like(f["valid"], dtype=np.bool_)),
                "foothold_y": (np.asarray(f["foothold_y"], dtype=np.float32)
                               if f.get("foothold_y") is not None
                               else np.zeros_like(f["slope"], dtype=np.float32)),
                "foothold_valid": (np.asarray(f["foothold_valid"], dtype=np.bool_)
                                   if f.get("foothold_valid") is not None
                                   else np.zeros_like(f["valid"], dtype=np.bool_)),
            },
            "origin": np.asarray(elev["origin"], dtype=np.float32),
            "resolution": float(elev["resolution"]),
        }
        self._elev_np_version += 1
        self._elev_jnp_cache = None

    def set_obstacle_costmap(self, cmap):
        """Inject the wall/obstacle 2D distance field for DIAL-MPC cost.

        cmap: CostMap2D or None. Fixed 80x80 at 0.2 m resolution; when None the
        grid is filled with dmax (far), which yields zero obstacle penalty (same
        fixed shape so the JAX trace does not change between no wall and wall).
        """
        if cmap is None:
            self._obst_np[:] = self._obst_dmax
            return
        d = np.asarray(cmap.d, dtype=np.float32)
        if d.shape != self._obst_np.shape:
            self._obst_np[:] = self._obst_dmax
            h = min(d.shape[0], self._obst_np.shape[0])
            w = min(d.shape[1], self._obst_np.shape[1])
            self._obst_np[:h, :w] = d[:h, :w]
        else:
            self._obst_np[:] = d
        self._obst_origin = np.asarray(cmap.origin, dtype=np.float32)
        self._obst_res = float(cmap.res)
        self._obst_dmax = float(cmap.dmax)

    def set_stair_action_bias(self, bias):
        # v168: 注入场驱动抬腿动作偏置（numpy (H+1,12) 或 (12,)，动作空间）。
        # 软先验：仅把采样均值推向"抬腿"方向，MPPI 权重/rollout cost 可覆盖。
        if bias is None:
            self._stair_action_bias = None
            return
        b = np.asarray(bias, dtype=np.float32)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        self._stair_action_bias = np.clip(b, -1.0, 1.0)

    def set_stair_ref(self, pitch_tar: float, base_z: float):
        # v162: STAIR known-map geometric profile override
        # pitch_tar negative = nose-up (project convention), base_z = world z
        self._stair_ref_set = True
        self._stair_pitch = float(pitch_tar)
        self._stair_base_z = float(base_z)

    def clear_stair_ref(self):
        self._stair_ref_set = False

    def set_ref_path(self, pts, valid=True):
        """注入参考路径（世界系 (N,2) 或 (N,3)），固定形状填充/截断到 REF_N。"""
        pts = np.asarray(pts, dtype=np.float32)
        if pts.size == 0 or not valid:
            self._ref_path[:] = 0.0
            self._ref_valid = False
            return
        pts = pts.reshape(-1, 2) if pts.shape[-1] == 2 else pts.reshape(-1, 3)
        if pts.shape[1] == 2:
            p3 = np.zeros((pts.shape[0], 3), dtype=np.float32)
            p3[:, :2] = pts
            pts = p3
        if pts.shape[0] >= self.env_config.ref_n:
            self._ref_path[:] = pts[:self.env_config.ref_n]
        else:
            self._ref_path[:] = 0.0
            self._ref_path[:pts.shape[0]] = pts
            # 末尾补齐为最后一个有效点（让"最近点"距离不为零噪声）
            self._ref_path[pts.shape[0]:] = pts[-1]
        self._ref_valid = True

    def _elev_jnp(self):
        """把 numpy 瓦片转 jnp（仅在主线程 plan_once 路径调用，规避并发 dispatch）。

        v184：结果按 _elev_np_version 缓存——set_elevation_map（8Hz）后才重建，
        plan_once（12Hz+）中间调用直接返回缓存，省掉每次 ~20ms 的 numpy→jnp
        转换/设备传输（H=25 总耗时 85ms→~65ms，实际频率 12Hz→~15Hz）。"""
        if self._elev_jnp_cache is not None:
            return self._elev_jnp_cache
        f = self._elev_np["features"]
        f = dict(f)
        f["valid"] = np.asarray(f["valid"], dtype=np.bool_)
        cache = {
            "heightmap": jnp.asarray(self._elev_np["heightmap"]),
            "features": {
                "valid": jnp.asarray(f["valid"]),
                "slope": jnp.asarray(f["slope"]),
                "roughness": jnp.asarray(f["roughness"]),
                "step": jnp.asarray(f["step"]),
                "step_flag": jnp.asarray(f["step_flag"]),
                "wheel_ref": jnp.asarray(f.get(
                    "wheel_ref",
                    np.zeros_like(np.asarray(f["slope"]), dtype=np.float32))),
                "wheel_ref_valid": jnp.asarray(f.get(
                    "wheel_ref_valid",
                    np.zeros_like(np.asarray(f["valid"]), dtype=np.bool_))),
                "foothold_y": jnp.asarray(f.get(
                    "foothold_y",
                    np.zeros_like(np.asarray(f["slope"]), dtype=np.float32))),
                "foothold_valid": jnp.asarray(f.get(
                    "foothold_valid",
                    np.zeros_like(np.asarray(f["valid"]), dtype=np.bool_))),
            },
            "origin": jnp.asarray(self._elev_np["origin"]),
            "resolution": self._elev_np["resolution"],
        }
        self._elev_jnp_cache = cache
        return cache

    def _ramped_ff_fwd(self, target):
        """前进轮速前馈斜坡：向上限速 S10_MPC_WHEEL_RAMP（默认 0.25 act/plan，
        0→1.0 约 1s，防起步满矩翘头）；向下 S10_MPC_WHEEL_RAMP_DOWN（默认 0.5，
        刹车响应快）。差速转向分量不斜坡，保持瞬时响应。"""
        ramp_up = float(os.environ.get("S10_MPC_WHEEL_RAMP", "0.25"))
        ramp_down = float(os.environ.get("S10_MPC_WHEEL_RAMP_DOWN", "0.5"))
        cur = self._ff_fwd
        d = float(target) - cur
        d = float(np.clip(d, -ramp_down, ramp_up))
        cur = cur + d
        self._ff_fwd = cur
        return cur

    def update_state(self, q: np.ndarray, qd: np.ndarray, t: float):
        self.state = self._set_state(self.state, q, qd, t)
        # 注入目标指令（遥控/导航共用）
        info = dict(self.state.info)
        info["vel_tar"] = jnp.concatenate([self.cmd_vel, jnp.array([0.0])])[:3]
        info["ang_vel_tar"] = jnp.concatenate([self.cmd_ang, jnp.array([0.0])])[:3]
        info["mode_stair"] = jnp.array(
            1.0 if getattr(self, "_mode", None) == "STAIR" else 0.0,
            dtype=jnp.float32)
        # v213/v214: 步态调度（S10_GAIT=1 固定序列 / S10_GAIT_UTIL=1 utility
        # 选腿）或 STAIR hard-mode 接触规划器（S10_STAIR_HARD_MODE=1，写
        # _gait_swing 为 0/1 轴级摆动）激活时才注入 gait_swing。仅当存在
        # 非零摆动信号时注入，否则省略该键 → rollout 回退纯 lift-need
        # 启发式摆动相（v211 基线行为，防止全零静音摆动/CRUISE 抬脊失效）。
        _gsw = getattr(self, "_gait_swing", np.zeros(4, dtype=np.float32))
        _gsw_active = bool(np.any(np.asarray(_gsw) > 0.5))
        # STAIR：始终注入 hard-mode 相位（含全零=全支撑），避免释放后回退
        # 旧 lift-need 启发式造成"该落地却继续悬空"（wp7 卡点：mode=0 释放后
        # 前轮悬空 15s 不落地）。CRUISE 行为不变（仍回退 lift-need 抬脊）。
        # S10_STAIR_ALWAYS_GWSW=0 可回退旧行为做 A/B。
        _stair_always = (os.environ.get("S10_STAIR_ALWAYS_GWSW", "1") == "1")
        if (os.environ.get("S10_GAIT", "0") == "1"
                or os.environ.get("S10_GAIT_UTIL", "0") == "1"
                or (getattr(self, "_mode", None) == "STAIR"
                    and (_stair_always or _gsw_active))):
            info["gait_swing"] = jnp.asarray(_gsw, dtype=jnp.float32)
        # v215d: 摆动邻近门控（轮距下一 riser 距离，m；默认 1e9=不启用）
        info["stair_prox"] = jnp.asarray(
            getattr(self, "_stair_prox", np.full(4, 1e9, dtype=np.float32)),
            dtype=jnp.float32)
        info["elevation_map"] = self._elev_jnp()
        info["obstacle_d"] = jnp.asarray(self._obst_np)
        info["obstacle_origin"] = jnp.asarray(self._obst_origin)
        info["obstacle_res"] = jnp.array(float(self._obst_res), dtype=jnp.float32)
        info["obstacle_dmax"] = jnp.array(float(self._obst_dmax), dtype=jnp.float32)
        # E3：地形自适应姿态目标（上坡仰头/下坡低头/过弯压弯）
        p_tar, r_tar = self._terrain_pose_targets()
        # v162：STAIR 已知地图几何剖面覆盖（pitch 负=仰头；base_z 世界系）
        if (getattr(self, "_stair_ref_set", False)
                and getattr(self, "_mode", None) == "STAIR"):
            p_tar = self._stair_pitch
        info["pitch_tar"] = jnp.array(p_tar, dtype=jnp.float32)
        info["roll_tar"] = jnp.array(r_tar, dtype=jnp.float32)


        # E4：参考路径
        info["ref_path"] = jnp.asarray(self._ref_path)
        info["ref_valid"] = jnp.array(bool(self._ref_valid))
        # 地形跟随高度目标（0806 §3.6）：目标 = 机下地形 + 站姿高 + clip(前方高差)。
        # 让 r_height 把机身"拉"向即将到达的地形高度，过楼梯/台阶（不依赖抬轮机制）。
        _zt = self._terrain_follow_z()
        if (getattr(self, "_stair_ref_set", False)
                and getattr(self, "_mode", None) == "STAIR"):
            _zt = self._stair_base_z
        info["pos_tar"] = jnp.array([0.0, 0.0, _zt],
                                    dtype=jnp.float32)
        self.state = self.state.replace(info=info)

    def _terrain_pose_targets(self):
        """E3：从高程瓦片算 pitch 目标（前方坡度），从指令算 roll 目标（压弯）。
        返回 (pitch_tar, roll_tar)，numpy 路径（update_state 调用，无 JAX）。"""
        e = self._elev_np
        hm = e["heightmap"]
        valid = e["features"]["valid"]
        ox, oy = float(e["origin"][0]), float(e["origin"][1])
        res = e["resolution"]
        d = self.state.pipeline_state.data
        bx, by = float(d.xpos[1][0]), float(d.xpos[1][1])
        xm = np.asarray(d.xmat[1]).reshape(3, 3)
        fx, fy = float(xm[0, 0]), float(xm[1, 0])   # 前向 = xmat 第一列
        fn = np.hypot(fx, fy) + 1e-9
        fx, fy = fx / fn, fy / fn                   # 水平归一化（抗俯仰缩短）

        def _h(x, y):
            i = int(np.floor((y - oy) / res))
            j = int(np.floor((x - ox) / res))
            if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                    and valid[i, j]):
                return float(hm[i, j])
            return None

        pitch_tar = 0.0
        la = self.env_config.pose_lookahead
        h_a = _h(bx + fx * la, by + fy * la)
        h_b = _h(bx - fx * la, by - fy * la)
        if h_a is not None and h_b is not None:
            pitch_tar = float(np.clip(
                np.arctan2(h_a - h_b, 2.0 * la), -0.4, 0.4))
        vx = float(self.cmd_vel[0])
        vyaw = float(self.cmd_ang[2])
        roll_tar = float(np.clip(
            self.env_config.pose_roll_gain * vyaw * abs(vx),
            -self.env_config.pose_roll_max, self.env_config.pose_roll_max))
        # v214j: 摆动轮联动 roll——HL（左后）摆动时左轮需卸荷抬升，允许
        # 车身右倾（roll 正=左高，v211 实测 roll 负=左轮低）；HR 摆动反向。
        # 软目标偏置（S10_SWING_ROLL 幅度），与 roll_level 回平通过权重平衡。
        _swing_roll = float(os.environ.get("S10_SWING_ROLL", "0.0"))
        if _swing_roll > 0.0:
            _gsw4 = np.asarray(
                getattr(self, "_gait_swing", np.zeros(4, dtype=np.float32)),
                dtype=np.float32)
            _hl = float(_gsw4[2]); _hr = float(_gsw4[3])
            roll_tar = float(np.clip(
                roll_tar + (_hl - _hr) * _swing_roll, -0.35, 0.35))
        # v215: STAIR 恒定 roll 偏置（S10_STAIR_ROLL_BIAS，默认 0）——
        # M10 左后轮 9mm 不对称，3 轮上台后 HL 被压（roll 负=左低）；
        # 正偏置抬高左侧卸载左轮（软目标，与 roll_level 通过权重平衡）。
        # v215n: 自适应欠抬差偏置（sim 节点注入 _stair_roll_override）
        if getattr(self, "_mode", None) == "STAIR":
            _srb = float(os.environ.get("S10_STAIR_ROLL_BIAS", "0.0"))
            _sro = getattr(self, "_stair_roll_override", 0.0)
            if abs(_srb) > 1e-6 or abs(_sro) > 1e-6:
                roll_tar = float(np.clip(
                    roll_tar + _srb + float(_sro), -0.35, 0.35))
        # v199: curve lean lookahead (moto-style). Base roll on path curvature
        # ahead of the robot instead of only the current command, so the body
        # pre-leans before entering the corner. S10_ROLL_FF_DIST=0 disables.
        _rfd = float(os.environ.get("S10_ROLL_FF_DIST", "1.2"))
        if _rfd > 0.0:
            _rc = self._ref_curvature()
            if _rc is not None:
                _cum, _k_s, _pts = _rc
                _k0 = int(np.argmin(np.sum(
                    (_pts - np.array([bx, by])) ** 2, axis=1)))
                _s_ahead = float(_cum[_k0]) + _rfd
                _k_a = float(np.interp(_s_ahead, _cum, _k_s))
                _mix = float(os.environ.get("S10_ROLL_FF_MIX", "0.5"))
                _roll_ff = (self.env_config.pose_roll_gain
                            * abs(vx) * (vx * _k_a))
                roll_tar = float(np.clip(
                    (1.0 - _mix) * roll_tar + _mix * _roll_ff,
                    -self.env_config.pose_roll_max,
                    self.env_config.pose_roll_max))
        # v217v: roll 目标速率限制（S10_POSE_ROLL_RATE>0）——S 弯左右压弯
        # 符号翻转太快时身体惯性跟不上会翻（wp2→3 实测）。平滑过渡。
        _rl = float(os.environ.get("S10_POSE_ROLL_RATE", "0.0"))
        if _rl > 0.0:
            _prev_rt = getattr(self, "_last_roll_tar", 0.0)
            roll_tar = float(np.clip(
                roll_tar, _prev_rt - _rl, _prev_rt + _rl))
        self._last_roll_tar = roll_tar
        return pitch_tar, roll_tar

    def _ref_curvature(self):
        # v199: cumulative arc + signed smoothed curvature from the injected
        # ref path (world frame). Returns (cum, kappa_signed, pts) or None.
        ref = np.asarray(self._ref_path, dtype=np.float64)
        if ref.shape[0] < 8 or not getattr(self, "_ref_valid", False):
            return None
        dxy = np.diff(ref[:, :2], axis=0)
        seg = np.linalg.norm(dxy, axis=1)
        dup = np.argmax(seg < 1e-6) if np.any(seg < 1e-6) else len(seg)
        nv = max(4, int(dup) + 1)
        pts = ref[:nv, :2]
        if pts.shape[0] < 4:
            return None
        dxy = np.diff(pts, axis=0)
        seg = np.maximum(np.linalg.norm(dxy, axis=1), 1e-6)
        heading = np.arctan2(dxy[:, 1], dxy[:, 0])
        dhead = np.unwrap(np.diff(heading))
        kappa = dhead / seg[1:]
        kappa = np.concatenate([[0.0], kappa, [0.0]])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        k_s = np.convolve(kappa, np.ones(3) / 3.0, mode="same")
        return cum, k_s, pts

    def _adaptive_sigma_node(self):
        # v199: MPPI-DBaS style adaptive sampling variance. Per-node sigma
        # multiplier from path curvature: shrink on straights (samples stick
        # to the nominal straight action -> straighter lines), widen before
        # and inside curves (explore turning, paired with early decel+lean).
        # v213: cruise 默认开启（直线 σ 0.5）；S10_ADAPTIVE_SIGMA=0 关闭；
        # STAIR 模式始终不调制（楼梯行为不变）。返回 (Hnode+1,) float32 mults.
        n = self.dial_config.Hnode + 1
        if (getattr(self, "_mode", None) == "STAIR"
                or os.environ.get("S10_ADAPTIVE_SIGMA", "1") == "0"):
            return np.ones(n, dtype=np.float32)
        _rc = self._ref_curvature()
        if _rc is None:
            return np.ones(n, dtype=np.float32)
        cum, k_s, pts = _rc
        k_abs = np.abs(k_s)
        # widen within a radius ahead of each high-curvature point so the
        # sampling starts exploring slightly before the corner
        _rad = float(os.environ.get("S10_ADAPTIVE_SIGMA_RADIUS", "0.8"))
        rw = max(1, int(_rad / max(float(np.median(np.diff(cum))), 1e-3)))
        kk = np.zeros_like(k_abs)
        for k in range(len(k_abs)):
            lo = max(0, k - rw)
            hi = min(len(k_abs), k + rw + 1)
            kk[k] = float(np.max(k_abs[lo:hi]))
        k_th = float(os.environ.get("S10_ADAPTIVE_SIGMA_KAPPA", "0.35"))
        s_straight = float(os.environ.get(
            "S10_ADAPTIVE_SIGMA_STRAIGHT", "0.45"))
        s_curve = float(os.environ.get(
            "S10_ADAPTIVE_SIGMA_CURVE", "1.2"))
        d = self.state.pipeline_state.data
        bx, by = float(d.xpos[1][0]), float(d.xpos[1][1])
        _k0 = int(np.argmin(np.sum(
            (pts - np.array([bx, by])) ** 2, axis=1)))
        vx = max(abs(float(self.cmd_vel[0])), 0.1)
        node_dt = float(self.mbdpi.node_dt)
        out = np.ones(n, dtype=np.float32)
        for j in range(n):
            s_j = float(cum[_k0]) + vx * node_dt * j
            k_j = float(np.interp(s_j, cum, kk))
            t = float(np.clip(
                (k_j - k_th * 0.6) / max(k_th * 0.8, 1e-6), 0.0, 1.0))
            t = t * t * (3.0 - 2.0 * t)
            out[j] = float(s_straight + (s_curve - s_straight) * t)
        return out

    def _terrain_follow_z(self) -> float:
        """从 numpy 高程瓦片算地形跟随目标高度（update_state 调用，无 JAX）。"""
        e = self._elev_np
        hm = e["heightmap"]
        valid = e["features"]["valid"]
        ox, oy = float(e["origin"][0]), float(e["origin"][1])
        res = e["resolution"]
        d = self.state.pipeline_state.data
        bx, by = float(d.xpos[1][0]), float(d.xpos[1][1])
        xm = np.asarray(d.xmat[1]).reshape(3, 3)
        fx, fy = float(xm[0, 0]), float(xm[1, 0])   # 前向 = xmat 第一列
        fn = np.hypot(fx, fy) + 1e-9
        fx, fy = fx / fn, fy / fn                   # 水平归一化（抗俯仰缩短）

        def _h(x, y):
            i = int(np.floor((y - oy) / res))
            j = int(np.floor((x - ox) / res))
            if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                    and valid[i, j]):
                return float(hm[i, j])
            return None

        h_now = _h(bx, by)
        if h_now is None:
            return float(self.env_config.height_tar)
        la = self.env_config.height_lookahead
        h_ahead = _h(bx + fx * la, by + fy * la)
        if h_ahead is None:
            h_ahead = h_now
        lift = float(np.clip(h_ahead - h_now, 0.0,
                             self.env_config.height_lift_cap))
        z_tar = h_now + self.env_config.height_tar + lift
        # 转弯蹲低（2026-08-08，用户"不再锁死蹲姿"）：转向指令大时降低
        # 目标高度 → 降低重心 → 提高弯道侧翻极限（摩托压弯压低身体）。
        # S10_TURN_CROUCH 每 rad/s 蹲低米数，S10_TURN_CROUCH_MAX 上限。
        _ck = float(os.environ.get("S10_TURN_CROUCH", "0.0"))
        if _ck > 0.0:
            _vyaw = float(np.asarray(self.cmd_ang)[2])
            _crouch = min(abs(_vyaw) * _ck,
                          float(os.environ.get("S10_TURN_CROUCH_MAX", "0.10")))
            z_tar -= _crouch
        return z_tar

    def set_cmd(self, vx: float, vy: float, vyaw: float):
        """遥控：目标线速度 (vx,vy) 与偏航角速度 vyaw。"""
        self.cmd_vel = jnp.array([vx, vy, 0.0])
        self.cmd_ang = jnp.array([0.0, 0.0, vyaw])
        # 命令阶跃时把轮速度解预热到前馈值（vx 前进 + vyaw 差速转向）：
        #   act_vx = vx / (vel_scale * r_wheel)
        #   act_ya = vyaw * half_track / (vel_scale * r_wheel)
        # 轮 action[12:] = [fl, fr, hl, hr]；左轮(+y)减差速、右轮加差速
        ff_vx = float(np.clip(
            vx / (self.env_config.vel_scale * 0.081), -1.0, 1.0))
        # yaw 前馈放大：物理公式给的是稳态轮速差，但轮矩滞后导致实测转向
        # 低于理论值。放大增益补偿滞后。
        # 速度自适应增益（竞速）：低速/原地转向用大增益（实测 4.5 rad/s），
        # 高速时减小（实测高速大差速会触发 MPC 防侧翻拒转 + 物理 LTR 极限），
        # 以 vx_cmd 线性过渡：0.5m/s→40，3m/s→15。
        yaw_gain_lo = float(
            self._yaw_gain_lo_override
            if self._yaw_gain_lo_override is not None
            else os.environ.get("S10_MPC_YAW_FF_GAIN", "50.0"))
        yaw_gain_hi = float(os.environ.get("S10_MPC_YAW_FF_GAIN_HI_SPD", "15.0"))
        vx_cmd = abs(float(vx))
        if vx_cmd >= 3.0:
            yaw_gain = yaw_gain_hi
        elif vx_cmd > 0.5:
            t = (vx_cmd - 0.5) / 2.5
            yaw_gain = yaw_gain_lo * (1.0 - t) + yaw_gain_hi * t
        else:
            yaw_gain = yaw_gain_lo
        ff_yaw = float(np.clip(
            yaw_gain * vyaw * 0.05 / (self.env_config.vel_scale * 0.081),
            -1.0, 1.0))
        if (self.state is not None
                and (abs(vx - self._last_vx) > 0.1
                     or abs(vyaw - self._last_vyaw) > 0.1)):
            # 回退到 Mode B e2e 实测有效的差速符号（act_w=[-0.37,0.37] → CCW）
            ff_vx_r = self._ramped_ff_fwd(ff_vx)
            self.Y = self.Y.at[:, 12:].set(
                jnp.array([ff_vx_r - ff_yaw, ff_vx_r + ff_yaw,
                           ff_vx_r - ff_yaw, ff_vx_r + ff_yaw],
                          dtype=jnp.float32))
        self._last_vx = float(vx)
        self._last_vyaw = float(vyaw)

    def set_mode(self, mode: str):
        """CRUISE / STAIR_SEQUENCE 双模式 reward 权重切换（用户方案 2.3/2.4）。

        ctx["cfg"] 是 jit 动态输入（参数化后），改值不 retrace——reward 权重
        随模式切换：CRUISE 防趴低+轻蹲姿、STAIR 放开屈膝+抬腿引导+做功奖励。
        """
        if mode == getattr(self, "_mode", None):
            return
        self._mode = mode
        # 模式化视界（用户方案）：STAIR 用长视界 MBDPI（H=20），
        # CRUISE 用短视界（H=14）。Hnode 相同，Y 状态直接复用。
        self.mbdpi = (self.mbdpi_h20 if mode == "STAIR"
                      else self.mbdpi_h14)
        cfg = self.env._ctx["cfg"]
        # 恒定项（chain 64 验证基线，两种模式共用）：轮-地形贴合 300、
        # 机身净高 60——避免 CRUISE 退化（chain 68 r1 东漂复现：
        # ground 120/clear 10 削弱横脊通过能力）。
        cfg["terrain_w_ground"] = 300.0
        cfg["w_clear"] = 60.0
        # 模式化采样 σ（用户方案 6"退火 z 向关节 sigma 放大"）：STAIR 腿维
        # 噪声放大让 MPC 探索到大幅抬腿；CRUISE 小 σ 保平地/横脊稳定。
        # sigma_dim 已参数化（reverse_once 动态参数），切换无 retrace。
        # 2026-08-07 解耦：CRUISE/STAIR 用独立 env（S10_STAIR_LEG_SIGMA 曾
        # 同时覆盖两者——σ0.6 爬梯时巡航段也变"乱"，横脊失败率升高）。
        if mode == "STAIR":
            leg_sigma = float(os.environ.get(
                "S10_STAIR_LEG_SIGMA", "2.0"))
            wheel_sigma = float(os.environ.get(
                "S10_STAIR_WHEEL_SIGMA", "1.0"))
        else:
            leg_sigma = float(os.environ.get(
                "S10_CRUISE_LEG_SIGMA", "0.3"))
            wheel_sigma = float(os.environ.get(
                "S10_CRUISE_WHEEL_SIGMA", "1.0"))
        self.mbdpi.sigma_dim = jnp.asarray(
            [leg_sigma] * 12 + [wheel_sigma] * 4, dtype=jnp.float32)
        # v200: DBaS 基线快照（自适应在其上缩放，模式切换时重置状态）
        self._sigma_dim_base = np.asarray(
            self.mbdpi.sigma_dim, dtype=np.float32)
        self._ada_se = None
        self._elite_sigma = None
        self._elite_bias = None
        # 采样偏置（用户 2026-08-07 平衡项）：STAIR 时给 Y 腿节点注入
        # 软偏置（动作空间）——前膝缩回（抬前轮）、后膝弯曲（抬后轮），
        # 让"抬腿"成为采样均值方向；扩散/M PPI 权重可覆盖（非门控）。
        if mode == "STAIR":
            _b = os.environ.get(
                "S10_STAIR_LEG_BIAS",
                "0,0,-0.45,0,0,-0.45,0,0,0.45,0,0,0.45")
            self._leg_bias = np.asarray(
                [float(x) for x in _b.replace(" ", "").split(",")],
                dtype=np.float32)
            if self._leg_bias.shape[0] != 12:
                self._leg_bias = np.zeros(12, dtype=np.float32)
        else:
            self._leg_bias = np.zeros(12, dtype=np.float32)
        # 模式化腿动作尺度（2026-08-07 根因修复）：rollout 采样动作被
        # clip 到 ±1，leg_action_scale=0.45 → 膝角可摆范围仅 ±0.45 rad；
        # 前轮爬 0.125m riser 需缩膝 ~0.71 rad（2.30→1.59），超出动作
        # 下限 1.85 → 抬腿动作在采样空间内不可达，狗只能顶死打滑。
        # STAIR 放大尺度让 swing 可达；CRUISE 恢复小尺度防乱甩。
        # 必须同时改 ctx cfg（rollout）与 env._config（主仿真 act2tau）。
        if mode == "STAIR":
            leg_as = float(os.environ.get(
                "S10_STAIR_LEG_ACTION_SCALE", "0.45"))
        else:
            leg_as = float(os.environ.get(
                "S10_LEG_ACTION_SCALE", "0.45"))
        cfg["leg_action_scale"] = leg_as
        if getattr(self.env, "_config", None) is not None:
            self.env._config.leg_action_scale = leg_as
        if mode == "STAIR":
            def _w(name, default):
                v = os.environ.get(name)
                return float(v) if v is not None else default
            overrides = {
                # 放开屈膝（w_crouch=0 同义）：腿自由伸展爬梯
                "terrain_w_leg": _w("S10_STAIR_W_LEG", 0.0),
                "w_crouch": _w("S10_STAIR_W_CROUCH", 0.0),
                # 抬腿引导（chain 64 验证基线）；push/swing/z_smooth
                # 临时关（chain 75 A/B：新 reward 可能干扰爬梯）
                # 2026-08-07 恢复组合：r_ext=30（伸展引导）+ lockpush=8
                # （顶死锁轮）——纯 foot_place 卡底部（r_ext 必要），
                # kp=2.0 时能到 riser 3-4（3/3 到 wp7 稳定）。
                "leg_ext_w": _w("S10_STAIR_LEG_EXT_W", 30.0),
                "lockpush_w": _w("S10_STAIR_LOCKPUSH_W", 8.0),
                # 轮速参考保持（2026-08-07）：v5b 实测无效（-57~+29 仍
                # 振荡且整体变差 wp7/6/7/6 vs v4 4/4）——轮速振荡是 riser
                # 边缘物理打滑，非 reward 可压；回退 0 恢复 v4 最优。
                "w_wheel_ref": _w("S10_STAIR_W_WHEEL_REF", 0.0),
                # 后轮蹬做功 + 抬轮到位微奖（2026-08-07 开）：foot_place
                # 抬前轮后卡 riser 顶（kp=2.0 3/3 复现），r_push 激励后轮
                # 蹬推、r_swing_ok 奖励抬到位。
                "w_swing_ok": _w("S10_STAIR_W_SWING", 2.0),
                "w_push": _w("S10_STAIR_W_PUSH", 1.0),
                "w_z_smooth": _w("S10_STAIR_W_ZSMOOTH", 0.0),
                "terrain_w_attdamp": _w("S10_STAIR_W_ATTDAMP", 2.0),
                # 爬升中直立加强（2026-08-06）：σ=2.0 腿伸展时车身侧倾
                # 累积（batch v28 r1 z=1.01 roll -1.16 侧翻），r_upright
                # 25→40 抑制爬升中侧翻。
                "terrain_w_upright": _w("S10_STAIR_W_UPRIGHT", 40.0),
                # v162c：轮-地形贴合/过抬权重可配（场目标下 300 过强 →
                # bang-bang 过冲翻车；100~150 温和跟随；过抬惩罚加强防甩高）
                "terrain_w_ground": _w("S10_STAIR_W_GROUND", 300.0),
                "terrain_w_overlift": _w("S10_STAIR_W_OVERLIFT", 200.0),
                # v164：r_ground 单向（只罚低于目标，高于目标交给 overlift）——
                # 消除"低于目标猛抬、高于目标猛压"的 bang-bang 过冲振荡
                "ground_oneway": float(os.environ.get(
                    "S10_STAIR_GROUND_ONEWAY", "0")),
                # v169：分相 ground——摆动相只罚没抬到位、支撑相只罚悬空
                "ground_phase": float(os.environ.get(
                    "S10_STAIR_GROUND_PHASE", "0")),
                # 机身抬升强化（2026-08-07）：卡点分析——后腿近直腿
                # （body 0.95-后轮 0.62=0.33 vs 直腿 0.36），后轮上
                # riser 需 body 先抬到 r_clear 目标 1.125；r_clear 60→120、
                # w_path_z 0→40 强制机身逐级抬升。
                "w_clear": _w("S10_STAIR_W_CLEAR", 120.0),
                "w_path_z": _w("S10_STAIR_W_PATH_Z", 40.0),
                # 爬梯时路径/航向跟踪加强（2026-08-06 修复 wp7 西漂）：
                # r_clear/r_ext 在台阶区主导时 MPC 忽视 yaw 指令 → 向西
                # 漂移侧翻（full_course_27）；权重翻倍让 MPC 兼顾导航。
                "ang_vel_weight": _w("S10_STAIR_W_ANG", 40.0),
                "w_path": _w("S10_STAIR_W_PATH", 40.0),
                # v139c：MPCC 进度项——奖励沿路径切线的推进速度，直接对抗
                # "轮子空转但车不前进"的 riser 卡点振荡（v136 r1 卡 8.5s）。
                "w_prog": _w("S10_STAIR_W_PROG", 0.0),
                # v184：STAIR 线速度跟踪权重可配（默认 25；死锁时提高到 40
                # 打破"轮子到位、无推进"局部最优）
                "vel_weight": _w("S10_STAIR_VEL_W", 25.0),
                # 航向跟踪加强（2026-08-07 用户问题 1）：run3 爬梯西漂 4m
                # （x -15→-19），w_path_head 25→40 让 MPC 在爬梯时保持
                # 路径切线航向，抑制横向漂移。
                "w_path_head": _w("S10_STAIR_W_PATH_HEAD", 60.0),
                "stair_pitch_w": _w("S10_STAIR_PITCH_W", 30.0),
                "stair_pitch_tar": float(os.environ.get(
                    "S10_STAIR_PITCH_TAR", "-0.45")),
                "stair_sym_w": _w("S10_STAIR_SYM_W", 80.0),
                "stair_air_w": _w("S10_STAIR_AIR_W", 0.0),
                "w_obstacle": 0.0,
            }
        else:  # CRUISE
            overrides = {
                "terrain_w_leg": 0.3,
                "w_crouch": 15.0,
                # CRUISE 小权重 r_ext（2026-08-06）：横脊 0.13m > 轮半径
                # 0.081 需抬腿，但 r_ext=0 时 σ=0.3 采样不抬 → 靠动量方差
                # 大（wp4→5 横脊通过率 ~60%）。r_ext=5 在横脊（lift_on）
                # 引导适度伸展，平地不触发（lift_on=False）。
                "leg_ext_w": 5.0,
                "lockpush_w": 0.0,
                "w_wheel_ref": 0.0,
                "w_swing_ok": 0.0,
                "w_push": 0.0,
                "w_z_smooth": 0.0,
                "w_prog": float(os.environ.get("S10_MPC_W_PROG", "0.0")),
                "terrain_w_attdamp": 0.8,
                "ang_vel_weight": 10.0,
                # 2026-08-07 修复 CRUISE 污染：STAIR 覆盖过的字段必须恢复，
                # 否则泄漏到 CRUISE（w_path_head=40/upright=40/w_path_z=40
                # 让 CRUISE 横脊/弯道退化）。恢复为外部/默认值。
                "terrain_w_upright": 25.0,
                "w_path": float(os.environ.get("S10_MPC_W_PATH", "15.0")),
                "w_path_head": float(os.environ.get(
                    "S10_MPC_W_PATH_HEAD", "20.0")),
                "w_path_z": 0.0,
                # DIAL-MPC wall/obstacle soft cost (USER 2026-08-17). Cruise only;
                # STAIR keeps 0 to avoid fighting the stair corridor geometry.
                "w_obstacle": float(os.environ.get("S10_MPC_W_OBS", "0.0")),
            }
        cfg.update(overrides)

    def set_yaw_gain_lo(self, gain):
        """覆盖低速 yaw 前馈增益（自动导航用 15 防过冲；None 恢复默认 50）。"""
        self._yaw_gain_lo_override = gain

    def _reinject_wheel_ff(self):
        """每轮规划前把轮速前馈重新写入 Y（对抗 MPPI 的轮速衰减）。

        实测：采样 MPC 在短视界内会把轮速指令逐轮砍低（0.99->0.3），
        导致实际速度远低于指令。这里把轮速按命令前馈固定（开环轮速），
        腿部仍由扩散采样优化（保持姿态/稳定性）；开环实测 knee=2.30
        满轮速可稳定 4 m/s。
        """
        if self.state is None:
            return
        vx = float(self.cmd_vel[0])
        vyaw = float(self.cmd_ang[2])
        if abs(vx) < 0.05 and abs(vyaw) < 0.05:
            # 空闲：让前馈平滑归零（刹车由速度伺服完成）
            self._ramped_ff_fwd(0.0)
            return
        ff_vx = float(np.clip(
            vx / (self.env_config.vel_scale * 0.081), -1.0, 1.0))
        # v215m: 爬越时加轮速（用户提示）——STAIR 模式轮速前馈 ×
        # S10_STAIR_WHEEL_FF_BOOST（>1）：楼梯区强制更高轮速（滚过棱角
        # 需要动量），巡航不变。软先验：采样仍可覆盖。
        if getattr(self, "_mode", None) == "STAIR":
            _wf = float(os.environ.get("S10_STAIR_WHEEL_FF_BOOST", "1.0"))
            if abs(_wf - 1.0) > 1e-6:
                ff_vx = float(np.clip(ff_vx * _wf, -1.0, 1.0))
        # yaw 前馈放大（与 set_cmd 一致）：速度自适应增益，见 set_cmd 注释
        yaw_gain_lo = float(
            self._yaw_gain_lo_override
            if self._yaw_gain_lo_override is not None
            else os.environ.get("S10_MPC_YAW_FF_GAIN", "50.0"))
        yaw_gain_hi = float(os.environ.get("S10_MPC_YAW_FF_GAIN_HI_SPD", "15.0"))
        vx_cmd = abs(float(vx))
        if vx_cmd >= 3.0:
            yaw_gain = yaw_gain_hi
        elif vx_cmd > 0.5:
            t = (vx_cmd - 0.5) / 2.5
            yaw_gain = yaw_gain_lo * (1.0 - t) + yaw_gain_hi * t
        else:
            yaw_gain = yaw_gain_lo
        ff_yaw = float(np.clip(
            yaw_gain * vyaw * 0.05 / (self.env_config.vel_scale * 0.081),
            -1.0, 1.0))
        # v216: 轮锁靠 reward（S10_STAIR_WHEEL_LOCK_W）实现——接近段保留
        # 前馈滚动，riser 面前 reward 锁自然把轮速拉 0（腿足狗式爬梯）。
        # 前馈不清零（整 STAIR 区清零会连接近段一起停，v216 实测卡 y=37.4）。
        ff_vx_r = self._ramped_ff_fwd(ff_vx)
        self.Y = self.Y.at[:, 12:].set(
            jnp.array([ff_vx_r - ff_yaw, ff_vx_r + ff_yaw,
                       ff_vx_r - ff_yaw, ff_vx_r + ff_yaw],
                      dtype=jnp.float32))

    # ---- 单步规划（扩散采样）----
    def _ada_update(self, info) -> None:
        """v200: DBaS 自适应采样方差（arXiv 2502.14387 公式 Se=mu*ln(e+C_B)）。

        每 plan 结束后，用当前 scan 最后一级扩散迭代的样本奖励统计标称轨迹
        代价（softmax 加权平均，锚点 = Ybar 样本奖励），卡台阶（代价高）时
        放大下一轮腿维采样 sigma，顺利通行时收敛。EMA 平滑防抖；默认只在
        STAIR 模式启用（巡航已单独调优，不动）。host 侧 numpy，无 JAX 开销。
        """
        if (not getattr(self, "_ada_enabled", False)
                and not getattr(self, "_ada_bias_on", False)):
            return
        if (self._ada_stair_only and not getattr(self, "_ada_bias_on", False)
                and getattr(self, "_mode", None) != "STAIR"):
            return
        rews = np.asarray(info["rews"])[-1]           # 末级扩散 (Nsample+1,)
        std = float(rews.std()) + 1e-6
        rew_ybar = float(rews[-1])                     # 标称轨迹（Ybar）奖励
        w = np.exp(np.clip((rews - rew_ybar) / std
                           / self.dial_config.temp_sample, -50.0, 50.0))
        w = w / float(w.sum())
        c = max(0.0, -float((w * rews).sum()) - self._ada_ref)
        # v200b: 奖励型 C 在卡台阶时实测不变（rew_bar 巡航/卡死同处 ~-350），
        # DBaS 的"约束代价"需要直接物理量：轮子目标转速大而实际速度≈0 =
        # 顶死打滑（前轮挂 riser、后轮空转）。slip=1-v_real/v_cmd 在
        # 严重打滑（>S10_ADA_SLIP_GATE）时叠加 C，Se 随 ln(e+C) 放大探索；
        # 正常跟踪 slip≈0 不激活（巡航探索保持 v176 基线）。
        v_real = 0.0
        if self.state is not None:
            try:
                d0 = self.state.pipeline_state.data
                v_real = float(np.linalg.norm(np.asarray(d0.cvel[1, 3:])))
            except Exception:
                v_real = 0.0
        v_cmd = abs(getattr(self, "_last_vx", 0.0))
        slip = 0.0
        # v_cmd>0.5 才判滑移：站起/起步阶段 v=0 且 vcmd=0 是正常静止，
        # 不能当"打滑"（v200b 误触发实测：warmup 时 slip=1.0 → Se 飙到 2.5）。
        self._ada_slip = 0.0
        self._ada_slip_active = False
        if v_cmd > float(os.environ.get("S10_ADA_VCMD_GATE", "0.5")):
            slip = max(0.0, 1.0 - v_real / v_cmd)
            self._ada_slip = float(slip)
            if slip > float(os.environ.get("S10_ADA_SLIP_GATE", "0.5")):
                self._ada_slip_active = True
                c = max(c, float(os.environ.get(
                    "S10_ADA_VEL_K", "3000.0")) * slip)
        se = self._ada_mu * float(np.log(np.e + c / max(self._ada_cscale, 1.0)))
        se = min(se, self._ada_max)
        if self._ada_se is None:
            self._ada_se = se
        else:
            self._ada_se = ((1.0 - self._ada_ema) * se
                            + self._ada_ema * self._ada_se)
        if self._ada_enabled:
            base = self._sigma_dim_base
            if base is None:
                base = np.asarray(self.mbdpi.sigma_dim, dtype=np.float32)
            sigma = base.copy()
            if self._ada_leg_only:
                sigma[:12] = sigma[:12] * self._ada_se
            else:
                sigma = sigma * self._ada_se
            sigma = np.clip(sigma, 0.05, 5.0)
            self.mbdpi.sigma_dim = jnp.asarray(sigma, dtype=jnp.float32)
        # v214: 摆动轮采样方差放大（utility 选腿）——摆动轮的腿关节
        # sigma × S10_GAIT_SIGMA_BOOST（>1 启用），让采样器更可能搜到
        # 抬腿轨迹；主摆动轮全量放大、对角次选按 utility 比例，其余轮
        # 保持紧致（用户"抬腿 sample variance 调大"落地，纯软探索）。
        # v215j: HL 关节 sigma 单独放大（S10_HL_SIGMA_BOOST>1）——左后轮
        # 抬升动作（膝 act≈1.0 满幅）离采样均值 3+σ，bias 满幅会翻车；
        # 只放大探索方差，让采样器能搜到满幅抬腿而不推均值。
        _hl_sb = float(os.environ.get("S10_HL_SIGMA_BOOST", "0"))
        if _hl_sb > 1.0:
            _sig = np.asarray(self.mbdpi.sigma_dim, dtype=np.float32).copy()
            _sig[6:9] = _sig[6:9] * _hl_sb
            _sig = np.clip(_sig, 0.05, 6.0)
            self.mbdpi.sigma_dim = jnp.asarray(_sig, dtype=jnp.float32)
        _boost = float(os.environ.get("S10_GAIT_SIGMA_BOOST", "0"))
        _gsw2 = getattr(self, "_gait_swing", None)
        if _boost > 1.0 and _gsw2 is not None and float(np.max(_gsw2)) > 0.0:
            _sw = np.asarray(_gsw2, dtype=np.float32)
            _per_j = np.repeat(_sw, 3)
            _per_j = _per_j / (float(np.max(_per_j)) + 1e-6)
            _sig = np.asarray(self.mbdpi.sigma_dim, dtype=np.float32).copy()
            _sig[:12] = _sig[:12] * (1.0 + (_boost - 1.0) * _per_j)
            _sig = np.clip(_sig, 0.05, 6.0)
            self.mbdpi.sigma_dim = jnp.asarray(_sig, dtype=jnp.float32)
        if os.environ.get("S10_MPC_DEBUG"):
            print(f"[ADA] C={c:.0f} Se={se:.3f} "
                  f"se_ema={self._ada_se:.3f} "
                  f"leg_sigma={float(self.mbdpi.sigma_dim[0]):.2f} "
                  f"rew_bar={float((w*rews).sum()):.0f} "
                  f"slip={slip:.2f} v={v_real:.2f} vcmd={abs(getattr(self, '_last_vx', 0.0)):.2f}",
                  flush=True)

    def _elite_adapt(self, info) -> None:
        """v209: MPOPI 式精英协方差自适应（host numpy，零 JAX 开销）。

        取 top-K 精英样本动作的每维均值/方差：sigma_dim 与基线混合更新
        （防塌缩），精英均值作软偏置注入下一次 Y（采样均值向好样本区域
        走，CMA 式学习循环）。只在 STAIR 模式启用。
        """
        if not getattr(self, "_elite_ada", False):
            return
        if getattr(self, "_mode", None) != "STAIR":
            return
        try:
            rews = np.asarray(info["rews"])[-1]          # (Ns+1,)
            Y0s = np.asarray(info["Y0s"])[-1]            # (Ns+1,H+1,16)
            K = max(2, int(self.dial_config.Nsample * self._elite_frac))
            idx = np.argsort(-rews)[:K]
            elite = Y0s[idx]
            em = elite.mean(axis=(0, 1))                 # (16,)
            es = elite.std(axis=(0, 1))                  # (16,)
            if self._elite_sigma is None:
                self._elite_sigma = es
            else:
                self._elite_sigma = (self._elite_alpha * self._elite_sigma
                                     + (1.0 - self._elite_alpha) * es)
            base = self._sigma_dim_base
            if base is None:
                base = np.asarray(self.mbdpi.sigma_dim, dtype=np.float32)
            sigma = np.clip(0.5 * base + 0.5 * self._elite_sigma,
                            0.05, 5.0)
            self.mbdpi.sigma_dim = jnp.asarray(
                sigma, dtype=jnp.float32)
            self._elite_bias = np.clip(em, -1.0, 1.0)
            if os.environ.get("S10_MPC_DEBUG"):
                print(f"[ELITE] K={K} es0={es[0]:.3f} es7={es[7]:.3f} "
                      f"em7={em[7]:+.3f} sigma7={float(sigma[7]):.3f}",
                      flush=True)
        except Exception as e:
            print(f"[ELITE] fail {e}", flush=True)

    def plan_once(self, q: np.ndarray, qd: np.ndarray, t: float) -> jnp.ndarray:
        """返回当前最优 action（16 维，供 act2tau 转力矩）。

        dial-mpc 主循环是 shift+reverse：先时间平移（把上次优化过的控制推进到
        第一个节点），再扩散采样优化未来节点。缺 shift 会导致 Y[0] 恒为初始 0。
        """
        if self.state is None:       # 防御：任何路径进入规划时确保 state 已初始化
            self.init_state(q, qd)
        _t_start = __import__("time").perf_counter()
        self.update_state(q, qd, t)
        _t_upd = __import__("time").perf_counter() - _t_start
        # 1) shift：按真实 plan 间隔推进（方案 C，2026-08-07）：执行
        # 零阶保持周期 = plan 周期（如 0.05s=2.5 ctrl_dt），shift 必须
        # 推进相同步数，否则 Y 序列相位每轮错位（原始 dial-mpc 用 delta_step）。
        # 修正（2026-08-07 复核）：原实现 shift_n 后又多执行一次无条件
        # shift（编辑残留），导致每轮多推进 1 步、视界被更快耗尽；改为
        # 小数累加器（Bresenham 式 2/3 交替），平均推进 = 实际流逝步数。
        _last_t = getattr(self, "_last_plan_t", None)
        if _last_t is None:
            n_shift_f = 1.0
        else:
            n_shift_f = (t - _last_t) / self.env_config.dt
        _acc = getattr(self, "_shift_accum", 0.0) + n_shift_f
        n_shift = int(round(_acc))
        if n_shift < 1:
            n_shift, _acc = 1, 0.0
        else:
            _acc -= n_shift
        self._shift_accum = _acc
        n_shift = max(1, min(n_shift, self.dial_config.Hnode))
        _t0 = __import__("time").perf_counter()
        self.Y = self.mbdpi.shift_n(
            self.Y, jnp.asarray(n_shift, dtype=jnp.int32))
        _t_shift = __import__("time").perf_counter() - _t0
        self._last_plan_t = t
        # 轮速前馈持续注入（对抗优化器衰减；见 _reinject_wheel_ff）
        _t0 = __import__("time").time()
        self._reinject_wheel_ff()
        # STAIR 采样偏置注入（软先验，扩散采样可覆盖）。v168：优先用场驱动
        # 的时变偏置（每节点不同），否则回退静态 _leg_bias。
        # v176：偏置用**混合收敛**（Y += λ(bias−Y)，λ=S10_STAIR_BIAS_BLEND）
        # 替代直接相加——直接相加每 plan 累积导致过抬（v171 前轮 1.0+ 根因）。
        _lb = getattr(self, "_stair_action_bias", None)
        if _lb is None:
            _lb = getattr(self, "_leg_bias", None)
        if _lb is not None and bool(np.any(_lb)):
            _lb_a = jnp.asarray(_lb, dtype=jnp.float32)
            if _lb_a.ndim == 1:
                _lb_a = _lb_a[None, :]
            _blend = float(os.environ.get("S10_STAIR_BIAS_BLEND", "0.30"))
            # v200e: 卡住打滑时放大抬腿先验（软先验；仅 STAIR 且启用时）
            if (getattr(self, "_ada_bias_on", False)
                    and getattr(self, "_ada_slip_active", False)
                    and getattr(self, "_mode", None) == "STAIR"):
                _blend = self._ada_bias_stuck
            # v209: 精英均值软偏置（CMA 式学习：采样均值向好样本区域走）
            _eb = getattr(self, "_elite_bias", None)
            if _eb is not None:
                _blend = max(_blend, self._elite_bias_blend)
                _lb_a2 = jnp.asarray(_eb, dtype=jnp.float32)
                self.Y = self.Y.at[:, :12].set(jnp.clip(
                    self.Y[:, :12]
                    + self._elite_bias_blend * (_lb_a2[None, :12] - self.Y[:, :12]),
                    -1.0, 1.0))
            _Yl = self.Y[:, :12]
            self.Y = self.Y.at[:, :12].set(jnp.clip(
                _Yl + _blend * (_lb_a - _Yl), -1.0, 1.0))

        n_diffuse = self.dial_config.Ndiffuse
        if self._first:
            print("[MPC] 首次 JIT 扩散采样 ...")
            n_diffuse = self.dial_config.Ndiffuse_init
            self._first = False
        factors = (
            self.mbdpi.sigma_control
            * self.dial_config.traj_diffuse_factor
            ** (jnp.arange(n_diffuse))[:, None]
            * jnp.asarray(self._adaptive_sigma_node(),
                          dtype=jnp.float32)[None, :]
        )
        rng, Y0, st = self.rng, self.Y, self.state
        _t0 = __import__("time").perf_counter()
        (rng, Y0, st), info = jax.lax.scan(
            self._scan_body, (rng, Y0, st), factors)
        _t_scan = __import__("time").perf_counter() - _t0
        if os.environ.get("S10_MPC_DEBUG"):
            import numpy as _np
            rew = _np.asarray(info["rews"])
            d0 = self.state.pipeline_state.data
            v_real = _np.linalg.norm(_np.asarray(d0.cvel[1, 3:]))
            print(f"[DBG] v_real={v_real:.2f} "
                  f"frac_bad={(rew < -1e5).mean():.2f} "
                  f"max|rew|={_np.abs(rew).max():.1e} "
                  f"Yw_in={_np.asarray(self.Y[0, 12]):.2f} "
                  f"Yw_out={_np.asarray(Y0[0, 12]):.2f} "
                  f"Yw1={_np.asarray(Y0[1, 12]):.2f} "
                  f"rew_mean={float(rew.mean()):.0f}",
                  flush=True)
        self.rng, self.Y, self.state = rng, Y0, st
        _t0 = __import__("time").perf_counter()
        self.Y = self.Y.block_until_ready()
        _t_sync = __import__("time").perf_counter() - _t0
        _t_plan = (__import__("time").perf_counter() - _t_start)
        self._last_plan_times = dict(
            total_ms=_t_plan * 1000.0, upd_ms=_t_upd * 1000.0,
            shift_ms=_t_shift * 1000.0, scan_ms=_t_scan * 1000.0,
            sync_ms=_t_sync * 1000.0)
        if os.environ.get("S10_MPC_TIMING"):
            print(f"[PLAN-T] total={_t_plan*1000:.1f}ms "
                  f"upd={_t_upd*1000:.1f} shift={_t_shift*1000:.1f} "
                  f"scan={_t_scan*1000:.1f} sync={_t_sync*1000:.1f} "
                  f"mode={getattr(self, '_mode', '?')}",
                  flush=True)
        # v200: DBaS 自适应方差（下一次 plan 的 reverse_once 使用）。
        # 放在计时区之后：内部 np.asarray(info["rews"]) 会同步 GPU，计入
        # plan 时间会污染频率测量（v200 实测 total 91ms 触发 GPU 守卫）。
        self._ada_update(info)
        # v209: MPOPI 式精英协方差自适应（同样在计时区之后）
        self._elite_adapt(info)
        self._last_plan_t = t
        return self.Y[0]

    # ---- 控制输出 ----
    def get_tau(self, action: jnp.ndarray) -> np.ndarray:
        """action → 16 维关节力矩（leg PD + wheel 直接力矩）。"""
        tau = self.env.act2tau(action, self.state.pipeline_state)
        return np.asarray(tau)

    def compute_tau(self, action, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """用当前仿真状态 numpy 重算力矩（不触发 JAX，可在 200Hz 主循环每步调用）。

        action = 12 腿位置残差 + 4 轮速度目标；与 env.act2tau 语义一致，
        但直接从仿真 mj_data 的 qpos/qvel 计算，避免 MPC 状态过期。
        """
        import numpy as _np

        from dial_mpc.envs.s10_env import LEG_IDX_NP, WHEEL_IDX_NP

        a = _np.asarray(action, dtype=_np.float32)
        q = _np.asarray(q, dtype=_np.float32)
        qd = _np.asarray(qd, dtype=_np.float32)
        _env_cfg = getattr(self.env, '_config', self.env_config)
        leg_target = (
            _np.asarray(self.env._default_leg)
            + a[:12] * _env_cfg.leg_action_scale
        )
        q_leg = q[7:][LEG_IDX_NP]
        qd_leg = qd[6:][LEG_IDX_NP]
        tau_leg = self.env_config.kp * (leg_target - q_leg) - self.env_config.kd * qd_leg
        qd_wheel = qd[6:][WHEEL_IDX_NP]
        if self.env_config.wheel_control == "velocity":
            vel_ref = -a[12:] * self.env_config.vel_scale
            tau_wheel = self.env_config.kd_wheel * (vel_ref - qd_wheel)
        else:
            tau_wheel = -a[12:] * self.env_config.wheel_tau_scale
        tau = _np.zeros(16, dtype=_np.float32)
        tau[LEG_IDX_NP] = tau_leg
        tau[WHEEL_IDX_NP] = tau_wheel
        ctrl = _np.asarray(self.env.joint_torque_range)
        return _np.clip(tau, ctrl[:, 0], ctrl[:, 1])

    # ---- 异步规划线程（仿真不阻塞；~2s/次更新指令）----
    def start_planning(self, q: np.ndarray, qd: np.ndarray):
        """后台线程持续规划：不断更新 latest_action / latest_tau。
        仿真主循环每步读取 latest_tau 施加，无需等待 MPC。"""
        import threading
        if self.state is None:
            self.init_state(q, qd)
        self._plan_lock = threading.Lock()
        self.latest_tau = np.zeros(16, dtype=np.float32)
        self.latest_action = np.zeros(16, dtype=np.float32)
        self._plan_q = np.asarray(q, dtype=np.float32)
        self._plan_qd = np.asarray(qd, dtype=np.float32)
        self._plan_t = 0.0
        self._plan_stop = threading.Event()
        self._plan_thread = threading.Thread(
            target=self._plan_loop, daemon=True)
        self._plan_thread.start()

    def update_plan_state(self, q: np.ndarray, qd: np.ndarray, t: float):
        """仿真线程每步调用：更新最新状态快照供规划线程读取。"""
        with self._plan_lock:
            self._plan_q = np.asarray(q, dtype=np.float32)
            self._plan_qd = np.asarray(qd, dtype=np.float32)
            self._plan_t = float(t)

    def _plan_loop(self):
        """规划线程：每轮用最新状态快照做一次 plan_once，更新输出。"""
        first = True
        while not self._plan_stop.is_set():
            with self._plan_lock:
                q = self._plan_q.copy()
                qd = self._plan_qd.copy()
                t = self._plan_t
            try:
                act = self.plan_once(q, qd, t)
                tau = self.get_tau(act)
                if np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
                    continue   # NaN 防护：保留上次有效 tau
                with self._plan_lock:
                    self.latest_action = np.asarray(act)
                    self.latest_tau = tau
                if first:
                    print(f"[MPC] 规划线程就绪（首次 {t:.2f}s）", flush=True)
                    first = False
            except Exception as e:
                import traceback
                print(f"[MPC] 规划线程异常: {e}", flush=True)
                traceback.print_exc()

    def stop_planning(self):
        if hasattr(self, "_plan_stop"):
            self._plan_stop.set()
