"""
 * @file mujoco_simulation.py
 * @brief simulation in mujoco
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-05
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

import os
import sys
import time
import socket
import struct
import threading
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData, JointDataCmd

# LiDAR相关导入
try:
    from mujoco_lidar import MjLidarWrapper, scan_gen
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
    LIDAR_AVAILABLE = True
except ImportError as e:
    LIDAR_AVAILABLE = False
    print(f"[WARNING] LiDAR dependencies not available: {e}")

# 高程图模块（纯 numpy，真机可移植；路径为 S10_sdk_deploy 包根）
S10_PKG_DIR = Path(__file__).resolve().parents[3]
if str(S10_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(S10_PKG_DIR))
try:
    from perception.points_to_heightmap import points_to_heightmap, ElevationMapConfig
    ELEVATION_AVAILABLE = True
except ImportError as e:
    ELEVATION_AVAILABLE = False
    print(f"[WARNING] Elevation map module not available: {e}")
try:
    from perception.local_map import LocalMap, LocalMapConfig, compute_terrain_features
    LOCAL_MAP_AVAILABLE = True
except ImportError as e:
    LOCAL_MAP_AVAILABLE = False
    print(f"[WARNING] Local map (voxel elevation tile) module not available: {e}")

# ================= LiDAR 配置（集中管理） =================
# 注意：WSL 中 taichi GPU 后端需要 libcuda.so（一般缺失且初始化时 core dump，
# 无法被 try/except 捕获），故默认用 cpu；有 CUDA 库的环境可 S10_LIDAR_BACKEND=taichi
LIDAR_BACKEND   = os.environ.get("S10_LIDAR_BACKEND", "cpu")  # cpu | taichi | jax | warp
LIDAR_FREQ      = float(os.environ.get("S10_LIDAR_FREQ", "10"))
LIDAR_CUTOFF    = float(os.environ.get("S10_LIDAR_CUTOFF", "20.0"))
# Airy-96 结构角表：垂直 ±7°（真机规格）→ 仿真加宽到 ±20° 以容忍机身俯仰。
# 依据（实测）：前倾 19.5° + FOV ±7° 仅覆盖前方 0.70~1.58m；狗抬头 10° 时近处
# 0~1.05m 全丢。FOV ±20° 后：抬头 10° 覆盖 0.62m~∞，低头 10° 覆盖 0.30~2.1m。
# 垂直行数 16→24→48：48 线角度分辨率 40°/48 ≈ 0.83°/行，远场台阶覆盖更好。
# 实测（tmp/bench_lidar_phi.py）：24 线 23.4ms/帧、48 线 45.9ms/帧（均 < 10Hz 周期
# 100ms），且 mj_multiRay 释放 GIL（主线程放大 1.0x），不影响 MPC 主线程规划。
# 射线 120×48=5760。
LIDAR_THETA_N   = int(os.environ.get("S10_LIDAR_THETA_N", "120"))    # 水平列数
LIDAR_PHI_N     = int(os.environ.get("S10_LIDAR_PHI_N", "48"))       # 垂直线数
LIDAR_FOV_H_DEG = float(os.environ.get("S10_LIDAR_FOV_H_DEG", "90"))  # 水平半角（度），180=360°全景
# 垂直半 FOV（度）：2026-08-06 按实机修正 20→45——配合 site euler 0.79
# （前上 45°）在爬坡 pitch 下覆盖地面最佳（诊断见 S10.xml 注释）。
LIDAR_PHI_DEG   = float(os.environ.get("S10_LIDAR_PHI_DEG", "45.0"))
LIDAR_MAX_PUB_POINTS = int(os.environ.get("S10_LIDAR_MAX_PUB_POINTS", "16384"))
# 高程图配置（16×10，0.1 m，前向 1.6 m × 侧向 1.0 m）
ELEVATION_CFG = ElevationMapConfig()
VIZ_ELEVATION = os.environ.get("S10_VIZ_ELEVATION", "1") == "1"   # viewer 中显示高程图面片

# ---- 世界对齐高程瓦片（感知-voxel，待办3 定稿规格，见 doc/0806.md §2.1）----
# 锚定瓦片 8×8m @0.1m（含 1m 重叠带），有效输出区 6×6m -> (60,60)；
# 机器人距瓦片中心 >2m 重锚定；重叠带继承；最近观测优先 + min-z 去噪。
LOCAL_MAP_CFG = LocalMapConfig(
    resolution=float(os.environ.get("S10_LOCAL_RES", "0.1")),
    tile_size=float(os.environ.get("S10_LOCAL_TILE", "8.0")),
    effective_size=float(os.environ.get("S10_LOCAL_EFFECTIVE", "6.0")),
    voxel_size=float(os.environ.get("S10_LOCAL_VOXEL", "0.05")),
    reanchor_dist=float(os.environ.get("S10_LOCAL_REANCHOR", "2.0")),
    max_hang=float(os.environ.get("S10_LOCAL_MAX_HANG", "1.5")),
    max_drop=float(os.environ.get("S10_LOCAL_MAX_DROP", "1.5")),
    max_age=float(os.environ.get("S10_LOCAL_MAX_AGE", "0.0")),
    inpaint_iter=int(os.environ.get("S10_LOCAL_INPAINT", "3")),
)
VIZ_LOCAL_MAP = os.environ.get("S10_VIZ_LOCAL_MAP", "0") == "1"   # 预留：瓦片可视化
# 高程图可视化分档（相对"脚下地面"参考 Δh = h - h0，单位 m）。
# 依据（对照 DIAL-MPC full-order 能力与 S10 机构学）：
#   - S10 腿部正运动学：轮心相对 hipy 站姿约 -0.12 m，最高可抬至 +0.33 m（行程 0.45 m）
#   - 轮子半径 0.081 m：≤ 0.08 m 的坎轮子可直接滚过（无需抬腿）
#   - DIAL-MPC 官方 unitree_go2_crate_climb 示例可爬 0.6 m 高箱子（full-order 采样）
#   → S10 轮足机构极限约 0.3 m；但**稳健运行**建议 ≤ 0.15 m 才放心，
#     0.15~0.3 m 属高风险区（MPC 可尝试，但 cost 应重罚/减速）。
# 分档：绿=平地，黄=轮子滚过，橙=单腿抬轮翻越（容易），红=需多腿爬台（高风险）。
# 注意：颜色仅影响可视化，heightmap 数值不变；"可通行/可翻越"判定在 MPC cost 中实现。
ELEV_BAND_FLAT = float(os.environ.get("S10_ELEV_BAND_FLAT", "0.03"))    # ±0.03 m = 平地
ELEV_BAND_ROLL = float(os.environ.get("S10_ELEV_BAND_ROLL", "0.08"))    # 轮子直接滚过
ELEV_BAND_CLIMB = float(os.environ.get("S10_ELEV_BAND_CLIMB", "0.15"))  # 稳健翻越上限（高风险分界）
# 自检模式：>0 时仿真运行该秒数后打印 LiDAR/高程图统计并自动退出（无头验证用）
SELF_TEST_SECONDS = float(os.environ.get("S10_SELF_TEST_SECONDS", "0"))
# s10_mpc.xml 轮体 id（fl/fr/hl/hr），运动学地面注入用（见 _lidar_worker）
WHEEL_CONTACT_IDS = [5, 9, 13, 17]
WHEEL_RADIUS = 0.081

# ================= dial-mpc 控制模式（模式 B 遥控） =================
# S10_MPC_ENABLE=1 时启用：仿真窗口内 z 站起 / c 进入 MPC 遥控 / wasd 移动 / qe 转向
MPC_ENABLE = os.environ.get("S10_MPC_ENABLE", "0") == "1"
MPC_YAML = os.environ.get("S10_MPC_YAML",
                          str(S10_PKG_DIR.parent.parent / "doc" / "s10_mpc_deploy.yaml"))
# 站起（StandUp）：腿 PD 位置控制拉到站姿角（与 JOINT_INIT 一致）
MPC_STAND_KP = 80.0
MPC_STAND_KD = 2.0
MPC_STAND_WHEEL_KD = 0.3   # 站姿时轮子只施加刹车力矩，避免被腿 PD 力矩误驱
MPC_STAND_TIME = 2.0      # 站起保持时间（秒）
# 遥控目标速度上限（与 yaml remote 节一致）
MPC_VX_MAX = float(os.environ.get("S10_MPC_VX_MAX", "4.5"))                               
MPC_VY_MAX = float(os.environ.get("S10_MPC_VY_MAX", "0.5"))
MPC_VYAW_MAX = float(os.environ.get("S10_MPC_VYAW_MAX", "2.0"))
# MBDPI 主线程规划间隔（仿真步数）：50 步 = 每 1s 规划一次（~0.25s 阻塞）
MPC_PLAN_INTERVAL = int(os.environ.get("S10_MPC_PLAN_INTERVAL", "50"))



MODEL_NAME = "S10"
# Get the directory of the current Python file
CURRENT_DIR = Path(__file__).resolve().parent
MJCF_DIR = (CURRENT_DIR / ".." / ".." / ".." / "S10_description" / "s10_mjcf" / "mjcf").resolve()

SCENE_XML_PATHS = {
    "track": MJCF_DIR / "S10_track.xml",
}
DEFAULT_SCENE_NAME = os.environ.get("S10_MUJOCO_SCENE", "track")
XML_PATH = str(SCENE_XML_PATHS.get(DEFAULT_SCENE_NAME, SCENE_XML_PATHS["track"]).resolve())
USE_VIEWER = os.environ.get("S10_USE_VIEWER", "1") == "1"  # 无头运行可设 0
TRACK_VIEWER = True
DT = 0.005
RENDER_INTERVAL = 10
TRACK_BODY_NAME = "base_link"
CAMERA_AZIMUTH = 90
CAMERA_ELEVATION = -25
CAMERA_DISTANCE = 10.0
COLLISION_GEOM_GROUP = 1
TRACK_START_BASE_POS = np.array([0.0, -2.5, 0.2])
TRACK_REACH_RADIUS = float(os.environ.get("S10_TRACK_REACH_RADIUS", "0.5"))
TRACK_DISTANCE_MODE = os.environ.get("S10_TRACK_DISTANCE_MODE", "xy").lower()
TRACK_WAYPOINT_PREFIX = "track_waypoint_"
TRACK_HEIGHT_POST_PREFIX = "track_height_post_"

# Calibaration parameters (for sim-to-real consistency)
JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
POS_OFFSET_DEG = np.array([-35, -145, 156, 0.,
                             35, -145, 156, 0,
                             -35, 145, -156, 0,
                             35, 145, -156, 0])
POS_OFFSET_RAD = POS_OFFSET_DEG / 180.0 * np.pi

JOINT_INIT = {
    "S10": np.array([-0.438, -1.16, 2.76, 0,
                     0.438, -1.16, 2.76, 0,
                     -0.438, 1.16, -2.76, 0,
                     0.438, 1.16, -2.76, 0], dtype=np.float32),
}


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run S10 MuJoCo ROS2 simulation.")
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_XML_PATHS),
        default=DEFAULT_SCENE_NAME if DEFAULT_SCENE_NAME in SCENE_XML_PATHS else "track",
        help="Built-in MJCF scene to load. Defaults to S10_MUJOCO_SCENE or 'track'.",
    )
    parser.add_argument(
        "--xml-path",
        default=os.environ.get("S10_MUJOCO_XML"),
        help="Custom MJCF path. Overrides --scene and S10_MUJOCO_SCENE.",
    )
    parser.add_argument("--model-key", default=MODEL_NAME, help="Robot key used for initial joint pose.")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def resolve_xml_path(scene_name: str, xml_path: str | None) -> str:
    if xml_path:
        return str(Path(xml_path).expanduser().resolve())
    return str(SCENE_XML_PATHS[scene_name].resolve())


class MuJoCoSimulationNode(Node):
    def __init__(self,
                 model_key: str = MODEL_NAME,
                 xml_path: str = XML_PATH):

        super().__init__('mujoco_simulation')

        # 加载 MJCF
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        # 机器人自由度列表
        self.actuator_ids = [a for a in range(self.model.nu)]  # 0..15
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for S10"

        # 初始化站立姿态
        self._set_initial_pose(model_key)
        self._init_track_progress()

        # 缓存
        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        # IMU
        self.last_base_linvel = np.zeros((3, 1), np.float64)
        self.timestamp = 0.0

        self.get_logger().info(f"[INFO] MuJoCo MJCF loaded: {xml_path}")
        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # ROS Publishers
        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)

        # ROS Subscriber
        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            '/JOINTS_CMD',
            self._cmd_callback,
            50
        )

        # 可视化 (先初始化viewer，再初始化LiDAR可视化)
        self.viewer = None
        if USE_VIEWER:
            # 键盘控制由 terminal pynput 接管（viewer key_callback 在标准 python3
            # 下不触发，且会与 mujoco 内置键冲突），这里不注册
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_viewer()

        # ========== LiDAR初始化 ==========
        if LIDAR_AVAILABLE:
            self._init_lidar()
            self.lidar_points_pub = self.create_publisher(PointCloud2, '/lidar_points', 10)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            self.static_tf_published = False
            if ELEVATION_AVAILABLE and self.lidar is not None:
                # 高程图话题：data = [x_min,y_min,resolution,nx,ny] + heightmap.flatten()
                self.elevation_map_pub = self.create_publisher(
                    Float32MultiArray, '/elevation_map', 10)
                self.em_msg = Float32MultiArray()
                self.em_msg.layout.dim = [
                    MultiArrayDimension(label="y", size=ELEVATION_CFG.ny, stride=ELEVATION_CFG.nx),
                    MultiArrayDimension(label="x", size=ELEVATION_CFG.nx, stride=1),
                ]
            if LOCAL_MAP_AVAILABLE and self.lidar is not None:
                # 世界对齐高程瓦片（voxel）：meta[4]=x0,y0,res,fill + height.flatten
                # + valid.flatten（0/1），保留 /elevation_map 兼容
                self.local_map_pub = self.create_publisher(
                    Float32MultiArray, '/local_map', 10)
                self.lm_msg = Float32MultiArray()
                self.lm_msg.layout.dim = [
                    MultiArrayDimension(label="y", size=LOCAL_MAP_CFG.n_out,
                                        stride=LOCAL_MAP_CFG.n_out),
                    MultiArrayDimension(label="x", size=LOCAL_MAP_CFG.n_out,
                                        stride=1),
                ]
        else:
            self.lidar = None
            self.get_logger().warn("[WARNING] LiDAR functionality disabled")

        # ========== dial-mpc 控制模式（模式 B 遥控） ==========
        self.mpc = None
        self.mpc_mode = False
        # ========== 模式 A：自动导航（S10_MODE=auto_nav） ==========
        self.auto_nav = os.environ.get("S10_MODE", "remote") == "auto_nav"
        self.auto_nav_active = False
        self.auto_stand_t0 = None
        self.follower = None
        self._stall_t = 0.0
        self._recovery_t = 0.0
        self._leg_assist = np.zeros(12, dtype=np.float32)
        # CTBC 式时序摆动抬腿状态（2026-08-05）：每轮独立
        # _lift_timer/_lift_active/_lift_peak——检测台阶→0.12s 抬升到峰值→
        # 保持跨越→到位/超时释放。峰值 = S10_LIFT_SWING（joint rad，±带符号）。
        self._lift_timer = np.zeros(4, dtype=np.float64)
        self._lift_active = np.zeros(4, dtype=bool)
        self._lift_peak = np.zeros(4, dtype=np.float64)
        self._lift_trigger = np.zeros(4, dtype=bool)
        # 确定性爬梯步态（S10_STAIR_GAIT=1 启用，默认关）：wp7 台阶区
        # （已知地图 5 级 0.13m 台阶）完全接管腿控——膝到限位抬轮 + 低速
        # 驱动，去掉采样 MPC 的方差。膝关节物理抬轮上限 ~0.12m < 0.13m，
        # 靠滚动+车身俯仰补足（见 0806 §3.10）。
        self._stair_gait_y0 = float(os.environ.get(
            "S10_STAIR_GATE_Y0", "34.5"))
        self._stair_gait_y1 = float(os.environ.get(
            "S10_STAIR_GATE_Y1", "41.6"))
        # 膝目标速率限制（rad/0.05s）：抬膝 3.0 阶跃会产生冲击，平滑之
        self._stair_knee_cur = np.array([2.3, 2.3, -2.3, -2.3],
                                         dtype=np.float64)
        self._stair_hipy_cur = np.array([-1.16, -1.16, 1.16, 1.16],
                                         dtype=np.float64)
        # wp7 台阶 riser 位置（已知地图，scan_wp7_3d.py 实测）与各级台阶顶高
        self._stair_risers = np.array(
            [38.4, 38.8, 39.4, 39.8, 40.2], dtype=np.float64)
        self._stair_tops = np.array(
            [0.67, 0.79, 0.92, 1.04, 1.17], dtype=np.float64)
        self.auto_finish_logged = False
        self.standup_phase = 0.0
        self.standup_done = False   # z 按下后置 True：站姿 PD 持续保持
        self._mpc_warmup_done = False
        self.last_act = None
        # 站起/驾驶姿态：hipx±0.05/hipy±1.16/knee±2.30 → base z≈0.204m，
        # 竞速蹲伏姿态（实测 4.04m/s 稳定；0.238m 高站姿 4m/s 前翻），
        # 初始 JOINT_INIT（hipx=0.438 外展）趴平 z≈0.082，按 z 明显站起
        _stand_hipx = float(os.environ.get("S10_STAND_HIPX", "0.05"))
        self.stand_target = np.array(
            [-_stand_hipx, -1.16, 2.30, 0.0,
              _stand_hipx, -1.16, 2.30, 0.0,
             -_stand_hipx,  1.16, -2.30, 0.0,
              _stand_hipx,  1.16, -2.30, 0.0], dtype=np.float64)
        self._mpc_key_thread = None
        if MPC_ENABLE:
            # MPC 构建（含预编译 30-40s）放后台线程：不阻塞仿真启动/viewer
            self.get_logger().info("[MPC] 后台构建 dial-mpc 控制器（JIT 预热在启动后自动执行）...")
            threading.Thread(target=self._init_mpc_background, daemon=True).start()
            # terminal 键盘监听（pynput，wasd 按下/释放；避开 mujoco viewer 内置键）
            self._start_terminal_keyboard()

    # ---- terminal 键盘（termios raw mode；复刻原始 keyboard_interface_sim 行为）----
    # 键位与原始一致：z 站起 / c 进遥控 / r 阻尼 / wasd 移动 / qe 转向
    # 行为与原始一致：按住 ramp 加速（每 repeat +0.1*max）、松开（超时 0.15s）归零
    def _start_terminal_keyboard(self):
        self._kb_vel = {k: 0.0 for k in "wasdqes"}   # 每键速度分量（s 复用占位）
        self._kb_vel = {"w": 0.0, "s": 0.0, "a": 0.0, "d": 0.0, "q": 0.0, "e": 0.0}
        self._kb_last = {}                            # 键 → 最后按下时间
        try:
            import termios, tty, select as _sel
            self._term_select = _sel
            fd = sys.stdin.fileno()
            if not os.isatty(fd):
                self.get_logger().warn("[MPC] stdin 非终端，键盘控制不可用（headless 模式）")
                return
            self._term_old = termios.tcgetattr(fd)
            self._kb_thread = threading.Thread(
                target=self._terminal_reader, daemon=True)
            self._kb_thread.start()
            self.get_logger().info(
                "[MPC] terminal 键盘就绪（raw mode，原始键位）："
                "z 站起 / c 遥控 / r 阻尼 / wasd 移动 / qe 转向（按住=加速，松开=停）")
        except Exception as e:
            self.get_logger().warn(f"[MPC] terminal 键盘初始化失败: {e}")

    def _terminal_reader(self):
        import termios, tty
        fd = sys.stdin.fileno()
        try:
            tty.setraw(fd)
            while True:
                r, _, _ = self._term_select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":      # Ctrl-C：恢复 terminal 后退出
                        break
                    if ch:
                        self._handle_term_key(ch)
                # 超时检测松开（原始模式：release 归零）
                now = time.time()
                changed = False
                for k in list(self._kb_last.keys()):
                    if now - self._kb_last[k] > 0.15:
                        self._kb_vel[k] = 0.0
                        del self._kb_last[k]
                        changed = True
                if changed:
                    self._update_mpc_cmd()
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._term_old)
            except Exception:
                pass
            self.get_logger().info("[MPC] terminal 键盘已退出（恢复正常输入）")

    def _handle_term_key(self, c):
        c = c.lower()
        if c == "z":
            # 站起（与原始 z 一致：WaitingForStand → StandingUp）
            if self.mpc_mode:
                self.mpc.stop_planning()
                self.mpc_mode = False
            self.standup_phase = MPC_STAND_TIME
            self.standup_done = True
            self.get_logger().info("[MPC] 站起（z）")
        elif c == "c":
            # 进遥控（与原始 c：StandingUp → RLControl）
            self._toggle_mpc_mode()
        elif c == "r":
            # 阻尼（与原始 r：JointDamping）→ 退出遥控 + 站姿兜底
            if self.mpc_mode:
                self.mpc.stop_planning()
                self.mpc_mode = False
            self.get_logger().info("[MPC] 阻尼（r）：已退出遥控，回站姿")
        elif c in "wasdqe":
            # 按住 ramp 加速（与原始 apply_key_direction：每 repeat +step）
            if not self.mpc_mode:
                return
            now = time.time()
            step = 0.2   # 竞速：按一次键更快到满速（原 0.1）
            mx = {"w": MPC_VX_MAX, "s": MPC_VX_MAX, "a": MPC_VYAW_MAX,
                  "d": MPC_VYAW_MAX, "q": MPC_VYAW_MAX, "e": MPC_VYAW_MAX}[c]
            self._kb_vel[c] = min(mx, self._kb_vel[c] + step * mx)
            self._kb_last[c] = now
            self._update_mpc_cmd()

    def _update_mpc_cmd(self):
        if self.mpc is None or not self.mpc_mode:
            return
        v = self._kb_vel
        vx = v["w"] - v["s"]
        vy = 0.0
        vyaw = (v["a"] - v["d"]) + (v["q"] - v["e"])
        self.mpc.set_cmd(vx, vy, vyaw)

    def _warmup_mpc(self):
        """主线程 JIT 预热：进入遥控前编译扩散采样（首次约 20-25s）。

        本环境 JAX 只能在主线程 dispatch（非主线程会卡死），且 GPU 上 JAX
        持久化编译缓存跨进程不生效，因此把首次 plan_once 提前到启动阶段执行：
        连续调用直到单次耗时 <0.5s（实测需 2-3 次编译），之后按 c 进入遥控
        立即可用（~0.1s/次）。
        """
        try:
            self.get_logger().info(
                "[MPC] JIT 预热编译中（首次约 20-25s，"
                "完成后按 c 即时进入遥控，机器人保持趴姿）...")
            q = np.asarray(self.data.qpos[:23], dtype=np.float32)
            qd = np.asarray(self.data.qvel[:22], dtype=np.float32)
            if self.mpc.state is None:
                self.mpc.init_state(q, qd)
            # 只编译后续实际使用的 Ndiffuse 变体（当前生产=1，见 yaml），
            # 跳过首次 Ndiffuse=10 预热
            self.mpc._first = False
            self.mpc.set_cmd(0.0, 0.0, 0.0)
            t0 = time.time()
            for k in range(4):
                t1 = time.time()
                self.mpc.plan_once(q, qd, self.timestamp + k * 0.02)
                dt = time.time() - t1
                if dt < 0.5:
                    break
            # 双视界预热（用户方案模式化 H）：STAIR 的 H20 MBDPI 也要编译，
            # 否则首次切 STAIR 时卡 15s 编译（爬梯中段致命）。切 STAIR 编译
            # 后切回 CRUISE。
            try:
                if hasattr(self.mpc, "mbdpi_h20"):
                    self.mpc.set_mode("STAIR")
                    for k in range(3):
                        t1 = time.time()
                        self.mpc.plan_once(
                            q, qd, self.timestamp + k * 0.02)
                        if time.time() - t1 < 0.5:
                            break
                    self.mpc.set_mode("CRUISE")
            except Exception:
                pass
            self.get_logger().info(
                f"[MPC] JIT 预热完成（{time.time()-t0:.1f}s），随时可按 c 进入遥控")
        except Exception as e:
            self.get_logger().error(f"[MPC] JIT 预热失败（仍可按 c 触发编译）: {e}")
        finally:
            self._mpc_warmup_done = True

    def _start_auto_nav(self):
        """模式 A 第一阶段：先站起（需 3s），站起完成后由
        _maybe_enter_auto_mpc 进入遥控并创建跟随器。"""
        if self.mpc is None or not self.track_enabled:
            self.get_logger().warn("[AUTO] 自动导航不可用：MPC 未就绪或无航点")
            return
        self.get_logger().info("[AUTO] 模式 A 启动：先站起（3s）...")
        self._handle_term_key("z")
        self.auto_stand_t0 = self.timestamp

    def _maybe_enter_auto_mpc(self):
        """模式 A 第二阶段：站起完成后进遥控 + 创建路径跟随器。"""
        if (self.auto_stand_t0 is None
                or self.timestamp - self.auto_stand_t0 < 3.0):
            return
        self._toggle_mpc_mode()
        # 自动导航：yaw 前馈切 1:1 增益（15）——大增益(50)会让反馈严重过冲
        # 形成航向极限环（实测 err 过冲 3rad）；遥控模式保留大增益。
        if self.mpc is not None:
            self.mpc.set_yaw_gain_lo(
                float(os.environ.get("S10_AUTO_YAW_FF_GAIN", "20.0")))
        from s10_mpc.auto_nav import AutoNavFollower
        self.follower = AutoNavFollower(
            self.track_waypoint_positions,
            max_speed=float(os.environ.get("S10_AUTO_VMAX", "4.5")),
            vyaw_max=float(os.environ.get("S10_AUTO_VYAW_MAX", "1.5")),
            yaw_gain=float(os.environ.get("S10_AUTO_YAW_GAIN", "3.0")),
            lookahead=float(os.environ.get("S10_AUTO_LOOKAHEAD", "1.5")),
        )
        if os.environ.get("S10_SKIP_RIDGE_SCAN") != "1":
            self._scan_ridge_zones()
        self.auto_nav_active = True
        # 测试快捷起点（2026-08-07 用户）：S10_AUTO_START_WP>0 时直接从
        # 楼梯前航点开始（跳过巡航段，加快爬梯迭代）。瞬移机器人、前置
        # 航点索引与路径弧长游标。
        start_wp = int(os.environ.get("S10_AUTO_START_WP", "0"))
        if start_wp > 0 and start_wp + 1 < len(self.track_waypoint_positions):
            wp_xy = self.track_waypoint_positions[start_wp]
            # 地面高度：垂直下扫 mj_ray（航点 z≈地形高但坡上不精确，
            # 直接瞬移悬空/嵌入会摔趴）。站姿高取 nominal 0.205。
            gid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                self.model, self.data,
                np.array([float(wp_xy[0]), float(wp_xy[1]), 8.0]),
                np.array([0.0, 0.0, -1.0]),
                None, False, -1, gid)
            if dist > 0.0:
                h_ground = 8.0 - float(dist)
            else:
                h_ground = float(wp_xy[2])
            self.data.qpos[0] = float(wp_xy[0])
            self.data.qpos[1] = float(wp_xy[1])
            self.data.qpos[2] = h_ground + 0.205
            mujoco.mj_forward(self.model, self.data)
            # 起点航点视为已到达，目标 = 下一个航点（否则机器人就在
            # 起点半径内会被 _update_track_progress 立即推进）。
            self.track_next_index = start_wp + 1
            f = self.follower
            if f is not None and hasattr(f, "path_wp_s"):
                try:
                    f._s_cur = float(f.path_wp_s[min(
                        start_wp, len(f.path_wp_s) - 1)])
                except Exception:
                    pass
            self.get_logger().info(
                f"[AUTO] 测试起点 wp{start_wp} @ "
                f"({wp_xy[0]:.1f},{wp_xy[1]:.1f}) "
                f"ground={h_ground:.2f} next=wp{start_wp + 1}")
        # 先由跟随器给出首个指令，再立即规划——避免沿用进入遥控前的
        # 旧动作（零指令下优化器会生成弱反向差速）导致初始偏航
        self._update_auto_nav()

    def _scan_ridge_zones(self):
        """已知地图预扫描（2026-08-06）：沿全局平滑路径 mj_ray 扫地形，
        标出离散台阶（相邻路径点高差 >0.08m）的弧长区间 → path_vlim 限速
        step_vx。解决**航点段内横脊**（wp4→5 航点 z 相同 → step_zone 漏检
        → 0.13m 横脊高速斜撞西漂侧翻，full_course_17~20 复现）。"""
        try:
            f = self.follower
            if f is None or not hasattr(f, "path_pts"):
                return
            pts = f.path_pts
            hs = np.empty(len(pts), dtype=np.float64)
            for k, p in enumerate(pts):
                g = np.array([-1], dtype=np.int32)
                dist = mujoco.mj_ray(
                    self.model, self.data,
                    np.array([p[0], p[1], 8.0]),
                    np.array([0.0, 0.0, -1.0]),
                    None, False, -1, g)
                hs[k] = (8.0 - dist) if g[0] >= 0 else float(p[2])
            dh = np.abs(np.diff(hs))
            # 跳过 wp0→1 起步段（缓坡 z 升 0.475 不是离散台阶，2026-08-07）
            skip_s = float(f.path_wp_s[1]) - 2.0 if len(f.path_wp_s) > 1 else 0.0
            # 阈值 0.12（2026-08-07 巡航提速）：0.08 对赛道微地形太敏感，
            # 误标大量"横脊"把整段压到 1.5；真实横脊 0.13m 仍触发。
            ridge_dh = float(os.environ.get("S10_RIDGE_DH", "0.12"))
            ridge_idx = np.where(
                (dh > ridge_dh) & (f.path_cum[:len(dh)] > skip_s))[0]
            # 横脊限速 1.5（2026-08-06 实测）：2.0 高速斜撞 wp4→5 横脊
            # 侧翻（full_course_24）；1.5 慢速正对过（full_course_23 过了
            # wp4→5）。传播窗口 2m 提前减速（原 3m 覆盖前段太长）。
            step_vx = float(os.environ.get("S10_RIDGE_VX", "1.5"))
            n_ahead = int(float(os.environ.get(
                "S10_RIDGE_AHEAD", "2.0")) / f.path_res)
            n_after = int(float(os.environ.get(
                "S10_RIDGE_AFTER", "1.2")) / f.path_res)
            n_zone = 0
            for k in ridge_idx:
                lo = max(0, k - n_ahead)
                hi = min(len(f.path_vlim), k + n_after)
                f.path_vlim[lo:hi] = np.minimum(
                    f.path_vlim[lo:hi], step_vx)
                # 横脊所在航段标记 step_zone（2026-08-06）：脱困速度
                # 自动降为 RECOVERY_VX_STEP=1.2——2.5m/s 高速斜推横脊
                # 被导向西漂侧翻（full_course_22 复现）。
                s_ridge = float(f.path_cum[k])
                seg_idx = int(np.searchsorted(
                    f.path_wp_s, s_ridge, side="right") - 1)
                if 0 <= seg_idx < len(f.step_zone):
                    f.step_zone[seg_idx] = True
                n_zone += 1
            if n_zone:
                self.get_logger().info(
                    f"[AUTO] 预扫描发现 {n_zone} 处横脊/台阶，"
                    f"已限速 {step_vx} m/s（防高速斜撞西漂）")
        except Exception as _e:
            self.get_logger().warn(f"[AUTO] 横脊预扫描失败: {_e}")
        q = np.asarray(self.data.qpos[:23], dtype=np.float32)
        qd = np.asarray(self.data.qvel[:22], dtype=np.float32)
        self.last_act = self.mpc.plan_once(q, qd, self.timestamp)
        self.get_logger().info(
            f"[AUTO] 跟随器就绪：{len(self.track_waypoint_positions)} 航点，"
            f"终点 z={self.track_waypoint_positions[-1, 2]:.2f}m")

    def _update_auto_nav(self):
        """自动导航每 10 步：更新路径指令 + 推进航点。"""
        if not self.auto_nav_active or self.follower is None:
            return
        if self.track_complete:
            if not self.auto_finish_logged:
                self.auto_finish_logged = True
                self.mpc.set_cmd(0.0, 0.0, 0.0)
                self.get_logger().info("[AUTO] 到达终点，停车")
            return
        pos = self.data.xpos[self.track_body_id][:2]
        q = self.data.xquat[self.track_body_id]
        # 双模式仲裁（用户方案 2.3）：CRUISE / STAIR_SEQUENCE → MPC 权重集
        if (hasattr(self, "follower") and self.follower is not None
                and hasattr(self.mpc, "set_mode")):
            try:
                _prev_mode = getattr(self, "_last_auto_mode", None)
                _yaw = float(np.arctan2(
                    2.0 * (q[3] * q[0] + q[1] * q[2]),
                    1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
                self.follower.update_mode(
                    pos, self.track_next_index, yaw=_yaw,
                    local_map=self.get_local_map())
                self.mpc.set_mode(self.follower.mode)
                if _prev_mode != self.follower.mode:
                    self._last_auto_mode = self.follower.mode
                    self.get_logger().info(
                        f"[AUTO] mode → {self.follower.mode} @ "
                        f"y={pos[1]:.2f} next={self.track_next_index}")
            except Exception as _e:
                import traceback
                traceback.print_exc()
        yaw = float(np.arctan2(
            2.0 * (q[3] * q[0] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
        # roll 安全：侧倾过大立即降速（20Hz 检测，兜住高速转弯侧翻）
        qq = self.data.xquat[self.track_body_id]
        w, x, y, z = qq
        roll = float(np.arctan2(2.0 * (w * x + y * z),
                                1.0 - 2.0 * (x * x + y * y)))
        # roll 安全刹车阈值（S10_AUTO_ROLL_BRAKE，默认 0.45）：
        # 实测 wp7 台阶区斜撞 riser 后持续侧倾 -0.4（未翻），0.3 阈值把推力
        # 压到 0.15 m/s → 永久卡死。台阶区内可放宽到 0.45~0.5（配合限速兜底）。
        _rb = float(os.environ.get("S10_AUTO_ROLL_BRAKE", "0.45"))
        # 爬梯区（STAIR 模式）放宽（2026-08-06）：爬升时机身必然侧倾
        # （左右轮交替接触），0.45 频繁急刹 → 失去动量 → 爬升中侧翻
        # （batch v28 r1 z=1.01 时 roll -1.16 复现）；0.75 给爬梯容差。
        if getattr(self, "follower", None) is not None \
                and getattr(self.follower, "mode", "") == "STAIR":
            _rb = float(os.environ.get("S10_AUTO_ROLL_BRAKE_STAIR", "0.75"))
        if abs(roll) > _rb:
            self.mpc.set_cmd(0.15, 0.0, 0.0)
            self.get_logger().warn(
                f"[AUTO] 侧倾 {roll:.2f}rad（阈值 {_rb}），急刹降速（防侧翻）")
            return
        yaw_rate = float(self.data.cvel[self.track_body_id][2])
        vx, vyaw = self.follower.compute_cmd(
            pos, yaw, self.track_next_index,
            robot_z=float(self.data.xpos[self.track_body_id, 2]),
            yaw_rate=yaw_rate)

        # v162：STAIR 已知地图几何剖面（pitch 仰头 / 机身 z / 速度剖面）。
        # pitch 负=仰头（本工程约定）；base_z 世界系；vx 取 zone 限速与剖面较小值
        # （riser 前减速、踏面恢复）。CRUISE 清除覆盖回感知姿态目标。
        try:
            if getattr(self.follower, "mode", "") == "STAIR":
                _y = float(pos[1])
                _pt = float(self.follower.stair_pitch_ref(
                    np.array([_y]))[0])
                # v190：机身 z 参考前瞻（cruise ref_path_3d 同款 0.2m 前瞻，
                # 提前抬身给后轮跟抬留出腿部工作空间）
                _preview = float(os.environ.get("S10_STAIR_BZ_PREVIEW", "0.0"))
                _bz = float(self.follower.stair_base_z_ref(
                    np.array([_y + _preview]))[0])
                self.mpc.set_stair_ref(_pt, _bz)
                _v = float(self.follower.stair_v_ref(np.array([_y]))[0])
                vx = min(vx, _v)
            else:
                if hasattr(self.mpc, "clear_stair_ref"):
                    self.mpc.clear_stair_ref()
        except Exception:
            pass

        # v168：场驱动抬腿动作偏置（软先验）——每条腿按 wheel_ref 场"欠抬量"
        # （目标轮心高 - 当前轮心高）注入动作空间偏置：前膝缩回（抬前轮）、
        # 后膝弯曲（抬后轮），随进度自然切换（前轮先、后轮后）。仅移动采样
        # 均值，MPPI 权重/cost 可覆盖；CRUISE 清空。
        try:
            if getattr(self.follower, "mode", "") == "STAIR":
                _f = self.follower
                _wpos = self.data.xpos[[5, 9, 13, 17]]
                _wy = np.asarray(_wpos[:, 1], dtype=np.float64)
                _wz = np.asarray(_wpos[:, 2], dtype=np.float64)
                _wr = np.asarray(_f.stair_wheel_ref(_wy), dtype=np.float64)
                _lift = np.clip(_wr - _wz, 0.0, 0.25)
                # v171：时变偏置——按视界内轮子前移后的欠抬量生成逐节点抬腿
                # 剖面（H+1,12），给采样器完整"抬-落"先验；欠抬 <0.05m 视为
                # 到位不注入（防过早/过猛）。
                _Hn = int(self.mpc.Y.shape[0])
                _dt = float(getattr(self.mpc, "dt", 0.02))
                _vk = float(vx)
                _bH = np.zeros((_Hn, 12), dtype=np.float32)
                for _k in range(_Hn):
                    _yk = _wy + _vk * _k * _dt
                    _wrk = np.asarray(_f.stair_wheel_ref(_yk), dtype=np.float64)
                    _lk = np.clip(_wrk - _wz, 0.0, 0.25)
                    # v203: 抬升触发阈值参数化（S10_BIAS_LIFT_MIN，默认 0.05）。
                    # 卡点实测四轮距 ref 只差 2.7~3.3cm（<0.05 被静音），
                    # 降阈值让"临界欠抬"也注入 bias 先验。
                    _lk = np.where(
                        _lk < float(os.environ.get("S10_BIAS_LIFT_MIN", "0.05")),
                        0.0, _lk)
                    _nk = np.clip(_lk / 0.15, 0.0, 1.0)
                    _b12 = np.zeros(12, dtype=np.float32)
                    # v176：四轮完整偏置（Y 混合收敛后无累积过抬问题）
                    # v186：恢复 v176 实证有效配方（唯一越过第二级 riser 的
                    # 组合；膝偏置符号按 v176 原样，hipy 为主抬升驱动）。
                    # v201: bias 系数参数化（运动学校准，2026-08-08 卡点可达性分析）：
                    # 卡点姿态下（前轮挂 riser2/3、后轮在 riser1 面）局部导数：
                    #   前轮 hipy+/knee+ 抬升（knee+ 0.45rad -> +7.6cm）
                    #   后轮 hipy+ 抬升（+5.5cm）、knee- 抬升（+5.4cm）
                    # v176 原系数（前 knee -0.50 / 后 hipy -0.10 / 后 knee +0.45）
                    # 在该姿态方向相反（会把轮压向地面），与 ground cost 拉锯
                    # -> 轮速正反震荡死锁。默认仍保持 v176，可用 env 覆盖为
                    # 运动学方向（见 tmp/run_stair_kin.sh）。
                    _bc = np.zeros(8, dtype=np.float32)
                    _bc[0] = float(os.environ.get("S10_BIAS_FL_HIPY", "0.20"))
                    _bc[1] = float(os.environ.get("S10_BIAS_FL_KNEE", "-0.50"))
                    _bc[2] = float(os.environ.get("S10_BIAS_FR_HIPY", "0.20"))
                    _bc[3] = float(os.environ.get("S10_BIAS_FR_KNEE", "-0.50"))
                    _bc[4] = float(os.environ.get("S10_BIAS_HL_HIPY", "-0.10"))
                    _bc[5] = float(os.environ.get("S10_BIAS_HL_KNEE", "0.45"))
                    _bc[6] = float(os.environ.get("S10_BIAS_HR_HIPY", "-0.10"))
                    _bc[7] = float(os.environ.get("S10_BIAS_HR_KNEE", "0.45"))
                    _b12[1] = _bc[0] * _nk[0]
                    _b12[2] = _bc[1] * _nk[0]
                    _b12[4] = _bc[2] * _nk[1]
                    _b12[5] = _bc[3] * _nk[1]
                    _b12[7] = _bc[4] * _nk[2]
                    _b12[8] = _bc[5] * _nk[2]
                    _b12[10] = _bc[6] * _nk[3]
                    _b12[11] = _bc[7] * _nk[3]
                    _bH[_k] = _b12
                if hasattr(self.mpc, "set_stair_action_bias"):
                    self.mpc.set_stair_action_bias(_bH)
            else:
                if hasattr(self.mpc, "set_stair_action_bias"):
                    self.mpc.set_stair_action_bias(None)
        except Exception:
            pass

        # 高程图限速（世界系 local_map，姿态无关）：沿世界航向前方 0.5~2m
        # 采样，坡度大则减速。旧 10×16 机身系地图在俯仰时 xy 被 cos(pitch)
        # 压缩、坡度高估（0806 §3.6 姿态敏感点 1），已弃用。
        try:
            lm = self.get_local_map()
            if lm is not None:
                hm, valid = lm["heightmap"], lm["valid"]
                res = float(lm["resolution"])
                ox, oy = float(lm["origin"][0]), float(lm["origin"][1])
                fwd = self.data.xmat[self.track_body_id][::3][:2]
                fn = float(np.linalg.norm(fwd))
                if fn > 1e-6:
                    fwd = fwd / fn
                pos = self.data.xpos[self.track_body_id][:2]
                hs = []
                for d in (0.5, 1.0, 1.5, 2.0):
                    p = pos + fwd * d
                    i = int(np.floor((p[1] - oy) / res))
                    j = int(np.floor((p[0] - ox) / res))
                    if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                            and valid[i, j]):
                        hs.append((float(hm[i, j]), d))
                if len(hs) >= 2:
                    slope = max(
                        abs(hs[k][0] - hs[k - 1][0])
                        / max(1e-3, hs[k][1] - hs[k - 1][1])
                        for k in range(1, len(hs)))
                    vx = min(vx, vx / (1.0 + 3.0 * slope))
        except Exception:
            pass

        # 爬坡直线跟随：感知地图检测持续上升（2m 前瞻高 >0.12m）**或航点 z 兜底
        # step_zone（已知地图，不依赖 2m 前瞻覆盖）**且离路径近（|cte|<3m）时，
        # 锁定航段方向直线爬——pursuit 弧线 + 脱困反复会在坡上累积横向漂移
        # （实测西漂 7m 迷路；wp7 区 2m 前瞻格常无效，必须用航点 z 兜底）。
        try:
            lm = self.get_local_map()
            f = self.follower
            in_step_zone = False
            if f is not None and 0 < self.track_next_index <= len(
                    getattr(f, "step_zone", [])):
                in_step_zone = bool(
                    f.step_zone[self.track_next_index - 1])
            if (lm is not None and f is not None
                    and abs(getattr(f, "_last_cte", 99.0)) < 3.0
                    and (getattr(f, "_last_dwp", 0.0) > 1.2
                         or in_step_zone)
                    and 0 < self.track_next_index < len(f.heading)):
                rise_ok = in_step_zone
                if not rise_ok:
                    hm = lm["heightmap"]
                    valid = lm["valid"]
                    res = float(lm["resolution"])
                    ox, oy = float(lm["origin"][0]), float(lm["origin"][1])
                    fwd = self.data.xmat[self.track_body_id][::3][:2]
                    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
                    ahead = pos + fwd * 2.0
                    i = int(np.floor((pos[1] - oy) / res))
                    j = int(np.floor((pos[0] - ox) / res))
                    ia = int(np.floor((ahead[1] - oy) / res))
                    ja = int(np.floor((ahead[0] - ox) / res))
                    if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                            and valid[i, j]
                            and 0 <= ia < hm.shape[0] and 0 <= ja < hm.shape[1]
                            and valid[ia, ja]):
                        rise_ok = float(hm[ia, ja]) - float(hm[i, j]) > 0.12
                if rise_ok and pos[1] < float(os.environ.get(
                        "S10_AUTO_LOCK_END_Y", "39.0")):
                    # v157：锁原始航段方向（v137/v156 切线锁失控东跑 25m 已弃）。
                    # 顶缘前（y<39.0）直线锁防漂移；顶缘后关闭，交回正常
                    # pursuit 跟走廊路径曲线（v155 顶缘东漂 = 直线锁在路径
                    # 向西弯时仍锁北向）。S10_AUTO_LOCK_END_Y 可调。
                    seg_h = float(f.heading[self.track_next_index - 1])
                    err_l = float(np.arctan2(
                        np.sin(seg_h - yaw), np.cos(seg_h - yaw)))
                    cte = float(getattr(f, "_last_cte", 0.0))
                    # STAIR 区 cte 纠偏加倍（爬梯时路径 cost 被地形项压制，
                    # 横向漂移收敛慢——v136 r1 漂 0.7m 上脊的诱因之一）。
                    _cte_k = 2.0 if getattr(f, "mode", "") == "STAIR" else 1.0
                    cte_corr = -f.cte_gain * _cte_k * float(
                        np.clip(cte / 2.0, -1.0, 1.0))
                    z_ahead = float(self.data.xpos[self.track_body_id, 2])
                    elev_k = float(os.environ.get("S10_AUTO_ELEV_K", "0.6"))
                    ef = 1.0 / (1.0 + elev_k * max(0.0, z_ahead - 0.4))
                    vyaw_l = float(np.clip(
                        f.yaw_gain * err_l - f.yaw_damp * yaw_rate + cte_corr,
                        -f.vyaw_max * ef, f.vyaw_max * ef))
                    # 爬坡提速温和化：2.5 m/s 高速接近台阶会在平地弹跳翻车
                    # （实测 y≈26.2 翻车），用户判断"慢速更易抬腿"——默认 1.5。
                    # step_zone 兜底时也保证最低 1.5（与 S10_AUTO_STEP_VX 一致，
                    # 避免 pursuit 限速把台阶前速度压到 0 造成动量不足）。
                    vx = max(vx, float(os.environ.get(
                        "S10_AUTO_CLIMB_VX", "1.5")))
                    vyaw = vyaw_l
        except Exception:
            pass

        # 感知高程图引导抬腿（监督式）：检测轮前台阶，给对应膝目标加抬升偏置。
        # 轮前 0.3m 地形比轮下高 >0.05m → lift∈(0,1]，膝偏置 = lift·0.35 rad。
        self._compute_leg_assist()

        # 爬坡模式已停用：加速推力在坡顶会把机器人"发射"翻车（r27-r30 实测）。
        # 台阶靠"卡死→持续前推 2.5 m/s"缓慢但安全地爬（r26 实测可达 wp7）。

        # ---- 卡死检测与脱困 ----
        # 指令 vx>0.5 但实际速度持续 <0.3 达 5s（轮子空转/顶住台阶边缘）→
        # 中速前推脱困（腿部碰撞已移除：台阶靠持续推力+动量越过；
        # 3.5m/s 从静止硬推会发射翻车，2.5m/s 更稳）。
        rec_t = getattr(self, "_recovery_t", 0.0)
        if rec_t > 0.0:
            self._recovery_t = rec_t + 0.05
            # v204：脱困先"后退释放楔死"再前冲（导航层，无 MPC 门控）。
            # 卡点 = 前轮挂 riser2/3、后轮在 riser1 面的对角楔死；直接前推
            # 只磨轮子。先反向 0.8m/s 退 1s 释放接触，再按原前推逻辑再攻。
            _bk_vx = float(os.environ.get("S10_AUTO_RECOVERY_BACKUP_VX", "0.0"))
            _bk_t = float(os.environ.get("S10_AUTO_RECOVERY_BACKUP_T", "1.0"))
            if _bk_vx > 0.0 and rec_t < _bk_t:
                self.mpc.set_cmd(-_bk_vx, 0.0, 0.0)
                if int(rec_t * 20) % 5 == 0:
                    self.get_logger().warn(
                        f"[AUTO] 脱困后退 {rec_t:.1f}/{_bk_t:.1f}s "
                        f"vx={-_bk_vx:.1f} m/s 释放楔死")
                self._update_track_progress()
                return
            # 前推 + 航向纠偏（链 55 修复）：之前用"冻结的 _last_cmd_vyaw"
            # ——脱困 3.5s 内无横向修正，横脊前反复脱困累积西漂 → 斜撞
            # 横脊侧翻（chain 55 r1 复现）。改为**锁定当前航段 heading**
            # （爬坡直线方向）+ cte 横向修正，脱困时保持直线不漂移。
            f = getattr(self, "follower", None)
            yaw_now = float(np.arctan2(
                2.0 * (q[3] * q[0] + q[1] * q[2]),
                1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
            if (f is not None
                    and 0 < self.track_next_index < len(f.heading)):
                seg_h = float(f.heading[self.track_next_index - 1])
                err_l = float(np.arctan2(
                    np.sin(seg_h - yaw_now), np.cos(seg_h - yaw_now)))
                cte = float(getattr(f, "_last_cte", 0.0))
                cte_corr = -f.cte_gain * float(
                    np.clip(cte / 2.0, -1.0, 1.0))
                vyaw_rec = f.yaw_gain * err_l - f.yaw_damp * yaw_rate \
                    + cte_corr
            else:
                vyaw_rec = 0.0
            # 限幅：2.5 m/s 前推时急转会甩翻（wp8 前实测），最多 0.35 rad/s
            vyaw_rec = float(np.clip(vyaw_rec, -0.35, 0.35))
            # 脱困前推速度：默认 2.5；台阶区内（step_zone）降为 1.2——实测
            # 2.5 斜撞 riser 会侧倾卡死（roll -0.4），低速 + 抬腿更稳。
            rec_vx = float(os.environ.get("S10_AUTO_RECOVERY_VX", "2.5"))
            if (hasattr(self, "follower") and self.follower is not None
                    and 0 < self.track_next_index <= len(
                        getattr(self.follower, "step_zone", []))
                    and self.follower.step_zone[self.track_next_index - 1]):
                rec_vx = float(os.environ.get(
                    "S10_AUTO_RECOVERY_VX_STEP", "1.2"))
            self.mpc.set_cmd(rec_vx, 0.0, vyaw_rec)
            if rec_t >= float(os.environ.get(
                    "S10_AUTO_RECOVERY_BACKUP_T", "1.0")) + 5.0:
                self._recovery_t = 0.0
                self._stall_t = 0.0
            self.get_logger().warn(
                f"[AUTO] 脱困中 ({rec_t:.1f}/3.5s)：前推 {rec_vx:.1f} m/s 过台阶")
            self._update_track_progress()
            return
        v_now = float(np.linalg.norm(
            self.data.cvel[self.track_body_id][3:5]))
        stall_t = getattr(self, "_stall_t", 0.0) + 0.05
        if vx > 0.5 and v_now < 0.3:
            self._stall_t = stall_t
        else:
            self._stall_t = 0.0
        if self._stall_t > 5.0:
            self._recovery_t = 0.05
            self.get_logger().warn(
                f"[AUTO] 检测到卡死（vx={vx:.1f} v={v_now:.2f}），开始脱困")
            self._update_track_progress()
            return

        self.mpc.set_cmd(vx, 0.0, vyaw)
        # E4：把 3D 参考路径（minisnap 平滑 + 高程图 z 轨迹，用户指示 2）
        # 注入 MPC reward（S10_MPC_W_PATH/_HEAD/_Z > 0 时启用路径跟踪）。
        try:
            lm = self.get_local_map()
            ref = self.follower.ref_path_3d(
                pos, self.track_next_index, local_map=lm)
            self.mpc.set_ref_path(ref if ref is not None else [],
                                  valid=ref is not None)
        except Exception:
            self.mpc.set_ref_path([], valid=False)
        # 抬轮 reward 镜像调试（S10_LIFT_DEBUG=1）：与 s10_env._reward 相同
        # 采样逻辑（CPU 侧），打印每个轮子的 h_terrain/h_ahead/step_ahead。
        if os.environ.get("S10_LIFT_DEBUG") and int(self.timestamp * 2) % 4 == 0:
            try:
                lm = self.get_local_map()
                if lm is not None:
                    hm = lm["heightmap"]
                    valid = lm["valid"]
                    res = float(lm["resolution"])
                    ox, oy = float(lm["origin"][0]), float(lm["origin"][1])
                    stepf = (lm.get("features") or {}).get("step_flag")
                    fwd = self.data.xmat[self.track_body_id][::3][:2]
                    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
                    lifts = []
                    for bid in WHEEL_CONTACT_IDS:
                        xy = self.data.xpos[bid][:2]
                        wz = float(self.data.xpos[bid, 2])

                        def _samp(p):
                            i = int(np.floor((p[1] - oy) / res))
                            j = int(np.floor((p[0] - ox) / res))
                            if (0 <= i < hm.shape[0] and 0 <= j < hm.shape[1]
                                    and valid[i, j]):
                                return float(hm[i, j]), float(
                                    stepf[i, j]) if stepf is not None else 0.0
                            return None
                        h_now = _samp(xy)
                        h_ter = (h_now[0] if h_now is not None
                                 else wz - 0.081)
                        hs, ss = [], []
                        for dd in (0.15, 0.28, 0.4):
                            hv = _samp(xy + fwd * dd)
                            if hv is not None:
                                hs.append(hv[0])
                                ss.append(hv[1])
                        h_ahead = max(hs) if hs else -9.9
                        s_ahead = max(ss) if ss else 0.0
                        lifts.append(
                            f"h={h_ter:.2f}/a={h_ahead:.2f}/s={s_ahead:.1f}"
                            f"({'ON' if (h_ahead - h_ter > 0.05 and s_ahead > 0.3) else '--'})")
                    print(f"[LIFT] y={pos[1]:.2f} " + " ".join(lifts),
                          flush=True)
            except Exception as _e:
                import traceback
                traceback.print_exc()
        self._last_cmd_vyaw = float(vyaw)
        self._update_track_progress()

    def _compute_leg_assist(self):
        """感知引导的静态膝偏置（S10_LEG_ASSIST=1 时启用；默认关）。

        2026-08-05 用户决策：**不用 CTBC 式时序摆动**（链 13~15 实测失败，
        CTBC/Chamarro 是 RL 训练出的行为，手写反射在采样 MPC 里拧巴）。
        恢复为链 5 曾 2/2 通过 wp5→wp6 双 riser 的静态比例偏置：
        前轮检出台阶（0.15~0.45m 窗口 + step_flag 门控）→ 膝偏置 ∝ 台阶净高；
        后轮"跟抬"（同侧前轮已上顶）默认关（S10_LIFT_REAR=0）。
        保留轮周 ±0.2m min-窗口地面基准（修复 riser 阴影空洞导致触发失效）。
        """
        assist_on = os.environ.get("S10_LEG_ASSIST", "0") == "1"
        if not assist_on:
            self._leg_assist = np.zeros(12, dtype=np.float32)
            return
        amp = float(os.environ.get("S10_LEG_ASSIST_AMP", "0.20"))
        amp_rear = float(os.environ.get("S10_LEG_ASSIST_AMP_REAR", "0.15"))
        dist = float(os.environ.get("S10_LEG_ASSIST_DIST", "0.30"))
        assist = np.zeros(12, dtype=np.float32)
        try:
            lm = self.get_local_map()
            if lm is None:
                self._leg_assist = assist
                return
            hm = lm["heightmap"]
            valid = lm["valid"]
            res = float(lm["resolution"])
            ox, oy = float(lm["origin"][0]), float(lm["origin"][1])
            fwd = self.data.xmat[self.track_body_id][::3][:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            # 轮前 0.15~0.45m 三点窗口采样（与 dial-mpc reward lift 同窗口风格），
            # 取最大高差；只有出现离散台阶（step_flag）才触发，避免坡道误抬。
            step_f = lm.get("features", {})
            step_flag = step_f.get("step_flag")
            step_thr = float(os.environ.get("S10_LEG_ASSIST_STEP_THR", "0.5"))
            for wi, (bid, knee_idx, sign) in enumerate(
                    ((5, 2, 1.0), (9, 5, 1.0), (13, 8, -1.0), (17, 11, -1.0))):
                xy = self.data.xpos[bid][:2]

                def _h(p):
                    i = int(np.floor((p[1] - oy) / res))
                    j = int(np.floor((p[0] - ox) / res))
                    if 0 <= i < hm.shape[0] and 0 <= j < hm.shape[1] \
                            and valid[i, j]:
                        return float(hm[i, j])
                    return None

                # 轮下地面基准（链 5 原样）：轮下地图格无效时用"轮心真实
                # 高度 − 轮半径"。曾改"轮周 ±0.2m min-窗口"回退，实测造成
                # reward 侧左右轮不对称侧翻（链 20），此处同步还原。
                h_now = _h(xy)
                if h_now is None:
                    h_now = float(self.data.xpos[bid, 2]) - 0.081
                step_top = h_now
                if bid in (13, 17):
                    # 后轮"跟抬"（2026-08-05 用户指导）：后轮前视窗口在 riser
                    # 阴影里恒空洞（LiDAR 被立面遮挡，探测证实），前视采样触发
                    # 不了；改为跟抬——同侧前轮已上台阶顶（前轮下地形高于后轮
                    # 下地形），就把后轮抬到前轮高度。后膝符号 -（几何实测：
                    # 后膝更负 = 抬轮，见 tmp 前向运动学验证）。
                    if os.environ.get("S10_LIFT_REAR", "1") == "1":
                        fid = 5 if bid == 13 else 9
                        fxy = self.data.xpos[fid][:2]
                        h_front = _h(fxy)
                        if h_front is None:
                            h_front = float(self.data.xpos[fid, 2]) - 0.081
                        diff = h_front - h_now
                        bxy = fxy - fwd * 0.15
                        bi = int(np.floor((bxy[1] - oy) / res))
                        bj = int(np.floor((bxy[0] - ox) / res))
                        bstep = 0.0
                        if step_flag is not None and 0 <= bi < hm.shape[0] \
                                and 0 <= bj < hm.shape[1] and valid[bi, bj]:
                            bstep = float(step_flag[bi, bj])
                        if diff > 0.05 and bstep >= step_thr:
                            wheel_z = float(self.data.xpos[bid, 2])
                            if wheel_z < h_front + 0.081 - 0.03:
                                lift = float(np.clip(
                                    (diff - 0.05) / 0.08, 0.0, 1.0))
                                assist[knee_idx] = sign * amp_rear * lift
                        continue
                else:
                    # 前轮：0.15~0.45m 三点窗口，离散台阶才触发（避免坡道误抬）
                    best = 0.0
                    best_flag = 0.0
                    for d in (dist * 0.5, dist * 0.75, dist):
                        p = xy + fwd * d
                        i = int(np.floor((p[1] - oy) / res))
                        j = int(np.floor((p[0] - ox) / res))
                        if not (0 <= i < hm.shape[0]
                                and 0 <= j < hm.shape[1]
                                and valid[i, j]):
                            continue
                        best = max(best, float(hm[i, j]) - h_now)
                        if step_flag is not None:
                            best_flag = max(
                                best_flag, float(step_flag[i, j]))
                    if best > 0.05 and best_flag >= step_thr:
                        wheel_z = float(self.data.xpos[bid, 2])
                        if wheel_z < best + h_now + 0.081 - 0.03:
                            lift = float(np.clip(
                                (best - 0.05) / 0.08, 0.0, 1.0))
                            assist[knee_idx] = sign * amp * lift
        except Exception:
            pass
        self._leg_assist = assist
        if int(self.timestamp * 2) % 2 == 0 and os.environ.get("S10_AUTO_DEBUG"):
            aw = np.asarray(self.mpc.Y[0, 12:])
            la = np.asarray(self.last_act) if self.last_act is not None else None
            tau_w = np.asarray(self.mpc.latest_tau)[[3, 7, 11, 15]]
            print(f"[LIFT] t={self.timestamp:.1f} "
                  f"active={self._lift_active.astype(int)} "
                  f"timer={np.round(self._lift_timer, 2)} "
                  f"assist={np.round(np.asarray(self._leg_assist)[[2, 5, 8, 11]], 2)} "
                  f"wz={np.round(np.asarray(self.data.xpos[[5, 9, 13, 17], 2]), 2)}",
                  flush=True)
            xyz = self.data.xpos[self.track_body_id]
            qq = self.data.xquat[self.track_body_id]
            yy = float(np.arctan2(
                2.0 * (qq[3] * qq[0] + qq[1] * qq[2]),
                1.0 - 2.0 * (qq[2] ** 2 + qq[3] ** 2)))
            cv = np.asarray(self.mpc.cmd_vel)
            ca = np.asarray(self.mpc.cmd_ang)
            print(f"[AUTO-DBG] t={self.timestamp:.1f} wp={self.track_next_index} "
                  f"pos=({xyz[0]:.1f},{xyz[1]:.1f}) yaw={yy:.2f} "
                  f"cmd=({cv[0]:.2f},{ca[2]:.2f}) Yw={np.round(aw, 2)} "
                  f"act_w={np.round(la[12:], 2) if la is not None else None} "
                  f"tau_w={np.round(tau_w, 1)} "
                  f"leg_amp={np.round(np.abs(la[:12]).max(), 2) if la is not None else None}",
                  flush=True)

    def _toggle_mpc_mode(self):
        if self.mpc is None:
            self.get_logger().warn("[MPC] 控制器仍在构建中，请稍后再按 c")
            return
        if not self.mpc_mode:
            q = np.asarray(self.data.qpos[:23], dtype=np.float32)
            qd = np.asarray(self.data.qvel[:22], dtype=np.float32)
            if self.mpc.state is None:
                self.mpc.init_state(q, qd)
            self.mpc_mode = True     # 先初始化 state，再置模式，避免主循环竞态
            self.last_act = None
            self.mpc.set_cmd(0.0, 0.0, 0.0)
            self.get_logger().info(
                "[MPC] 进入遥控模式（c）：主线程 MBDPI 规划"
                "（JIT 已在启动时预热），wasd 移动 qe 转向")
        else:
            self.mpc_mode = False
            if self.mpc is not None:
                self.mpc.set_yaw_gain_lo(None)   # 恢复遥控大增益
            self.get_logger().info("[MPC] 退出遥控模式")

    def _init_mpc_background(self):
        """后台构建 dial-mpc MBDPI 控制器（仅 CPU/模型构建，无 JAX dispatch）。

        注意：本环境 JAX 不允许在非主线程 dispatch（会卡死），所以预编译
        plan_once 不放这里；JIT 预热在启动时主线程执行。
        """
        try:
            from s10_mpc.mpc_controller import MPCController
            self.mpc = MPCController(MPC_YAML)
            self.mpc.set_cmd(0.0, 0.0, 0.0)
            self.mpc.ready = False  # 首次 plan_once 在启动预热 JIT 编译后置 True
            self.get_logger().info(
                "[MPC] dial-mpc 控制器构建完成（启动后自动 JIT 预热）。"
                "terminal 键盘（原始键位）："
                "z 站起 / c 遥控 / r 阻尼 / wasd 移动 qe 转向（按住=加速，松开=停）")
        except Exception as e:
            import traceback
            self.get_logger().error(f"[MPC] 初始化失败: {e}")
            traceback.print_exc()
            self.mpc = None

    def _set_initial_pose(self, key: str):
        """关节位置设置为与 PyBullet 脚本一致的初始角度"""
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = JOINT_INIT[key]  # ,3-6 basequat，0-2 basepos
        qpos0[:3] = TRACK_START_BASE_POS
        qpos0[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def _track_geom_index(self, name: str, prefix: str):
        if not name or not name.startswith(prefix):
            return None
        suffix = name[len(prefix):]
        index_text = suffix.split("_", 1)[0]
        if not index_text.isdigit():
            return None
        return int(index_text)

    def _find_track_geoms(self):
        waypoint_geoms = {}
        point_related_geoms = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            waypoint_index = self._track_geom_index(name, TRACK_WAYPOINT_PREFIX)
            if waypoint_index is not None:
                waypoint_geoms[waypoint_index] = geom_id
                point_related_geoms.setdefault(waypoint_index, []).append(geom_id)
                continue

            post_index = self._track_geom_index(name, TRACK_HEIGHT_POST_PREFIX)
            if post_index is not None:
                point_related_geoms.setdefault(post_index, []).append(geom_id)

        return waypoint_geoms, point_related_geoms

    def _init_track_progress(self):
        self.track_enabled = False
        self.track_complete = False
        self.track_next_index = 0
        self.track_start_time = None
        self.track_finish_time = None
        self.track_waypoint_positions = np.empty((0, 3), dtype=np.float64)
        self.track_point_geom_ids = {}

        waypoint_geoms, point_related_geoms = self._find_track_geoms()
        if not waypoint_geoms:
            return

        expected_indices = list(range(max(waypoint_geoms) + 1))
        missing = [index for index in expected_indices if index not in waypoint_geoms]
        if missing:
            self.get_logger().warn(f"Track progress disabled; missing waypoint geoms: {missing}")
            return

        self.track_waypoint_geom_ids = [waypoint_geoms[index] for index in expected_indices]
        self.track_point_geom_ids = {
            index: point_related_geoms.get(index, [waypoint_geoms[index]])
            for index in expected_indices
        }
        self.track_waypoint_positions = np.array(
            [self.data.geom_xpos[geom_id].copy() for geom_id in self.track_waypoint_geom_ids],
            dtype=np.float64,
        )
        self.track_original_rgba = {
            geom_id: self.model.geom_rgba[geom_id].copy()
            for geom_ids in self.track_point_geom_ids.values()
            for geom_id in geom_ids
        }
        self.track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TRACK_BODY_NAME)
        if self.track_body_id < 0:
            self.get_logger().warn(f"Track progress disabled; cannot find body '{TRACK_BODY_NAME}'")
            return

        self.track_enabled = True
        self.get_logger().info(
            f"[INFO] Track progress enabled: {len(self.track_waypoint_positions)} waypoints, "
            f"radius={TRACK_REACH_RADIUS:.3f}m, distance_mode={TRACK_DISTANCE_MODE}"
        )

    def _hide_track_point(self, waypoint_index: int):
        for geom_id in self.track_point_geom_ids.get(waypoint_index, []):
            self.model.geom_rgba[geom_id, 3] = 0.0

    def _track_distance(self, robot_pos: np.ndarray, waypoint_pos: np.ndarray) -> float:
        if TRACK_DISTANCE_MODE == "xyz":
            return float(np.linalg.norm(robot_pos - waypoint_pos))
        return float(np.linalg.norm(robot_pos[:2] - waypoint_pos[:2]))

    def _update_track_progress(self):
        if not self.track_enabled or self.track_complete:
            return
        if self.track_next_index >= len(self.track_waypoint_positions):
            return

        robot_pos = self.data.xpos[self.track_body_id]
        waypoint_pos = self.track_waypoint_positions[self.track_next_index]
        distance = self._track_distance(robot_pos, waypoint_pos)
        if distance > TRACK_REACH_RADIUS:
            return

        reached_index = self.track_next_index
        self._hide_track_point(reached_index)

        if reached_index == 0 and self.track_start_time is None:
            self.track_start_time = self.timestamp
            self.get_logger().info(
                f"[TRACK] Timer started at waypoint 0, sim_time={self.track_start_time:.3f}s"
            )
        else:
            self.get_logger().info(
                f"[TRACK] Reached waypoint {reached_index}, sim_time={self.timestamp:.3f}s, "
                f"distance={distance:.3f}m"
            )

        self.track_next_index += 1
        if self.track_next_index >= len(self.track_waypoint_positions):
            self.track_complete = True
            self.track_finish_time = self.timestamp
            elapsed = 0.0 if self.track_start_time is None else self.track_finish_time - self.track_start_time
            self.get_logger().info(
                f"[TRACK] Final waypoint reached. Timer stopped at sim_time={self.track_finish_time:.3f}s, "
                f"elapsed={elapsed:.3f}s"
            )

    def _configure_viewer(self):
        with self.viewer.lock():
            track_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                TRACK_BODY_NAME,
            )
            if TRACK_VIEWER and track_body_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = track_body_id
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.lookat[:] = self.data.qpos[:3]

                if TRACK_VIEWER:
                    self.get_logger().warn(
                        f"Cannot find body '{TRACK_BODY_NAME}'; viewer camera tracking disabled"
                    )

            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.azimuth = CAMERA_AZIMUTH
            self.viewer.cam.elevation = CAMERA_ELEVATION
            self.viewer.cam.distance = CAMERA_DISTANCE

            if COLLISION_GEOM_GROUP < len(self.viewer.opt.geomgroup):
                self.viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0

    def _init_lidar(self):
        """初始化 Airy-96 LiDAR（360°×96 线，垂直 ±7°，site 前倾 ~19.5° 覆盖近场地面）。

        说明：
        - 库自带 generate_airy96() 的垂直角为 0°~89.5°（全朝上），无法扫到地面，
          故按真机规格生成等效角表：水平 360° × 垂直 96 线 ±7°。
        - site 前倾角见 S10.xml lidar_site euler="0 0.34 0"（本模型 compiler
          angle="radian"，0.34 rad ≈ 19.5°）。
        """
        self.lidar_type = "airy96"  # 结构对齐 RoboSense Airy-96
        self.lidar_backend = LIDAR_BACKEND  # 优先 taichi(GPU)，失败自动回退 cpu
        self.lidar_frequency = LIDAR_FREQ
        self.lidar_cutoff_dist = LIDAR_CUTOFF

        # 生成 Airy-96 结构角表：前向扇区 ±LIDAR_FOV_H_DEG × 垂直 LIDAR_PHI_N 线，
        # 垂直 FOV ±7°。水平仅覆盖前方（比赛只需要前方高程图，全局位姿已知无需全景）。
        fov_h = np.deg2rad(LIDAR_FOV_H_DEG)
        self.rays_theta, self.rays_phi = scan_gen.generate_grid_scan_pattern(
            num_ray_cols=LIDAR_THETA_N,
            num_ray_rows=LIDAR_PHI_N,
            theta_range=(-fov_h, fov_h),
            phi_range=(np.deg2rad(-LIDAR_PHI_DEG), np.deg2rad(LIDAR_PHI_DEG)),
        )
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        # 排除自身：只保留 group 0（地形/floor）。
        # group 1 = 机器人碰撞体，group 2 = 机器人视觉 + track_overlay（赛道线/航点），
        # 全部关闭，避免机器人本体与赛道指示物被扫入高程图。
        geomgroup = np.zeros((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[0] = 1
        bodyexclude = self.model.body("base_link").id

        # 创建 LiDAR wrapper（taichi → cpu 自动回退）
        self.lidar = None
        for backend in ([self.lidar_backend, "cpu"] if self.lidar_backend != "cpu" else ["cpu"]):
            try:
                self.lidar = MjLidarWrapper(
                    self.model,
                    site_name="lidar_site",
                    backend=backend,
                    cutoff_dist=self.lidar_cutoff_dist,
                    args={"bodyexclude": bodyexclude, "geomgroup": geomgroup},
                )
                self.lidar_backend = backend
                break
            except Exception as e:
                self.get_logger().warn(f"[WARN] LiDAR backend '{backend}' init failed: {e}")
        if self.lidar is None:
            self.get_logger().error("[ERROR] Failed to initialize LiDAR (all backends)")
            return

        self.get_logger().info(
            f"[INFO] LiDAR initialized: type={self.lidar_type}, "
            f"backend={self.lidar_backend}, rays={self.rays_theta.shape[0]}, "
            f"cutoff={self.lidar_cutoff_dist}m"
        )

        # 高程图状态
        self.base_body_id = self.model.body("base_link").id
        self.elevation_map = None
        self.elevation_valid = None
        if LOCAL_MAP_AVAILABLE:
            self.local_map = LocalMap(
                LOCAL_MAP_CFG,
                robot_xy=self.data.xpos[self.base_body_id][:2],
                stamp=self.timestamp)

        # 初始化PointCloud2消息模板
        self.pc_msg = PointCloud2()
        self.pc_msg.header.frame_id = "lidar_link"
        self.pc_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self.pc_msg.is_bigendian = False
        self.pc_msg.point_step = 12  # 3 * sizeof(float32)
        self.pc_msg.height = 1
        self.pc_msg.is_dense = True

        # LiDAR发布时间控制
        self.last_lidar_pub_time = -1.0

        # 可视化射线 + 高程图面片（在MuJoCo viewer中显示）
        self._init_lidar_visualization()
        if VIZ_ELEVATION:
            self._init_elevation_visualization()

        # 后台采集线程：独立 MjData + mj_forward 同步主线程状态后 trace，
        # 避免阻塞 200 Hz 主循环（mujoco C 调用释放 GIL 可并行，但必须隔离 data）。
        self._lidar_lock = threading.Lock()
        self._lidar_state = None
        self._lidar_next_time = time.monotonic()
        self._lidar_worker_data = mujoco.MjData(self.model)
        self._lidar_thread = threading.Thread(target=self._lidar_worker, daemon=True)
        self._lidar_thread.start()
        self.get_logger().info(f"[INFO] LiDAR worker thread started ({self.lidar_frequency} Hz)")
    
    def _init_lidar_visualization(self):
        """初始化LiDAR射线在MuJoCo viewer中的可视化"""
        if self.viewer is None:
            self.get_logger().info("[INFO] LiDAR visualization skipped (no viewer)")
            return
        
        try:
            # 创建可视化几何体（用于显示射线碰撞点）
            num_vis_rays = min(100, self.rays_theta.shape[0])  # 减少到100个点，避免过载
            indices = np.linspace(0, self.rays_theta.shape[0] - 1, num_vis_rays, dtype=int)
            self.vis_ray_indices = indices
            
            # 在 viewer 锁内设置自定义几何（线程安全）
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = num_vis_rays
                for i in range(num_vis_rays):
                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],  # 小球表示碰撞点
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 1, 0, 0.8]),  # 绿色半透明
                    )
            
            self.get_logger().info(f"[INFO] LiDAR visualization initialized: {num_vis_rays} rays")
        except Exception as e:
            self.get_logger().warn(f"[WARNING] Failed to initialize LiDAR visualization: {e}")
            self.vis_ray_indices = None
    
    def _sim_stamp(self):
        """基于仿真时间的 ROS 时间戳（与 IMU/关节一致，供多传感器对齐）"""
        sec = int(self.timestamp)
        return Time(sec=sec, nanosec=int((self.timestamp - sec) * 1e9))

    def _lidar_worker(self):
        """后台线程：按 lidar_frequency 执行射线追踪并生成高程图。

        使用独立 MjData（self._lidar_worker_data），每帧从主线程 data 快照
        qpos/qvel（numpy 复制在 GIL 下原子）再 mj_forward，与主循环 mj_step
        完全隔离（共享同一 MjData 会导致 mj_collideTree 崩溃）。
        只计算并写共享状态，不发布、不碰 viewer。
        """
        period = 1.0 / self.lidar_frequency
        while rclpy.ok() and not getattr(self, "_lidar_stop", False):
            now = time.monotonic()
            if now < self._lidar_next_time:
                time.sleep(min(0.02, self._lidar_next_time - now))
                continue
            self._lidar_next_time = now + period
            try:
                # 快照主线程状态（GIL 下原子）→ 同步到独立 data → mj_forward
                self._lidar_worker_data.qpos[:] = self.data.qpos
                self._lidar_worker_data.qvel[:] = self.data.qvel
                mujoco.mj_forward(self.model, self._lidar_worker_data)

                self.lidar.trace_rays(self._lidar_worker_data,
                                      self.rays_theta, self.rays_phi)
                pts = self.lidar.get_hit_points()
                dist = self.lidar.get_distances()
                if pts is None or len(pts) == 0:
                    continue
                if dist is not None:
                    pts = pts[np.asarray(dist).flatten() > 0.0]
                if len(pts) == 0:
                    continue

                # 传感器系 → 世界系（local_map 与 base 系高程图共用）
                sensor_rot = self.lidar.sensor_rotation.copy()
                sensor_pos = self.lidar.sensor_position.copy()
                points_world = (pts @ sensor_rot.T + sensor_pos)
                # 运动学地面注入（S10_KIN_GROUND，默认 1）：轮子与地面接触时
                # 把"轮下地面点"补进点云——真机同款做法（Bjelonic 系列用最近
                # 接触位置拟合地形平面），修复 riser 阴影导致轮下地图恒空洞、
                # r_ground/ref-z 失效卡死（2026-08-06 链 41 诊断：wp7 台阶区
                # valid_sample=0，轮下/路径起点 z 全走 0.205 兜底）。
                if os.environ.get("S10_KIN_GROUND", "1") == "1":
                    try:
                        wd = self._lidar_worker_data
                        f_z = np.abs(wd.cfrc_ext[WHEEL_CONTACT_IDS, 2]) \
                            if hasattr(wd, "cfrc_ext") else None
                        if f_z is not None and f_z.shape[0] >= 4:
                            kin_pts = []
                            for k, bid in enumerate(WHEEL_CONTACT_IDS):
                                if f_z[k] > 5.0:   # 法向支撑 >5N = 着地
                                    p = wd.xpos[bid].copy()
                                    p[2] -= 0.081   # 轮心 → 地面接触点
                                    kin_pts.append(p)
                            if kin_pts:
                                points_world = np.concatenate(
                                    [points_world,
                                     np.asarray(kin_pts, dtype=np.float64)],
                                    axis=0)
                    except Exception:
                        pass

                # 旧契约（base 系 10×16 快照，供 auto_nav 限速等既有消费者；保留兼容）
                # 世界系 → base 系：xy 用 base 系保持机身前向；z 用**重力方向**相对
                # 机身高度，避免机身 roll/pitch 时平坦地面在 base 系被误判为斜面。
                elevation_map = elevation_valid = None
                if ELEVATION_AVAILABLE:
                    R_base = self._lidar_worker_data.xmat[self.base_body_id].reshape(3, 3)
                    t_base = self._lidar_worker_data.xpos[self.base_body_id]
                    p_rot = (points_world - t_base) @ R_base
                    points_base = np.column_stack(
                        [p_rot[:, :2], points_world[:, 2] - t_base[2]])
                    elevation_map, elevation_valid = points_to_heightmap(
                        points_base, ELEVATION_CFG)

                # 感知-voxel 世界对齐瓦片：3D 体素栅格 → 2.5D 高程瓦片（rollout 用）
                local_tile = None
                if LOCAL_MAP_AVAILABLE:
                    base_xy = self._lidar_worker_data.xpos[self.base_body_id][:2]
                    base_z = self._lidar_worker_data.xpos[self.base_body_id][2]
                    local_tile = self.local_map.update(
                        points_world, base_xy, base_z, self.timestamp)
                    # 感知-voxel 瓦片注入 dial-mpc（rollout 地形代价，8Hz 更新）
                    mpc = getattr(self, "mpc", None)
                    if mpc is not None and local_tile is not None:
                        try:
                            # v162：已知地图轮心 z 参考场注入（楼梯段世界系几何剖面）。
                            # 纯 numpy、固定形状，随瓦片 8Hz 更新；区外 valid=False，
                            # rollout 回退感知/接触机制。
                            _fl = getattr(self, "follower", None)
                            if (_fl is not None
                                    and hasattr(_fl, "stair_wheel_ref_grid")):
                                _wr, _wr_ok = _fl.stair_wheel_ref_grid(
                                    float(local_tile["origin"][0]),
                                    float(local_tile["origin"][1]),
                                    int(local_tile["nx"]),
                                    int(local_tile["ny"]),
                                    float(local_tile["resolution"]))
                                local_tile["features"]["wheel_ref"] = _wr
                                local_tile["features"][
                                    "wheel_ref_valid"] = _wr_ok
                            # v203 P2.1：已知几何瓦片覆盖（S10_KNOWN_TERRAIN=1
                            # 启用，默认关）。楼梯带内 heightmap 用 stair_terrain
                            # 精确值，slope/roughness/step 置 0（已知可爬，不
                            # 当障碍），消除 LiDAR 俯仰浮空/空洞对 cost 的污染。
                            if os.environ.get("S10_KNOWN_TERRAIN", "0") == "1":
                                _kt = _fl.stair_known_tile(
                                    float(local_tile["origin"][0]),
                                    float(local_tile["origin"][1]),
                                    int(local_tile["nx"]),
                                    int(local_tile["ny"]),
                                    float(local_tile["resolution"]))
                                if _kt is not None:
                                    _mk = _kt["valid"]
                                    local_tile["heightmap"] = np.where(
                                        _mk, _kt["heightmap"],
                                        local_tile["heightmap"])
                                    local_tile["valid"] = (
                                        local_tile["valid"] | _mk)
                                    _f = local_tile["features"]
                                    for _k in ("slope", "roughness",
                                               "step", "step_flag"):
                                        _f[_k] = np.where(
                                            _mk, _kt[_k], _f[_k])
                            mpc.set_elevation_map(local_tile)
                        except Exception:
                            pass

                with self._lidar_lock:
                    self._lidar_state = {
                        "stamp": self.timestamp,
                        "points_local": pts,
                        "sensor_rot": sensor_rot,
                        "sensor_pos": sensor_pos,
                        "elevation_map": elevation_map,
                        "elevation_valid": elevation_valid,
                        "local_tile": local_tile,
                    }
            except Exception as e:
                self.get_logger().error(f"[ERROR] LiDAR worker: {e}")

    def _publish_lidar_data(self):
        """发布最新 LiDAR 点云并更新高程图可视化（数据由后台线程采集）"""
        with self._lidar_lock:
            st = self._lidar_state
        if st is None:
            return
        # 10 Hz 节流（后台线程同频更新，避免重复发布相同帧）
        if self.timestamp - self.last_lidar_pub_time < 1.0 / self.lidar_frequency:
            return
        self.last_lidar_pub_time = self.timestamp

        points_local = st["points_local"]
        self.elevation_map = st["elevation_map"]
        self.elevation_valid = st["elevation_valid"]

        # 发布点云（lidar 系，保持 /lidar_points 兼容）
        if points_local.shape[0] > LIDAR_MAX_PUB_POINTS:
            idx = np.linspace(0, points_local.shape[0] - 1, LIDAR_MAX_PUB_POINTS, dtype=int)
            points_local = points_local[idx]
        self.pc_msg.header.stamp = self._sim_stamp()
        self.pc_msg.width = points_local.shape[0]
        self.pc_msg.row_step = self.pc_msg.point_step * points_local.shape[0]
        self.pc_msg.data = points_local.astype(np.float32).tobytes()
        self.lidar_points_pub.publish(self.pc_msg)

        # 高程图可视化（主线程内更新，数据来自后台线程）
        if ELEVATION_AVAILABLE and self.elevation_map is not None:
            self._update_elevation_visualization()
            # 发布 /elevation_map：元数据(5) + heightmap.flatten()(160)
            meta = np.array([ELEVATION_CFG.x_min, ELEVATION_CFG.y_min,
                             ELEVATION_CFG.resolution,
                             ELEVATION_CFG.nx, ELEVATION_CFG.ny], dtype=np.float32)
            self.em_msg.data = np.concatenate(
                [meta, self.elevation_map.flatten()]).astype(np.float32).tolist()
            self.elevation_map_pub.publish(self.em_msg)

        # 发布 /local_map：meta[4]=x0,y0,res,fill + height.flatten(3600)
        # + valid.flatten(3600, 0/1)；形状固定，下游按值注入（零 retrace）
        if LOCAL_MAP_AVAILABLE and st.get("local_tile") is not None:
            lt = st["local_tile"]
            meta = np.array([lt["origin"][0], lt["origin"][1],
                             lt["resolution"], LOCAL_MAP_CFG.fill_value],
                            dtype=np.float32)
            valid_f = lt["valid"].astype(np.float32)
            self.lm_msg.data = np.concatenate(
                [meta, lt["heightmap"].flatten(), valid_f.flatten()]
            ).astype(np.float32).tolist()
            self.local_map_pub.publish(self.lm_msg)

        # 射线可视化（用与点云同一扫描的姿态，避免竞态浮空）
        self._visualize_lidar_rays(
            points_local, st.get("sensor_rot"), st.get("sensor_pos"))

    def get_elevation_map(self):
        """获取最新局部高程图（供下游 DIAL-MPC 使用）。

        数据契约（冻结，与 perception/points_to_heightmap.py 一致）：
          heightmap: (ny=10, nx=16) float32
                    行 = y（左→右，第 0 行 y=y_min+0.05）
                    列 = x（近→远，第 0 列 x=x_min+0.05）
                    值 = 该格内点云 min-z，即**相对机身 z 的落地面高度**；
                         空洞格 = 10.0（配合 valid 判断）
          valid:     (10,16) bool，True=该格有有效测量
          x_min/y_min/resolution: ROI 原点（相对机身）与栅格边长（0.1 m）
          frame: "base_link"（前向 x、左向 y、z 相对机身）
        返回 dict 或 None（首帧高程图尚未生成）。
        """
        with self._lidar_lock:
            st = self._lidar_state
        if st is None or st["elevation_map"] is None:
            return None
        return {
            "heightmap": st["elevation_map"],      # (10,16) float32
            "valid": st["elevation_valid"],        # (10,16) bool
            "x_min": ELEVATION_CFG.x_min,          # 0.0
            "y_min": ELEVATION_CFG.y_min,          # -0.5
            "resolution": ELEVATION_CFG.resolution,  # 0.1
            "nx": ELEVATION_CFG.nx,                # 16
            "ny": ELEVATION_CFG.ny,                # 10
            "stamp": st["stamp"],                  # 仿真时间
            "frame": "base_link",
        }

    def get_local_map(self):
        """获取最新世界对齐高程瓦片（感知-voxel，供 DIAL-MPC rollout 使用）。

        数据契约（冻结，与 perception/local_map.py 一致）：
          heightmap: (60,60) float32 世界系落地面高度（绝对 z，单位 m）
          valid:     (60,60) bool，True=该格有有效测量（空洞/未知=False）
          origin:    (2,) float32，输出瓦片左下角世界坐标 (x0, y0)
          resolution: 0.1 m
          features:  slope/roughness/step/step_flag (60,60) 派生网格（cost 直接 gather）
          stamp/frame: 仿真时间 / "map"
        返回 dict 或 None（首帧瓦片尚未生成）。
        """
        with self._lidar_lock:
            st = self._lidar_state
        if st is None or st.get("local_tile") is None:
            return None
        return st["local_tile"]
    
    def _visualize_lidar_rays(self, points_local: np.ndarray,
                              sensor_rot=None, sensor_pos=None):
        """在MuJoCo viewer中可视化LiDAR射线碰撞点。

        2026-08-06 修复：之前主线程读 self.lidar.sensor_rotation（工作线程
        共享、无锁），扫描与可视化之间可能插入新扫描 → 用新姿态变换旧点云
        → 机器人运动/俯仰时点云"浮空"。现在用与点云同帧存储的姿态。
        """
        if self.viewer is None or not hasattr(self, 'vis_ray_indices') or self.vis_ray_indices is None:
            return
        
        try:
            if sensor_rot is None:
                sensor_rot = self.lidar.sensor_rotation
            if sensor_pos is None:
                sensor_pos = self.lidar.sensor_position
            # 转换到世界坐标系
            points_world = (
                points_local @ sensor_rot.T + sensor_pos
            )
            
            # 更新可视化几何体位置（只更新部分点）
            num_vis = min(len(self.vis_ray_indices), points_world.shape[0])
            
            # viewer 锁内更新（线程安全）
            with self.viewer.lock():
                for i in range(num_vis):
                    idx = self.vis_ray_indices[i] if i < len(self.vis_ray_indices) else i
                    if idx < points_world.shape[0]:
                        self.viewer.user_scn.geoms[i].pos[:] = points_world[idx]
                        # 根据距离设置颜色（近红远绿）
                        dist = np.linalg.norm(points_local[idx])
                        if dist < 2.0:
                            color = np.array([1, 0, 0, 0.8])  # 红色 - 近距离
                        elif dist < 5.0:
                            color = np.array([1, 1, 0, 0.8])  # 黄色 - 中距离
                        else:
                            color = np.array([0, 1, 0, 0.8])  # 绿色 - 远距离
                        self.viewer.user_scn.geoms[i].rgba[:] = color
                    
        except Exception as e:
            pass  # 可视化失败不影响主要功能

    def _init_elevation_visualization(self):
        """初始化高程图面片可视化（10×16 小球，颜色按高度，灰色=空洞）"""
        if self.viewer is None or not ELEVATION_AVAILABLE:
            return
        try:
            nx, ny = ELEVATION_CFG.nx, ELEVATION_CFG.ny
            n = nx * ny
            with self.viewer.lock():
                self.elev_viz_base_idx = self.viewer.user_scn.ngeom
                self.viewer.user_scn.ngeom += n
                for i in range(n):
                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.elev_viz_base_idx + i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0, 0],
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.5, 0.5, 0.5, 0.3]),
                    )
            self.get_logger().info(f"[INFO] Elevation map viz initialized: {n} cells")
        except Exception as e:
            self.get_logger().warn(f"[WARNING] Failed to init elevation viz: {e}")

    def _update_elevation_visualization(self):
        """把高程图渲染到 viewer（base 系坐标 → 世界系）"""
        if self.viewer is None or not hasattr(self, "elev_viz_base_idx"):
            return
        if self.elevation_map is None:
            return
        try:
            nx, ny = ELEVATION_CFG.nx, ELEVATION_CFG.ny
            R = self.data.xmat[self.base_body_id].reshape(3, 3)
            t = self.data.xpos[self.base_body_id]
            hm, valid = self.elevation_map, self.elevation_valid
            # 脚下地面参考 h0 = 有效格低分位（避免单个异常低点），颜色按 Δh=h-h0 分档
            if valid.any():
                h0 = float(np.percentile(hm[valid], 10))
            else:
                h0 = 0.0
            with self.viewer.lock():
                k = 0
                for ci in range(ny):
                    for cj in range(nx):
                        x = ELEVATION_CFG.x_min + (cj + 0.5) * ELEVATION_CFG.resolution
                        y = ELEVATION_CFG.y_min + (ci + 0.5) * ELEVATION_CFG.resolution
                        p_world = np.array([x, y, float(hm[ci, cj])]) @ R.T + t
                        g = self.viewer.user_scn.geoms[self.elev_viz_base_idx + k]
                        g.pos[:] = p_world
                        if valid[ci, cj]:
                            dh = float(hm[ci, cj]) - h0
                            if dh < -ELEV_BAND_FLAT:
                                col = np.array([0.2, 0.4, 1.0, 0.7])   # 蓝：低洼/坑
                            elif dh <= ELEV_BAND_FLAT:
                                col = np.array([0.2, 1.0, 0.3, 0.7])   # 绿：平地（可通行）
                            elif dh <= ELEV_BAND_ROLL:
                                col = np.array([1.0, 1.0, 0.2, 0.7])   # 黄：小坡/小坎（轮子可滚过）
                            elif dh <= ELEV_BAND_CLIMB:
                                col = np.array([1.0, 0.6, 0.1, 0.7])   # 橙：中台阶（需抬腿/减速）
                            else:
                                col = np.array([1.0, 0.2, 0.2, 0.7])   # 红：高障碍（轮足难翻越）
                        else:
                            col = np.array([0.6, 0.6, 0.6, 0.25])      # 灰：空洞
                        g.rgba[:] = col
                        k += 1
        except Exception:
            pass  # 可视化失败不影响主要功能

    def _publish_lidar_tf(self):
        """发布 LiDAR 静态 TF（base_link -> lidar_link，含前倾姿态）"""
        if not LIDAR_AVAILABLE or self.lidar is None:
            return

        # 发布静态变换 (base_link -> lidar_link)
        if not self.static_tf_published:
            try:
                lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
                if lidar_site_id < 0:
                    return

                lidar_pos = self.model.site_pos[lidar_site_id]
                lidar_quat = self.model.site_quat[lidar_site_id]  # (w,x,y,z) 相对 base_link

                static_tf = TransformStamped()
                static_tf.header.stamp = self._sim_stamp()
                static_tf.header.frame_id = "base_link"
                static_tf.child_frame_id = "lidar_link"

                static_tf.transform.translation.x = float(lidar_pos[0])
                static_tf.transform.translation.y = float(lidar_pos[1])
                static_tf.transform.translation.z = float(lidar_pos[2])
                static_tf.transform.rotation.w = float(lidar_quat[0])
                static_tf.transform.rotation.x = float(lidar_quat[1])
                static_tf.transform.rotation.y = float(lidar_quat[2])
                static_tf.transform.rotation.z = float(lidar_quat[3])

                self.static_tf_broadcaster.sendTransform(static_tf)
                self.static_tf_published = True

                self.get_logger().info("[INFO] Published static TF: base_link -> lidar_link")
            except Exception as e:
                self.get_logger().warn(f"[WARNING] Failed to publish LiDAR TF: {e}")

    def _print_self_test_summary(self):
        """自检模式（S10_SELF_TEST_SECONDS>0）：打印 LiDAR/高程图统计（无头验证用）"""
        print("\n" + "=" * 56)
        print(f"[SELF-TEST] 仿真运行到 t={self.timestamp:.1f} s")
        if self.lidar is None:
            print("[SELF-TEST] LiDAR 未初始化（依赖缺失）")
            print("=" * 56)
            return
        print(f"[SELF-TEST] LiDAR: backend={self.lidar_backend}, "
              f"rays={self.rays_theta.shape[0]}, freq={self.lidar_frequency} Hz")
        with self._lidar_lock:
            st = self._lidar_state
        if st is not None:
            print(f"[SELF-TEST] 最新点云点数: {len(st['points_local'])}")
            em = self.get_elevation_map()
            if em is not None:
                hm, valid = em["heightmap"], em["valid"]
                print(f"[SELF-TEST] 高程图: valid={int(valid.sum())}/160, "
                      f"覆盖率={100 * valid.mean():.0f}%")
                if valid.any():
                    print(f"[SELF-TEST] 高程图 z 范围: {hm[valid].min():.3f} ~ "
                          f"{hm[valid].max():.3f} m (相对机身, 空洞={ELEVATION_CFG.fill_value})")
            else:
                print("[SELF-TEST] 高程图: 尚未生成")
            lm = self.get_local_map()
            if lm is not None:
                h, v = lm["heightmap"], lm["valid"]
                f = lm["features"]
                print(f"[SELF-TEST] 本地瓦片: {h.shape[0]}x{h.shape[1]} "
                      f"origin=({lm['origin'][0]:.1f},{lm['origin'][1]:.1f}) "
                      f"res={lm['resolution']} valid={int(v.sum())}/{h.size} "
                      f"覆盖={100 * v.mean():.0f}%")
                if v.any():
                    print(f"[SELF-TEST] 瓦片高度: {h[v].min():.3f} ~ {h[v].max():.3f} m "
                          f"(世界绝对 z)")
                    sl = f["slope"][v]
                    sl = sl[np.isfinite(sl)]
                    if len(sl):
                        print(f"[SELF-TEST] 瓦片坡度: 最大 {sl.max():.2f} m/m")
            else:
                print("[SELF-TEST] 本地瓦片: 尚未生成")
        else:
            print("[SELF-TEST] LiDAR 后台线程尚无输出")
        print("=" * 56)

    def _cmd_callback(self, msg: JointsDataCmd):
        """Convert received (published) positions/velocities to internal (raw)"""
        if len(msg.data.joints_data) != 16:
            self.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        pub_pos = np.zeros(self.dof_num, dtype=np.float32)
        pub_vel = np.zeros(self.dof_num, dtype=np.float32)
        for i in range(self.dof_num):
            joint_cmd = msg.data.joints_data[i]
            self.kp_cmd[i] = joint_cmd.kp
            self.kd_cmd[i] = joint_cmd.kd
            pub_pos[i] = joint_cmd.position
            pub_vel[i] = joint_cmd.velocity
            self.tau_ff[i] = joint_cmd.torque  # tau_ff no processing

        # Convert: raw = published * dir + offset_rad
        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR

    def start(self):
        # 主模拟循环
        step = 0
        last_time = time.time()
        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1
                # 控制律
                self._apply_joint_torque()
                # 模拟一步
                mujoco.mj_step(self.model, self.data)
                # 关节角 clip（2026-08-07 关键修复）：MPC σ 大时采样极端
                # 动作 → 主仿真关节转飞（wp7 台阶区 hl 后轮 knee 实测
                # -1148 rad）→ 数值爆炸卡死/侧翻。qpos[7:] 16 关节限制在
                # jnt_range（freejoint 跳过），与 rollout 内 clip 一致。
                # 12 个腿关节（range 非零驱动关节）；wheel 自由转不 clip
                if getattr(self, "_joint_clip", None) is None:
                    _act_j = self.model.actuator_trnid[:, 0]
                    _mask = (self.model.jnt_range[_act_j, 0] != 0) | \
                        (self.model.jnt_range[_act_j, 1] != 0)
                    _j = _act_j[_mask]
                    self._joint_clip = (
                        self.model.jnt_qposadr[_j],
                        self.model.jnt_range[_j])
                _qidx, _rng = self._joint_clip
                self.data.qpos[_qidx] = np.clip(
                    self.data.qpos[_qidx], _rng[:, 0], _rng[:, 1])

                self.timestamp = step * DT

                if self.mpc_mode and self.mpc is not None:
                    # —— dial-mpc MBDPI 主线程规划：每 MPC_PLAN_INTERVAL 步
                    # 规划一次（~0.25s 阻塞），每步用当前状态 numpy 重算力矩 ——
                    q = np.asarray(self.data.qpos[:23], dtype=np.float32)
                    qd = np.asarray(self.data.qvel[:22], dtype=np.float32)
                    if self.auto_nav_active and step % 10 == 0:
                        self._update_auto_nav()
                    plan_interval = int(os.environ.get(
                        "S10_MPC_PLAN_INTERVAL_AUTO", "10")
                        if self.auto_nav_active
                        else str(MPC_PLAN_INTERVAL))
                    if (step % plan_interval == 0
                            or self.last_act is None):
                        self.last_act = self.mpc.plan_once(q, qd, self.timestamp)
                    if self.auto_nav_active:
                        # 自动模式：限制腿部动作幅值，防止 MPC 腿动甩偏重心
                        la = np.asarray(self.last_act).copy()
                        la[:12] += np.asarray(self._leg_assist, np.float32)
                        la[:12] = np.clip(
                            la[:12],
                            -float(os.environ.get("S10_AUTO_LEG_CLIP", "0.30")),
                            float(os.environ.get("S10_AUTO_LEG_CLIP", "0.30")))
                        self.last_act = la
                    self.mpc.latest_tau = self.mpc.compute_tau(
                        self.last_act, q, qd)
                    if step % 500 == 0:  # 每 2.5s 打印一次 MPC 位姿（调试/验证用）
                        self.get_logger().info(
                            f"[MPC] t={self.timestamp:.1f}s x={self.data.qpos[0]:.2f} "
                            f"z={self.data.qpos[2]:.2f}")
                    if step % 25 == 0:    # 8Hz 低频 Python（状态+LiDAR 发布）
                        self._publish_robot_state(step)
                        # 遥控中也要实时更新高程图（此前只在非 MPC 分支发布，
                        # 导致按 c 后高程图冻结、按 z 才跳到机头位置）
                        if LIDAR_AVAILABLE and self.lidar is not None:
                            self._publish_lidar_data()
                            self._publish_lidar_tf()
                        if self.viewer and step % RENDER_INTERVAL == 0:
                            self.viewer.sync()
                else:
                    self._update_track_progress()

                    # 主线程 JIT 预热：按 c 前先把扩散采样编译完，
                    # 避免进入遥控时卡 10-30s 编译（仅启动后执行一次）
                    if self.mpc is not None and not self._mpc_warmup_done:
                        self._warmup_mpc()
                    # 模式 A：预热完成后自动启动自动导航
                    if (self.auto_nav and not self.auto_nav_active
                            and self._mpc_warmup_done):
                        if self.auto_stand_t0 is None:
                            self._start_auto_nav()
                        else:
                            self._maybe_enter_auto_mpc()

                    # 自检模式：跑满指定秒数后打印统计并退出
                    if SELF_TEST_SECONDS > 0.0 and self.timestamp >= SELF_TEST_SECONDS:
                        self._print_self_test_summary()
                        break

                    # 采样 & 发送观测 (every 5 steps for 200 Hz)
                    if step % 5 == 0:
                        self._publish_robot_state(step)
                        # 发布LiDAR数据
                        if LIDAR_AVAILABLE and self.lidar is not None:
                            self._publish_lidar_data()
                            self._publish_lidar_tf()

                    # 可视化
                    if self.viewer and step % RENDER_INTERVAL == 0:
                        self.viewer.sync()

            # Handle ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.0)

    def _key_callback(self, keycode: int):
        """viewer 键盘：h 站起 / g 遥控 / i k 前后 / j l u o 转向。

        键位避开 mujoco viewer 内置键（wasd 相机平移、qe 相机旋转、zc 等），
        避免冲突。toggle 式：按 i 前进、再按 i 停止。
        注：固定轮无法侧移，j/l 与 u/o 均映射为转向（与 terminal 键位一致）。
        """
        if self.mpc is None:
            return
        ch = chr(keycode) if 32 <= keycode < 127 else ""
        c = ch.lower()
        if c == "h":
            # 站起：腿 PD 拉站姿，MPC 暂停
            if self.mpc_mode:
                self.mpc.stop_planning()
                self.mpc_mode = False
            self.standup_phase = MPC_STAND_TIME
            self.get_logger().info("[MPC] 站起（h）")
        elif c == "g":
            if not self.mpc_mode:
                self.mpc_mode = True
                self.last_act = None
                self.mpc.set_cmd(0.0, 0.0, 0.0)
                self.get_logger().info("[MPC] 进入遥控模式（g）：i/k 前后 j/l 平移 u/o 转向")
            else:
                self.mpc_mode = False
                self.get_logger().info("[MPC] 退出遥控模式")
        elif self.mpc_mode and c in ("i", "j", "k", "l", "u", "o"):
            # toggle 式更新目标速度
            self._mpc_toggle = getattr(self, "_mpc_toggle", {})
            self._mpc_toggle[c] = not self._mpc_toggle.get(c, False)
            t = self._mpc_toggle
            vx = (MPC_VX_MAX if t.get("i") else 0.0) - (MPC_VX_MAX if t.get("k") else 0.0)
            vy = 0.0
            vyaw = ((MPC_VYAW_MAX if t.get("j") else 0.0)
                    - (MPC_VYAW_MAX if t.get("l") else 0.0)
                    + (MPC_VYAW_MAX if t.get("u") else 0.0)
                    - (MPC_VYAW_MAX if t.get("o") else 0.0))
            self.mpc.set_cmd(vx, vy, vyaw)
            self.get_logger().info(f"[MPC] cmd vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}")

    def _apply_joint_torque(self):
        # MPC 遥控：直接施加 MPC 最新力矩（腿 PD + 轮力矩已含在 act2tau）
        if self.mpc_mode and self.mpc is not None:
            # 确定性爬梯步态优先（S10_STAIR_GAIT=1）：wp7 台阶区完全接管
            gait_tau = self._stair_gait_tau()
            if gait_tau is not None:
                self.data.ctrl[:] = gait_tau
                return
            tau = np.asarray(
                getattr(self.mpc, "latest_tau",
                        np.zeros(16, dtype=np.float32)), dtype=np.float64)
            if np.any(tau) and not (np.any(np.isnan(tau)) or np.any(np.isinf(tau))):
                # 膝力矩接管（S10_RISER_LIFT=1，默认关）：检测到轮前台阶时
                # 直接替换膝 PD 目标到限位（确定性抬轮），其余关节仍由 MPC
                if self.auto_nav_active:
                    tau = self._riser_lift_override(tau)
        # 台阶区航向软修正（S10_STAIR_HEAD_LOCK=1，默认关）：y>36 且目标
        # wp7 时在 MPC 轮力矩上**叠加**差速修正（软性，不替换——硬接管实测
        # 与 MPC 打架导致漂移到另一侧，链37 复现）
                if self.auto_nav_active:
                    tau = self._stair_heading_tau_bias(tau)
                # go2 式足位 P 偏置（默认关）：在 MPC 力矩上叠加膝 PD 目标平移
                if self.auto_nav_active:
                    tau = tau + self._foot_place_tau_bias()
                self.data.ctrl[:] = tau
                return
            # MPC 无有效输出（规划中/NaN）：站姿 PD 兜底防塌
            q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
            dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
            tau = (
                MPC_STAND_KP * (self.stand_target.reshape(-1, 1) - q)
                - MPC_STAND_KD * dq).flatten()
            tau[3::4] = -MPC_STAND_WHEEL_KD * dq[3::4].flatten()
            self.data.ctrl[:] = tau
            return
        # 非 MPC 模式：z 按下前零力矩（保持初始趴姿）；z 后站姿 PD 持续保持
        # （r 阻尼也回这里）。此前一直 PD → z 变成无意义；0 力矩后需 z 触发。
        if not (self.standup_phase > 0.0 or self.standup_done):
            self.data.ctrl[:] = 0.0
            return
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        tau = (
            MPC_STAND_KP * (self.stand_target.reshape(-1, 1) - q)
            - MPC_STAND_KD * dq).flatten()
        tau[3::4] = -MPC_STAND_WHEEL_KD * dq[3::4].flatten()
        self.data.ctrl[:] = tau

    def _riser_lift_override(self, tau):
        """膝/髋力矩接管（S10_RISER_LIFT=1 启用，默认关；非时序摆动）。

        2026-08-05 决定性重写（全航点 wp4→5 横脊卡死复现后）：
        - 前轮：轮前 0.15~0.55m 窗口检出台阶（step_flag + 高差）且轮心未到
          台阶顶+轮半径 → 膝目标打到限位（前 +3.0）+ 髋前摆（-0.70），
          速率限制平滑；到位自动回落。
        - 后轮：**纯运动学跟抬**——同侧前轮比后轮高 >0.06m（前轮已上台阶/脊
          顶）且前轮"前视不升"（非斜坡：前轮处于局部高点）→ 抬后轮
          （膝 -3.0 + 髋 +0.70）。不依赖后轮前方地图（riser 阴影空洞）或
          step_flag（缓脊 step_flag≈0 导致跟抬失效——全航点卡死根因）。
        MPC 的轮/偏航不受影响。
        """
        if os.environ.get("S10_RISER_LIFT", "0") != "1":
            return tau
        try:
            lm = self.get_local_map()
            if lm is None:
                return tau
            hm = lm["heightmap"]
            valid = lm["valid"]
            res = float(lm["resolution"])
            ox, oy = float(lm["origin"][0]), float(lm["origin"][1])
            fwd = self.data.xmat[self.track_body_id][::3][:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            step_flag = (lm.get("features") or {}).get("step_flag")
            # 2026-08-06 用户："门控越少越好"——step_flag 门控 0.5→0.2
            step_gate = float(os.environ.get(
                "S10_RISER_LIFT_STEP_THR", "0.2"))
            kp, kd = 80.0, 2.0
            r = 0.081

            def _h(p):
                i = int(np.floor((p[1] - oy) / res))
                j = int(np.floor((p[0] - ox) / res))
                if 0 <= i < hm.shape[0] and 0 <= j < hm.shape[1] \
                        and valid[i, j]:
                    return (float(hm[i, j]),
                            float(step_flag[i, j])
                            if step_flag is not None else 0.0)
                return None

            # 1) 前轮：地图窗口检测台阶
            front_lift = {}    # fid -> True/False
            front_h = {}       # fid -> (h_now, best_h)
            wheel_slow = float(os.environ.get(
                "S10_RISER_LIFT_WHEEL_SLOW", "0.0"))
            slow_wheels = set()
            for bid in (5, 9):
                wi = 0 if bid == 5 else 1
                xy = self.data.xpos[bid][:2]
                wz = float(self.data.xpos[bid, 2])
                h_now = _h(xy)
                h_now = h_now[0] if h_now is not None else wz - r
                best_h = None
                best_flag = 0.0
                for dd in (0.15, 0.25, 0.35, 0.45, 0.55):
                    hv = _h(xy + fwd * dd)
                    if hv is not None:
                        if best_h is None or hv[0] > best_h:
                            best_h = hv[0]
                            best_flag = hv[1]
                front_h[bid] = (h_now, best_h)
                deficit = (best_h + r) - wz if best_h is not None else 0.0
                lift_on = (best_h is not None and best_flag >= 0.5
                           and (best_h - h_now) > 0.05 and deficit > 0.02)
                if best_flag >= step_gate:
                    lift_on = (best_h is not None
                               and (best_h - h_now) > 0.05
                               and deficit > 0.02)
                front_lift[bid] = lift_on
                self._set_lift_joints(tau, wi, bid, lift_on)
                if lift_on and wheel_slow > 0:
                    slow_wheels.add(3 if bid == 5 else 7)

            # 2) 后轮：纯运动学跟抬（前轮高于后轮 + 前轮后方有台阶边界）
            for bid, wi in ((13, 2), (17, 3)):
                fid = 5 if bid == 13 else 9
                fz = float(self.data.xpos[fid, 2])
                rz = float(self.data.xpos[bid, 2])
                diff = fz - rz
                # 前轮后方 0.1~0.45m 的 step_flag（刚爬过的台阶边界在轮后；
                # 连续台阶区：前轮上第 N 级后，第 N 级 riser 就在其后方）。
                # 斜坡无 step_flag → 不跟抬（避免长坡上后轮乱抬失去抓地）。
                back_flag = 0.0
                for dd in (0.10, 0.20, 0.30, 0.45):
                    hv = _h(self.data.xpos[fid][:2] - fwd * dd)
                    if hv is not None:
                        back_flag = max(back_flag, hv[1])
                # 组合门控：① 前轮后方有台阶边界（台阶区 riser 在轮后）；
                # ② 或前轮处于"局部高点"（前方不再上升——平滑横脊无 step_flag
                #   但脊后是平地，全航点 wp4→5 卡死即此情形）。
                # 斜坡（前方持续上升且后方无边界）两种情况都不满足 → 不跟抬。
                f_h_now, f_best_h = front_h[fid]
                rise_ahead = (f_best_h is not None
                              and (f_best_h - f_h_now) > 0.05)
                lift_on = diff > 0.06 and (
                    back_flag >= 0.5 or not rise_ahead)
                self._set_lift_joints(tau, wi, bid, lift_on)
                if lift_on and wheel_slow > 0:
                    slow_wheels.add(11 if bid == 13 else 15)
            # 抬轮中的轮子减速（"轮子滚动+抬腿"协同，防撞立面翻车）
            if wheel_slow > 0:
                for wid in slow_wheels:
                    tau[wid] = tau[wid] * wheel_slow

            # 3) 台阶区航向锁定（y>36 且目标 wp7）：轮差速强制朝 wp6→wp7
            #    方向（1.55 rad，已知地图）。全航点 #6 在台阶区 yaw 偏 90°
            #    横着卡死——pursuit 漂移后纠不回。此处直接接管轮差速。
            y = float(self.data.xpos[self.track_body_id, 1])
            if y > 36.0 and self.track_next_index == 7:
                qq = self.data.xquat[self.track_body_id]
                yaw = float(np.arctan2(
                    2.0 * (qq[3] * qq[0] + qq[1] * qq[2]),
                    1.0 - 2.0 * (qq[2] ** 2 + qq[3] ** 2)))
                err_yaw = float(np.arctan2(
                    np.sin(1.55 - yaw), np.cos(1.55 - yaw)))
                kd_w = 1.2
                r = 0.081
                vx = float(os.environ.get("S10_STAIR_GATE_VX", "1.2"))
                vel_ref = -vx / (self.mpc.env_config.vel_scale * r)
                turn = float(np.clip(1.2 * err_yaw, -0.8, 0.8))
                # 左轮减速（vel_ref 为负=前进，加正差速=前进变慢）；右轮加速
                for wi, wid in ((0, 3), (1, 7), (2, 11), (3, 15)):
                    side = 1.0 if wi in (0, 2) else -1.0   # 左轮加正差速=减速
                    qd_w = float(self.data.qvel[6 + wid])
                    tau[wid] = kd_w * (vel_ref + side * turn - qd_w)
            return tau
        except Exception:
            return tau

    def _stair_heading_tau_bias(self, tau):
        """台阶区航向锁定（S10_STAIR_HEAD_LOCK=1 启用，默认关）。

        链 65 修正：楼梯区内（y∈[37.5,41.5]）锁定**垂直台阶边缘**（+y，
        1.57 rad）直爬——斜爬（边爬边转向 wp8 西北）时轮子被 riser 边缘
        导向侧面 → 西漂 + 侧翻（chain 64 爬升中 x -15→-16 复现）。
        出楼梯区后恢复 pursuit 转向。只接管轮差速，不碰腿。
        """
        if os.environ.get("S10_STAIR_HEAD_LOCK", "0") != "1":
            return tau
        try:
            y = float(self.data.xpos[self.track_body_id, 1])
            # v202: 航向锁区间/增益参数化（默认 37.5~41.5 / 1.5）：入口提前
            # 锁直（避免 riser1→2 斜楔西漂）+ 卡住时更强纠偏
            _y0 = float(os.environ.get("S10_STAIR_HEADLOCK_Y0", "37.5"))
            _y1 = float(os.environ.get("S10_STAIR_HEADLOCK_Y1", "41.5"))
            _gain = float(os.environ.get("S10_STAIR_HEADLOCK_GAIN", "1.5"))
            if not (_y0 < y < _y1):
                return tau
            qq = self.data.xquat[self.track_body_id]
            yaw = float(np.arctan2(
                2.0 * (qq[3] * qq[0] + qq[1] * qq[2]),
                1.0 - 2.0 * (qq[2] ** 2 + qq[3] ** 2)))
            err_yaw = float(np.arctan2(
                np.sin(1.57 - yaw), np.cos(1.57 - yaw)))
            # v189：增强航向锁定增益（原 0.8/±0.5 太弱，长爬升西漂失控）
            turn = float(np.clip(_gain * err_yaw, -0.8, 0.8))
            for wi, wid in ((0, 3), (1, 7), (2, 11), (3, 15)):
                side = 1.0 if wi in (0, 2) else -1.0
                # 叠加差速力矩：左轮减速/右轮加速（左转）
                tau[wid] = tau[wid] + 1.2 * side * turn
            return tau
        except Exception:
            return tau

    def _set_lift_joints(self, tau, wi, bid, lift_on):
        """按轮设置膝/髋力矩（接管）：lift_on 时膝到限位+髋前摆，否则回落。"""
        legs = {5: (2, 1, 1.0, -0.70), 9: (5, 4, 1.0, -0.70),
                13: (8, 9, -1.0, 0.70), 17: (11, 12, -1.0, 0.70)}
        knee_idx, hipy_idx, sign, hipy_swing = legs[bid]
        kp, kd = 80.0, 2.0
        cur = self._stair_knee_cur[wi]
        knee_tar = float(np.clip(
            (sign * 3.0 if lift_on else sign * 2.3),
            cur - 0.035, cur + 0.035))
        self._stair_knee_cur[wi] = knee_tar
        cur_h = self._stair_hipy_cur[wi]
        # 髋前摆开关（S10_RISER_LIFT_HIPY，默认 1）：实测髋摆可能把轮子
        # 别进 riser 立面导致翻车（全航点 #11），可关掉只做膝抬。
        if os.environ.get("S10_RISER_LIFT_HIPY", "1") == "1":
            hipy_target = hipy_swing if lift_on else sign * -1.16
        else:
            hipy_target = sign * -1.16
        hipy_tar = float(np.clip(
            hipy_target,
            cur_h - 0.03, cur_h + 0.03))
        self._stair_hipy_cur[wi] = hipy_tar
        q_knee = float(self.data.qpos[7 + knee_idx])
        qd_knee = float(self.data.qvel[6 + knee_idx])
        q_hipy = float(self.data.qpos[7 + hipy_idx])
        qd_hipy = float(self.data.qvel[6 + hipy_idx])
        tau[knee_idx] = kp * (knee_tar - q_knee) - kd * qd_knee
        tau[hipy_idx] = kp * (hipy_tar - q_hipy) - kd * qd_hipy

    def _stair_gait_tau(self):
        """确定性爬梯步态（S10_STAIR_GAIT=1 启用，默认关；非时序摆动）。

        wp7 台阶区（已知地图：y∈[38.2,41.6]，riser 38.4/38.8/39.4/39.8/40.2，
        各级顶 0.67/0.79/0.92/1.04/1.17）接管腿控：某轮接近其前方最近 riser
        且轮心仍低于台阶顶+轮半径时，膝 PD 目标直接打到限位（前 +3.0 /
        后 −3.0，物理抬轮 ~0.12m）；轮心到位后回落默认。四轮同时 1.2 m/s
        驱动，差速 0（台阶区航向直北）。出区返回 None → MPC 恢复。
        """
        if os.environ.get("S10_STAIR_GAIT", "0") != "1":
            return None
        if not self.auto_nav_active:
            return None
        y = float(self.data.xpos[self.track_body_id, 1])
        if not (self._stair_gait_y0 <= y <= self._stair_gait_y1):
            return None
        r = 0.081
        kp, kd = 80.0, 2.0
        kd_w = 1.2
        vx = float(os.environ.get("S10_STAIR_GATE_VX", "1.2"))
        vel_ref = -vx / (self.mpc.env_config.vel_scale * r)
        tau = np.zeros(16, dtype=np.float64)
        legs = ((5, 2, 3, 1.0), (9, 5, 7, 1.0),
                (13, 8, 11, -1.0), (17, 11, 15, -1.0))
        # 默认腿姿态（hipx/hipy 保持蹲姿；膝由步态决定）
        for (bid, hx, i_hipx, i_hipy) in ((5, -0.05, 1, 2), (9, 0.05, 4, 5),
                                          (13, -0.05, 8, 9),
                                          (17, 0.05, 12, 13)):
            hy = -1.16 if bid in (5, 9) else 1.16
            tau[i_hipx] = kp * (hx - float(self.data.qpos[7 + i_hipx])) \
                - kd * float(self.data.qvel[6 + i_hipx])
            tau[i_hipy] = kp * (hy - float(self.data.qpos[7 + i_hipy])) \
                - kd * float(self.data.qvel[6 + i_hipy])
        for bid, knee_idx, wheel_idx, sign in legs:
            wy = float(self.data.xpos[bid, 1])
            wz = float(self.data.xpos[bid, 2])
            # 找该轮前方最近 riser 及对应台阶顶
            ahead = self._stair_risers > (wy - 0.08)
            if not ahead.any():
                knee_tar = sign * 2.3        # 默认蹲姿
            else:
                k = int(np.argmax(ahead))
                y_r = float(self._stair_risers[k])
                top = float(self._stair_tops[k])
                deficit = (top + r) - wz
                if (y_r - wy) < 0.55 and deficit > 0.02:
                    knee_tar = sign * 3.0    # 抬轮到限位
                else:
                    knee_tar = sign * 2.3    # 默认蹲姿
            # 膝目标速率限制：等效 0.35 rad/0.05s（200Hz 下每步 0.035 rad）
            wi = (0 if bid == 5 else 1 if bid == 9
                  else 2 if bid == 13 else 3)
            cur = self._stair_knee_cur[wi]
            knee_tar = float(np.clip(
                knee_tar, cur - 0.035, cur + 0.035))
            self._stair_knee_cur[wi] = knee_tar
            q_knee = float(self.data.qpos[7 + knee_idx])
            qd_knee = float(self.data.qvel[6 + knee_idx])
            tau[knee_idx] = kp * (knee_tar - q_knee) - kd * qd_knee
            # 轮速度伺服（velocity 模式：act 目标速度 = -v/(vel_scale*r)）
            qd_w = float(self.data.qvel[6 + wheel_idx])
            tau[wheel_idx] = kd_w * (vel_ref - qd_w)
        # 打印调试
        if os.environ.get("S10_AUTO_DEBUG") and int(self.timestamp * 2) % 4 == 0:
            print(f"[STAIR] t={self.timestamp:.1f} y={y:.2f} "
                  f"tau_k={np.round(tau[[2, 5, 8, 11]], 1)} "
                  f"wz={np.round(self.data.xpos[[5, 9, 13, 17], 2], 2)}",
                  flush=True)
        return tau

    def _foot_place_tau_bias(self):
        """go2 式足位 P 偏置（S10_FOOT_PLACE=1 时启用，默认关）。

        2026-08-05 用户决策"不要 CTBC 式时序摆动"后，用此替代：不做时序
        状态机，而是**足位比例 P 控制**（与 go2 落脚点同类）——前轮接近
        台阶（0.1~0.5m 窗口 + step_flag 门控）时，膝 PD 目标按
        `clip(k_p × (台阶顶+轮半径 − 轮心高), 0, max)` 平移，把轮子抬向
        台阶顶；到位（轮心 ≥ 台阶顶+轮半径−0.02）后欠高为负 → 自动释放。
        力矩层叠加（不受 S10_AUTO_LEG_CLIP 限制），MPC 照常管平衡与推进。
        后轮"跟抬"：同侧前轮已上顶时抬后轮（S10_LIFT_REAR=1）。
        """
        if os.environ.get("S10_FOOT_PLACE", "0") != "1":
            return np.zeros(16, dtype=np.float64)
        try:
            lm = self.get_local_map()
            if lm is None or self.mpc is None:
                return np.zeros(16, dtype=np.float64)
            hm = lm["heightmap"]
            valid = lm["valid"]
            res = float(lm["resolution"])
            ox, oy = float(lm["origin"][0]), float(lm["origin"][1])
            fwd = self.data.xmat[self.track_body_id][::3][:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            step_flag = (lm.get("features") or {}).get("step_flag")
            step_thr = float(os.environ.get("S10_FOOT_PLACE_STEP_THR", "0.5"))
            # 链 60：kp=2.0/max=0.5（前轮 hipy+knee 同步抬，力度足够）
            # ——chain 48 用同力度在横脊猛抬侧翻，但当时 foot_place 无
            # 楼梯门控；现在只在连续楼梯区（前轮基准门控）触发。
            k_p = float(os.environ.get("S10_FOOT_PLACE_KP", "2.0"))
            max_bias = float(os.environ.get("S10_FOOT_PLACE_MAX", "0.5"))
            trig_deficit = float(os.environ.get(
                "S10_FOOT_PLACE_TRIG", "0.06"))
            r = 0.081
            tau_bias = np.zeros(16, dtype=np.float64)
            kp = self.mpc.env_config.kp

            def _h(p):
                i = int(np.floor((p[1] - oy) / res))
                j = int(np.floor((p[0] - ox) / res))
                if 0 <= i < hm.shape[0] and 0 <= j < hm.shape[1] \
                        and valid[i, j]:
                    return float(hm[i, j]), float(
                        step_flag[i, j]) if step_flag is not None else 0.0
                return None

            front_step = {}   # fid -> h_step（前轮检测到的台阶顶）
            front_deficit = {}   # fid -> deficit（左右同步用）
            for wi, (bid, knee_idx, sign) in enumerate(
                    ((5, 2, 1.0), (9, 5, 1.0), (13, 8, -1.0), (17, 11, -1.0))):
                xy = self.data.xpos[bid][:2]
                wheel_z = float(self.data.xpos[bid, 2])
                if bid in (5, 9):   # 前轮：前视 0.1~0.5m 窗口
                    # 2026-08-06：去掉楼梯门控——单级横脊（0.13m > 轮半径
                    # 0.081）也需抬轮，chain 49（kp=1.0 温和）实证横脊能过；
                    # chain 48 猛抬侧翻是 kp=2.0 过强，用温和参数即可。
                    best_h = None
                    best_flag = 0.0
                    hs_list = []
                    for dd in (0.1, 0.2, 0.3, 0.4, 0.5):
                        hv = _h(xy + fwd * dd)
                        if hv is not None:
                            hs_list.append(hv[0])
                            if best_h is None or hv[0] > best_h:
                                best_h = hv[0]
                                best_flag = hv[1]
                    # 陡升梯度（2026-08-06）：step_flag 依赖网格对齐（探针可能
                    # 整段落在台面上错过边界格），OR 相邻探针梯度>0.6 判定
                    # 离散台阶（0.125m riser≈1.2、20% 坡≈0.2，天然区分）。
                    max_grad = 0.0
                    for _a, _b in zip(hs_list[:-1], hs_list[1:]):
                        max_grad = max(max_grad, (_b - _a) / 0.1)
                    if best_h is not None and (
                            best_flag >= step_thr or max_grad > 0.6):
                        front_step[bid] = best_h
                        front_deficit[bid] = (best_h + r) - wheel_z
                else:               # 后轮：同侧前轮已上台阶顶（跟抬）
                    # 2026-08-07 拆分：foot_place 后轮跟抬用独立开关
                    # S10_FOOT_PLACE_LIFT_REAR（默认 0，防 v38 东漂）；
                    # reward 层 r_ground 后轮 lift 由 S10_LIFT_REAR 控制。
                    if os.environ.get(
                            "S10_FOOT_PLACE_LIFT_REAR", "0") != "1":
                        continue
                    fid = 5 if bid == 13 else 9
                    hs = front_step.get(fid)
                    if hs is None:
                        continue
                    deficit = (hs + r) - wheel_z
                    if deficit > trig_deficit:
                        bias = float(np.clip(k_p * deficit, 0.0, max_bias))
                        tau_bias[knee_idx] = kp * sign * bias
            # 前轮左右同步（链 50）：fl/fr deficit 取 max → 同步抬升，
            # 防地图格对齐导致的左右不对称侧翻（chain 49 roll=1.95 复现）。
            if 5 in front_deficit or 9 in front_deficit:
                d_sync = max(
                    front_deficit.get(5, 0.0), front_deficit.get(9, 0.0))
                if d_sync > trig_deficit:
                    bias = float(np.clip(k_p * d_sync, 0.0, max_bias))
                    for bid, knee_idx, sign in ((5, 2, 1.0), (9, 5, 1.0)):
                        tau_bias[knee_idx] = kp * sign * bias
                        # hipy 同步抬（链 60）：膝+胯协同，前轮抬升力足够
                        # （0.45rad 动作尺度下单膝抬 0.08m 不够 0.13m riser）
                        tau_bias[knee_idx - 1] = kp * sign * bias * 0.7
            if os.environ.get("S10_FOOT_PLACE_DEBUG") \
                    and int(self.timestamp * 2) % 4 == 0:
                _pos = self.data.xpos[self.track_body_id][:2]
                print(f"[FPLACE] y={_pos[1]:.2f} "
                      f"front_step={list(front_step.keys())} "
                      f"deficit={ {k: round(v, 3) for k, v in front_deficit.items()} } "
                      f"tau={np.round(tau_bias[[1, 2, 4, 5]], 1)}",
                      flush=True)
            return tau_bias
        except Exception:
            return np.zeros(16, dtype=np.float64)

    # --------------------------------------------------------
    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = q

        # roll (X-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        # pitch (Y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # 防止数值漂移导致 |t2|>1
        pitch = np.arcsin(t2)

        # yaw (Z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    # --------------------------------------------------------

    def _publish_robot_state(self, step: int):
        # ----- IMU -----
        q_world = self.data.sensordata[:4]  # quaternion (w, x, y, z) in MuJoCo convention
        rpy_rad = self.quaternion_to_euler(q_world)  # returns [roll, pitch, yaw] in radians

        # Convert to degrees
        rpy_deg = [angle * (180.0 / 3.141592653589793) for angle in rpy_rad]

        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]  # body frame

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
        imu_msg.data.omega_x = float(angvel_b[0])
        imu_msg.data.omega_y = float(angvel_b[1])
        imu_msg.data.omega_z = float(angvel_b[2])
        imu_msg.data.acc_x = float(body_acc[0])
        imu_msg.data.acc_y = float(body_acc[1])
        imu_msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        # ----- 关节 -----
        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()

        # Convert raw to published: published = (raw - offset_rad) * dir
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = tau * JOINT_DIR  # Torque also needs direction flip
        
        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]  # Dummy name (four spaces)
            joint.data_id = 0  # Dummy
            joint.status_word = 1  # Normal
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0  # Dummy normal temp
            joint.driver_temp = 45.0  # Dummy normal temp
        self.joints_pub.publish(joints_msg)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    cli_args, ros_args = parse_cli_args()
    rclpy.init(args=ros_args)
    sim_node = MuJoCoSimulationNode(
        model_key=cli_args.model_key,
        xml_path=resolve_xml_path(cli_args.scene, cli_args.xml_path),
    )
    sim_node.start()
    sim_node.destroy_node()
    rclpy.shutdown()
