"""cruise_vmc_noros.py — v218 方案无 ROS 独立测试（wp0→MAX_WP，原始赛道）。

结构：导航（AutoNavFollower pursuit/vlim）→ 身体层 MPPI [vx,ω]（20Hz）
     → VMC/阻抗腿层（200Hz）→ mujoco。
已知地图：地形高 mj_ray 逐轮查询；横脊预扫描 dh>0.12 → 弧长表 → 抬轮前馈。
"""
import os, sys, time
import numpy as np
import mujoco
try:
    import mujoco.viewer
    _HAS_VIEWER = True
except Exception:
    _HAS_VIEWER = False

PKG = '/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy'
sys.path.insert(0, PKG)
from s10_mpc.stair_auto_nav import AutoNavFollower
from s10_mpc.body_mppi import BodyMPPI
from s10_mpc.stair_vmc_legs import (VMCController, CarVMC, LEG_ATTACH, WHEEL_BODY,
    WHEEL_Q_IDX, LidarTerrain)
from s10_mpc.stair_wbc import StairWBC

DT = 0.005
MAX_SIM = float(os.environ.get('S10_TEST_MAX_SIM', '90'))
MAX_WP = int(os.environ.get('S10_AUTO_MAX_WP', '8'))
STOP_AT = int(os.environ.get('S10_STOP_AT_WP', '0'))
XML = os.environ.get('S10_XML',
    f'{PKG}/S10_description/s10_mjcf/mjcf/S10_track.xml')

# v862: ????=CarVMC ???????knee 1.90/hipy 1.10?S10_CAR_SQUAT=1??
# ????(??2.30)->???(??1.90)?????0.5s ??? body ???
# ???????????/??????? body 0.31->0.75 ? 2s?????
# ???????wp0->1 ?? 5.0s??
STAND_TARGET = np.array([-0.05, -1.10, 1.90, 0.0,
                          0.05, -1.10, 1.90, 0.0,
                         -0.05,  1.10, -1.90, 0.0,
                          0.05,  1.10, -1.90, 0.0], dtype=np.float64)


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    d.qpos[0:3] = [0.0, -2.5, float(os.environ.get('S10_INIT_Z', '0.2'))]
    iy = float(os.environ.get('S10_INIT_YAW', '0'))
    if abs(iy) > 1e-3:
        d.qpos[3:7] = [np.cos(iy/2), 0, 0, np.sin(iy/2)]
    else:
        d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:23] = np.array([-0.438, -1.16, 2.45, 0.0,
                              0.438, -1.16, 2.45, 0.0,
                             -0.438,  1.16, -2.45, 0.0,
                              0.438,  1.16, -2.45, 0.0])
    mujoco.mj_forward(m, d)

    _viewer = None
    if os.environ.get('S10_USE_VIEWER', '0') == '1' and _HAS_VIEWER:
        try:
            _tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
            _viewer = mujoco.viewer.launch_passive(m, d)
            with _viewer.lock():
                _viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                _viewer.cam.trackbodyid = _tb if _tb >= 0 else 1
            print('[VMC] viewer 已打开（关窗口即停止）', flush=True)
        except Exception as _e:
            print('[VMC] viewer 打开失败，无头运行:', _e, flush=True)
            _viewer = None

    wps = []
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith('track_waypoint_'):
            idx = int(name[len('track_waypoint_'):].split('_')[0])
            wps.append((idx, d.geom_xpos[gid].copy()))
    wps.sort()
    wp = np.array([w for _, w in wps])
    # v643: 测试用 S10_START_WP——从指定航点起跑（跳过楼梯等卡点，
    # 用于验证其余赛段 wp7→33；0=正常从 wp0 起跑）
    START_WP = int(os.environ.get('S10_START_WP', '0'))
    if START_WP > 0 and START_WP < len(wp):
        # v886: 起始高度用起跑点实际地形+半蹲站立高——原 wp6 用 wp6.z+0.21
        # 且腿用伸展位姿（膝 2.45 轮只低髋 0.12m），在 0.63m 坡面悬浮起步
        # （轮离地→v870 ground_f=0 无驱动→被迫 GF=1.0 兜底→进梯离地差速
        # 对转）。半蹲位姿（膝 1.90，轮低髋 0.207m）+ terrain+0.24 让轮落地，
        # 恢复 v870 离地阻尼。
        _sbk = float(os.environ.get('S10_START_BACK', '1.0'))
        _sz0 = float(wp[START_WP][2]) + 0.21
        try:
            _g0 = np.array([-1], dtype=np.int32)
            _dist0 = np.zeros(1); _nrm0 = np.zeros(3)
            _hit0 = mujoco.mj_ray(m, d,
                                  [float(wp[START_WP][0]),
                                   float(wp[START_WP][1]) - _sbk, 8.0],
                                  [0, 0, -1], None, True, -1, _g0, _nrm0)
            if _hit0 > 0:
                _sz0 = (8.0 - _hit0) + 0.24
        except Exception:
            pass
        # v893(方案3): 台架出生点直接放 step1 台面(y=38.0)——只测
        # 0.125m 单阶，排除首级小台阶与后轮爬小台阶干扰
        if float(os.environ.get('S10_STAIR_BENCH', '0')) > 0:
            # 出生在走廊 x=-14.5（wp6.x=-15.12 在 x=-15.0 柱子上）
            # v894: riser2(38.34)前 1.2m 平地(y37.14)——riser1 已抹平，
            # 地面 0.479 一直平到 riser2 单阶
            d.qpos[0:3] = [-14.50, 37.14, 0.72]
            _iy = 1.5708
        elif _sbk <= 0.0:
            d.qpos[0:3] = [float(wp[START_WP][0]),
                           float(wp[START_WP][1]), _sz0]
        else:
            d.qpos[0:3] = [float(wp[START_WP][0]),
                           float(wp[START_WP][1]) - _sbk, _sz0]
        if START_WP + 1 < len(wp):
            _dy = wp[START_WP + 1][1] - wp[START_WP][1]
            _dx = wp[START_WP + 1][0] - wp[START_WP][0]
            _iy = float(np.arctan2(_dy, _dx))
        else:
            _iy = 1.5708
        d.qpos[3:7] = [np.cos(_iy / 2), 0, 0, np.sin(_iy / 2)]
        # v886: 半蹲位姿（STAND_TARGET）——伸展位姿轮高离地悬浮
        d.qpos[7:23] = STAND_TARGET.copy()
        mujoco.mj_forward(m, d)
        print(f'[VMC] 从 wp{START_WP} 起跑（跳过 wp0→{START_WP}）', flush=True)

    # v894(方案1): 台架抹平 riser1(0.061m)——内存中把 step1 顶条(y≈37.9-38.3,
    # 顶面0.54)下压到地面0.479，只留 riser2 单级 0.125m；不改任何 XML/STL
    if float(os.environ.get('S10_STAIR_BENCH', '0')) > 0:
        try:
            _flat_cnt = 0
            for _gi in range(m.ngeom):
                if int(m.geom_type[_gi]) != 7 or int(m.geom_group[_gi]) != 0:
                    continue
                _xp = d.geom_xpos[_gi]
                _sz = m.geom_size[_gi]
                if (37.9 <= float(_xp[1]) <= 38.3
                        and 0.52 <= float(_xp[2]) + float(_sz[2]) <= 0.56
                        and float(_sz[0]) > 3.0):
                    m.geom_pos[_gi, 2] = float(_xp[2]) - (
                        float(_xp[2]) + float(_sz[2]) - 0.479)
                    _flat_cnt += 1
            mujoco.mj_forward(m, d)
            print('[VMC] BENCH 抹平 riser1: %d 个 strip' % _flat_cnt, flush=True)
        except Exception as e:
            print('[VMC] BENCH 抹平失败', e, flush=True)

    fol = AutoNavFollower(
        wp,
        max_speed=float(os.environ.get('S10_AUTO_VMAX', '6.0')),
        vyaw_max=float(os.environ.get('S10_AUTO_VYAW_MAX', '3.5')),
        yaw_gain=float(os.environ.get('S10_AUTO_YAW_GAIN', '2.5')),
        lookahead=float(os.environ.get('S10_AUTO_LOOKAHEAD', '1.5')))
    # v893: 台架出生在 y37.9(s≈5.5)，s_cur 默认 0 会让导航目标指向身后
    # 狗掉头南跑——初始化 s_cur 到出生点最近路径弧长
    if float(os.environ.get('S10_STAIR_BENCH', '0')) > 0:
        try:
            _sb_p = int(np.argmin(np.sum(
                (fol.path_pts[:, :2] - d.qpos[0:2]) ** 2, axis=1)))
            fol._s_cur = float(fol.path_cum[_sb_p])
            fol._k_near = _sb_p
            print('[VMC] BENCH s_cur 初始化到 %.2f' % fol._s_cur, flush=True)
        except Exception as e:
            print('[VMC] BENCH s_cur 初始化失败', e, flush=True)

    # v826e: ref_path 导出（S10_REF_DUMP 路径；S10_DUMP_ONLY=1 只出图
    # 不仿真——每次测试前先生成 ref_path 供用户审阅质量）
    _dump_p = os.environ.get('S10_REF_DUMP', '')
    if _dump_p:
        try:
            np.savez(_dump_p,
                     path_pts=fol.path_pts,
                     path_vlim=fol.path_vlim,
                     path_wp_s=fol.path_wp_s,
                     path_cum=fol.path_cum,
                     path_curv=fol.path_curv_signed,
                     wp=wp)
            print(f'[VMC] ref_path dump -> {_dump_p}', flush=True)
        except Exception as _e:
            print('[VMC] ref_path dump 失败', _e, flush=True)
    if os.environ.get('S10_DUMP_ONLY', '0') == '1':
        return

    # 已知地图横脊预扫描（与节点 _scan_ridge_zones 同法）
    ridge_arcs = []
    _ridge_signed = {}
    try:
        pts = fol.path_pts
        hs = np.empty(len(pts))
        for k, p in enumerate(pts):
            g = np.array([-1], dtype=np.int32); dist = np.zeros(1); nrm = np.zeros(3)
            hit = mujoco.mj_ray(m, d, [p[0], p[1], 8.0], [0, 0, -1],
                                None, False, -1, g, nrm)
            hs[k] = (8.0 - hit) if g[0] >= 0 else float(p[2])
        dh_s = np.diff(hs)
        dh = np.abs(dh_s)
        skip_s = float(fol.path_wp_s[1]) - 2.0
        ridge_idx = np.where((dh > 0.12) & (fol.path_cum[:len(dh)] > skip_s))[0]
        ridge_arcs = [(float(fol.path_cum[k]), float(dh[k])) for k in ridge_idx]
        _ridge_signed = {float(fol.path_cum[k]): float(dh_s[k])
                         for k in ridge_idx}
        fol.ridge_s = [float(fol.path_cum[k]) for k in ridge_idx]
        print(f'[VMC] 预扫描横脊 {len(ridge_arcs)} 处', flush=True)
        # v825: 删除横脊限速/下降沿限速（用户指令）

    except Exception as e:
        print('[VMC] 横脊预扫描失败', e, flush=True)
    # v236: 台阶几何预扫描——wp6->7 楼梯区 riser 弧长表（已知地图，供
    # 相位步态）。wp5->6 台阶间距 2m 与相位窗不匹配（v447 卡第一级），
    # 仍由连续抬轮处理。
    stair_risers = []
    try:
        _s6 = float(fol.path_wp_s[6]); _s7 = float(fol.path_wp_s[7])
        # v587: 楼梯 riser 只取**上升沿**（dh_s>=0.12）——wp5→6 第二级是
        # 0.60 平台降到 0.48 的**下降沿**，|dh| 误判为台阶导致 CPG 空转。
        # v627: 楼梯区 riser 阈值 0.12→0.05——y=37.94 的 0.06m 小台阶是
        # 第一个真障碍（前轮上平台后后轮卸载爬不动）；下降沿仍排除。
        # v630: 直接用 signed dh_s 扫全路径 [s6,s7]——旧法从 ridge_arcs
        # 派生（ridge 扫描 |dh|>0.12），0.06m 小台阶进不了表
        # v699: 楼梯表只留**真楼梯**（段末 4m 内，y≥37.4）——wp5→6 第二级
        # （y=32.88）也是 0.12m 台阶但由巡航处理，误归入楼梯表会让
        # FootPlace 过早接管翻车（v696-698）
        _stair_idx = np.where((dh_s >= 0.05)
                              & (fol.path_cum[:len(dh_s)] >= _s7 - 4.0)
                              & (fol.path_cum[:len(dh_s)] <= _s7))[0]
        stair_risers = [(float(fol.path_cum[k]), float(dh_s[k]))
                        for k in _stair_idx]
        print(f'[VMC] 楼梯区 riser {len(stair_risers)} 处', flush=True)
    except Exception as e:
        print('[VMC] 楼梯预扫描失败', e, flush=True)
    # v247: 横脊世界坐标表（供物理距离触发，替代 s_cur 投影——转向时 s_cur
    # 滞后导致步态漏触发，wp4→5 撞脊翻车实测）
    ridge_world = []
    try:
        for (sr, dhv) in ridge_arcs:
            _k = int(np.searchsorted(fol.path_cum, sr, side='right') - 1)
            _k = min(max(_k, 0), len(fol.path_pts) - 1)
            _pt = fol.path_pts[_k, :2].copy()
            # v259: 存脊点处路径切线（脊近似垂直路径；距离沿切线投影，
            # 忽略横向偏移——狗偏西 0.5m 时欧氏距离误判 0.44m 不触发抬轮）
            _tng = np.zeros(2)
            if _k < len(fol.path_heading):
                _th = float(fol.path_heading[_k])
                _tng = np.array([np.cos(_th), np.sin(_th)])
            ridge_world.append((_pt, _tng, sr, dhv))
    except Exception as e:
        print('[VMC] 横脊坐标表失败', e, flush=True)
    # v571: 楼梯 riser 世界坐标表（供几何同步抬放窗——s_cur 投影在转弯/
    # 卡死时不可靠，用轮轴到 riser 面的沿路径物理距离触发）
    stair_world = []
    try:
        # v629: 直接从 stair_risers（0.05 阈值）构建——旧版从 ridge_world
        # 派生（0.12 阈值），0.06m 小台阶不在 ridge_world，CPG 看不到它
        for (sr, dhv) in stair_risers:
            _k = int(np.searchsorted(fol.path_cum, sr, side='right') - 1)
            _k = min(max(_k, 0), len(fol.path_pts) - 2)
            _pt = fol.path_pts[_k, :2].copy()
            _th = float(fol.path_heading[_k])
            _tng = np.array([np.cos(_th), np.sin(_th)])
            _g9 = np.array([-1], dtype=np.int32)
            _d9 = np.zeros(1); _n9 = np.zeros(3)
            _hit9 = mujoco.mj_ray(
                m, d, [_pt[0] + _tng[0] * 0.06,
                      _pt[1] + _tng[1] * 0.06, 8.0],
                [0, 0, -1], None, False, -1, _g9, _n9)
            _top9 = (8.0 - _hit9) if _hit9 > 0 else 0.0
            stair_world.append((_pt, _tng, sr, float(dhv), float(_top9)))
        print(f'[VMC] 楼梯世界坐标 {len(stair_world)} 处', flush=True)
    except Exception as e:
        print('[VMC] 楼梯坐标表失败', e, flush=True)
    # v891(方案3): 单级台架模式——只保留第一个高 riser(0.125m)，
    # 排除首级小台阶与后续多级，定位单级爬升动力学
    if float(os.environ.get('S10_STAIR_BENCH', '0')) > 0:
        try:
            # v895: 台架表直接手工构造 riser2 单级（预扫描被出生狗体污染，
            # 造出狗身边假 riser → 全轮 SWING → 卡死实测）
            _k2 = int(np.argmin(np.abs(fol.path_pts[:, 1] - 38.34)))
            _pt2 = fol.path_pts[_k2, :2].copy()
            _hd2 = float(fol.path_heading[_k2])
            _tng2 = np.array([np.cos(_hd2), np.sin(_hd2)])
            _arc2 = float(fol.path_cum[_k2])
            stair_world = [(_pt2, _tng2, _arc2, 0.125, 0.666)]
            stair_risers = [(_arc2, 0.125)]
            print('[VMC] BENCH 手工 riser2 单级: y=%.2f arc=%.2f'
                  % (float(_pt2[1]), _arc2), flush=True)
        except Exception as e:
            print('[VMC] BENCH 手工表失败', e, flush=True)
    # v871: 几何表单源化——预扫描 stair_world 回填导航 stair 表（消除
    # 硬编码 STAIR_RISERS/TOPS 双源，换地图不失效）
    try:
        if stair_world:
            _sw_y = np.array([float(_p[1]) for (_p, _t, _s, _h, _z)
                              in stair_world])
            _sw_t = np.array([float(_z) for (_p, _t, _s, _h, _z)
                              in stair_world])
            _ord = np.argsort(_sw_y)
            fol.STAIR_RISERS = _sw_y[_ord].astype(np.float64)
            fol.STAIR_TOPS = _sw_t[_ord].astype(np.float64)
            fol._stair_last_arc = float(stair_world[-1][2])
            print('[VMC] stair 表回填 %d 级: y=%s top=%s'
                  % (len(_sw_y), np.round(fol.STAIR_RISERS, 3),
                     np.round(fol.STAIR_TOPS, 3)), flush=True)
    except Exception as e:
        print('[VMC] stair 表回填失败', e, flush=True)
    self_fk_r = 0.081

    mppi = BodyMPPI(
        N=int(os.environ.get('VMC_MPPI_N', '4096')),
        H=int(os.environ.get('VMC_MPPI_H', '40')),
        vx_max=float(os.environ.get('S10_AUTO_VMAX', '6.0')))
    _vmode = os.environ.get('S10_VMC_MODE', 'wbc')
    if _vmode == 'pd':
        from s10_mpc.stair_vmc_legs import LegPDDrive
        vmc = LegPDDrive()
        print('[VMC] LegPDDrive 模式（腿锁蹲姿+轮驱动）', flush=True)
    elif _vmode == 'car':
        vmc = CarVMC()
        print('[VMC] CarVMC 模式（车化：轮驱动/差速，腿=主动悬架姿态）', flush=True)
    elif _vmode == 'place':
        from s10_mpc.stair_vmc_legs import FootPlaceVMC
        vmc = FootPlaceVMC()
        print('[VMC] FootPlaceVMC 模式（逐轮 IK 落脚点位置控制）', flush=True)
    elif _vmode == 'dual2':
        # v695: 双技能2——巡航 CarVMC，楼梯区 FootPlaceVMC（逐轮 IK 落脚点）
        from s10_mpc.stair_vmc_legs import FootPlaceVMC
        vmc_car = CarVMC()
        vmc_fp = FootPlaceVMC()
        vmc = vmc_car
        print('[VMC] 双技能2：CRUISE=CarVMC, STAIR=FootPlaceVMC', flush=True)
    elif _vmode == 'dual':
        # v466: 双技能执行器——巡航用 CarVMC（已调优），STAIR 模式用
        # VMCController（WBC 全身力控，老 dial-MPC 时代爬台阶执行器）。
        vmc_car = CarVMC()
        vmc_wbc = VMCController()
        # v480: WBC 独立轮参数（CarVMC 共享 env 会被改坏巡航）——WBC 轮
        # 驱动被阻尼抵消（t_wheel≈0 打滑空转卡楼梯），加大 wheel_k 减小
        # wheel_d 给足突破推力。
        vmc_wbc.wheel_k = float(os.environ.get(
            'S10_VMC_WBC_WHEEL_K', '10'))
        vmc_wbc.wheel_d = float(os.environ.get(
            'S10_VMC_WBC_WHEEL_D', '0.02'))
        vmc = vmc_car
        print('[VMC] 双技能模式：CRUISE=CarVMC, STAIR=VMCController(WBC)', flush=True)
    elif _vmode == 'stairwbc':
        # v871: 终版 StairWBC（位置基全身控制）——巡航用 CarVMC
        vmc_car = CarVMC()
        vmc = vmc_car
        print('[VMC] 双技能：CRUISE=CarVMC, STAIR=StairWBC(终版)', flush=True)
    else:
        vmc = VMCController()

    # v871: StairWBC 实例（S10_VMC_MODE=stairwbc 时 STAIR 区启用）
    vmc_stw = StairWBC()
    vmc_stw.stair_world = stair_world
    vmc_stw.stair = fol

    # 站起
    t = 0.0
    # v868: stand with CarVMC leg control - IK keeps wheels grounded,
    # body reaches cruise height; fixed-joint PD left hipy forward (-0.07
    # vs target -1.10) and body stuck at 0.24. Wheels locked (cmd vx=0).
    t = 0.0
    while t < 0.5:
        qpos_s = np.asarray(d.qpos, dtype=np.float64)
        qvel_s = np.asarray(d.qvel, dtype=np.float64)
        wheel_xyz_s = np.asarray([d.xpos[WHEEL_BODY[i]] for i in range(4)])
        wheel_vel_s = np.asarray([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])
        terr_s = np.zeros(4)
        for _li in range(4):
            _g = np.array([-1], dtype=np.int32)
            _dist = np.zeros(1); _nrm = np.zeros(3)
            _hit = mujoco.mj_ray(
                m, d, [wheel_xyz_s[_li, 0], wheel_xyz_s[_li, 1], 8.0],
                [0, 0, -1], None, True, -1, _g, _nrm)
            terr_s[_li] = (8.0 - _hit) if _hit > 0 else 0.0
        cmd_s = dict(vx=0.0, omega=0.0, roll_tar=0.0, pitch_tar=0.0)
        tau = vmc.compute_tau(qpos_s, qvel_s, wheel_xyz_s, wheel_vel_s,
                              cmd_s, terr_s, DT)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT

    # v219f: 地形感知来源。ray=上帝视角实时 raycast（调试，零噪声）；
    # lidar=lidar_site 扇形射线局部栅格（传感器视角，10Hz 更新，部署同款）
    # v871: 修复并发改动丢失的 if 头——恢复 lidar/ray 分支
    if os.environ.get('S10_VMC_TERRAIN', 'ray') == 'lidar':
        lterr = LidarTerrain(m, d)
        _lupd = -1.0
        def terrain_at(x, y):
            nonlocal _lupd
            if t - _lupd >= 0.1:
                lterr.update()
                _lupd = t
            _h = lterr.height(x, y)
            # v275: 高架伪影抑制——lidar 在起步坡看到上方高架盒底面
            # （2.16m 读数，实测抬轮误触发/腿阻抗过激侧翻）；读数高于
            # 机体+1.0m 视为伪影，用运动学地面（机高-0.55）兜底
            if _h > body_pos[2] + 1.0:
                return float(body_pos[2] - 0.55)
            return _h
    else:
        def terrain_at(x, y):
            g = np.array([-1], dtype=np.int32); dist = np.zeros(1); nrm = np.zeros(3)
            hit = mujoco.mj_ray(m, d, [x, y, 8.0], [0, 0, -1],
                                None, True, -1, g, nrm)
            return (8.0 - hit) if hit > 0 else 0.0

    next_idx = START_WP if 'START_WP' in dir() else 0
    # v893: 台架模式跳过 wp6（出生在 step1，直接以 wp7 为目标）
    if float(os.environ.get('S10_STAIR_BENCH', '0')) > 0 and next_idx == START_WP:
        next_idx = START_WP + 1
    # v822: 布尔几何相位状态（用户方案：位置基全身控制，硬切换非 sin²）
    _sp_f = 0.0; _sp_r = 0.0
    _sp_f_top = 0.0; _sp_r_top = 0.0
    _sp_f_t0 = None; _sp_r_t0 = None
    _exit_cnt = [0.0]
    wp_times = {}
    t_start = None
    traj = []
    prev_u = np.zeros(2)
    dbg = 0
    last_log = 0.0
    _last_dbg_t = -9.0
    # v220a: 单步跨越状态机（0=off, 1=前轮抬, 2=后轮抬）
    _step_state = 0
    _step_t0 = 0.0
    _terr_f = None
    # v568: CPG 步态相位（台阶技能）——按前进距离推进相位（每跨一个
    # 梯面 2π），前轴在 [0,π) 抬放、后轴在 [π,2π)，sin² 波形平滑衔接。
    _cpg_phase = 0.0
    # v567: 卡死超时——15s 无航点推进即退出（用户：不需要等很久）
    _last_adv_t = 0.0
    _stuck_timeout = float(os.environ.get('S10_STUCK_TIMEOUT', '15.0'))
    # v292: 力矩合规统计（腿 ±50 Nm / 轮 ±14 Nm，连续超限>0.5s 不合格）
    _max_tau_leg = 0.0
    _max_tau_wh = 0.0
    _over_run = 0.0
    _over_worst = 0.0
    _over_total = 0.0
    # v839: ref_path 生成频率实测
    _freq_t0 = time.perf_counter()
    _ctrl_cnt = 0
    _rp_cnt = 0
    _rp_tot = 0.0
    _rp_max = 0.0
    _mp_cnt = 0
    _mp_tot = 0.0
    _mp_max = 0.0
    _last_freq_t = -1e9
    # 主循环仿真起点（站起阶段 t=0→2 不计入）
    _freq_sim0 = t
    while t < MAX_SIM:
        qpos = np.asarray(d.qpos, dtype=np.float64)
        qvel = np.asarray(d.qvel, dtype=np.float64)
        body_pos = d.xpos[1]
        quat = d.xquat[1]
        yaw = float(np.arctan2(
            2.0*(quat[3]*quat[0]+quat[1]*quat[2]),
            1.0-2.0*(quat[2]**2+quat[3]**2)))
        # v291d: continuous roll safety envelope for crest-driven rear
        # lift boost (banked approach must not get extra rear lift).
        _roll_now = float(np.arctan2(
            2.0*(quat[0]*quat[1]+quat[2]*quat[3]),
            1.0-2.0*(quat[1]**2+quat[2]**2)))
        _roll_env = float(np.clip((0.35 - abs(_roll_now)) / 0.15, 0.0, 1.0))
        wheel_xyz = np.asarray([d.xpos[WHEEL_BODY[i]] for i in range(4)])
        wheel_vel = np.asarray([d.cvel[WHEEL_BODY[i]][0:3] for i in range(4)])

        # 导航 + MPPI 更新率（S10_NAV_HZ，默认 2）
        # v437: 原 int(t*20)%10==0 实为 2Hz（0.5s 一次）——方向翻转后 0.5s
        # 才更新，2.5m/s 下已转过 0.75rad，过弯振荡的直接放大器。
        # MPPI N=4096/H=40 单次 89ms → 实际上限 ~10Hz；默认 2 保 v410。
        _nav_period = max(1, int(round(200.0 / float(os.environ.get(
            'S10_NAV_HZ', '2')))))
        if int(t * 200) % _nav_period == 0 and (
                dbg == 0 or t - last_log >= 0.05):
            _rp_t0 = time.perf_counter()
            pos2 = body_pos[:2]
            # v462: 双模式判定（此前从未调用——STAIR 技能从不激活，wp6→7
            # 楼梯全程用巡航参数）
            # v871: 感知确认（lidar step_flag）+ 几何出口传入技能状态机
            try:
                _percept_ok = bool(lterr.stair_confirmed(pos2, yaw))
            except Exception:
                _percept_ok = False
            try:
                _exit_geom = False
                if stair_world:
                    (_lp, _ltng, _larc, _ldhv, _ltop) = stair_world[-1]
                    _fwdx = np.array([d.xmat[1][0], d.xmat[1][3]])
                    _fwdn = float(np.hypot(_fwdx[0], _fwdx[1])) + 1e-9
                    _fxr, _fyr = _fwdx[0] / _fwdn, _fwdx[1] / _fwdn
                    _rax9 = body_pos[:2] - np.array(
                        [_fxr * 0.228, _fyr * 0.228])
                    _d_rear = float(np.dot(_rax9 - _lp, _ltng))
                    _wz_rear = float(np.mean([
                        d.xpos[WHEEL_BODY[i], 2] for i in (2, 3)]))
                    _exit_geom = (_d_rear < -0.05
                                  and _wz_rear >= _ltop + self_fk_r - 0.02)
                if _exit_geom:
                    _exit_cnt[0] += _nav_period * DT
                else:
                    _exit_cnt[0] = 0.0
                _exit_ok = _exit_cnt[0] >= 0.1
            except Exception:
                _exit_ok = False
            try:
                fol.update_mode(pos2, next_idx, yaw=yaw,
                                percept_confirmed=_percept_ok,
                                stair_exit=_exit_ok)
            except Exception:
                pass
            vx, vyaw = fol.compute_cmd(
                pos2, yaw, next_idx,
                robot_z=float(body_pos[2]), yaw_rate=float(qvel[5]))
            v_ref = fol._last_vlim
            # 路径参考轨迹（弧长采样：当前位置起 8m，步长 0.5m）
            _ref = []
            _s0 = float(fol._s_cur)
            # v558: ref 段截止——vmax 4 时 12m ref 让 MPPI 提前看到下一
            # 航点后的弯（wp1→2 看到 wp2 S 弯提前转蛇形翻车实测）。截止到
            # 下一个航点+1.5m 出口余量，MPPI 只规划当前段。
            _ref_end = fol.path_total
            if next_idx < len(fol.path_wp_s):
                _ref_end = min(
                    _ref_end, float(fol.path_wp_s[next_idx]) + 1.5)
            for _ds in np.arange(0.0, 12.0, 0.5):
                _sp = _s0 + _ds
                if _sp >= _ref_end or _sp >= fol.path_total:
                    break
                _k = int(np.searchsorted(fol.path_cum, _sp, side="right") - 1)
                _k = min(max(_k, 0), len(fol.path_pts) - 2)
                _t = ((_sp - fol.path_cum[_k])
                      / max(fol.path_cum[_k + 1] - fol.path_cum[_k], 1e-6))
                _x = fol.path_pts[_k, 0] + _t * (fol.path_pts[_k + 1, 0] - fol.path_pts[_k, 0])
                _y = fol.path_pts[_k, 1] + _t * (fol.path_pts[_k + 1, 1] - fol.path_pts[_k, 1])
                _hd = fol.path_heading[min(_k, len(fol.path_heading) - 1)]
                _ref.append([_x, _y, _hd])
            _ref = np.array(_ref) if len(_ref) else np.array([[pos2[0], pos2[1], yaw]])
            st = np.array([pos2[0], pos2[1], yaw,
                           float(d.cvel[1][3]), float(d.cvel[1][4]), float(qvel[5])])
            _rp_dt = time.perf_counter() - _rp_t0
            _rp_cnt += 1
            _rp_tot += _rp_dt
            _rp_max = max(_rp_max, _rp_dt)
            if os.environ.get('S10_NAV_DEBUG', '0') == '1' and next_idx <= 6:
                print('[NAV] t=%.1f pos=(%.2f,%.2f) yaw=%.2f err=%.2f '
                      'tgt=(%.2f,%.2f) s_cur=%.2f vyaw=%.2f cte=%.2f'
                      % (t, body_pos[0], body_pos[1], yaw,
                         getattr(fol, '_last_err', 0.0),
                         getattr(fol, '_last_tgt', [0, 0, 0])[0],
                         getattr(fol, '_last_tgt', [0, 0, 0])[1],
                         getattr(fol, '_s_cur', 0.0), vyaw,
                         getattr(fol, '_last_cte', 0.0)), flush=True)
            # v871/v874: STAIR 爬升窗内无 MPPI（终版：纯导航），接近段保留
            # MPPI 纠偏（v828 靠 MPPI 保持走廊线）；S10_VMC_USE_NAV=1 全局关
            _mppi_off = False
            try:
                if fol.mode == 'STAIR' and stair_world:
                    # v879: MPPI 关闭窗用 stair_world[0]（第 1 级小台阶）——
                    # 实测用高 riser 时接近段 MPPI 多跑 0.4s 提前失速(y37.7)，
                    # 回到能到 y38 的旧窗口
                    (_rp0, _tng0, _sr0, _dh0, _top0) = stair_world[0]
                    _d_f0 = float(np.dot(pos2 - _rp0, _tng0))
                    _mppi_off = abs(_d_f0) < float(os.environ.get(
                        'S10_STAIR_MPPI_OFF_D', '0.5'))
            except Exception:
                _mppi_off = False
            if (os.environ.get('S10_VMC_USE_NAV', '0') == '1'
                    or _mppi_off):
                vx_c, om_c = vx, vyaw   # 直接导航指令（无 MPPI 随机性）
            else:
                # v270: MPPI 采样中心加曲率前馈 κ·v_ref（导航放开、MPPI
                # 约束兜底；样本围绕正确转向率，约束仍在摩擦锥内）
                # v315: MPPI 采样中心 = 导航完整转向指令（err + 曲率FF + cte）
                # ——纯路径跟踪会切内弯错过航点（wp1 最近 0.6m 实测）；导航的
                # 瞄航点逻辑保证 0.3m 判点，MPPI 负责平滑 + 摩擦锥约束兜底。
                _g_om = float(vyaw) if vyaw is not None else 0.0
                _mp_t0 = time.perf_counter()
                vx_c, om_c = mppi.plan(
                    st, _ref, v_ref, prev_u, guide_om=_g_om)
                _mp_dt = time.perf_counter() - _mp_t0
                _mp_cnt += 1
                _mp_tot += _mp_dt
                _mp_max = max(_mp_max, _mp_dt)
                if (os.environ.get('S10_NAV_DEBUG', '0') == '1'
                        and next_idx <= 6):
                    print('[MPPI] g_om=%.2f out=(%.2f,%.2f) vref=%.2f'
                          % (_g_om, vx_c, om_c, v_ref), flush=True)
            # v218p: omega 上限匹配 VMC yaw 能力（防指令远超执行导致振荡）
            # v245: speed-dependent cap - lateral accel envelope a_lat=w*v
            # （实测 YAW_TMAX 滑移权威下 ω 可达 3.6+，v=1.9 时 a_lat 7m/s2 翻车）
            _omcap = float(os.environ.get("S10_VMC_OM_CAP", "0.5"))
            _latmax = float(os.environ.get("S10_AUTO_LAT_MAX", "5.0"))
            _omcap = min(_omcap, _latmax / max(abs(vx_c), 0.5))
            om_c = float(np.clip(om_c, -_omcap, _omcap))
            # v599: 楼梯区导航 omega 置零——楼梯是直道且横向走廊宽，导航
            # 的 yaw 振荡只会让后轮左右对转空耗推力（tauW ±13.5 对转实测）；
            # 航向保持交给 WBC 反馈（om_f=0 时差速≈0，四轮统一向前推）。
            # v871: 技能门控改 fol.mode（z 先验 stair_zone 不再作触发）
            if fol.mode == 'STAIR':
                om_c *= 0.5
            # v263: 回退 v261/v262 脊前对准（起步被扰动 + wp4→5 仍不稳，
            # 脆）。过脊靠地形前瞻（ERR_GATE 提高后转弯中保持生效）+ 慢速。
            # 近脊仅保留速度相关 omcap（v245）。
            prev_u = np.array([vx_c, om_c])
            last_log = t
        else:
            vx_c, om_c = prev_u

        # 地形 + 前瞻（v222: 连续地形响应，无门控）——
        # 每轮高程取"轮前方 _lk 处"，腿垂直阻抗连续跟随前方地形：
        # 接近横脊时高程升 → 腿伸 → 轮抬，过脊自然回落
        _lk = float(os.environ.get('S10_VMC_TERRAIN_LOOKAHEAD', '0.0'))
        if _lk > 0.0:
            _bx = d.xmat[1][0]
            _by = d.xmat[1][3]
            _bn = float(np.hypot(_bx, _by)) + 1e-9
            _bx, _by = _bx / _bn, _by / _bn
            # v223g: 前瞻权重随导航 yaw 误差连续衰减——弯道未完成(err大)
            # 前瞻≈0(防转向中抬轮失控)，直线对准(err小)前瞻满(过脊抬轮)
            _w = float(os.environ.get('S10_VMC_TERRAIN_AHEAD_W', '1.0'))
            _err = float(getattr(fol, '_last_err', 0.0))
            _w_eff = _w * float(np.clip(
                1.0 - abs(_err) / float(os.environ.get(
                    'S10_VMC_TERRAIN_ERR_GATE', '0.7')), 0.0, 1.0))
            terr_foot = np.array([terrain_at(wheel_xyz[i, 0],
                                             wheel_xyz[i, 1])
                                  for i in range(4)])
            # v231: 运动学 fallback 默认关（改变 wp4→5 部分格值致混沌翻车）；
            # 需要时 S10_VMC_TERRAIN_KIN=1（wp5→6 缓坡无数据区）
            if (os.environ.get('S10_VMC_TERRAIN_KIN', '0') == '1'
                    and os.environ.get('S10_VMC_TERRAIN', 'ray') == 'lidar'):
                for _i in range(4):
                    if not lterr.has(wheel_xyz[_i, 0], wheel_xyz[_i, 1]):
                        terr_foot[_i] = float(wheel_xyz[_i, 2] - 0.081)
            # v223h: 前瞻用车体中心前方单值，四腿共用——逐腿前瞻在狗斜向
            # 接近脊时右轮先到→单侧抬轮侧翻（wz 0.85/0.84 实测）
            _hx = body_pos[0] + _bx * _lk
            _hy = body_pos[1] + _by * _lk
            terr_ahead = np.full(4, terrain_at(_hx, _hy))
            terr = (1.0 - _w_eff) * terr_foot + _w_eff * terr_ahead
        else:
            terr = np.array([terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                             for i in range(4)])
        # v223b: 地形低通（lidar 栅格稀疏/噪声，防腿抖）
        _tlp = float(os.environ.get('S10_VMC_TERRAIN_LP', '0.0'))
        if _tlp > 0.0:
            if _terr_f is None:
                _terr_f = terr.copy()
            else:
                _terr_f = _tlp * terr + (1.0 - _tlp) * _terr_f
            terr = _terr_f
        # v595: 楼梯落脚点——轴接近 riser 时把该轴轮下地形设为台面高
        # （z_des 随地形中点升、pitch 用轴距坡度，前髋抬高跨棱、后轮
        # 仍贴地受力；raw 地面会导致 z 控制把轮子悬空、后轮失牵引）
        _iszn9 = (fol.mode == 'STAIR')
        # v885: stairwbc 也走原始地形分支——此前用 WBC 分支的
        # stair_wheel_ref（棱前 0.14m ramp 提前抬升→离地→yaw 对转实测）
        _fp_active = os.environ.get('S10_VMC_MODE', 'wbc') in (
            'place', 'dual2', 'stairwbc')
        # v746: lidar 高程图在楼梯区失真。但轮下地形与抬轮目标要分开——
        # WBC（力控）：轮下地形用运动学地面（轮心 z - 半径），支撑腿贴地
        # 承重，body 高度不被台面表顶起（z_des 过高→四轮悬空 fn=0 实测）；
        # 抬轮目标由 place_z 单独给（见 CPG 分支）。
        # FootPlace（IK 放轮）：保持台面表覆盖（它把轮直接放台面上）。
        if _iszn9 and _fp_active:
            try:
                # v884: 小台阶用轮下实际地形（原始 lidar/ray）——stair_terrain
                # 是 y 阶跃，轮还没到就把腿目标抬到台面顶 → 狗提前离地失抓地
                # → 差速对转 yaw 失控（实测）。0.061m 小台阶自然滚过；
                # 0.125m 大台阶由 StairWBC 贴面爬升 place_z 显式抬升。
                terr = np.array(
                    [terrain_at(wheel_xyz[i, 0], wheel_xyz[i, 1])
                     for i in range(4)], dtype=np.float64)
            except Exception:
                pass
        elif _iszn9:
            try:
                # v755: 显式轮心 z 参考（文献:轮轨迹显式化）——stair_wheel_ref
                # 由 riser 表解析生成（棱前 RAMP_A ramp 平滑抬到台面顶+r），
                # 替代"轮位-半径"自证地面（悬空轮跟随轮位不被纠正死锁实测）。
                # 轮心目标 = terr + r = stair_wheel_ref，支撑腿沿参考轨迹抬升。
                # v755d: 左右轮参考对称化——yaw≠1.57 时左/右轮 y 不同→ramp
                # 相位差→单侧过抬 roll 侧翻实测（wz 0.78/0.95）；楼梯走廊
                # 横向平坦，用轴心 y 生成参考，左右一致。
                _wy_sym = np.array(
                    [float(np.mean(wheel_xyz[0:2, 1]))] * 2
                    + [float(np.mean(wheel_xyz[2:4, 1]))] * 2)
                terr = np.asarray(fol.stair_wheel_ref(
                    _wy_sym), dtype=np.float64) - 0.081
            except Exception:
                pass
        # v755b: 旧覆盖（棱前 0.30m 把 terr 抬到台面顶+0.02）删除——提前抬
        # 高把前轮卸载 fn=0、轮矩失效死锁实测；轮心 z 参考由 stair_wheel_ref
        # 的 ramp（棱前 0.14m 起）显式生成，棱处达台面顶+r，不再需要覆盖。
        s_cur = float(getattr(fol, '_s_cur', 0.0))
        # v292: 台阶窗 vx 连续插值到 STAIR_WIN_VX（默认 1.8，不归零）——
        # 窗内只换腿控制（几何相位）与轮力矩模式，vx 参考保持连续
        # v871: vx 楼梯窗世界坐标化——用 stair_world 物理距离（与布尔相位
        # 同源），s_cur 在倾斜/卡死时漂移不再影响
        if stair_world:
            _sramp = float(os.environ.get('S10_STAIR_VX_RAMP', '1.0'))
            _win_vx = float(os.environ.get('S10_STAIR_WIN_VX', '1.8'))
            _p0, _t0, _, _, _ = stair_world[0]
            _pl, _tl, _, _, _ = stair_world[-1]
            _d_in = float(np.dot(body_pos[:2] - _p0, _t0))
            _d_out = float(np.dot(body_pos[:2] - _pl, _tl))
            _f_in = float(np.clip((_d_in + _sramp) / max(_sramp, 1e-6),
                                  0.0, 1.0))
            _f_out = float(np.clip(
                (2.0 + _sramp - _d_out) / max(_sramp, 1e-6), 0.0, 1.0))
            _f = min(_f_in, _f_out)
            vx_c = _win_vx * _f + vx_c * (1.0 - _f)

        # v236: 台阶相位步态——按弧长对每级 riser 调度前轴/后轴抬放
        # （几何已知，无硬模式：仅当台阶 riser 在前方窗口内才产生抬放量）。
        # 前轴窗口 = 棱边前 0.40m -> 棱边后 0.30m；后轴按半轴距 0.228m 延后。
        step_lift = np.zeros(4)
        place_z = np.zeros(4)   # v732: 每腿落脚点台面高（CPG 抬升用）
        stair_lift_flag = 0.0
        # v573: 楼梯区标识——stair_zone 内由 CPG 步态独占抬轮，
        # 巡航横脊抬轮/后轮跟抬/地形 bump 全部禁用（防前后轴双抬+塌身）。
        _in_stairzone_now = (fol.mode == 'STAIR')
        # v871: 执行器解耦——STAIR 模式 + 前轴距首级 riser < S10_STAIR_EXEC_D
        # （默认 1.0m）才切位置基执行器；接近段保持 CarVMC 平稳减速，
        # 避免高速切换 FP 弹跳
        _stair_exec = False
        if _in_stairzone_now and stair_world:
            try:
                # v879(批准 A+B): 执行器只对"高 riser"(dh>0.085) 生效——
                # 第 1 级小台阶(0.061m)保持 CarVMC 动量滚上，第 2 级起
                # (0.125m)才切 StairWBC 贴面爬升
                _d_first = 1e9
                for (_rp0, _tng0, _sr0, _dh0, _top0) in stair_world:
                    if _dh0 <= 0.085:
                        continue
                    _dd0 = float(np.dot(body_pos[:2] - _rp0, _tng0))
                    if abs(_dd0) < abs(_d_first):
                        _d_first = _dd0
                _stair_exec = abs(_d_first) < float(os.environ.get(
                    'S10_STAIR_EXEC_D', '0.5'))
            except Exception:
                _stair_exec = True
        # v789: 恢复 v752 USC 关键姿态插值因子——前轴到最近棱距离连续量：
        # 棱前 0.05m 起、过棱 0.05m 满。驱动前腿伸长+后腿收缩（几何前倾），
        # 前轮贴台面接触（当年卡"接触但不推进"，轮矩钳制已 v755b 解锁）。
        _stair_pose = 0.0
        if _in_stairzone_now and stair_world:
            _fwdsp = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fnsp = float(np.hypot(_fwdsp[0], _fwdsp[1])) + 1e-9
            _fxsp, _fysp = _fwdsp[0] / _fnsp, _fwdsp[1] / _fnsp
            _axsp = body_pos[:2] + np.array([_fxsp * 0.228, _fysp * 0.228])
            _dsp_min = 1e9
            for (_rp, _tng, _sr, _dhv, _top) in stair_world:
                _dsp = float(np.dot(_axsp - _rp, _tng))
                if abs(_dsp) < abs(_dsp_min):
                    _dsp_min = _dsp
            _stair_pose = float(np.clip(
                (_dsp_min + 0.05) / 0.10, 0.0, 1.0))
        # v822: 布尔几何相位（用户方案）——S10_STAIR_POSMODE=1 时楼梯段
        # 用硬切换状态机（前/后轴 |到棱|<0.05m 切位置抬升，过棱且轮落台面
        # 释放），替代 sin² CPG 连续窗。位置目标由 stair_world 台面顶给出。
        _posmode_st = float(os.environ.get('S10_STAIR_POSMODE', '0'))
        if _posmode_st > 0.0 and _stair_exec and stair_world:
            _fwd_p = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn_p = float(np.hypot(_fwd_p[0], _fwd_p[1])) + 1e-9
            _fx_p, _fy_p = _fwd_p[0] / _fn_p, _fwd_p[1] / _fn_p
            _fax_p = body_pos[:2] + np.array([_fx_p * 0.228, _fy_p * 0.228])
            _rax_p = body_pos[:2] - np.array([_fx_p * 0.228, _fy_p * 0.228])
            _df_p = 1e9; _dr_p = 1e9; _tf_p = 0.0; _tr_p = 0.0
            for (_rp, _tng, _sr, _dhv, _top) in stair_world:
                # 只对高 riser（>轮半径 0.085）触发抬升，小台阶纯滚
                if _dhv <= 0.085:
                    continue
                _dd_f = float(np.dot(_fax_p - _rp, _tng))
                _dd_r = float(np.dot(_rax_p - _rp, _tng))
                if abs(_dd_f) < abs(_df_p):
                    _df_p = _dd_f; _tf_p = float(_top)
                if abs(_dd_r) < abs(_dr_p):
                    _dr_p = _dd_r; _tr_p = float(_top)
            _wz4p = np.asarray([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
            # v875: 前轴相位机——触发提前到 d<0.15（轮半径 0.081+余量），
            # 原 |d|<0.05 时轮已贴 riser 立面卡死（0.081>0.05 死区实测）；
            # 过棱且前轮落台面顶+r 释放
            _sw_d = float(os.environ.get('S10_STAIR_SWING_D', '0.15'))
            if _sp_f <= 0.0:
                # v900: 前轴进入也加防双抬守卫（与后轴一致）
                if -_sw_d < _df_p < 0.05 and _sp_r <= 0.0:
                    _sp_f = 1.0; _sp_f_top = _tf_p
                    _sp_f_t0 = t
            elif (_df_p > 0.05
                  and float(np.mean(_wz4p[0:2])) >= _sp_f_top + self_fk_r):
                _sp_f = 0.0; _sp_f_t0 = None
            # v871: SWING 超时释放（1.5s 内未达台面顶则回 STANCE，防卡死空跳）
            if _sp_f > 0 and _sp_f_t0 is not None and t - _sp_f_t0 > float(
                    os.environ.get('S10_STAIR_SWING_TO', '1.5')):
                _sp_f = 0.0; _sp_f_t0 = None
            # 后轴相位机：前轴不在抬升时 d<0.15 才抬（防同抬）
            if _sp_r <= 0.0:
                if -_sw_d < _dr_p < 0.05 and _sp_f <= 0.0:
                    _sp_r = 1.0; _sp_r_top = _tr_p
                    _sp_r_t0 = t
            elif (_dr_p > 0.05
                  and float(np.mean(_wz4p[2:4])) >= _sp_r_top + self_fk_r):
                _sp_r = 0.0; _sp_r_t0 = None
            if _sp_r > 0 and _sp_r_t0 is not None and t - _sp_r_t0 > float(
                    os.environ.get('S10_STAIR_SWING_TO', '1.5')):
                _sp_r = 0.0; _sp_r_t0 = None
            step_lift[:] = [_sp_f, _sp_f, _sp_r, _sp_r]
            place_z[:] = [(_sp_f_top if _sp_f > 0 else 0.0),
                          (_sp_f_top if _sp_f > 0 else 0.0),
                          (_sp_r_top if _sp_r > 0 else 0.0),
                          (_sp_r_top if _sp_r > 0 else 0.0)]
            stair_lift_flag = max(_sp_f, _sp_r)
        # v264: 连续前瞻抬轮（无门控）——按"轴前 0.35m 地形高 - 轴下地形高"
        # 连续抬放（比例 clamp 0.15m）。纯几何连续量，替代 hop 冲量/横脊
        # 步态等离散触发（用户原则：除 cruise/stair 切换外无门控）。
        # v266: 抬轮前瞻独立于 terr 前瞻（LOOKAHEAD=0.5 在起步坡上使地形
        # 阻抗过激翻车实测；抬轮用自身 0.35m 窗口）
        # v276: 已知横脊连续抬放（替代 lidar rise——lidar 在起步坡误触发、
        # 近场盲区噪声；已知地图连续响应，与台阶 skill 同类）。0.5m 前起抬、
        # 过脊 0.3m 释放，前/后轴分别。
        if (float(os.environ.get('S10_VMC_RIDGE_LIFT_CONT', '1')) > 0
                and ridge_world
                and (not _in_stairzone_now
                     or float(os.environ.get('S10_STAIR_RIDGE_LIFT', '0')) > 0)):
            _fwd5 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn5 = float(np.hypot(_fwd5[0], _fwd5[1])) + 1e-9
            _fx5, _fy5 = _fwd5[0] / _fn5, _fwd5[1] / _fn5
            # v291d: crest state (front axle already above rear axle) selects
            # the long rear release window; otherwise rear keeps the proven
            # 0.30m window (v291b regression at the wp2->3 turn). Crest boost
            # also fades with body roll (continuous safety envelope).
            _wzc = np.asarray([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
            _crest_pre = (float(np.clip((float(np.max(_wzc[0:2]))
                                          - float(np.min(_wzc[2:4])) - 0.03)
                                         / 0.06, 0.0, 1.0)) * _roll_env)
            for _ai in range(2):
                _sgn = 1.0 if _ai == 0 else -1.0
                _ax = np.array([body_pos[0] + _fx5 * 0.228 * _sgn,
                                body_pos[1] + _fy5 * 0.228 * _sgn])
                _dmin_r = 1e9
                for (_rp, _tng, _sr, _dh) in ridge_world:
                    # v285: 只对 dh>=0.08 的脊抬轮（<0.08 微脊=轮滚微起伏，
                    # 腿阻抗吸收，不抬）
                    if _dh < 0.08 or _dh > 0.5:
                        # v684: dh>0.5 是墙/垂直障碍（如 wp11→12 的 3.55m 墙），
                        # 不是可爬台阶——抬轮无用且会翻车，跳过
                        continue
                    _dd = float(np.dot(_ax - _rp, _tng))
                    # v861: ????????????????0 ?????
                    _dn2 = float(np.hypot(_ax[0] - _rp[0], _ax[1] - _rp[1]))
                    _dlat = float(np.sqrt(max(_dn2 * _dn2 - _dd * _dd, 0.0)))
                    if _dlat > 1.5:
                        _dd = 1e9
                    if -0.3 <= _dd <= 0.8 and _dd < _dmin_r:
                        _dmin_r = _dd
                if _dmin_r < 0.8:
                    # v278: 紧窗口满幅——棱边前 0.25m 起抬、0.1m 前满幅、
                    # 过棱 0.1m 后保持、0.3m 后释放。0.141m 够清 0.12m 脊；
                    # v277 提前 0.5m 满幅→前轮长时间离地失驱动侧翻
                    # v291c: front axle keeps the proven 0.30m release
                    # (v277/v288: long front lift during turns loses drive);
                    # rear axle gets the 0.50m window only at crest state
                    # (front already on the ridge top).
                    if _ai == 0 or _crest_pre <= 0.0:
                        _release, _span = 0.30, 0.20
                    else:
                        _release, _span = 0.50, 0.25
                    _lstart = float(os.environ.get(
                        'S10_VMC_LIFT_START', '0.25'))
                    _lramp = float(os.environ.get(
                        'S10_VMC_LIFT_RAMP', '0.15'))
                    _lift_r = float(
                        np.clip((_lstart - _dmin_r) / _lramp, 0.0, 1.0)
                        * np.clip((_release - _dmin_r) / _span, 0.0, 1.0))
                    if _lift_r > 0.02:
                        if _ai == 0:
                            step_lift[0:2] = np.maximum(
                                step_lift[0:2], _lift_r)
                        else:
                            step_lift[2:4] = np.maximum(
                                step_lift[2:4], _lift_r)
            # v406: 前后轴防同抬——前轴抬升中连续抑制后轴（至少保持后轮
            # 接地保牵引），防 wp4->5 横脊上四轮全抬（sl=[1,1,1,1] 实测
            # 无抓地卡死/漂东翻车）。连续量：前轴 lift>0.3 起后轴按比例衰减。
            # v279: 抬轮幅度乘航向对准系数——转弯（|err| 大）时抬轮衰减，
            # 防"转弯+抬轮"叠加侧翻（wp2→3 后轮满抬转弯中翻车实测）。
            _err_l = abs(float(getattr(fol, '_last_err', 0.0)))
            _lift_align = float(np.clip(
                1.0 - _err_l / float(os.environ.get(
                    'S10_VMC_LIFT_ERR_GATE', '0.8')), 0.0, 1.0))
            # v288: 回退 v287——前轮抬轮转弯中同样不稳（wp1→2 sl满幅+ω4.08
            # 翻车）；wp4→5 真问题=偏航处导航命令反向，需 NAV_DEBUG 定位
            if _lift_align < 1.0:
                step_lift *= _lift_align
            if float(np.max(step_lift)) > 0.02:
                stair_lift_flag = 1.0
        # v264: lidar rise 抬轮（仅 S10_VMC_RIDGE_LIFT_CONT=0 时兜底）
        _lk_c = float(os.environ.get('S10_VMC_LIFT_LOOKAHEAD', '0.35'))
        if (float(os.environ.get('S10_VMC_RIDGE_LIFT_CONT', '1')) <= 0
                and _lk_c > 0.05 and not _in_stairzone_now):
            _fwd4 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn4 = float(np.hypot(_fwd4[0], _fwd4[1])) + 1e-9
            _fx4, _fy4 = _fwd4[0] / _fn4, _fwd4[1] / _fn4
            for _ai in range(2):
                _sgn = 1.0 if _ai == 0 else -1.0
                _ax = np.array([body_pos[0] + _fx4 * 0.228 * _sgn,
                                body_pos[1] + _fy4 * 0.228 * _sgn])
                _ha = terrain_at(_ax[0] + _fx4 * _lk_c,
                                 _ax[1] + _fy4 * _lk_c)
                _hb = terrain_at(_ax[0], _ax[1])
                if (os.environ.get('S10_LIFT_DEBUG', '0') == '1'
                        and next_idx == 5 and 20.5 < t < 24.0):
                    print('[LIFT] t=%.1f ai=%d ha=%.3f hb=%.3f ax=(%.2f,%.2f)'
                          % (t, _ai, _ha, _hb, _ax[0], _ax[1]), flush=True)
                # v265/v274: 上升量带通——只对 0.02~0.35m 可爬台阶响应
                # （0.12m 脊→0.8；起步坡 0.5m 伪尖峰→0；v273 前向修正后
                # 抬轮在起步坡误触发侧翻，收紧上限）
                _rise0 = float(_ha - _hb)
                _rise = 0.0
                if _rise0 > 0.08 and _rise0 < 0.35:
                    _rise = (float(np.clip(_rise0 / 0.15, 0.0, 1.0))
                             * float(np.clip((0.35 - _rise0) / 0.15, 0.0, 1.0)))
                if _rise > 0.02:
                    if _ai == 0:
                        step_lift[0:2] = np.maximum(step_lift[0:2], _rise)
                    else:
                        step_lift[2:4] = np.maximum(step_lift[2:4], _rise)
            if float(np.max(step_lift)) > 0.02:
                stair_lift_flag = 1.0
        if (float(os.environ.get('S10_VMC_STAIR_GAIT', '0')) > 0
                and stair_risers
                and stair_risers[0][0] - 1.0 <= s_cur
                <= stair_risers[-1][0] + 2.0):
            _fl_max, _rl_max = 0.0, 0.0
            # v238: 收窄抬放窗口（梯面0.4m≈轴距0.456m，宽窗口致前轮持续
            # 抬起不落地）——前轴 df∈[-0.08,0.18]、后轴 dr∈[-0.06,0.20]，
            # 每级 riser 前后轴快速衔接抬放；可选 FRONT_HOLD 连续挂前轮
            # v465: 前后抬升窗口可调（S10_STAIR_FRONT_START/REAR_START）。
            # 原后轮窗起点 -0.06m 与前轮释放 +0.18m 之间有 0.22m 空档——
            # 空档里两轴都不抬，后轮 0.081m 半径撞 0.13m 台阶面失速（wp6→7
            # 卡 y=38 死锁）。后轮窗提前到 -0.20~-0.25 与前轮衔接。
            _fst = float(os.environ.get('S10_STAIR_FRONT_START', '0.08'))
            _rst = float(os.environ.get('S10_STAIR_REAR_START', '0.06'))
            for (sr, dhv) in stair_risers:
                _df = s_cur - (sr - 0.228)   # 前轴到棱边
                _dr = s_cur - (sr + 0.228)   # 后轴到棱边
                _fl = (float(np.clip((0.18 - _df) / 0.06, 0.0, 1.0))
                       * float(np.clip((_df + _fst) / 0.06, 0.0, 1.0)))
                _rl = (float(np.clip((0.20 - _dr) / 0.06, 0.0, 1.0))
                       * float(np.clip((_dr + _rst) / 0.06, 0.0, 1.0)))
                _fl_max = max(_fl_max, _fl); _rl_max = max(_rl_max, _rl)
            if (float(os.environ.get('S10_VMC_STAIR_FRONT_HOLD', '0')) > 0
                    and stair_risers[0][0] - 0.6 <= s_cur
                    <= stair_risers[-1][0] + 1.2):
                _fl_max = max(_fl_max, 1.0)
            # v470: 真正实现 v406 防同抬（原只有注释）——六级楼梯前后轴
            # 同时抬升（sl 都 0.8-1.0）→ 无轮接地卡死（v469 走廊修复后
            # 卡第一级实测）。交替：前轴抬升时连续抑制后轴（前轴回落时
            # 后轴接管），后轴同理。S10_STAIR_ANTI=0 关闭。
            if float(os.environ.get('S10_STAIR_ANTI', '0')) > 0:
                _fa = float(np.clip((_fl_max - 0.15) / 0.35, 0.0, 1.0))
                _ra = float(np.clip((_rl_max - 0.15) / 0.35, 0.0, 1.0))
                if _fa >= _ra:
                    _rl_max *= (1.0 - _fa)
                else:
                    _fl_max *= (1.0 - _ra)
            if _fl_max + _rl_max > 0.02:
                step_lift[:] = [_fl_max, _fl_max, _rl_max, _rl_max]
                stair_lift_flag = 1.0
        elif float(os.environ.get('S10_VMC_STEP_OVER', '0')) > 0:
            # v247: 横脊连续抬放（替代 v220a 状态机）——按前/后轴到横脊的
            # 世界坐标物理距离触发（s_cur 投影在转向时滞后会漏触发）：
            # 前轴 0.9m 前起抬、过脊即放；后轴同窗口。与台阶步态同模式。
            _fwdv = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn2 = float(np.hypot(_fwdv[0], _fwdv[1])) + 1e-9
            _fx2, _fy2 = _fwdv[0] / _fn2, _fwdv[1] / _fn2
            _fax = np.array([body_pos[0] + _fx2 * 0.228,
                             body_pos[1] + _fy2 * 0.228])
            _rax = np.array([body_pos[0] - _fx2 * 0.228,
                             body_pos[1] - _fy2 * 0.228])
            _fl2, _rl2 = 0.0, 0.0
            for (_rp, _tng, _sr, _dh) in ridge_world:
                # v259: 沿路径切线投影的距离（横向偏移不计入）
                _dfr = float(abs(np.dot(_fax - _rp, _tng)))
                _drr = float(abs(np.dot(_rax - _rp, _tng)))
                # v254/v258: 0.45m 前起抬；**满值保持到轮子越过棱边 0.25m**
                # （v254 过脊即放，失速时窗口掉到 0.3~0.4 部分抬升→后轮
                # 爬不上脊死锁，wp4→5 卡 90s 实测）
                _fl2 = max(_fl2, float(
                    np.clip((0.45 - _dfr) / 0.15, 0.0, 1.0)
                    * np.clip((_dfr + 0.25) / 0.20, 0.0, 1.0)))
                _rl2 = max(_rl2, float(
                    np.clip((0.45 - _drr) / 0.15, 0.0, 1.0)
                    * np.clip((_drr + 0.25) / 0.20, 0.0, 1.0)))
            if _fl2 + _rl2 > 0.02:
                # v256: 横脊步态激活时用轮下地形（禁前瞻）——前瞻抬轮 +
                # step_lift 抬轮双重叠加致腿饱和侧翻（wp4→5 实测）
                if 'terr_foot' in locals():
                    terr = terr_foot.copy()
                # v255: 横脊只抬前轮，后轮滚动/蹬过（斜过脊时后轮单侧先触
                # 脊，双侧同抬反而侧翻；"前挂后蹬"）
                if float(os.environ.get(
                        "S10_VMC_RIDGE_REAR_LIFT", "1.0")) > 0:
                    step_lift[:] = [_fl2, _fl2, _rl2, _rl2]
                else:
                    step_lift[:] = [_fl2, _fl2, 0.0, 0.0]

        # v221e: 车身抬升（过脊）——渐变：脊前 0.6m 起、0.2m 处满，
        # 过脊后 0.4m 内回落（防阶跃弹跳侧翻）
        _body_lift = 0.0
        # v221g: 物理检测——前轮前方 0.5m 真实地形高差>0.07 才抬身
        # （s_cur 投影推进快于物理位置，弯道出口误触发侧翻实测）
        if os.environ.get('S10_VMC_BODY_LIFT', '0') == '1':
            _fwd3 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fl = float(np.hypot(_fwd3[0], _fwd3[1])) + 1e-9
            _fx3, _fy3 = _fwd3[0] / _fl, _fwd3[1] / _fl
            _hf = terrain_at(body_pos[0] + _fx3 * 0.5,
                             body_pos[1] + _fy3 * 0.5)
            _h0 = terrain_at(body_pos[0], body_pos[1])
            if _hf - _h0 > 0.07:
                _body_lift = 1.0
        _lift = float(os.environ.get('S10_VMC_RIDGE_LIFT', '0.0'))
        _lift_act = 0.0
        if not _in_stairzone_now:
            for (sr, dhv) in ridge_arcs:
                ds = s_cur - sr
                # v219u: 短促 bump——前轮棱前 0.8m 抬、过棱即放。
                if -0.8 <= ds < 0.08:
                    f = float(np.clip((0.8 - abs(ds + 0.30)) / 0.8, 0.0, 1.0))
                    terr[:] = np.maximum(terr, terr + _lift * f)
                    _lift_act = max(_lift_act, f)
                elif _step_state == 0 and 0.08 <= ds < 0.55:
                    # v220i: 迈步时后轮不抬（前轮由 step_lift 抬 + 前轮 bump 保留，
                    # 组合抬升够脊顶；后轮必须着地推车身）
                    f = float(np.clip((0.55 - ds) / 0.47, 0.0, 1.0))
                    terr[2:] = np.maximum(terr[2:], terr[2:] + _lift * f)
                    _lift_act = max(_lift_act, f)
        # v219i: 后轮跟抬——前轮已上棱即抬后轮。加 s_cur 接近横脊的条件：
        # 弯道/上坡时前轮天然高于后轮（>0.05），无条件触发会误抬后轮
        # 导致失去驱动力（v219a 实测 wp3→4 卡死根因）。
        _wz = np.asarray([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
        _lf = max(_wz[0], _wz[1]); _lr = min(_wz[2], _wz[3])
        _in_ridge = any(-0.8 <= s_cur - sr <= 1.2 for (sr, dhv) in ridge_arcs)
        # v291: crest = continuous geometry (front axle already on ridge top,
        # rear axle at the face). Boosts rear terrain bump to ~1.4x _lift so
        # the rear wheel center target >= ridge top, and restores rear
        # step_lift to full (overrides yaw-alignment decay only for the rear).
        _crest = (float(np.clip((_lf - _lr - 0.03) / 0.06, 0.0, 1.0))
                  if (_in_ridge and _lf > _lr + 0.03
                         and not _in_stairzone_now) else 0.0) * _roll_env
        # v220b: 迈步期间禁用后轮跟抬——前轮被迈步抬起时 wz 天然高于后轮，
        # 会误把后轮也抬离地（全轮无推力死锁）
        if _step_state == 0 and _crest > 0.0:
            _rear_x = float(os.environ.get('S10_VMC_REAR_CREST_EXTRA', '0.6'))
            terr[2:] = np.maximum(
                terr[2:], terr[2:] + _lift * (0.8 + _rear_x * _crest))
            step_lift[2:4] = np.maximum(step_lift[2:4], _crest)


        # v234: 巡航转弯=差速为主 + hip 微 roll ±3.5°(0.06 rad)协调——
        # 不采用腿足式压弯（轮足深压弯→后轮打滑/无悬架抖动，用户总结）
        # v290: 压弯幅度可调（S10_CAR_ROLL_AMP，默认 0.06；测试 0.12 强化
        # 高速转弯向心/减侧翻）
        # v748: 真正的 lean-in 压弯——θ ≈ a_lat/g = ω·|vx|/g（重力横向
        # 分量辅助向心 + 重心内移降 LTR）。S10_CAR_ROLL_K 默认 0.10≈1/g；
        # 开大压弯须配 vmc_legs 弯外轮保载（S10_CAR_ROLL_MAX_DL/MIN_FRAC）。
        # 默认 = v746 精确行为：-0.06·ω·|vx_cmd|，clip ±0.06。大压弯
        # (S10_CAR_ROLL_AMP>0.06) 时建议同时设 S10_CAR_ROLL_K=0.10、
        # S10_CAR_ROLL_ERR_GATE=0.5、S10_CAR_ROLL_VGATE=1.5（配 vmc_legs
        # 弯外轮保载 ROLL_MAX_DL/MIN_FRAC）。
        _ramp = float(os.environ.get("S10_CAR_ROLL_AMP", "0.06"))
        _lean_k = float(os.environ.get("S10_CAR_ROLL_K", "0.06"))
        roll_tar = float(np.clip(-_lean_k * om_c * abs(vx_c), -_ramp, _ramp))
        # v854: 删除 ROLL_VGATE/ROLL_ERR_GATE 门控（用户：无离散门控）——
        # 压弯 roll_tar 直接随 vx·ω 生成
        pitch_tar = 0.0
        # v285: 脊前速度**微缩**（用户"横脊=连续扰动非障碍"）——±0.15m 内
        # 线性从 vlim 缩到 min(vlim,2.5)，脊后即恢复；严禁硬砍 1.0 制造
        # 速度凹坑
        if ridge_world:
            _fwd6 = np.array([d.xmat[1][0], d.xmat[1][3]])
            _fn6 = float(np.hypot(_fwd6[0], _fwd6[1])) + 1e-9
            _fx6, _fy6 = _fwd6[0] / _fn6, _fwd6[1] / _fn6
            _dd6_min = 1e9
            for _sgn6 in (1.0, -1.0):
                _ax6 = np.array([body_pos[0] + _fx6 * 0.228 * _sgn6,
                                 body_pos[1] + _fy6 * 0.228 * _sgn6])
                for (_rp, _tng, _sr, _dh) in ridge_world:
                    _dd6 = float(np.dot(_ax6 - _rp, _tng))
                    if abs(_dd6) < abs(_dd6_min):
                        _dd6_min = _dd6
            _d6 = abs(_dd6_min)
            if _d6 < 0.15:
                _soft = float(os.environ.get("S10_RIDGE_SOFT_VX", "2.5"))
                vx_c = min(vx_c, _soft + max(vx_c - _soft, 0.0) * _d6 / 0.15)
        # v825: 删除下降区位置直判减速（用户指令）
        try:
            fwd = np.array([d.xmat[1][0], d.xmat[1][3]])
            fx, fy = fwd[0], fwd[1]
            h_a = terrain_at(body_pos[0] + fx*0.6, body_pos[1] + fy*0.6)
            h_b = terrain_at(body_pos[0] - fx*0.6, body_pos[1] - fy*0.6)
            # v225/v583: 巡航段爬坡前倾匹配坡度（轮推力对准）；楼梯区
            # 反向——车身**抬头**匹配台阶坡度，抬升前髋让抬轮过 0.13m 棱
            # （低头会把前髋压低，前轮差 1cm 卡棱实测）。
            if _in_stairzone_now:
                # v789: 楼梯 pitch 可切换——默认 1=抬头（v765，前轮压面/
                # 后轮贴地推）；2=低头（v752，配 stair_pose 前腿伸长，前轮
                # 贴台面、后轮地面推，当年实测前轮接触台面）。
                _pfr = float(np.mean(terr[0:2])); _prr = float(np.mean(terr[2:4]))
                if float(os.environ.get("S10_STAIR_PITCH_MODE", "1")) == "2":
                    pitch_tar = float(np.clip(
                        np.arctan2(_prr - _pfr, 0.456) + _stair_pose * 0.08,
                        -0.30, 0.08))
                else:
                    pitch_tar = float(np.clip(
                        np.arctan2(_pfr - _prr, 0.456), 0.10, 0.30))
            else:
                pitch_tar = float(np.clip(
                    np.arctan2(h_a - h_b, 1.2), -0.35, 0.35))
            # v218o: 横脊抬前轮时顺坡仰头（防 pitch 控制器对抗抬升导致腿饱和）
            if _lift_act > 0.05:
                pitch_tar = max(pitch_tar, 0.25 * _lift_act)
            # v285: 撤 v282（静态后轮抗抬头会顶起车身）——横脊靠 kd_pitch
            # 俯仰率阻尼吸收（削峰不硬顶）
        except Exception:
            pass

        if os.environ.get('VMC_STAND', '0') == '1':
            cmd = dict(vx=0.0, omega=0.0, roll_tar=0.0, pitch_tar=0.0)
        else:
            # v220n: 前轮地形前瞻 hop——只在横脊区触发（防平地毛刺误触发，
            # wp3→4 高差毛刺曾导致 hop=300 侧翻）
            hop = np.zeros(4, dtype=np.float64)
            _in_hop_zone = any(
                float(abs(np.dot(
                    np.array([body_pos[0] + d.xmat[1][0] * 0.228,
                              body_pos[1] + d.xmat[1][3] * 0.228])
                    - _rp, _tng))) < 1.5 for (_rp, _tng, _sr, _dh) in ridge_world)
            # v495: riser 精确跳步（替代 v487 sl 触发）——前/后轴到 riser
            # 前 0.12m 时给垂直冲量，轮子快速越过 0.13m 台阶面并落在台面
            # （悬空时间最短，避免 hoist 悬空死锁）。hop 在 CarVMC 加进
            # 腿垂直力、WBC 在 (1-sl) 后加 fw[2]，两者通用。
            if fol.mode == 'STAIR':
                _shf = float(os.environ.get('S10_VMC_STAIR_HOP_F', '0'))
                if _shf > 0.0 and stair_world:
                    # v624: hop 改用**物理轴-棱距离**触发——原 s_cur 超前 6m
                    # 导致 _dfh 恒在窗外（-7.8），hop 从未生效
                    _fwdh = np.array([d.xmat[1][0], d.xmat[1][3]])
                    _fnh = float(np.hypot(_fwdh[0], _fwdh[1])) + 1e-9
                    _fxh, _fyh = _fwdh[0] / _fnh, _fwdh[1] / _fnh
                    _faxh = body_pos[:2] + np.array([_fxh * 0.228, _fyh * 0.228])
                    _raxh = body_pos[:2] - np.array([_fxh * 0.228, _fyh * 0.228])
                    _fl_top, _rl_top = 0.0, 0.0
                    for (_rp, _tng, _sr, _dhv, _top) in stair_world:
                        _dfh = float(np.dot(_faxh - _rp, _tng))
                        _drh = float(np.dot(_raxh - _rp, _tng))
                        if -0.12 <= _dfh <= 0.02:
                            hop[0:2] = _shf
                        if -0.12 <= _drh <= 0.02:
                            hop[2:4] = _shf
            if _in_hop_zone and stair_lift_flag <= 0.0:
                _fwd2 = np.array([d.xmat[1][0], d.xmat[1][3]])
                _fx, _fy = _fwd2[0], _fwd2[1]
                _fn = float(np.hypot(_fx, _fy)) + 1e-9
                _fx, _fy = _fx / _fn, _fy / _fn
                for _wi in (0, 1):
                    _w0 = wheel_xyz[_wi]
                    # v220l: hop 检测用原始地形（terr 已被 bump 抬高，差值<0.08）
                    _h0 = terrain_at(_w0[0], _w0[1])
                    _ha = terrain_at(_w0[0] + _fx * 0.80,
                                     _w0[1] + _fy * 0.80)
                    if _ha - _h0 > 0.08:
                        hop[_wi] = float(os.environ.get('S10_VMC_HOP_F', '0.0'))
            # v449: 软抬轮技能划分——台阶/陡升段（step_zone/stair_zone，
            # 永久升面需保牵引爬升）用 S10_VMC_LIFT_F_SCALE_STEP；巡航平脊
            # 段（z 不升，动量冲过）用默认 1.0 硬抬轮。地形属性=技能切换。
            _lfs = 1.0
            _lsw = float(os.environ.get('S10_VMC_LIFT_SWING', '0.66'))
            if (next_idx >= 2 and next_idx - 1 < len(fol.step_zone)
                    and fol.step_zone[next_idx - 1]):
                _lfs = float(os.environ.get(
                    'S10_VMC_LIFT_F_SCALE_STEP', '0.3'))
                # v475: stair_zone 滚上模式（S10_VMC_STAIR_NO_LIFT=1）——
                # 连续六级台阶用"腿伸长压面+动量滚上"替代悬空抬轮（hoist
                # 让前轮悬空 0.06m 下不来卡死实测）；仅 stair_zone 生效，
                # step_zone（wp5→6）保持抬轮。
                if (float(os.environ.get('S10_VMC_STAIR_NO_LIFT', '0')) > 0
                        and fol.mode == 'STAIR'):
                    step_lift[:] = 0.0
                # v473: 摆动按台阶类型分档——step_zone（wp5→6 双级，2m 间距）
                # 用 1.0（大摆动会过度旋转卡死）；stair_zone（wp6→7 六级
                # 连续 riser）用 S10_VMC_LIFT_SWING_STAIR（前轮需抬足 0.13m）。
                if fol.mode == 'STAIR':
                    _lsw = float(os.environ.get(
                        'S10_VMC_LIFT_SWING_STAIR', '1.4'))
                else:
                    _lsw = float(os.environ.get(
                        'S10_VMC_LIFT_SWING_STEP', '1.0'))
                # v568/v571: CPG 步态（S10_STAIR_CPG=1）——台阶区按**每个
                # riser 物理位置**同步的抬放窗（sin² 波形），轮轴到棱面
                # 距离 ±S10_STAIR_CPG_SWING(0.15m) 内抬放，棱面处达峰。
                # 轴距 0.456m > riser 间距 0.4m：前轮跨 Rk+1 与后轮跨 Rk
                # 仅差 0.056m，必须几何同步（自由相位会漂移错位）。相位
                # 由实际前进/轴-棱距离驱动，卡死时冻结，无原地空转。
                _in_stairzone = (fol.mode == 'STAIR')
                if (_in_stairzone
                        and float(os.environ.get('S10_STAIR_CPG', '0')) > 0
                        and float(os.environ.get('S10_STAIR_POSMODE', '0')) <= 0
                        and stair_world):
                    _swing = float(os.environ.get(
                        'S10_STAIR_CPG_SWING', '0.15'))
                    _fwd8 = np.array([d.xmat[1][0], d.xmat[1][3]])
                    _fn8 = float(np.hypot(_fwd8[0], _fwd8[1])) + 1e-9
                    _fx8, _fy8 = _fwd8[0] / _fn8, _fwd8[1] / _fn8
                    _fl_cpg, _rl_cpg = 0.0, 0.0
                    _fax8 = body_pos[:2] + np.array([_fx8 * 0.228, _fy8 * 0.228])
                    _rax8 = body_pos[:2] - np.array([_fx8 * 0.228, _fy8 * 0.228])
                    _hold8 = float(os.environ.get('S10_STAIR_CPG_HOLD', '0.05'))
                    _sw8 = _swing + _hold8
                    _fl_top, _rl_top = 0.0, 0.0
                    _fl_dmin, _rl_dmin = 1e9, 1e9
                    _wzc = np.asarray([d.xpos[WHEEL_BODY[i], 2] for i in range(4)])
                    # v732: 前轮"踩实"因子——前轮实际轮高接近当前级台面+半径
                    # 才允许抬下一级（防身体未抬升时目标不可达 IK 翻转）
                    _fl_ground = float(np.mean(fol.stair_terrain(
                        np.asarray([d.xpos[WHEEL_BODY[i], 1] for i in range(2)]))))
                    _fl_set = float(np.clip(
                        (float(np.mean(_wzc[0:2])) - (_fl_ground + self_fk_r)) / 0.05,
                        0.0, 1.0)) if False else 1.0
                    for (_rp, _tng, _sr, _dhv, _top) in stair_world:
                        # v769: 台阶高 ≤ 轮半径(0.081) 的 riser 不抬轮——
                        # 纯滚动可过，提前抬轮反而让前轮离地失牵引卡死实测
                        # （首级 0.063m 小台阶 < r，历史上是第一个卡点）。
                        if _dhv <= 0.085:
                            continue
                        _df8 = float(np.dot(_fax8 - _rp, _tng))
                        _dr8 = float(np.dot(_rax8 - _rp, _tng))
                        if -_sw8 <= _df8 <= _sw8:
                            if _df8 < -_hold8:
                                _l8 = float(np.sin(
                                    0.5 * np.pi * (-_hold8 - _df8) / _swing)
                                    ** 2)
                            elif _df8 <= _hold8:
                                _l8 = 1.0
                            else:
                                _l8 = float(np.sin(
                                    0.5 * np.pi * (_sw8 - _df8) / _swing)
                                    ** 2)
                            _l8 *= float(np.clip(0.5 + 0.5 * _dhv / 0.13, 0.5, 1.0))
                            _fl_cpg = max(_fl_cpg, _l8)
                            if abs(_df8) < _fl_dmin:
                                _fl_dmin = abs(_df8)
                                _fl_top = float(_top)
                        if -_sw8 <= _dr8 <= _sw8:
                            if _dr8 < -_hold8:
                                _l8 = float(np.sin(
                                    0.5 * np.pi * (-_hold8 - _dr8) / _swing)
                                    ** 2)
                            elif _dr8 <= _hold8:
                                _l8 = 1.0
                            else:
                                _l8 = float(np.sin(
                                    0.5 * np.pi * (_sw8 - _dr8) / _swing)
                                    ** 2)
                            _l8 *= float(np.clip(0.5 + 0.5 * _dhv / 0.13, 0.5, 1.0))
                            _rl_cpg = max(_rl_cpg, _l8)
                            if abs(_dr8) < _rl_dmin:
                                _rl_dmin = abs(_dr8)
                                _rl_top = float(_top)
                    if _fl_cpg + _rl_cpg > 0.02:
                        # v736: 前轴优先交替 + 抬轮幅度上限 0.55——轮微抬
                        # 避卡棱，靠轮驱动推力滚上台面（大抬轮悬空失支撑
                        # 侧翻实测）。台面高由 terr 覆盖+身体随轮自然升起。
                        _rl_cpg *= (1.0 - _fl_cpg) ** 2
                        place_z[:] = [_fl_top, _fl_top, _rl_top, _rl_top]
                        # v755e: 抬轮上限随台阶高（0.125m 台阶需轮心≥台面+r
                        # =0.747；0.55 微抬只够首级 0.063m，二级以上卡棱实测）
                        _lcap9 = float(os.environ.get(
                            "S10_STAIR_LIFT_CAP", "0.75"))
                        _fl_cpg = min(_fl_cpg, _lcap9)
                        _rl_cpg = min(_rl_cpg, _lcap9)
                        step_lift[:] = [_fl_cpg, _fl_cpg, _rl_cpg, _rl_cpg]
                        stair_lift_flag = 1.0
                        # v755: 踩实释放（接触反馈调制相位，Kimura/ETH 腿效用
                        # 同源）——前轮物理上台面（wz 达 place_z+r）且前轴已过
                        # 棱边后，CPG 投影滞后仍给抬轮→屏蔽前轮承重→单点支撑
                        # 死锁实测；轮高到位即连续释放抬升，无离散门控。
                        # v757: 踩实释放参考=刚越过的棱（df>0 最近）的台面高——
                        # 旧版用"最近棱"=下一级（riser2），前轮落 stair1 后不
                        # 释放悬空卡死实测；落地后阻抗压实恢复 fn/推力。
                        # 下一棱 <0.10m 前停止释放，让下一级抬升正常起量。
                        _fdf8 = 1e9; _fdf8_pos = 1e9
                        _ftop_past = 0.0; _fd_next = 1e9
                        for (_rp2, _tng2, _sr2, _dhv2, _top2) in stair_world:
                            _d2 = float(np.dot(_fax8 - _rp2, _tng2))
                            if abs(_d2) < abs(_fdf8):
                                _fdf8 = _d2
                            if 0.0 < _d2 < _fdf8_pos:
                                _fdf8_pos = _d2
                                _ftop_past = float(_top2)
                            if _d2 < 0.0 and -_d2 < _fd_next:
                                _fd_next = -_d2
                        if (_ftop_past > 0.0 and _fdf8_pos < 0.80
                                and _fd_next > 0.20):
                            _pzr = _ftop_past + 0.081 - 0.03
                            _fl_rel = float(np.clip(
                                (float(np.mean(_wzc[0:2])) - _pzr) / 0.04,
                                0.0, 1.0))
                            if _fl_rel > 0.0:
                                step_lift[0:2] *= (1.0 - _fl_rel)
                        if os.environ.get('S10_STAIR_DEBUG', '0') == '1':
                            print('[CPG2] t=%.1f y=%.2f fl=%.2f rl=%.2f '
                                  'pzF=%.3f pzR=%.3f' % (
                                      t, body_pos[1], _fl_cpg, _rl_cpg,
                                      _fl_top, _rl_top), flush=True)
                    if os.environ.get('S10_STAIR_DEBUG', '0') == '1':
                        print('[CPG] t=%.1f pos=(%.2f,%.2f) yaw=%.2f fl=%.2f rl=%.2f swing=%.2f n_riser=%d'
                              % (t, body_pos[0], body_pos[1], yaw,
                                 _fl_cpg, _rl_cpg, _swing, len(stair_world)),
                              flush=True)
                elif (_in_stairzone
                        and float(os.environ.get('S10_STAIR_ANTI', '0')) > 0):
                    _fln = float(np.mean(step_lift[0:2]))
                    _rln = float(np.mean(step_lift[2:4]))
                    _fa = float(np.clip((_fln - 0.15) / 0.35, 0.0, 1.0))
                    _ra = float(np.clip((_rln - 0.15) / 0.35, 0.0, 1.0))
                    _afloor = float(os.environ.get(
                        'S10_STAIR_ANTI_FLOOR', '0.4'))
                    if _fa >= _ra:
                        step_lift[2:4] = np.maximum(
                            step_lift[2:4] * (1.0 - _fa), _afloor)
                    else:
                        step_lift[0:2] = np.maximum(
                            step_lift[0:2] * (1.0 - _ra), _afloor)
            if os.environ.get('S10_STAIR_DEBUG', '0') == '1' and _in_stairzone_now:
                _rr0 = terrain_at(wheel_xyz[0, 0], wheel_xyz[0, 1])
                print('[TERR] t=%.1f y=%.2f terr=%s ray0=%.3f'
                      % (t, body_pos[1], np.round(terr, 3), _rr0),
                      flush=True)
            # v756: NO_LIFT 最终清零——CPG 分支在其后覆盖 step_lift，
            # 纯滚动（USC/Go2-W 路线）需在楼梯抬放逻辑末尾强制归零。
            if (float(os.environ.get('S10_VMC_STAIR_NO_LIFT', '0')) > 0
                    and _in_stairzone_now):
                step_lift[:] = 0.0
                stair_lift_flag = 1.0
            # v813: 计算每轮爬升窗掩码（轮世界 y 与 riser y 距离）
            _climb_mask = np.zeros(4)
            if _in_stairzone_now and stair_world:
                for _cl in range(4):
                    _wy = float(wheel_xyz[_cl, 1])
                    for (_rp, _tng, _sr, _dhv, _top) in stair_world:
                        _dcl = float(np.dot(
                            wheel_xyz[_cl, :2] - _rp, _tng))
                        if -0.15 <= _dcl <= 0.05:
                            _climb_mask[_cl] = 1.0
                            break
            # v817: 楼梯区 WBC z_des 偏移蹲低（S10_VMC_Z_DES_OFFSET 动态覆盖）
            # ——body 蹲低让后轮压实得牵引（前轮抬升时后轮悬空 fn 弱无推力
            # 死锁实测；WBC 每拍读该 env，CarVMC 不读故此前 DROP_SQUAT 无效）
            _sqt = float(os.environ.get('S10_STAIR_SQUAT', '0.0'))
            if _in_stairzone_now and _sqt > 0.0:
                os.environ['S10_VMC_Z_DES_OFFSET'] = str(0.205 - _sqt)
            elif _sqt > 0.0:
                os.environ['S10_VMC_Z_DES_OFFSET'] = '0.205'
            # v732: 楼梯区 body z 目标 = 轮下台面均值 + 站立高（随楼梯逐级升）
            _zd = 0.0
            if _in_stairzone_now:
                try:
                    _zd = float(np.mean(fol.stair_terrain(
                        wheel_xyz[:, 1]))) + 0.205
                except Exception:
                    _zd = 0.0
            # v746: 到最近横脊的路径距离（连续量）——横脊窗口内轮矩恢复
            # μN 钳制防弹跳（wp4→5 发卡+横脊 err 门控切换侧翻实测）
            _ridge_d = 99.0
            try:
                _rs_arr = getattr(fol, 'ridge_s', [])
                # v755b: 楼梯区跳过横脊钳制——台阶 riser 也落在 ridge 表里，
                # μN·r 把轮矩压到 1.7Nm 无推力卡死实测；stair 技能内轮矩
                # 保持上限 13.5Nm（用户原则:轮力矩越大越好）。
                if _rs_arr and not _in_stairzone_now:
                    _rd = [abs(s_cur - _r) for _r in _rs_arr]
                    _ridge_d = float(min(_rd))
            except Exception:
                pass
            cmd = dict(vx=vx_c, omega=om_c, roll_tar=roll_tar,
                      pitch_tar=pitch_tar,
                      yaw_scale=(1.0 - float(np.clip(
                          float(np.max(step_lift)) * 0.5, 0.0, 0.55))
                          if _in_stairzone_now else 1.0 - _lift_act),
                      ridge_dist=_ridge_d,
                      hop=hop,
                      step_lift=step_lift,
                      body_lift=_body_lift,
                      stair_lift=stair_lift_flag,
                      lift_f_scale=_lfs,
                      z_min=1.0 if _in_stairzone_now else 0.0,
                      fp_place=1.0 if _in_stairzone_now else 0.0,
                      wheel_press=(float(os.environ.get(
                          'S10_WHEEL_PRESS', '0.25'))
                                   if _in_stairzone_now else 0.0),
                      kd_scale=(float(os.environ.get(
                          'S10_STAIR_KD_SCALE', '2.5'))
                                if _in_stairzone_now else 1.0),
                      att_scale=(float(os.environ.get(
                          'S10_STAIR_ATT_SCALE', '2.5'))
                                 if _in_stairzone_now else 1.0),
                      pure_fwd=(1.0 if (float(os.environ.get(
                          'S10_STAIR_PURE_FWD', '0')) > 0
                                        and _in_stairzone_now
                                        and float(np.max(step_lift)) > 0.3)
                                else 0.0),
                      lift_swing=_lsw,
                      stair_pose=_stair_pose,
                      z_des=_zd,
                      # v813: 爬升窗掩码——每轮 y 在任一 riser 爬升窗
                      # [y_r-0.15, y_r+0.05] 内=1（双侧沿面拉高）；台面=0
                      # （单侧贴地抓地）
                      climb_mask=_climb_mask,
                      place_z=place_z,
                      place_margin=float(os.environ.get(
                          'S10_STAIR_LIFT_MARGIN', '0.04')))
        if (os.environ.get('S10_VMC_MODE', 'wbc') == 'dual'):
            vmc = vmc_wbc if fol.mode == 'STAIR' else vmc_car
        if os.environ.get('S10_VMC_MODE', 'wbc') == 'dual2':
            vmc = vmc_fp if _stair_exec else vmc_car
        if os.environ.get('S10_VMC_MODE', 'wbc') == 'stairwbc':
            vmc = vmc_stw if _stair_exec else vmc_car
        tau = vmc.compute_tau(qpos, qvel, wheel_xyz, wheel_vel, cmd, terr, DT)
        _tleg = float(np.abs(tau[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]).max())
        _twh = float(np.abs(tau[[3, 7, 11, 15]]).max())
        _max_tau_leg = max(_max_tau_leg, _tleg)
        _max_tau_wh = max(_max_tau_wh, _twh)
        if _tleg >= 50.0 or _twh >= 14.0:
            _over_run += DT
            _over_total += DT
            _over_worst = max(_over_worst, _over_run)
        else:
            _over_run = 0.0
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        t += DT
        _ctrl_cnt += 1
        if t - _last_freq_t >= 2.0:
            _last_freq_t = t
            _wnow = time.perf_counter()
            _wel = _wnow - _freq_t0
            _tsim = t - _freq_sim0
            print('[FREQ] t=%.1fs | ref_path: %d次 %.1fHz avg=%.2fms max=%.2fms | '
                  'MPPI: %d次 %.1fHz avg=%.2fms max=%.2fms | 控制环 %.0fHz %.2fms/step'
                  % (t, _rp_cnt, _rp_cnt / max(_tsim, 1e-9),
                     1e3 * _rp_tot / max(_rp_cnt, 1), 1e3 * _rp_max,
                     _mp_cnt, _mp_cnt / max(_tsim, 1e-9),
                     1e3 * _mp_tot / max(_mp_cnt, 1), 1e3 * _mp_max,
                     _ctrl_cnt / max(_wel, 1e-9), 1e3 * _wel / max(_ctrl_cnt, 1)),
                flush=True)
        # v782: 密集轨迹记录（S10_TRAJ_DENSE=1 时每个控制周期 5ms 记录，
        # 供速度着色轨迹图；默认关保持原行为）
        if os.environ.get('S10_TRAJ_DENSE', '0') == '1':
            traj.append([t, body_pos[0], body_pos[1], float(d.cvel[1][3])])
        if _viewer is not None:
            if not _viewer.is_running():
                print('[VMC] viewer 已关闭，结束', flush=True)
                break
            _viewer.sync()

        # 航点推进（0.5m + v204 捷径）
        if next_idx < len(wp):
            rp = d.xpos[1][:2]
            dist = float(np.linalg.norm(rp - wp[next_idx][:2]))
            # v294: 判据=质心投影 xy 进入航点 0.3m（机器狗任意一点经过的等效简化）
            _adv = float(os.environ.get('S10_WP_ADVANCE_DIST', '0.3'))
            reached = dist <= _adv
            if reached:
                if next_idx == 0 and t_start is None:
                    t_start = t
                wp_times[next_idx] = t
                _last_adv_t = t
                print(f'[VMC-T] wp{next_idx} @ t={t:.2f}s', flush=True)
                next_idx += 1
                # v253: 过航点后强制弧长游标越过该航点——切弯时 s_cur 投影
                # 滞后会令前视目标回指后方（wp4→5 指令反向侧翻实测）
                try:
                    _s_min = float(fol.path_wp_s[next_idx - 1]) - 0.05
                    if fol._s_cur < _s_min:
                        fol._s_cur = _s_min
                except Exception:
                    pass
                if STOP_AT > 0 and next_idx > STOP_AT:
                    print(f'[VMC-T] 到达 wp{STOP_AT}，结束', flush=True)
                    break
                if next_idx >= MAX_WP:
                    print('[VMC-T] 到达最大航点，结束', flush=True)
                    break

        if int(t * 200) % 100 == 0:
            roll = float(np.arctan2(
                2.0*(quat[0]*quat[1]+quat[2]*quat[3]),
                1.0-2.0*(quat[1]**2+quat[2]**2)))
            print(f'[VMC-T] t={t:.0f}s wp={next_idx} pos=({body_pos[0]:.1f},'
                  f'{body_pos[1]:.1f},{body_pos[2]:.2f}) yaw={yaw:.2f} vx_w={float(d.cvel[1][3]):.2f} '
                  f'roll={roll:.2f} cmd=({vx_c:.2f},{om_c:.2f}) '
                  f'vref={v_ref:.2f} '
                  f'wz={np.round([d.xpos[WHEEL_BODY[i],2] for i in range(4)],2)} '
                  f'tauH={np.round(tau[[0,4,8,12]],0)} tauY={np.round(tau[[1,5,9,13]],0)} tauK={np.round(tau[[2,6,10,14]],0)} '
                  f'om={float(qvel[5]):.2f} tauW={np.round(tau[[3,7,11,15]],1)} '
                  f'stp={_step_state} sl={np.round(step_lift,1)}', flush=True)
            if abs(roll) > 0.9 or body_pos[2] < 0.12:
                print('[VMC-T] *** 侧翻/摔倒 ***', flush=True)
                break
            if _in_stairzone_now and os.environ.get('S10_STAIR_DEBUG', '0') == '1':
                _stq = np.asarray(d.xquat[1])
                _stpitch = float(np.arctan2(
                    2.0 * (_stq[0] * _stq[2] - _stq[3] * _stq[1]),
                    1.0 - 2.0 * (_stq[1] ** 2 + _stq[2] ** 2)))
                _stroll = float(np.arctan2(
                    2.0 * (_stq[0] * _stq[1] + _stq[2] * _stq[3]),
                    1.0 - 2.0 * (_stq[1] ** 2 + _stq[2] ** 2)))
                _fn9 = [float(d.cfrc_ext[_gb][2]) for _gb in WHEEL_BODY]
                if t - _last_dbg_t >= 0.5:
                    print('[STAIRDBG] t=%.1f pos=(%.2f,%.2f) pitch=%.2f roll=%.2f '
                          'bz=%.3f wz=%s sl=%s terrF=%.3f terrR=%.3f fn=%s '
                          'cmd=(%.2f,%.2f)' % (t, body_pos[0], body_pos[1],
                             _stpitch, _stroll, body_pos[2],
                             np.round([d.xpos[WHEEL_BODY[i], 2]
                                       for i in range(4)], 2),
                             np.round(step_lift, 1), terr[0], terr[2],
                             np.round(_fn9, 0), vx_c, om_c), flush=True)
                    _last_dbg_t = t
            traj.append([t, body_pos[0], body_pos[1], float(d.cvel[1][3])])
            if _stuck_timeout > 0.0 and t - _last_adv_t > _stuck_timeout:
                print('[VMC-T] *** 卡死 %.0fs 无航点推进 (wp=%d) ***'
                      % (t - _last_adv_t, next_idx), flush=True)
                _vb = np.asarray(d.cvel[1][0:6], dtype=np.float64)
                _wq4 = [float(qvel[WHEEL_Q_IDX[i]]) for i in range(4)]
                _wz4 = [float(d.xpos[WHEEL_BODY[i], 2]) for i in range(4)]
                # v722: 接触力诊断用 cfrc_ext（mujoco 标准外力 API）
                _fn = [float(d.cfrc_ext[_gb][2]) for _gb in WHEEL_BODY]
                print('[STUCKDBG] body_v=%.3f,%.3f,%.3f om=%.3f wq=%s wz=%s '
                      'fn=%s'
                      % (_vb[0], _vb[1], _vb[2], _vb[5],
                         np.round(_wq4, 1), np.round(_wz4, 2),
                         np.round(_fn, 0)), flush=True)
                break
    print('=== VMC 全航点结果 ===')
    print(f'完成: {next_idx >= MAX_WP}，最终 wp={next_idx}/{MAX_WP}')
    if t_start is not None:
        print(f'wp0→wp{min(next_idx-1, MAX_WP-1)} 用时 {wp_times.get(max(next_idx-1,0),0)-t_start:.1f}s')
    print('[VMC] 力矩合规: 腿max|tau|=%.1fNm(限50) 轮max|tau|=%.1fNm(限14) '
          '连续超限最长 %.2fs / 累计 %.2fs%s' % (
              _max_tau_leg, _max_tau_wh, _over_worst, _over_total,
              '  [超0.5s不合格!]' if _over_worst > 0.5 else ''), flush=True)
    if os.environ.get('VMC_TRAJ'):
        np.save(os.environ['VMC_TRAJ'], np.array(traj))
    if _viewer is not None:
        try:
            _viewer.close()
        except Exception:
            pass
        # v293: mujoco.viewer 在 WSLg 下正常退出会段错误，改用 os._exit
        sys.stdout.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
