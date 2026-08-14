# DiAL 分层落脚点/步态方案（v2，20+200Hz）

## 0. 结论先放前面

采用 **“上层连续相位/落脚点规划 + 下层 DiAL 扭矩采样”**，不把步态/落脚点塞进 DiAL 的暴力采样维度。

- DiAL 是底层执行器，直接顶掉 `NMPC+WBC`，不是接在两者之间。
- 步态和落脚点由 20Hz 上层输出**连续软权重与目标场**，DiAL 仍只搜索 16 维动作（12 腿 + 4 轮）。
- 废弃 `s10_dial_mpc.py` 里的 DDP/SRBD 骨架，方向已经错了，不再修。

---

## 1. 总体架构

```text
lidar 高程图                    10 Hz
   |
riser/高程特征                  10 Hz
   |
AutoNavFollower                 20 Hz  -> vx, yaw, 路径弧长
   |
StairContactPlanner（新增）      20 Hz  -> 连续 swing 权重 + 落脚点目标
   |
DiAL-MPC（MBDPI）               20 Hz  -> 全阶扭矩级采样，固定软相位
   |
act2tau                         200 Hz -> 16 维关节/轮力矩
   |
MuJoCo                          200 Hz
```

CRUISE 段保持不变；只有 STAIR 段切到 `StairContactPlanner + DiAL`。

---

## 2. 数据管线

| 层 | 频率 | 输入 | 输出 | 代码位置 |
|---|---|---|---|---|
| 感知 | 10Hz | lidar 点云/高程 | 高程栅格、riser 特征 | `lidar_terrain_v2.py` |
| 导航 | 20Hz | 当前位姿、waypoint | `vx`, `vyaw`, `s_cur` | `stair_auto_nav.py` |
| 接触规划 | 20Hz | 高程栅格、riser 表、轮世界坐标 | `w_swing[4]`、`foothold_xy[4]`、`foothold_z[4]` | 新增 `stair_contact_planner.py` |
| DiAL | 20Hz | 状态、目标、软相位、地形场 | 16D action | `mpc_controller.py` |
| 力矩 | 200Hz | action、qpos、qvel | 16D torque | `mpc_controller.compute_tau()` |

---

## 3. 步态和落脚点怎么表达，而不是暴力搜索

### 3.1 步态：连续 swing 权重

每条腿不输出 `0/1`，输出连续权重：

```text
d_i = 轮 i 到目标 riser 的切向投影距离
w_swing_i = sigmoid((d_i - d_trigger_i) / sigma)
```

- `w_swing_i ∈ [0,1]`，由几何和速度自然连续过渡。
- 前轴、后轴各有目标 riser；前后轴天然错开半阶。
- **不新增布尔门控**，只新增一个平滑参数 `sigma`（默认 `0.05m`），或者直接复用现有 `S10_SWING_PROX`/`S10_SWING_THRESH` 的逻辑。

### 3.2 落脚点：连续目标场

每条腿输出一个落脚点目标，不是离散枚举：

```text
stance 轮：
    p_i_xy = 当前轮 xy
    p_i_z  = terrain(wheel_xy) + R

swing 轮：
    p_i_xy = 下一级踏面中心（由 riser 表给）
    p_i_z  = 下一级台面顶 + R
```

其中 `R = 0.081m`。这样 DiAL 的代价只惩罚“轮实际位置偏离目标场”，不会让采样器去枚举“落哪一级”。

### 3.3 DiAL 内部只搜扭矩

DiAL 动作空间仍保持：

```text
u = [leg_hipy, leg_thigh, leg_calf] * 4 + [wheel] * 4
```

总维度 16，不新增接触/落脚点变量。

代价里加入三项即可：

```text
J += w_swing_i * ||wheel_z_i - foothold_z_i||^2       # 摆动相抬到位
J += (1 - w_swing_i) * ||wheel_z_i - ground_z_i||^2   # 支撑相贴地
J += w_foothold * w_swing_i * ||wheel_xy_i - foothold_xy_i||^2
```

现有 `mpc_controller.py` 已经具备 `gait_swing`、`foothold_y`、`foothold_valid` 注入入口，方案不需要重写 DiAL 内核。

---

## 4. 控制频率

保持现在确认的配置：

| 模块 | 频率 |
|---|---|
| lidar 建图 | 10Hz |
| AutoNav | 20Hz |
| StairContactPlanner | 20Hz |
| DiAL 规划 | 20Hz（`plan_interval=10`，`DT=0.005`） |
| 力矩输出 | 200Hz |

不要追 50Hz DiAL。当前 20Hz 规划 + 200Hz 力矩是稳定基线。

---

## 5. 门控与参数清理

### 保留的门控

只有一个：

```text
CRUISE <-> STAIR
```

STAIR 内部不再有：

- HOVER 子态
- 前轴等后轴相位门
- 相位门超时强制 SWING
- 距离窗布尔 SWING

全部由 `w_swing_i` 连续权重替代。

### 删除/归档的代码

- `s10_dial_mpc.py`：DDP/SRBD 骨架，方向错误，归档或 `git rm`。
- 旧 NMPC+WBC 的距离窗参数：`SWING_D`、`HOVER_*`、`Z_OFF`、`S10_NMPC_SWING_D` 等，不再进入新 STAIR 分支。
- 旧 WBC 的位置基抬轮、力控软切换逻辑不在新链路里运行。

### 最小参数集

DiAL 核心参数保持现有：

```text
Nsample, Hsample, Hnode, Ndiffuse, temp_sample,
update_method, sigma_scale, dt
```

接触规划相关只保留：

```text
swing_prox        # 摆动相邻近距离
swing_thresh      # 抬升判定阈值
w_foothold        # 落脚点前拉权重
ground_phase      # 分相 ground 开关
```

不新增一堆可调权重。

---

## 6. 文件改动方案

不改比赛原文件，不碰 `cruise_noros.py` 源文件；复制出来做 STAIR 版本。

### 新增

1. `src/S10_sdk_deploy/s10_mpc/stair_contact_planner.py`

   职责：
   - 从 lidar 高程/riser 表计算 `w_swing[4]`。
   - 计算 `foothold_xy[4]`、`foothold_z[4]`。
   - 纯 numpy/JAX 可测，不依赖 MuJoCo。

2. `src/S10_sdk_deploy/scripts/stair_dial_noros.py`

   由 `cruise_noros.py` 复制而来：
   - CRUISE 行为与源文件一致。
   - `fol.mode == STAIR` 时创建并调用 `StairContactPlanner`。
   - 把 `w_swing` 和 `foothold` 传给 `MPCController`。

### 修改

3. `src/S10_sdk_deploy/s10_mpc/mpc_controller.py`

   只增加/补全接口：

   ```python
   set_gait_swing(w: np.ndarray)          # 4 维 [0,1]
   set_foothold(xy: np.ndarray, z: np.ndarray, valid: np.ndarray)
   ```

   内部已有 `_gait_swing`、`foothold_y` 注入，改成从 planner 直接写入。

4. `dial-mpc/dial_mpc/envs/s10_env.py`

   代价里接入 `gait_swing` 和 `foothold` 字段，确保软相位正确进入 rollout。当前已有基础，按本方案收紧。

### 明确不修改

- `s10_mpc/s10_nmpc_wbc.py`
- `s10_mpc/stair_wbc.py`
- `s10_mpc/stair_vmc_legs.py`
- `scripts/cruise_noros.py`
- 比赛原始 XML 和赛道文件

---

## 7. 实施顺序

按 1→7 完成，逐步可回退。

1. **归档 DDP 骨架**
   `git mv` 或注释标记 `s10_dial_mpc.py` 废弃，不删历史。

2. **确认 MBDPI 基线**
   用 `tmp/run_v658_test.sh` 记录当前 STAIR 卡点，保存日志和参数快照。

3. **实现 `StairContactPlanner`**
   先离线单测：给定 riser 表和轮坐标，输出连续 `w_swing`，验证前后轴自然交替、无硬切跳变。

4. **接入 `stair_dial_noros.py`**
   只在 STAIR 分支启用 planner；CRUISE 仍走原逻辑。

5. **打通 `gait_swing + foothold` 到 DiAL cost**
   检查 rollout 中是否真正收到 soft phase，而不是回退到旧启发式。

6. **清理旧参数和门控**
   删除/禁用旧距离窗、HOVER、相位门、NMPC+WBC 相关 STAIR 配置。

7. **真原图验证**
   先跑 `wp6→7` 连续 4 次成功，再跑全段回归。

---

## 8. 验证判据

STAIR 段通过的硬指标：

| 指标 | 目标 |
|---|---|
| 前轮越过 riser2 | `wheel_z >= 台面顶 + R - 0.02` |
| 接触后轮地力 | `fn > 10N` |
| 航向 | `|yaw_err| < 5°` |
| 俯仰 | `|pitch| < 0.5 rad` |
| 侧倾 | `|roll| < 0.5 rad` |
| 速度 | `vx >= 0.8 m/s`，无长期停滞 |
| 力矩 | 腿 `±48Nm`，轮 `±13.5Nm` |

最终目标：**wp6→7 连续成功 4 次**。

---

## 9. 风险与回退

**若 DiAL 连续软相位仍搜不到抬轮**：
优先调整 `swing_prox`、`w_foothold`、`ground_phase`，不新增布尔门控。

**若软相位确实不够**：
再升级为“低维上层离散选择器 + 固定 mode DiAL”。上层只搜 `4 腿接触相位 + 4 落脚点`，下层 DiAL 固定 mode 做扭矩优化；这才是把离散量独立出去的正确位置，仍然不是让 DiAL 暴力枚举。

**不重开的方向**：
旧 NMPC+WBC 距离窗、力控软切换、车化动量冲阶。

---

一句话版本：**把步态做成连续 swing 权重，把落脚点做成连续目标场，DiAL 仍只采样 16 维扭矩；20Hz 规划接触、200Hz 出力，废弃 DDP 骨架，STAIR 分支用复制出来的新脚本接入。**
