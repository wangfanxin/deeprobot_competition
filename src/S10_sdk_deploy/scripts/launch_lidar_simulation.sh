#!/bin/bash

# S10 LiDAR Simulation Launch Script
# 启动MuJoCo仿真和RViz2可视化

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}S10 LiDAR Simulation Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$PACKAGE_DIR")")")"

# 配置文件路径
RVIZ_CONFIG="$PACKAGE_DIR/config/s10_lidar.rviz"

# Source ROS2环境
echo -e "${YELLOW}[INFO] Sourcing ROS2 environment...${NC}"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f "/opt/ros/iron/setup.bash" ]; then
    source /opt/ros/iron/setup.bash
else
    echo -e "${RED}[ERROR] ROS2 environment not found!${NC}"
    exit 1
fi

# Source工作空间
if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    echo -e "${YELLOW}[INFO] Sourcing workspace: $WORKSPACE_DIR${NC}"
    source "$WORKSPACE_DIR/install/setup.bash"
fi

# 检查mujoco-lidar是否安装
echo -e "${YELLOW}[INFO] Checking mujoco-lidar installation...${NC}"
python3 -c "from mujoco_lidar import MjLidarWrapper, scan_gen" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] mujoco-lidar not installed!${NC}"
    echo -e "${YELLOW}[INFO] Installing mujoco-lidar[taichi]...${NC}"
    pip install "mujoco-lidar[taichi]"
    
    # 再次检查
    python3 -c "from mujoco_lidar import MjLidarWrapper, scan_gen" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to install mujoco-lidar!${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}[OK] mujoco-lidar is installed${NC}"

# 启动RViz2
echo -e "${GREEN}[INFO] Launching RViz2...${NC}"
if [ -f "$RVIZ_CONFIG" ]; then
    rviz2 -d "$RVIZ_CONFIG" &
    RVIZ_PID=$!
    echo -e "${GREEN}[OK] RViz2 launched with config: $RVIZ_CONFIG${NC}"
else
    echo -e "${YELLOW}[WARN] RViz config not found, launching default RViz2${NC}"
    rviz2 &
    RVIZ_PID=$!
fi

# 等待RViz启动
sleep 2

# 启动MuJoCo仿真
echo -e "${GREEN}[INFO] Launching MuJoCo simulation with LiDAR...${NC}"
SIMULATION_SCRIPT="$PACKAGE_DIR/interface/robot/simulation/mujoco_simulation_ros2.py"

if [ -f "$SIMULATION_SCRIPT" ]; then
    python3 "$SIMULATION_SCRIPT"
else
    echo -e "${RED}[ERROR] Simulation script not found: $SIMULATION_SCRIPT${NC}"
    kill $RVIZ_PID 2>/dev/null
    exit 1
fi

# 清理
echo -e "${YELLOW}[INFO] Shutting down...${NC}"
kill $RVIZ_PID 2>/dev/null
echo -e "${GREEN}[OK] Simulation stopped${NC}"
