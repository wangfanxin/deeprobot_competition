# Implementation Plan: Three-Layer Architecture Deployment

## Overview

This implementation plan breaks down the three-layer perception-based control architecture for the S10 quadruped robot into discrete coding tasks. The architecture enables sensor-based navigation through a waypoint track using MuJoCo simulation, with three operational layers: Perception (sensor processing), Navigation (optional path planning), and Locomotion (motion control).

The implementation follows this sequence:
1. MuJoCo sensor integration (lidar and depth camera simulation)
2. Perception layer implementation (sensor management and processing)
3. Navigation layer implementation (waypoint tracking, DWA planning, obstacle avoidance)
4. Locomotion layer implementation (RL policy inference and PD control)
5. System integration (state machine, ROS 2 nodes, configuration management)
6. Testing and validation

## Tasks

- [ ] 1. Set up project structure and MuJoCo sensor integration
  - [ ] 1.1 Create MuJoCo sensor simulation script for lidar and depth camera
    - Create `src/S10_sdk_deploy/interface/robot/simulation/mujoco_sensors.py`
    - Implement lidar raycast sensor publishing sensor_msgs/PointCloud2 at 10 Hz
    - Implement depth camera sensor publishing sensor_msgs/Image at 20 Hz
    - Add configurable sensor mounting positions and orientations
    - Apply Gaussian noise models to sensor outputs
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ] 1.2 Modify MuJoCo simulation ROS 2 integration
    - Update `src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py`
    - Import and initialize sensor simulation module
    - Publish sensor data alongside existing IMU and joint data
    - Ensure timestamp synchronization with simulation time
    - _Requirements: 1.1, 1.2, 1.5, 7.6_

  - [ ] 1.3 Create ROS 2 message definitions for perception and navigation
    - Create `src/perception_msgs/msg/PerceptionData.msg` with obstacle cloud, waypoint direction, occupancy grid
    - Create `src/navigation_msgs/msg/ControlCommand.msg` with linear/angular velocity and control mode
    - Create `src/navigation_msgs/msg/NavigationStatus.msg` with waypoint tracking and planning metrics
    - Update CMakeLists.txt to build custom messages
    - _Requirements: 2.5, 3.3, 5.2, 5.3_

- [ ] 2. Checkpoint - Verify sensor simulation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Implement Perception Layer components
  - [ ] 3.1 Implement Sensor Manager component
    - Create `src/S10_sdk_deploy/perception/sensor_manager.hpp` and `.cpp`
    - Implement ROS 2 subscribers for /sensor/lidar/scan, /sensor/camera/depth, /IMU_DATA, /JOINTS_DATA
    - Implement time-synchronized data buffering with deque structures
    - Implement getSynchronizedData() method with 50ms sync threshold
    - Provide ground truth position access from MuJoCo ground truth topic
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 5.1_

  - [ ] 3.2 Implement Perception Processor component
    - Create `src/S10_sdk_deploy/perception/perception_processor.hpp` and `.cpp`
    - Implement point cloud transformation to world frame using IMU orientation
    - Implement obstacle extraction within 10m radius using voxel grid downsampling and RANSAC ground removal
    - Implement waypoint marker detection using height filtering and template matching
    - Implement occupancy grid generation (200x200 cells, 0.1m resolution)
    - Publish PerceptionData at sensor input rate
    - Track and publish processing latency metrics
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 8.1_

  - [ ]* 3.3 Write unit tests for Perception Layer
    - Test sensor data buffering and synchronization with varying timestamps
    - Test point cloud transformation accuracy
    - Test obstacle detection with known point cloud fixtures
    - Test waypoint detection with synthetic marker geometries
    - Test occupancy grid generation
    - _Requirements: 9.6_

- [ ] 4. Implement Navigation Layer components
  - [ ] 4.1 Implement Waypoint Tracker component
    - Create `src/S10_sdk_deploy/navigation/waypoint_tracker.hpp` and `.cpp`
    - Load waypoint sequence from configuration or topic
    - Implement updatePosition() and checkWaypointReached() with 0.2m radius
    - Compute direction and distance to next waypoint
    - Track timing statistics and log waypoint arrival events
    - _Requirements: 3.2, 8.5_

  - [ ] 4.2 Implement Local Planner component with DWA algorithm
    - Create `src/S10_sdk_deploy/navigation/local_planner.hpp` and `.cpp`
    - Implement velocity sampling within dynamic window (v_max=1.0 m/s, w_max=1.5 rad/s)
    - Implement trajectory prediction over 2-second horizon with 0.1s time step
    - Implement cost evaluation (heading, distance, velocity, obstacle costs)
    - Select optimal velocity command and publish ControlCommand at 50 Hz
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 8.2_

  - [ ] 4.3 Implement Obstacle Avoidance component
    - Create `src/S10_sdk_deploy/navigation/obstacle_avoidance.hpp` and `.cpp`
    - Implement artificial potential field repulsive force computation
    - Apply avoidance forces to modify planned velocity commands
    - Implement collision imminence detection with 0.3m emergency stop threshold
    - Maintain safe clearance distance of 1.0m
    - _Requirements: 3.4, 10.2_

  - [ ]* 4.4 Write unit tests for Navigation Layer
    - Test waypoint detection and sequencing logic
    - Test DWA velocity sampling and trajectory prediction
    - Test cost function computation with known scenarios
    - Test obstacle avoidance force computation
    - Test emergency stop triggering conditions
    - _Requirements: 9.6_

- [ ] 5. Checkpoint - Verify navigation planning
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement Locomotion Layer components
  - [ ] 6.1 Implement Locomotion Policy component
    - Create `src/S10_sdk_deploy/locomotion/locomotion_policy.hpp` and `.cpp`
    - Load trained RL policy model from ONNX or TorchScript file
    - Implement buildObservation() constructing 83-dim vector (robot state + control command + perception features)
    - Implement infer() method running policy network inference
    - Implement directPerceptionFallback() for navigation-disabled mode
    - Implement safeStopAction() for failure handling
    - _Requirements: 4.1, 4.4, 10.2_

  - [ ] 6.2 Implement Joint Controller component with PD control
    - Create `src/S10_sdk_deploy/locomotion/joint_controller.hpp` and `.cpp`
    - Implement computePDControl() with per-joint Kp/Kd gains
    - Implement checkJointLimits() and checkVelocityLimits() for safety enforcement
    - Implement applyTorqueLimits() clamping torques to max values
    - Implement detectUnsafeState() checking position errors and base tilt
    - Convert RobotAction to drdds::msg::JointsDataCmd format
    - Publish to /JOINTS_CMD at 200 Hz
    - _Requirements: 4.2, 4.3, 4.5, 4.6, 5.4, 10.3, 10.4, 10.6_

  - [ ] 6.3 Create configuration structures and YAML parser
    - Create `src/S10_sdk_deploy/config/config_types.hpp` with SensorConfig, PerceptionConfig, NavigationConfig, LocomotionConfig structs
    - Create `src/S10_sdk_deploy/config/config_loader.hpp` and `.cpp` to parse YAML configuration files
    - Implement validation for configuration parameters (bounds checking, required fields)
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 6.4 Write unit tests for Locomotion Layer
    - Test policy observation vector construction
    - Test PD control computation with known inputs
    - Test joint limit enforcement
    - Test unsafe state detection logic
    - Test configuration loading and validation
    - _Requirements: 9.6_

- [ ] 7. Integrate three layers with State Machine
  - [ ] 7.1 Modify State Machine for three-layer initialization
    - Update `src/S10_sdk_deploy/src/state_machine.cpp` to add perception/navigation/locomotion initialization in RLControl state
    - Add layer health monitoring and timeout detection
    - Implement safety transitions to JointDamping state on failures
    - _Requirements: 5.5, 5.6, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ] 7.2 Create ROS 2 node for Perception Layer
    - Create `src/S10_sdk_deploy/nodes/perception_node.cpp`
    - Initialize SensorManager and PerceptionProcessor
    - Set up ROS 2 subscribers and publishers
    - Implement main control loop at sensor input rate
    - Add diagnostics publishing
    - _Requirements: 5.1, 5.2, 8.1_

  - [ ] 7.3 Create ROS 2 node for Navigation Layer
    - Create `src/S10_sdk_deploy/nodes/navigation_node.cpp`
    - Initialize WaypointTracker, LocalPlanner, and ObstacleAvoidance
    - Set up ROS 2 subscribers for PerceptionData and publishers for ControlCommand
    - Implement main control loop at 50 Hz
    - Add diagnostics and status publishing
    - Make node optional via launch parameter
    - _Requirements: 5.2, 5.3, 5.7, 8.2_

  - [ ] 7.4 Create ROS 2 node for Locomotion Layer
    - Create `src/S10_sdk_deploy/nodes/locomotion_node.cpp`
    - Initialize LocomotionPolicy and JointController
    - Set up ROS 2 subscribers for ControlCommand/PerceptionData and publisher for /JOINTS_CMD
    - Implement main control loop at 200 Hz
    - Handle timeout detection for perception/navigation failures
    - Add logging and diagnostics
    - _Requirements: 4.6, 4.7, 5.3, 5.4, 8.3, 10.1, 10.2_

  - [ ]* 7.5 Write integration tests for three-layer system
    - Test end-to-end data flow from sensor input to joint command output
    - Test navigation-enabled and navigation-disabled modes
    - Test timeout handling and safe state transitions
    - Test configuration reloading
    - _Requirements: 9.2, 9.3, 9.7_

- [ ] 8. Create launch files and deployment infrastructure
  - [ ] 8.1 Create ROS 2 launch file for full three-layer system
    - Create `src/S10_sdk_deploy/launch/three_layer_system.launch.py`
    - Launch perception_node, navigation_node, locomotion_node with proper parameters
    - Launch MuJoCo simulation with sensor plugins
    - Configure topic remapping and namespace management
    - _Requirements: 6.1, 6.2, 5.7_

  - [ ] 8.2 Create configuration YAML templates
    - Create `config/three_layer_config.yaml` with default parameters for all layers
    - Create `config/perception_only_config.yaml` for navigation-disabled mode
    - Document all configuration parameters and valid ranges
    - _Requirements: 6.3, 6.4, 6.5, 6.6_

  - [ ] 8.3 Create deployment documentation
    - Create README.md documenting system architecture, launch procedures, and configuration options
    - Document sensor simulation setup in MuJoCo
    - Document troubleshooting procedures for common failures
    - _Requirements: 6.7_

- [ ] 9. Implement monitoring and diagnostic tools
  - [ ] 9.1 Create performance monitoring node
    - Create `src/S10_sdk_deploy/nodes/monitor_node.cpp`
    - Subscribe to diagnostic topics from all layers
    - Publish aggregated system health status
    - Detect message queue overflows and dropped messages
    - Log timing statistics and bottlenecks
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.4_

  - [ ] 9.2 Create visualization tools
    - Create RViz configuration displaying sensor data, perception outputs, and planned trajectories
    - Create data flow visualization script showing topic latencies
    - _Requirements: 9.3_

  - [ ] 9.3 Implement ROS 2 services for runtime control
    - Create service definitions for enable/disable verbose logging
    - Create service definition for emergency stop
    - Implement service handlers in each layer node
    - _Requirements: 8.4, 10.7_

  - [ ]* 9.4 Create rosbag recording and playback utilities
    - Create launch file for recording critical topics
    - Create playback analysis script for offline debugging
    - _Requirements: 8.6_

- [ ] 10. Checkpoint - Final system validation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Implement test infrastructure and validation
  - [ ] 11.1 Create synthetic sensor data generator for testing
    - Create `test/synthetic_data_generator.cpp`
    - Generate fake lidar point clouds with known obstacles
    - Generate fake depth images with known geometry
    - Publish on sensor topics for layer testing without simulation
    - _Requirements: 9.1_

  - [ ] 11.2 Create end-to-end race simulation test
    - Create `test/race_simulation_test.cpp`
    - Load a test waypoint track configuration
    - Run full three-layer system through waypoint sequence
    - Validate waypoint reaching and elapsed time computation
    - Test with navigation enabled and disabled modes
    - _Requirements: 3.7, 8.7_

  - [ ]* 11.3 Create performance benchmarking suite
    - Measure perception processing latency under various point cloud sizes
    - Measure navigation planning time under various obstacle densities
    - Measure control loop timing stability over extended runs
    - Generate performance report with latency histograms
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 12. Final checkpoint - System ready for deployment
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements from the requirements document for traceability
- Checkpoints ensure incremental validation and provide opportunities for user feedback
- The implementation uses C++ for performance-critical components (perception, control) and Python for simulation integration
- ROS 2 message definitions must be built before dependent nodes can compile
- The locomotion policy model file must be trained separately and provided as an input artifact
- Configuration files use YAML format for human readability and runtime reloading
- All timestamps use simulation time (not wall clock time) for deterministic replay
- The system supports both full three-layer mode (with navigation) and direct perception-to-locomotion mode

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2", "3.1", "6.3"] },
    { "id": 2, "tasks": ["3.2", "11.1"] },
    { "id": 3, "tasks": ["3.3", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3"] },
    { "id": 5, "tasks": ["4.4", "6.1"] },
    { "id": 6, "tasks": ["6.2"] },
    { "id": 7, "tasks": ["6.4", "7.1"] },
    { "id": 8, "tasks": ["7.2", "7.3", "7.4"] },
    { "id": 9, "tasks": ["7.5", "8.1", "8.2"] },
    { "id": 10, "tasks": ["8.3", "9.1"] },
    { "id": 11, "tasks": ["9.2", "9.3", "9.4"] },
    { "id": 12, "tasks": ["11.2", "11.3"] }
  ]
}
```
