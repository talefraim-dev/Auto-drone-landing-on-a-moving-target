\<div align="center">

\<h1 align="center">Autonomous UAV Landing on a Moving Target\</h1>

\<p align="center">

  Developed by \<b>Tal Efraim\</b> and \<b>Daniel Rosman\</b>

\</p>

**### A vision-guided, dual-agent reinforcement-learning system for autonomous target tracking and cooperative landing**





![Python]\(https\://img.shields.io/badge/Python-RL%20%26%20Vision-3776AB?logo=python&logoColor=white)

![PPO]\(https\://img.shields.io/badge/Reinforcement%20Learning-PPO-7B61FF)

![Unreal Engine]\(https\://img.shields.io/badge/Simulation-Unreal%20Engine-0E1128?logo=unrealengine&logoColor=white)

![React]\(https\://img.shields.io/badge/UI-React-61DAFB?logo=react&logoColor=black)

![Electron]\(https\://img.shields.io/badge/Desktop-Electron-47848F?logo=electron&logoColor=white)

\</div>

\---

**## Overview**

This project proposes an autonomous approach for landing a UAV on a moving target whose position is not known in advance.

The system detects a target selected at runtime, preserves its visual identity, tracks it while it moves, aligns the UAV above the landing surface, and performs a controlled descent until physical contact is achieved.

The project combines **\*\*computer vision\*\***, **\*\*visual tracking\*\***, **\*\*distance sensing\*\***, **\*\*motion control\*\***, **\*\*reinforcement learning\*\***, and a desktop **\*\*Ground Control Station\*\*** within an Unreal Engine and AirSim-based simulation environment.

\---

**## Engineering Challenge**

Landing on a moving platform requires the UAV to solve several coupled problems in real time:

\- Detect and identify a target inside a visually complex scene.

\- Preserve the target identity while both the UAV and target are moving.

\- Predict target motion during temporary visual instability.

\- Align the UAV in the horizontal plane and in yaw.

\- Decide when the system is ready to begin the landing phase.

\- Descend while maintaining alignment above a moving surface.

\- Continue operating when close-range vision becomes less reliable.

The main engineering objective is to achieve this behavior without requiring the target location to be predefined and without relying on a single monolithic controller for the entire task.

\---

**## Proposed Architecture**

The system uses a **\*\*dual-agent PPO architecture\*\*** that separates the task into two specialized learning problems.

\`\`\`text

Selected Target

      │

      ▼

YOLO Detection + ResNet-18 Features

      │

      ▼

Target Memory + Visual Tracker + Kalman Prediction

      │

      ▼

Agent 1: Tracking, XY Control and Yaw Alignment

      │

      ├── Not Ready ──► Continue Tracking

      │

      ▼

Landing-Ready Handoff

      │

      ▼

Agent 1: Horizontal Corrections

Agent 2: Vertical Control and Descent

      │

      ▼

Range Sensors + Vision + Safety Logic

      │

      ▼

Physical Contact and Landing Validation

\`\`\`

**### Agent 1 — Tracking and Alignment**

Agent 1 is responsible for:

\- Target detection and identity verification.

\- Visual tracking and motion prediction.

\- Horizontal movement along the X and Y axes.

\- Yaw alignment.

\- Preparing the UAV for landing.

\- Determining when the landing-ready conditions have been reached.

Its perception pipeline combines:

\- YOLO-based object detection.

\- ResNet-18 visual feature extraction.

\- Target-memory management.

\- Bounding-box tracking.

\- Kalman filtering.

\- Motion-based prediction during temporary target loss.

**### Agent 2 — Cooperative Landing**

Agent 2 is responsible for:

\- Vertical motion control.

\- Controlled descent.

\- Processing downward-facing range sensors.

\- Maintaining a safe and stable landing sequence.

\- Reaching physical contact with the selected target.

During the cooperative landing phase, Agent 1 continues to correct horizontal position while Agent 2 controls the vertical axis. This decomposition reduces the action-space complexity and allows each policy to specialize in a more focused task.

**### Supporting Components**

The architecture also includes:

\- LiDAR and range-finder processing.

\- Observation-vector construction for both agents.

\- Safety filtering and training-safety constraints.

\- Separate reward modules for following, identity stability, and touchdown behavior.

\- Atomic paired-checkpoint management.

\- Statistical benchmark scripts for repeated landing trials.

\- CSV telemetry for performance and failure analysis.

\---

**## Repository Structure**

The tree below intentionally shows only the main project source files and the most important model and backup directories. Generated dependencies, IDE metadata, caches, build artifacts, and other auxiliary files are omitted for clarity.

\`\`\`text

Project Root/
│
├── RL_training/
│   ├── agent1p2_env.py
│   ├── agent2_landing_env.py
│   ├── alternating_cotraining_env.py
│   ├── calibrate_and_test_range_finders.py
│   ├── cooperative_training_config.py
│   ├── drone_env.py
│   ├── follow_reward.py
│   ├── identity_stability_reward.py
│   ├── lidar_processor.py
│   ├── manual_drone_tester_live.py
│   ├── object_tracker.py
│   ├── observation_builder.py
│   ├── paired_checkpoint_manager.py
│   ├── preflight_cooperative_training.py
│   ├── range_finder_array.py
│   ├── resnet_yolo_tracker.py
│   ├── run_final_object_tracker.py
│   ├── Run_train_alternating_agents.py
│   ├── Run_train_cooperative_final.py
│   ├── safety_filter.py
│   ├── test_final_models_100_landings_csv.py
│   ├── touchdown_time_reward.py
│   ├── training_safety.py
│   ├── weights_config.py
│   │
│   ├── config/
│   │   └── flow_config.py
│   │
│   ├── tracking/
│   │   ├── bbox_utils.py
│   │   ├── kalman_bbox.py
│   │   ├── target_memory.py
│   │   ├── target_tracker_manager.py
│   │   └── tracking_state.py
│   │
│   └── models/
│       └── FINAL_MODELS/
│
├── drone_user_interface/
│   ├── electron/
│   │   ├── main.js
│   │   └── preload.js
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   │
│   │   ├── components/
│   │   │   ├── Footer.jsx
│   │   │   ├── Minimap.jsx
│   │   │   └── VideoFeed.jsx
│   │   │
│   │   └── hooks/
│   │       ├── useDroneLogic.js
│   │       ├── useLogger.js
│   │       ├── usePixelStreaming.js
│   │       └── useTelemetry.js
│   │
│   └── package.json
│
├── telemetry_server/
│   └── airsim_api.py
│
└── Agents_backup/
    ├── AGENT_1P2/
    ├── ALTERNATING_CO_TRAINING/
    ├── COOPERATIVE_FINAL/
    ├── FINAL_MODELS/
    └── PPO_Tracker/

\`\`\`

**### \`RL\_training/\`**

Contains the main UAV system, including:

\- Agent environments and cooperative wrappers.

\- Vision and target-tracking modules.

\- Sensor and safety processing.

\- Reward logic and training configuration.

\- Final trained models.

\- Benchmark and statistical experiment data.

**### \`drone\_user\_interface/\`**

Contains the React and Electron Ground Control Station used to visualize the simulation, select targets, display telemetry, and send control commands to Unreal Engine through Pixel Streaming interactions.

**### \`Agents\_backup/\`**

Contains historical checkpoints, paired models, TensorBoard event files, training snapshots, manifests, and source snapshots from the different training stages.

This directory is intended mainly for backup, experiment recovery, and training-history preservation. The final runtime models are stored separately in \`RL\_training/models/FINAL\_MODELS/\`.

\---

**## Main Project Files**

**### Core Environments and Training**

\| File | Description |

\|---|---|

\| \`RL\_training/drone\_env.py\` | Main Agent 1 environment for target tracking, vision, horizontal control, and yaw alignment. |

\| \`RL\_training/agent2\_landing\_env.py\` | Agent 2 environment for vertical control, range sensing, descent, and landing logic. |

\| \`RL\_training/agent1p2\_env.py\` | Shared environment connecting the two agents. |

\| \`RL\_training/alternating\_cotraining\_env.py\` | Wrappers used for alternating and cooperative agent execution. |

\| \`RL\_training/Run\_train\_alternating\_agents.py\` | Entry point for alternating co-training. |

\| \`RL\_training/Run\_train\_cooperative\_final.py\` | Entry point for the final cooperative training stage. |

\| \`RL\_training/preflight\_cooperative\_training.py\` | Preflight validation before cooperative training. |

\| \`RL\_training/cooperative\_training\_config.py\` | Configuration for cooperative training. |

\| \`RL\_training/weights\_config.py\` | Main environment, reward, threshold, and movement configuration. |

**### Vision and Target Tracking**

\| File | Description |

\|---|---|

\| \`RL\_training/resnet\_yolo\_tracker.py\` | YOLO detection and ResNet-based feature extraction. |

\| \`RL\_training/object\_tracker.py\` | Main visual-tracking interface. |

\| \`RL\_training/tracking/target\_tracker\_manager.py\` | Manages target tracking, prediction, loss, and recovery states. |

\| \`RL\_training/tracking/kalman\_bbox.py\` | Kalman filtering for bounding-box motion prediction. |

\| \`RL\_training/tracking/target\_memory.py\` | Stores the selected target identity and visual representation. |

\| \`RL\_training/tracking/bbox\_utils.py\` | Bounding-box helper functions. |

\| \`RL\_training/tracking/tracking\_state.py\` | Tracking-state definitions. |

**### Sensors, Observations, Rewards, and Safety**

\| File | Description |

\|---|---|

\| \`RL\_training/observation\_builder.py\` | Builds the observation vectors supplied to the agents. |

\| \`RL\_training/range\_finder\_array.py\` | Manages the downward-facing range-sensor array. |

\| \`RL\_training/lidar\_processor.py\` | Processes LiDAR measurements. |

\| \`RL\_training/safety\_filter.py\` | Filters unsafe movement commands. |

\| \`RL\_training/training\_safety.py\` | Adds safety constraints during training. |

\| \`RL\_training/follow\_reward.py\` | Reward logic for following and tracking behavior. |

\| \`RL\_training/identity\_stability\_reward.py\` | Reward logic for preserving target identity. |

\| \`RL\_training/touchdown\_time\_reward.py\` | Reward logic related to touchdown timing. |

\| \`RL\_training/calibrate\_and\_test\_range\_finders.py\` | Calibration and validation utility for range sensors. |

**### Models, Configuration, and Evaluation**

\| File or Directory | Description |

\|---|---|

\| \`RL\_training/settings.json\` | AirSim vehicle, camera, and sensor configuration. |

\| \`RL\_training/config/flow\_config.py\` | Controls the selected execution and training flow. |

\| \`RL\_training/config/bottom\_bbox\_center\_calibration.json\` | Bottom-camera bounding-box center calibration. |

\| \`RL\_training/config/range\_finder\_calibration.json\` | Range-sensor calibration data. |

\| \`RL\_training/models/FINAL\_MODELS/agent1\_final.zip\` | Final Agent 1 PPO model. |

\| \`RL\_training/models/FINAL\_MODELS/agent2\_final.zip\` | Final Agent 2 PPO model. |

\| \`RL\_training/test\_final\_models\_100\_landings\_csv.py\` | Repeated landing benchmark with CSV telemetry output. |

\| \`RL\_training/statistics\_data/\` | Experimental datasets collected at different landing-surface scales. |

\| \`RL\_training/paired\_checkpoint\_manager.py\` | Atomic management of paired Agent 1 and Agent 2 checkpoints. |

\---

**## User Interface**

The desktop user interface is located in:

\`\`\`text

drone\_user\_interface/

\`\`\`

It is implemented using:

\- React

\- Vite

\- Electron

\- Leaflet

\- Unreal Engine Pixel Streaming frontend libraries

**### Interface Capabilities**

The interface provides:

\- Live Unreal Engine video streaming.

\- Click-based target selection through the video feed.

\- Target acquisition and lock indicators.

\- Altitude, speed, pitch, and roll display.

\- Simulator minimap and UAV position display.

\- Manual and autonomous mode controls.

\- Hover, return-to-home, landing, synchronization, and emergency-abort commands.

\- System-status indicators and an event log.

**### Main Interface Files**

\| File | Description |

\|---|---|

\| \`drone\_user\_interface/src/App.jsx\` | Main Ground Control Station application component. |

\| \`drone\_user\_interface/src/components/VideoFeed.jsx\` | Live video area, HUD, attitude display, and target-selection overlay. |

\| \`drone\_user\_interface/src/components/Minimap.jsx\` | Simulator map and UAV-position display. |

\| \`drone\_user\_interface/src/components/Footer.jsx\` | Footer and mission-status information. |

\| \`drone\_user\_interface/src/hooks/usePixelStreaming.js\` | Connection to the Unreal Engine Pixel Streaming server. |

\| \`drone\_user\_interface/src/hooks/useTelemetry.js\` | Receives and processes telemetry messages. |

\| \`drone\_user\_interface/src/hooks/useDroneLogic.js\` | Handles UI modes, target selection, logs, and simulator commands. |

\| \`drone\_user\_interface/electron/main.js\` | Electron desktop-window entry point. |

\| \`drone\_user\_interface/electron/preload.js\` | Isolated bridge between Electron and the renderer. |

\| \`drone\_user\_interface/package.json\` | Frontend dependencies and development scripts. |

\> **\*\*Note:\*\*** The interface contains both simulator-connected telemetry handling and prototype fallback or simulated status values. Final packaging should validate every telemetry and command path against the intended Unreal Engine runtime configuration.

\---

**## Final Models and Experimental Data**

The final frozen policies are located in:

\`\`\`text

RL\_training/models/FINAL\_MODELS/

├── agent1\_final.zip

├── agent2\_final.zip

├── final\_models.json

├── training\_config\_snapshot.json

└── touchdown\_time\_reward\_state.json

\`\`\`

The collected landing experiments are stored under:

\`\`\`text

RL\_training/statistics\_data/BMW/

\`\`\`

The available datasets include tests performed with multiple landing-surface scales, enabling analysis of:

\- Landing success rate.

\- Landing-position error.

\- Flight and landing duration.

\- Visual match, prediction, and target-loss percentages.

\- Landing-ready handoff behavior.

\- Failure reasons and vision-related operating limits.

\---

**## Project Scope**

This repository represents a simulation-based research and engineering prototype. Its primary contribution is the design and evaluation of a modular perception-and-control architecture for autonomous UAV landing on a moving target.

The project demonstrates that separating target tracking and vertical landing into specialized policies can simplify learning and provide interpretable performance metrics. The current operating limits are primarily evaluated within the supplied simulation, camera, tracker, and sensor configuration.

\---

\<div align="center">

**\*\*Autonomous perception · Target tracking · Cooperative reinforcement learning · Precision landing\*\***

\</div>