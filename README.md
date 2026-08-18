rosman
rosman13
Invisible
﻿
talefraim96 — Yesterday at 13:30
@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:842 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:842
useTelemetry.js:63 WebSocket connection to 'ws://127.0.0.1:4001/' failed: WebSocket is closed before the connection is established.
(anonymous) @ useTelemetry.js:63
@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:653 Level: Error
Msg: WebSocket error
Caller: 
    at i.GetStackTrace (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:629:16)
    at f.handleOn (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:801:15)
    at webSocket.onerror (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:779:141)
Error @ @epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:653
@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:641 Level: Log
Msg: Disconnected to the signalling server via WebSocket: 1006 - 
Caller: 
    at i.GetStackTrace (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:629:16)
    at f.handleOnClose (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:804:13)
    at webSocket.onclose (@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:779:196)
useTelemetry.js:50 Telemetry WS Error: Event
(anonymous) @ useTelemetry.js:50
useTelemetry.js:49 Connected to Telemetry Server
talefraim96 — Yesterday at 13:47
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
usePixelStreaming.js:42 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842
closeSignalingServer @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:2759
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:2766
disconnect @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:3101

message.txt
18 KB
"C:\Program Files\Git\bin\bash.exe" start_system.sh
rosman — Yesterday at 13:56
SignallingWebServer
talefraim96 — Yesterday at 13:57
C:\Program Files\Epic Games\UE_5.5\Engine\Plugins\Media\PixelStreaming\Resources\WebServers\SignallingWebServer
C:\Program Files\Epic Games\UE_5.5\Engine\Plugins\Media\PixelStreaming2\Resources\WebServers\SignallingWebServer
Directory of C:\Program Files\Epic Games\UE_5.5\Engine\Plugins\Media\PixelStreaming\Resources\WebServers\SignallingWebServer

05/28/2026  09:55 PM    <DIR>          .
05/28/2026  09:55 PM    <DIR>          ..
05/28/2026  09:55 PM                57 .gitignore
05/28/2026  09:55 PM               114 .lintstagedrc.mjs
05/28/2026  09:55 PM               318 .prettierignore
05/28/2026  09:55 PM    <DIR>          apidoc
05/28/2026  09:55 PM               542 config.json
05/28/2026  09:55 PM             1,098 Dockerfile
05/28/2026  09:55 PM               985 eslint.config.mjs
05/28/2026  09:55 PM             1,192 from_cirrus.md
05/28/2026  09:55 PM             1,964 package.json
05/28/2026  09:55 PM    <DIR>          platform_scripts
05/28/2026  09:55 PM             6,558 README.md
05/28/2026  09:55 PM    <DIR>          src
05/28/2026  09:55 PM            12,475 tsconfig.json
05/28/2026  09:55 PM    <DIR>          www
              10 File(s)         25,303 bytes
               6 Dir(s)  132,365,746,176 bytes free

C:\Program Files\Epic Games\UE_5.5\Engine\Plugins\Media\PixelStreaming\Resources\WebServers\SignallingWebServer>
"C:\Program Files\Git\bin\bash.exe" start_system.sh
talefraim96 — Yesterday at 14:06

C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target>"C:\Program Files\Git\bin\bash.exe" start_system.sh
Initializing AeroGuard System Startup...

[1/3] Starting Pixel Streaming Signalling Server...

message.txt
8 KB
"C:\Program Files\Git\bin\bash.exe" start_system.sh

C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target>git pull
Already up to date.

C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target>git pull
remote: Enumerating objects: 5, done.

message.txt
8 KB
talefraim96 — Yesterday at 14:15
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
usePixelStreaming.js:42 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842
closeSignalingServer @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:2759
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:2766
disconnect @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:3101

message.txt
18 KB
rosman — Yesterday at 14:19
{
  "type": "Telemetry",
  "X": {X},
  "Y": {Y},
  "Z": {Z},
  "Speed": {Speed},
  "Pitch": {Pitch},
  "Roll": {Roll},
  "Yaw": {Yaw}
}
talefraim96 — Yesterday at 14:43


rosman — Yesterday at 14:49
{"type": "Telemetry", "X": {X}, "Y": {Y}, "Z": {Z}, "Speed": {SPEED}, "Pitch": {PITCH}, "Roll": {ROLL}, "Yaw": {YAW}}
talefraim96 — Yesterday at 15:02
103useTelemetry.js:45 Failed to parse telemetry from WS: SyntaxError: Unexpected non-whitespace character after JSON at position 12 (line 1 column 13)
    at JSON.parse (<anonymous>)
    at ws.onmessage (useTelemetry.js:30:31)
rosman — Yesterday at 15:03
{"type": "Telemetry", "X": {X}, "Y": {Y}, "Z": {Z}, "Speed": {SPEED}, "Pitch": {PITCH}, "Roll": {ROLL}, "Yaw": {YAW}}
talefraim96 — Yesterday at 15:07
69useTelemetry.js:45 Failed to parse telemetry from WS: SyntaxError: Unexpected token ',', ", "X": 140"... is not valid JSON
    at JSON.parse (<anonymous>)
    at ws.onmessage (useTelemetry.js:30:31)
Received from Unreal: , "X": 140, "Y": 20, "Z": 19.714, "Speed": 0, "Pitch": 0, "Roll": 0, "Yaw": 0 }
rosman
 started a call that lasted 44 minutes. — Yesterday at 15:49
talefraim96 — Yesterday at 15:54
Received from Unreal: , "X": 140, "Y": 20, "Z": 19.714, "Speed": 0, "Pitch": 0, "Roll": 0, "Yaw": 0 }
rosman — Yesterday at 16:03
{"type": "Telemetry", "X": {X}, "Y": {Y}, "Z": {Z}, "Speed": {SPEED}, "Pitch": {PITCH}, "Roll": {ROLL}, "Yaw": {YAW}}
"type": "Telemetry", "X": {X}, "Y": {Y}, "Z": {Z}, "Speed": {SPEED}, "Pitch": {PITCH}, "Roll": {ROLL}, "Yaw": {YAW}
talefraim96 — Yesterday at 16:08
177
useTelemetry.js:49 Failed to parse telemetry from WS: SyntaxError: Unexpected non-whitespace character after JSON at position 6 (line 1 column 7)
    at JSON.parse (<anonymous>)
    at ws.onmessage (useTelemetry.js:31:31)
(anonymous)    @    useTelemetry.js:49
[11, 22, 33, 44, 55, 66, 77]
rosman — Yesterday at 16:15
"type": "Telemetry", "X": {X}, "Y": {Y}, "Z": {Z}, "Speed": {Speed}, "Pitch": {Pitch}, "Roll": {Roll}, "Yaw": {Yaw}
"C:\Program Files\Git\bin\bash.exe" start_system.sh
talefraim96 — Yesterday at 16:21
import json

def parse_drone_data(body: str):
    values = json.loads(body)

    keys = ["x", "y", "z", "speed", "roll", "pitch", "yaw"]

    if len(values) != len(keys):
        raise ValueError(
            f"Expected {len(keys)} values, received {len(values)}"
        )

    return dict(zip(keys, values))
{
    "x": 140,
    "y": 20,
    "z": 19.582,
    "speed": 0,
    "roll": 0,
    "pitch": 0,
    "yaw": 0
}
talefraim96 — Yesterday at 16:33
{
    "x": 140,
    "y": 20,
    "z": 19.582,
    "speed": 0,
    "roll": 0,
    "pitch": 0,
    "yaw": 0
}
 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
 WebSocket connection to 'ws://127.0.0.1:4001/' failed: WebSocket is closed before the connection is established.
 Failed to parse telemetry from WS: 
 Raw string that caused error: [2,454.064,-79.713,181.277,0,-0,0.85,-2.041]
 Failed to parse telemetry from WS: 
 Raw string that caused error: [2,429.58,-78.84,181.243,0,-0,0.676,-2.041]... (94 KB left)

message.txt
144 KB
rosman
 started a call. — 10:06
talefraim96 — 10:25
import cosysairsim as airsim
client = airsim.MultirotorClient()
client.confirmConnection()
state = client.getMultirotorState()
vx = state.kinematics_estimated.linear_velocity.x_val
vy = state.kinematics_estimated.linear_velocity.y_val
vz = state.kinematics_estimated.linear_velocity.z_val
speed_mps = math.sqrt(vx2 + vy2 + vz**2
speed_kmh = speed_mps * 3.6
talefraim96 — 10:51
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842
@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:653 Level: Error
Msg: WebSocket error
Caller: ... (2 KB left)

message.txt
52 KB
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
usePixelStreaming.js:42 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842
closeSignalingServer @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:2759
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:2766
disconnect @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:3101... (32 KB left)

message.txt
82 KB
talefraim96 — 11:18
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:842 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:842
@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:653 Level: Error
Msg: WebSocket error
Caller: 
    at i.GetStackTrace (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:629:16)
    at f.handleOn (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:801:15)
    at webSocket.onerror (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:779:141)
Error @ @epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:653
@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:641 Level: Log
Msg: Disconnected to the signalling server via WebSocket: 1006 - 
Caller: 
    at i.GetStackTrace (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:629:16)
    at f.handleOnClose (@epicgames-ps_lib-pixelstreamingfrontend-ue53.js?v=aa329037:804:13)
    at webSocket.onclose (@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:779:196)
useTelemetry.js:74 Connected to Telemetry Server
React Developer Tools – React
The library for web and native user interfaces
React Developer Tools – React
"C:\Program Files\Git\bin\bash.exe" start_system.sh
talefraim96 — 11:28
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842
@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:653 Level: Error
Msg: WebSocket error
Caller: 

message.txt
11 KB
rosman — 11:30
http://127.0.0.1:8000/api/speed
talefraim96 — 11:40
{"error":"module 'cosysairsim' has no attribute 'to_eularian_angles'"}
Traceback (most recent call last):
  File "C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target\telemetry_server\airsim_api.py", line 2, in <module>
    from cosysairsim.utils import to_eularian_angles
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ImportError: cannot import name 'to_eularian_angles' from 'cosysairsim.utils' (C:\Users\Tal Efraim\AppData\Local\Programs\Python\Python312\Lib\site-packages\cosysairsim\utils.py)
talefraim96 — 12:04
"""
Live manual multirotor controller for Cosys-AirSim.

Controls:
W = forward
S = backward

manual_drone_tester_live.py
3 KB
talefraim96 — 12:13
chunk-TYILIMWK.js?v=aa329037:21551 Download the React DevTools for a better development experience: https://reactjs.org/link/react-devtools
@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842 WebSocket connection to 'ws://127.0.0.1/' failed: WebSocket is closed before the connection is established.
close @ @epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:842
@epicgames-ps_lib-pixelstreamingfrontend-ue5__3.js?v=aa329037:653 Level: Error
Msg: WebSocket error
Caller: 

message.txt
22 KB
Traceback (most recent call last):
  File "C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target\telemetry_server\airsim_api.py", line 7, in <module>
    import keyboard
ModuleNotFoundError: No module named 'keyboard'
Traceback (most recent call last):
  File "C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target\telemetry_server\airsim_api.py", line 7, in <module>
    import keyboard
ModuleNotFoundError: No module named 'keyboard'
"C:\Program Files\Git\bin\bash.exe" start_system.sh
talefraim96 — 12:31
{
  "error": "There is no current event loop in thread 'AnyIO worker thread'."
}
rosman — 12:37
http://127.0.0.1:8000/api/telemetry
talefraim96 — 13:24
Redesign the existing UAV Mission Control / Autonomous Landing dashboard.

The goal is to improve the **information hierarchy, component placement, operator readability, and use of screen space**, while preserving the existing dark aerospace-style visual language.

Do NOT redesign the application into a completely different product.

message.txt
16 KB
talefraim96 — 13:53
<div align="center">

<h1 align="center">Autonomous UAV Landing on a Moving Target</h1>

<p align="center">
  Developed by <b>Tal Efraim</b> and <b>Daniel Rosman</b>

README.md
13 KB
﻿
<div align="center">

<h1 align="center">Autonomous UAV Landing on a Moving Target</h1>

<p align="center">
  Developed by <b>Tal Efraim</b> and <b>Daniel Rosman</b>
</p>

### A vision-guided, dual-agent reinforcement-learning system for autonomous target tracking and cooperative landing



![Python](https://img.shields.io/badge/Python-RL%20%26%20Vision-3776AB?logo=python&logoColor=white)
![PPO](https://img.shields.io/badge/Reinforcement%20Learning-PPO-7B61FF)
![Unreal Engine](https://img.shields.io/badge/Simulation-Unreal%20Engine-0E1128?logo=unrealengine&logoColor=white)
![React](https://img.shields.io/badge/UI-React-61DAFB?logo=react&logoColor=black)
![Electron](https://img.shields.io/badge/Desktop-Electron-47848F?logo=electron&logoColor=white)

</div>

---

## Overview

This project proposes an autonomous approach for landing a UAV on a moving target whose position is not known in advance.

The system detects a target selected at runtime, preserves its visual identity, tracks it while it moves, aligns the UAV above the landing surface, and performs a controlled descent until physical contact is achieved.

The project combines **computer vision**, **visual tracking**, **distance sensing**, **motion control**, **reinforcement learning**, and a desktop **Ground Control Station** within an Unreal Engine and AirSim-based simulation environment.

---

## Engineering Challenge

Landing on a moving platform requires the UAV to solve several coupled problems in real time:

- Detect and identify a target inside a visually complex scene.
- Preserve the target identity while both the UAV and target are moving.
- Predict target motion during temporary visual instability.
- Align the UAV in the horizontal plane and in yaw.
- Decide when the system is ready to begin the landing phase.
- Descend while maintaining alignment above a moving surface.
- Continue operating when close-range vision becomes less reliable.

The main engineering objective is to achieve this behavior without requiring the target location to be predefined and without relying on a single monolithic controller for the entire task.

---

## Proposed Architecture

The system uses a **dual-agent PPO architecture** that separates the task into two specialized learning problems.

```text
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
```

### Agent 1 — Tracking and Alignment

Agent 1 is responsible for:

- Target detection and identity verification.
- Visual tracking and motion prediction.
- Horizontal movement along the X and Y axes.
- Yaw alignment.
- Preparing the UAV for landing.
- Determining when the landing-ready conditions have been reached.

Its perception pipeline combines:

- YOLO-based object detection.
- ResNet-18 visual feature extraction.
- Target-memory management.
- Bounding-box tracking.
- Kalman filtering.
- Motion-based prediction during temporary target loss.

### Agent 2 — Cooperative Landing

Agent 2 is responsible for:

- Vertical motion control.
- Controlled descent.
- Processing downward-facing range sensors.
- Maintaining a safe and stable landing sequence.
- Reaching physical contact with the selected target.

During the cooperative landing phase, Agent 1 continues to correct horizontal position while Agent 2 controls the vertical axis. This decomposition reduces the action-space complexity and allows each policy to specialize in a more focused task.

### Supporting Components

The architecture also includes:

- LiDAR and range-finder processing.
- Observation-vector construction for both agents.
- Safety filtering and training-safety constraints.
- Separate reward modules for following, identity stability, and touchdown behavior.
- Atomic paired-checkpoint management.
- Statistical benchmark scripts for repeated landing trials.
- CSV telemetry for performance and failure analysis.

---

## Repository Structure

Each ZIP archive supplied with the project should be extracted as a folder directly inside the repository root.

```text
Project Root/
│
├── RL_training/
│   ├── config/
│   ├── tracking/
│   ├── models/
│   │   └── FINAL_MODELS/
│   ├── statistics_data/
│   ├── UE_env/
│   └── *.py
│
├── drone_user_interface/
│   ├── electron/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   └── hooks/
│   ├── dist/
│   └── package.json
│
└── Agents_backup/
    ├── AGENT_1P2/
    ├── ALTERNATING_CO_TRAINING/
    └── COOPERATIVE_FINAL/
```

### `RL_training/`

Contains the main UAV system, including:

- Agent environments and cooperative wrappers.
- Vision and target-tracking modules.
- Sensor and safety processing.
- Reward logic and training configuration.
- Final trained models.
- Benchmark and statistical experiment data.

### `drone_user_interface/`

Contains the React and Electron Ground Control Station used to visualize the simulation, select targets, display telemetry, and send control commands to Unreal Engine through Pixel Streaming interactions.

### `Agents_backup/`

Contains historical checkpoints, paired models, TensorBoard event files, training snapshots, manifests, and source snapshots from the different training stages.

This directory is intended mainly for backup, experiment recovery, and training-history preservation. The final runtime models are stored separately in `RL_training/models/FINAL_MODELS/`.

---

## Main Project Files

### Core Environments and Training

| File | Description |
|---|---|
| `RL_training/drone_env.py` | Main Agent 1 environment for target tracking, vision, horizontal control, and yaw alignment. |
| `RL_training/agent2_landing_env.py` | Agent 2 environment for vertical control, range sensing, descent, and landing logic. |
| `RL_training/agent1p2_env.py` | Shared environment connecting the two agents. |
| `RL_training/alternating_cotraining_env.py` | Wrappers used for alternating and cooperative agent execution. |
| `RL_training/Run_train_alternating_agents.py` | Entry point for alternating co-training. |
| `RL_training/Run_train_cooperative_final.py` | Entry point for the final cooperative training stage. |
| `RL_training/preflight_cooperative_training.py` | Preflight validation before cooperative training. |
| `RL_training/cooperative_training_config.py` | Configuration for cooperative training. |
| `RL_training/weights_config.py` | Main environment, reward, threshold, and movement configuration. |

### Vision and Target Tracking

| File | Description |
|---|---|
| `RL_training/resnet_yolo_tracker.py` | YOLO detection and ResNet-based feature extraction. |
| `RL_training/object_tracker.py` | Main visual-tracking interface. |
| `RL_training/tracking/target_tracker_manager.py` | Manages target tracking, prediction, loss, and recovery states. |
| `RL_training/tracking/kalman_bbox.py` | Kalman filtering for bounding-box motion prediction. |
| `RL_training/tracking/target_memory.py` | Stores the selected target identity and visual representation. |
| `RL_training/tracking/bbox_utils.py` | Bounding-box helper functions. |
| `RL_training/tracking/tracking_state.py` | Tracking-state definitions. |

### Sensors, Observations, Rewards, and Safety

| File | Description |
|---|---|
| `RL_training/observation_builder.py` | Builds the observation vectors supplied to the agents. |
| `RL_training/range_finder_array.py` | Manages the downward-facing range-sensor array. |
| `RL_training/lidar_processor.py` | Processes LiDAR measurements. |
| `RL_training/safety_filter.py` | Filters unsafe movement commands. |
| `RL_training/training_safety.py` | Adds safety constraints during training. |
| `RL_training/follow_reward.py` | Reward logic for following and tracking behavior. |
| `RL_training/identity_stability_reward.py` | Reward logic for preserving target identity. |
| `RL_training/touchdown_time_reward.py` | Reward logic related to touchdown timing. |
| `RL_training/calibrate_and_test_range_finders.py` | Calibration and validation utility for range sensors. |

### Models, Configuration, and Evaluation

| File or Directory | Description |
|---|---|
| `RL_training/settings.json` | AirSim vehicle, camera, and sensor configuration. |
| `RL_training/config/flow_config.py` | Controls the selected execution and training flow. |
| `RL_training/config/bottom_bbox_center_calibration.json` | Bottom-camera bounding-box center calibration. |
| `RL_training/config/range_finder_calibration.json` | Range-sensor calibration data. |
| `RL_training/models/FINAL_MODELS/agent1_final.zip` | Final Agent 1 PPO model. |
| `RL_training/models/FINAL_MODELS/agent2_final.zip` | Final Agent 2 PPO model. |
| `RL_training/test_final_models_100_landings_csv.py` | Repeated landing benchmark with CSV telemetry output. |
| `RL_training/statistics_data/` | Experimental datasets collected at different landing-surface scales. |
| `RL_training/paired_checkpoint_manager.py` | Atomic management of paired Agent 1 and Agent 2 checkpoints. |

---

## User Interface

The desktop user interface is located in:

```text
drone_user_interface/
```

It is implemented using:

- React
- Vite
- Electron
- Leaflet
- Unreal Engine Pixel Streaming frontend libraries

### Interface Capabilities

The interface provides:

- Live Unreal Engine video streaming.
- Click-based target selection through the video feed.
- Target acquisition and lock indicators.
- Altitude, speed, pitch, and roll display.
- Simulator minimap and UAV position display.
- Manual and autonomous mode controls.
- Hover, return-to-home, landing, synchronization, and emergency-abort commands.
- System-status indicators and an event log.

### Main Interface Files

| File | Description |
|---|---|
| `drone_user_interface/src/App.jsx` | Main Ground Control Station application component. |
| `drone_user_interface/src/components/VideoFeed.jsx` | Live video area, HUD, attitude display, and target-selection overlay. |
| `drone_user_interface/src/components/Minimap.jsx` | Simulator map and UAV-position display. |
| `drone_user_interface/src/components/Footer.jsx` | Footer and mission-status information. |
| `drone_user_interface/src/hooks/usePixelStreaming.js` | Connection to the Unreal Engine Pixel Streaming server. |
| `drone_user_interface/src/hooks/useTelemetry.js` | Receives and processes telemetry messages. |
| `drone_user_interface/src/hooks/useDroneLogic.js` | Handles UI modes, target selection, logs, and simulator commands. |
| `drone_user_interface/electron/main.js` | Electron desktop-window entry point. |
| `drone_user_interface/electron/preload.js` | Isolated bridge between Electron and the renderer. |
| `drone_user_interface/package.json` | Frontend dependencies and development scripts. |

> **Note:** The interface contains both simulator-connected telemetry handling and prototype fallback or simulated status values. Final packaging should validate every telemetry and command path against the intended Unreal Engine runtime configuration.

---

## Final Models and Experimental Data

The final frozen policies are located in:

```text
RL_training/models/FINAL_MODELS/
├── agent1_final.zip
├── agent2_final.zip
├── final_models.json
├── training_config_snapshot.json
└── touchdown_time_reward_state.json
```

The collected landing experiments are stored under:

```text
RL_training/statistics_data/BMW/
```

The available datasets include tests performed with multiple landing-surface scales, enabling analysis of:

- Landing success rate.
- Landing-position error.
- Flight and landing duration.
- Visual match, prediction, and target-loss percentages.
- Landing-ready handoff behavior.
- Failure reasons and vision-related operating limits.

---

## Project Scope

This repository represents a simulation-based research and engineering prototype. Its primary contribution is the design and evaluation of a modular perception-and-control architecture for autonomous UAV landing on a moving target.

The project demonstrates that separating target tracking and vertical landing into specialized policies can simplify learning and provide interpretable performance metrics. The current operating limits are primarily evaluated within the supplied simulation, camera, tracker, and sensor configuration.

---

<div align="center">

**Autonomous perception · Target tracking · Cooperative reinforcement learning · Precision landing**

</div>
README.md
13 KB