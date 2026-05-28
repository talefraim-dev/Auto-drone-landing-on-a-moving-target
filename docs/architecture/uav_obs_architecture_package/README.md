# UAV Final Project — Observation Architecture Package

Project: **Vision-Based Autonomous UAV Landing on a Moving Platform using Reinforcement Learning**

This package summarizes the current RL observation design and image-space kinematics plan for three separate agents:

1. Follow Agent — target tracking while maintaining safe distance
2. Static Landing Agent — landing on a stationary target
3. Moving Landing Agent — landing on a moving target

Core design principle:

- The UAV state may come from simulated onboard sensors / simulator ground truth.
- The target state should be estimated from vision: bounding box center, size, area, motion, and image-space dynamics.
- The same base observation vector is recommended for all agents, while each agent differs by reward function, curriculum, target behavior, and success/termination criteria.

