# Clawd-bot / Territorial.io AI Bot

An advanced AI bot for [Territorial.io](https://territorial.io) with modular, extensible architecture.

## Overview

This bot uses computer vision, deep reinforcement learning, and strategic AI to play Territorial.io autonomously.

## Architecture

```
territorial_bot/
├── config.py                    # All tunable settings
├── vision_system.py             # Screen capture + CNN territory classifier
├── brain_system.py              # DQN decision-making AI
├── action_system.py             # Playwright browser controller
├── bot_controller.py            # Main orchestrator (integrates all systems)
├── enemy_tracker.py             # Individual enemy tracking & behavior analysis
├── smart_strategy.py            # Advanced strategic decision making
├── learning_system.py           # Performance tracking & adaptive learning
├── performance_optimizer.py     # CPU/memory optimization & adaptive quality
├── anti_detection.py            # Human-like behavior simulation
├── map_analyzer.py              # Advanced spatial analysis & expansion planning
├── stats_dashboard.py           # Real-time statistics & reporting
├── requirements.txt             # Python dependencies
├── models/                      # Trained model weights
├── kaggle_notebooks/            # Kaggle training notebooks
├── screenshots/                 # Debug screenshots
└── logs/                        # Bot logs & session reports
```

## Quick Start

```bash
cd territorial_bot
pip install -r requirements.txt
playwright install chromium

# Run with all systems enabled
python bot_controller.py --name "MyBot"

# Run with online training
python bot_controller.py --train --name "MyBot"

# Run headless with debug overlay
python bot_controller.py --headless --debug
```

## Systems

| System | Description |
|--------|-------------|
| **VisionSystem** | CNN + OpenCV territory detection |
| **BrainSystem** | DQN reinforcement learning decisions |
| **ActionSystem** | Playwright browser automation |
| **EnemyTracker** | Per-enemy behavior profiling |
| **SmartStrategy** | Defense, corner, and alliance strategies |
| **LearningSystem** | Win/loss tracking and adaptive parameters |
| **PerformanceOptimizer** | Adaptive quality and frame skipping |
| **AntiDetection** | Human-like mouse movement and timing |
| **MapAnalyzer** | Cluster detection, chokepoints, expansion paths |
| **StatsDashboard** | Real-time metrics and session reports |

See [`territorial_bot/README.md`](territorial_bot/README.md) for full documentation.
