"""
Territorial.io AI Bot Package
==============================
A complete AI bot for Territorial.io with modular systems:

  - VisionSystem      : Real-time screen capture and territory classification (CNN)
  - BrainSystem       : Strategic decision making (Deep Q-Network / DQN)
  - ActionSystem      : Browser automation via Playwright
  - BotController     : Main orchestrator with start/stop CLI
  - EnemyTracker      : Individual enemy tracking and behavior analysis
  - SmartStrategy     : Advanced strategic decision making
  - LearningSystem    : Performance tracking and adaptive learning
  - PerformanceOptimizer : CPU/memory optimization and adaptive quality
  - AntiDetection     : Human-like behavior simulation
  - MapAnalyzer       : Advanced spatial analysis and expansion planning
  - StatsDashboard    : Real-time statistics aggregation and reporting

Quick Start:
    from territorial_bot.bot_controller import BotController
    bot = BotController(headless=False, enable_training=True)
    bot.start()
"""

# Core data structures (always importable)
from territorial_bot.vision_system import GameState, TerritoryCell
from territorial_bot.brain_system import Decision, Action

# Main systems
from territorial_bot.vision_system import VisionSystem
from territorial_bot.brain_system import BrainSystem
from territorial_bot.action_system import ActionSystem
from territorial_bot.bot_controller import BotController

# Advanced systems
from territorial_bot.enemy_tracker import EnemyTracker, EnemyPlayer
from territorial_bot.smart_strategy import SmartStrategy, StrategyType, StrategicSituation
from territorial_bot.learning_system import LearningSystem, GameRecord, StrategyPerformance
from territorial_bot.performance_optimizer import PerformanceOptimizer, SmartFrameSelector
from territorial_bot.anti_detection import AntiDetection, HumanBehaviorProfile

# New systems added in this integration
from territorial_bot.map_analyzer import MapAnalyzer, MapAnalysis, TerritoryCluster
from territorial_bot.stats_dashboard import StatsDashboard, SessionStats, EpisodeStats

# Configuration
from territorial_bot.config import (
    GAME_URL,
    BROWSER_CONFIG,
    VISION_CONFIG,
    BRAIN_CONFIG,
    ACTION_CONFIG,
    CONTROLLER_CONFIG,
    TRAINING_CONFIG,
    PLAYER_COLORS,
    COLOR_MATCH_TOLERANCE,
)

__version__ = "2.0.0"
__author__ = "Territorial.io AI Bot"

__all__ = [
    # Core data structures
    "GameState",
    "TerritoryCell",
    "Decision",
    "Action",

    # Main systems
    "VisionSystem",
    "BrainSystem",
    "ActionSystem",
    "BotController",

    # Advanced systems
    "EnemyTracker",
    "EnemyPlayer",
    "SmartStrategy",
    "StrategyType",
    "StrategicSituation",
    "LearningSystem",
    "GameRecord",
    "StrategyPerformance",
    "PerformanceOptimizer",
    "SmartFrameSelector",
    "AntiDetection",
    "HumanBehaviorProfile",

    # New systems
    "MapAnalyzer",
    "MapAnalysis",
    "TerritoryCluster",
    "StatsDashboard",
    "SessionStats",
    "EpisodeStats",

    # Configuration
    "GAME_URL",
    "BROWSER_CONFIG",
    "VISION_CONFIG",
    "BRAIN_CONFIG",
    "ACTION_CONFIG",
    "CONTROLLER_CONFIG",
    "TRAINING_CONFIG",
    "PLAYER_COLORS",
    "COLOR_MATCH_TOLERANCE",
]
