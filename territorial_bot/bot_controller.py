"""
Bot Controller for Territorial.io
====================================
Main orchestrator that ties together all bot systems:
  - VisionSystem       – processes screenshots into GameState
  - BrainSystem        – converts GameState into Decisions (DQN)
  - ActionSystem       – executes Decisions in the browser
  - EnemyTracker       – tracks individual enemies and their behavior
  - SmartStrategy      – advanced strategic decision making
  - LearningSystem     – tracks performance and adapts strategy
  - PerformanceOptimizer – adaptive quality and frame management
  - AntiDetection      – human-like behavior simulation
  - MapAnalyzer        – advanced spatial analysis
  - StatsDashboard     – real-time statistics and reporting

Usage:
    python bot_controller.py                  # Run with defaults
    python bot_controller.py --headless       # Run headless
    python bot_controller.py --train          # Enable online training
    python bot_controller.py --debug          # Show debug overlay
    python bot_controller.py --no-anti-detect # Disable anti-detection
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import CONTROLLER_CONFIG, BROWSER_CONFIG, ACTION_CONFIG, BRAIN_CONFIG, VISION_CONFIG
from vision_system import VisionSystem, GameState
from brain_system import BrainSystem, Decision
from action_system import ActionSystem
from enemy_tracker import EnemyTracker
from smart_strategy import SmartStrategy, StrategyType
from learning_system import LearningSystem
from performance_optimizer import PerformanceOptimizer, SmartFrameSelector
from anti_detection import AntiDetection
from map_analyzer import MapAnalyzer, MapAnalysis
from stats_dashboard import StatsDashboard

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: str = "logs/bot.log"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger("BotController")


# ─────────────────────────────────────────────
# BOT CONTROLLER
# ─────────────────────────────────────────────

class BotController:
    """
    Main controller that orchestrates all bot systems.

    The main loop runs at CONTROLLER_CONFIG["tick_rate_ms"] intervals.
    Each tick:
      1. Capture screenshot (ActionSystem)
      2. Process frame (VisionSystem → GameState)
      3. Analyze map (MapAnalyzer → MapAnalysis)
      4. Track enemies (EnemyTracker)
      5. Assess situation (SmartStrategy)
      6. Make decision (BrainSystem, informed by strategy)
      7. Apply anti-detection (AntiDetection)
      8. Execute action (ActionSystem)
      9. Record tick (LearningSystem, StatsDashboard)
      10. Optimize performance (PerformanceOptimizer)
    """

    def __init__(
        self,
        headless: bool = False,
        enable_training: bool = False,
        debug_overlay: bool = False,
        player_name: str = "AIBot",
        enable_anti_detection: bool = True,
    ):
        self.headless = headless
        self.enable_training = enable_training
        self.debug_overlay = debug_overlay
        self.player_name = player_name
        self.enable_anti_detection = enable_anti_detection

        self.cfg = CONTROLLER_CONFIG

        # Override headless setting
        browser_cfg = dict(BROWSER_CONFIG)
        browser_cfg["headless"] = headless

        # ── Core Systems ──────────────────────────────────────────────────
        self.vision = VisionSystem(VISION_CONFIG)
        self.brain = BrainSystem(BRAIN_CONFIG)
        self.action = ActionSystem(browser_cfg, ACTION_CONFIG)

        # ── Advanced Systems ──────────────────────────────────────────────
        self.enemy_tracker = EnemyTracker()
        self.smart_strategy = SmartStrategy()
        self.learning = LearningSystem()
        self.perf_optimizer = PerformanceOptimizer(target_fps=10)
        self.frame_selector = SmartFrameSelector(motion_threshold=3.0)
        self.anti_detection = AntiDetection() if enable_anti_detection else None
        self.map_analyzer = MapAnalyzer()
        self.dashboard = StatsDashboard(
            log_interval_ticks=self.cfg.get("screenshot_save_interval", 50)
        )

        # Load previous stats
        self.dashboard.load_stats()

        # ── State Tracking ────────────────────────────────────────────────
        self._running = False
        self._paused = False
        self._tick_count = 0
        self._episode_count = 0
        self._start_time: Optional[float] = None
        self._prev_game_state: Optional[GameState] = None
        self._last_map_analysis: Optional[MapAnalysis] = None

        # Stats
        self._total_rewards = 0.0
        self._episode_rewards = 0.0
        self._best_territory = 0.0
        self._last_tick_time = 0.0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"BotController initialized | "
            f"headless={headless} | training={enable_training} | "
            f"debug={debug_overlay} | anti_detect={enable_anti_detection}"
        )

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and SIGTERM gracefully."""
        logger.info(f"Signal {signum} received. Stopping bot...")
        self.stop()

    # ── Public Controls ────────────────────────────────────────────────────

    def start(self):
        """Start the bot (blocking call)."""
        logger.info("Starting Territorial.io Bot...")
        asyncio.run(self._run())

    def stop(self):
        """Signal the bot to stop."""
        logger.info("Stop requested.")
        self._running = False

    def pause(self):
        """Pause the bot (keeps browser open)."""
        self._paused = True
        logger.info("Bot paused.")

    def resume(self):
        """Resume the bot after pausing."""
        self._paused = False
        logger.info("Bot resumed.")

    # ── Main Async Loop ────────────────────────────────────────────────────

    async def _run(self):
        """Main async entry point."""
        try:
            await self.action.launch()
            await self.action.join_game(self.player_name)
            await self.action.wait_for_game_load()

            self._running = True
            self._start_time = time.time()
            self._episode_count = 1

            # Start first episode tracking
            self.learning.start_new_game(f"game_{self._episode_count}")
            self.dashboard.start_episode(self._episode_count)

            logger.info("=" * 60)
            logger.info("Bot is now running. Press Ctrl+C to stop.")
            logger.info("=" * 60)

            await self._game_loop()

        except Exception as e:
            logger.error(f"Fatal error in bot: {e}", exc_info=True)
        finally:
            await self._shutdown()

    async def _game_loop(self):
        """Main game loop: capture → vision → analyze → decide → act → repeat."""
        tick_interval = self.cfg["tick_rate_ms"] / 1000.0
        max_runtime = self.cfg.get("max_runtime_seconds", 0)

        while self._running:
            loop_start = time.time()

            # Check max runtime
            if max_runtime > 0:
                elapsed = time.time() - self._start_time
                if elapsed >= max_runtime:
                    logger.info(f"Max runtime ({max_runtime}s) reached. Stopping.")
                    break

            # Pause handling
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)

            # Maintain tick rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, tick_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _tick(self):
        """Execute one bot tick: capture → process → analyze → decide → act."""
        tick_start = time.time()
        self._tick_count += 1

        # ── 1. Capture Screenshot ──────────────────────────────────────────
        screenshot = await self.action.capture_screenshot()
        if screenshot is None:
            logger.warning("Screenshot capture returned None; skipping tick.")
            return

        # ── 2. Performance: Frame Selection ───────────────────────────────
        # Use SmartFrameSelector to skip redundant frames
        if not self.frame_selector.should_process(screenshot):
            return

        # ── 3. Performance: Optimize Frame ────────────────────────────────
        optimized_frame = self.perf_optimizer.optimize_frame(screenshot)

        # ── 4. Vision Processing ───────────────────────────────────────────
        canvas_bbox = self.action.get_canvas_bbox()
        game_state = self.vision.process_frame(optimized_frame, canvas_bbox)

        # ── 5. Map Analysis (every 5 ticks for performance) ───────────────
        if self._tick_count % 5 == 0 or self._last_map_analysis is None:
            self._last_map_analysis = self.map_analyzer.analyze(game_state)

            # Override best_attack_target with map-analyzer's recommendation
            if self._last_map_analysis.best_expansion_paths:
                game_state.best_attack_target = (
                    self._last_map_analysis.best_expansion_paths[0].end
                )

        map_analysis = self._last_map_analysis

        # ── 6. Enemy Tracking ──────────────────────────────────────────────
        detected_enemies = self.enemy_tracker.detect_enemies(game_state)
        self.enemy_tracker.cleanup_stale_enemies()

        # ── 7. Strategic Situation Assessment ─────────────────────────────
        situation = self.smart_strategy.analyze_situation(
            game_state, self.enemy_tracker
        )

        # Get adaptive parameters from learning system
        adaptive_params = self.learning.get_adaptive_parameters()

        # Determine strategy
        strategy = self.smart_strategy.determine_strategy(
            situation, game_state, adaptive_params
        )
        self.smart_strategy.tick()

        # ── 8. Debug Overlay ───────────────────────────────────────────────
        if self.debug_overlay and self._tick_count % 5 == 0:
            overlay = self.vision.draw_debug_overlay(game_state)
            cv2.imshow("Territorial.io Bot Debug", overlay)
            cv2.waitKey(1)

        # ── 9. Save Screenshot (periodic) ─────────────────────────────────
        save_interval = self.cfg.get("screenshot_save_interval", 50)
        if self._tick_count % save_interval == 0:
            self._save_screenshot(screenshot, game_state)

        # ── 10. Check Game Over ────────────────────────────────────────────
        if await self.action.is_game_over():
            await self._handle_game_over(game_state)
            return

        # ── 11. Brain Decision ─────────────────────────────────────────────
        # Adjust brain's aggression based on strategy and adaptive params
        original_aggression = self.brain.cfg.get("aggression", 0.65)
        strategy_params = self.smart_strategy.get_strategy_action_params(strategy)

        # Temporarily blend strategy aggression with adaptive params
        blended_aggression = (
            strategy_params["aggression"] * 0.5
            + adaptive_params.get("aggression", original_aggression) * 0.5
        )
        self.brain.cfg["aggression"] = blended_aggression

        # Override attack target based on strategy
        if strategy == StrategyType.DEFEND and situation.is_being_attacked:
            defense_target = self.smart_strategy.get_defense_target(situation, game_state)
            if defense_target:
                game_state.best_attack_target = defense_target

        elif strategy == StrategyType.CORNER:
            corner_target = self.smart_strategy.get_corner_strategy_target(
                situation, game_state
            )
            if corner_target:
                game_state.best_attack_target = corner_target

        elif strategy == StrategyType.NEUTRAL_ALLIANCE:
            neutral_target = self.smart_strategy.get_neutral_alliance_target(
                situation, game_state
            )
            if neutral_target:
                game_state.best_attack_target = neutral_target

        decision = self.brain.decide(game_state)

        # Restore original aggression
        self.brain.cfg["aggression"] = original_aggression

        # ── 12. Anti-Detection: Apply Human-like Behavior ─────────────────
        if self.anti_detection and self.enable_anti_detection:
            self.anti_detection.record_action(decision.action_name)

            # Occasionally vary the action to avoid patterns
            if self.anti_detection.should_vary_pattern(decision.action_name):
                alt_action_name = self.anti_detection.get_alternative_action(
                    decision.action_name
                )
                logger.debug(
                    f"Anti-detection: varying {decision.action_name} → {alt_action_name}"
                )
                # Note: We log the variation but still execute the brain's decision
                # to avoid degrading gameplay quality. The variation is mainly for
                # timing and movement path randomization.

            # Apply human-like delay
            delay = self.anti_detection.get_action_delay(decision.action_name)
            if delay > 0.05:  # Only add meaningful delays
                await asyncio.sleep(min(delay, 0.3))  # Cap at 300ms

        # ── 13. Training: Store Transition ────────────────────────────────
        if self.enable_training and self._prev_game_state is not None:
            reward = self.brain.compute_reward(
                self._prev_game_state, game_state, done=False
            )
            self._episode_rewards += reward
            self._total_rewards += reward

            self.brain.store_transition(
                self._prev_game_state,
                decision.action_id,
                reward,
                game_state,
                done=False,
            )
            loss = self.brain.train_step()
            if loss is not None and self._tick_count % 100 == 0:
                logger.debug(f"Training loss: {loss:.4f}")
        else:
            reward = 0.0

        # ── 14. Execute Action ─────────────────────────────────────────────
        await self.action.execute_decision(decision)

        # ── 15. Update State ───────────────────────────────────────────────
        self._prev_game_state = game_state

        # Track best territory
        if game_state.own_territory_pct > self._best_territory:
            self._best_territory = game_state.own_territory_pct

        # ── 16. Record Tick in Learning System ────────────────────────────
        self.learning.record_game_tick(
            territory_pct=game_state.own_territory_pct,
            action=decision.action_name,
        )

        # ── 17. Update Stats Dashboard ────────────────────────────────────
        tick_duration = time.time() - tick_start
        fps = 1.0 / max(0.001, tick_duration)
        self.perf_optimizer.update_performance_metrics(tick_duration)

        self.dashboard.update(
            tick=self._tick_count,
            game_state=game_state,
            decision=decision,
            fps=fps,
            epsilon=self.brain.epsilon,
            vulnerability=map_analysis.vulnerability_score if map_analysis else 0.0,
            expansion_opportunity=map_analysis.expansion_opportunity if map_analysis else 0.0,
            reward=reward,
        )

        # ── 18. Periodic Logging ───────────────────────────────────────────
        if self._tick_count % 20 == 0:
            self._log_status(game_state, decision, strategy, map_analysis)

        # ── 19. Periodic Performance Report ───────────────────────────────
        if self._tick_count % 200 == 0:
            self.perf_optimizer.log_performance()

    # ── Game Over Handling ─────────────────────────────────────────────────

    async def _handle_game_over(self, game_state: GameState):
        """Handle end-of-game: store final transition, update epsilon, restart."""
        won = game_state.own_territory_pct > 0.5  # Heuristic: >50% = win
        logger.info(
            f"Game over! Episode {self._episode_count} | "
            f"Best territory: {self._best_territory:.1%} | "
            f"Episode reward: {self._episode_rewards:.2f} | "
            f"{'WIN' if won else 'LOSS'}"
        )

        # Store terminal transition
        if self.enable_training and self._prev_game_state is not None:
            reward = self.brain.compute_reward(
                self._prev_game_state, game_state, done=True
            )
            self.brain.store_transition(
                self._prev_game_state,
                0,  # STAY action for terminal state
                reward,
                game_state,
                done=True,
            )
            self.brain.train_step()
            self.brain.update_epsilon()
            self.brain.episode += 1

            # Update target network periodically
            if self._episode_count % self.brain.cfg["target_update_freq"] == 0:
                self.brain.update_target_network()
                logger.info("Target network updated.")

            # Save model checkpoint
            if self._episode_count % 10 == 0:
                self.brain.save_model()

        # Record game end in learning system
        detected_enemies = self.enemy_tracker.detect_enemies(game_state)
        self.learning.end_game(
            won=won,
            final_territory_pct=game_state.own_territory_pct,
            enemies_encountered=self.enemy_tracker.total_enemies_encountered,
            final_enemy_count=len(detected_enemies),
        )

        # Record episode end in dashboard
        self.dashboard.end_episode(
            won=won,
            final_territory=game_state.own_territory_pct,
            total_reward=self._episode_rewards,
        )

        # Reset episode stats
        self._episode_rewards = 0.0
        self._best_territory = 0.0
        self._prev_game_state = None
        self._episode_count += 1

        # Auto-restart
        if self.cfg.get("auto_restart", True) and self._running:
            restart_delay = self.cfg.get("restart_delay_seconds", 3)
            logger.info(f"Restarting in {restart_delay}s...")
            await asyncio.sleep(restart_delay)
            await self.action.restart_game()
            await self.action.join_game(self.player_name)

            # Start new episode tracking
            self.learning.start_new_game(f"game_{self._episode_count}")
            self.dashboard.start_episode(self._episode_count)

    # ── Shutdown ───────────────────────────────────────────────────────────

    async def _shutdown(self):
        """Clean up all resources."""
        logger.info("Shutting down bot...")

        if self.enable_training:
            self.brain.save_model()
            logger.info("Final model saved.")

        if self.debug_overlay:
            cv2.destroyAllWindows()

        # Export final stats report
        try:
            report_path = self.dashboard.export_full_report()
            logger.info(f"Session report exported to {report_path}")
        except Exception as e:
            logger.warning(f"Could not export report: {e}")

        # Print dashboard summary
        self.dashboard.print_summary()

        await self.action.close()

        runtime = time.time() - (self._start_time or time.time())
        logger.info(
            f"Bot stopped. "
            f"Runtime: {runtime:.0f}s | "
            f"Ticks: {self._tick_count} | "
            f"Episodes: {self._episode_count} | "
            f"Total reward: {self._total_rewards:.2f}"
        )

    # ── Logging ────────────────────────────────────────────────────────────

    def _log_status(
        self,
        game_state: GameState,
        decision: Decision,
        strategy: StrategyType,
        map_analysis: Optional[MapAnalysis],
    ):
        """Log current bot status with all system information."""
        summary = self.brain.get_strategy_summary(game_state)
        strategy_summary = self.smart_strategy.get_strategy_summary()
        live_status = self.dashboard.get_live_status_line()

        vuln = map_analysis.vulnerability_score if map_analysis else 0.0
        opp = map_analysis.expansion_opportunity if map_analysis else 0.0

        logger.info(
            f"Tick {self._tick_count:5d} | "
            f"Phase: {summary['game_phase']:5s} | "
            f"Own: {summary['own_territory']:6s} | "
            f"Enemy: {summary['enemy_territory']:6s} | "
            f"Action: {decision.action_name:12s} | "
            f"Strategy: {strategy_summary['current_strategy']:12s} | "
            f"Vuln: {vuln:.2f} | "
            f"Opp: {opp:.2f} | "
            f"ε: {summary['epsilon']} | "
            f"{live_status}"
        )

    def _save_screenshot(self, screenshot: np.ndarray, game_state: GameState):
        """Save a debug screenshot to disk."""
        try:
            Path("screenshots").mkdir(exist_ok=True)
            ts = int(time.time())
            path = f"screenshots/tick_{self._tick_count:06d}_{ts}.png"
            cv2.imwrite(path, screenshot)
        except Exception as e:
            logger.debug(f"Screenshot save failed: {e}")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Territorial.io AI Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot_controller.py                    # Run with visible browser
  python bot_controller.py --headless         # Run without browser window
  python bot_controller.py --train            # Enable online RL training
  python bot_controller.py --debug            # Show vision debug overlay
  python bot_controller.py --name MyBot       # Set player name
  python bot_controller.py --train --debug    # Train with debug view
  python bot_controller.py --no-anti-detect   # Disable anti-detection
        """
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run browser in headless mode (no visible window)"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Enable online reinforcement learning during gameplay"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show real-time vision debug overlay window"
    )
    parser.add_argument(
        "--name", type=str, default="AIBot",
        help="Player name to use in the game (default: AIBot)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level"
    )
    parser.add_argument(
        "--no-anti-detect", action="store_true",
        help="Disable anti-detection (human-like behavior simulation)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_logging(
        log_level=args.log_level,
        log_file=CONTROLLER_CONFIG.get("log_file", "logs/bot.log")
    )

    bot = BotController(
        headless=args.headless,
        enable_training=args.train,
        debug_overlay=args.debug,
        player_name=args.name,
        enable_anti_detection=not args.no_anti_detect,
    )

    bot.start()
