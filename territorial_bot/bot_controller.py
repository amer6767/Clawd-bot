"""
Bot Controller for Territorial.io
====================================
Main orchestrator that ties together the Vision, Brain, and Action systems.
Provides start/stop control and the main game loop.

Usage:
    python bot_controller.py                  # Run with defaults
    python bot_controller.py --headless       # Run headless
    python bot_controller.py --train          # Enable online training
    python bot_controller.py --debug          # Show debug overlay
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
    Main controller that orchestrates the three bot systems:
      1. VisionSystem  – processes screenshots into GameState
      2. BrainSystem   – converts GameState into Decisions
      3. ActionSystem  – executes Decisions in the browser

    The main loop runs at CONTROLLER_CONFIG["tick_rate_ms"] intervals.
    """

    def __init__(
        self,
        headless: bool = False,
        enable_training: bool = False,
        debug_overlay: bool = False,
        player_name: str = "AIBot",
    ):
        self.headless = headless
        self.enable_training = enable_training
        self.debug_overlay = debug_overlay
        self.player_name = player_name

        self.cfg = CONTROLLER_CONFIG

        # Override headless setting
        browser_cfg = dict(BROWSER_CONFIG)
        browser_cfg["headless"] = headless

        # Initialize systems
        self.vision = VisionSystem(VISION_CONFIG)
        self.brain = BrainSystem(BRAIN_CONFIG)
        self.action = ActionSystem(browser_cfg, ACTION_CONFIG)

        # State tracking
        self._running = False
        self._paused = False
        self._tick_count = 0
        self._episode_count = 0
        self._start_time: Optional[float] = None
        self._prev_game_state: Optional[GameState] = None

        # Stats
        self._total_rewards = 0.0
        self._episode_rewards = 0.0
        self._best_territory = 0.0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"BotController initialized | "
            f"headless={headless} | training={enable_training} | "
            f"debug={debug_overlay}"
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

            logger.info("=" * 60)
            logger.info("Bot is now running. Press Ctrl+C to stop.")
            logger.info("=" * 60)

            await self._game_loop()

        except Exception as e:
            logger.error(f"Fatal error in bot: {e}", exc_info=True)
        finally:
            await self._shutdown()

    async def _game_loop(self):
        """Main game loop: capture → vision → brain → action → repeat."""
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
        """Execute one bot tick: capture → process → decide → act."""
        self._tick_count += 1

        # ── 1. Capture Screenshot ──────────────────────────────────────────
        screenshot = await self.action.capture_screenshot()
        if screenshot is None:
            logger.warning("Screenshot capture returned None; skipping tick.")
            return

        # ── 2. Vision Processing ───────────────────────────────────────────
        canvas_bbox = self.action.get_canvas_bbox()
        game_state = self.vision.process_frame(screenshot, canvas_bbox)

        # ── 3. Debug Overlay ───────────────────────────────────────────────
        if self.debug_overlay and self._tick_count % 5 == 0:
            overlay = self.vision.draw_debug_overlay(game_state)
            cv2.imshow("Territorial.io Bot Debug", overlay)
            cv2.waitKey(1)

        # ── 4. Save Screenshot (periodic) ─────────────────────────────────
        save_interval = self.cfg.get("screenshot_save_interval", 50)
        if self._tick_count % save_interval == 0:
            self._save_screenshot(screenshot, game_state)

        # ── 5. Check Game Over ─────────────────────────────────────────────
        if await self.action.is_game_over():
            await self._handle_game_over(game_state)
            return

        # ── 6. Brain Decision ──────────────────────────────────────────────
        decision = self.brain.decide(game_state)

        # ── 7. Training: Store Transition ──────────────────────────────────
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

        # ── 8. Execute Action ──────────────────────────────────────────────
        await self.action.execute_decision(decision)

        # ── 9. Update State ────────────────────────────────────────────────
        self._prev_game_state = game_state

        # Track best territory
        if game_state.own_territory_pct > self._best_territory:
            self._best_territory = game_state.own_territory_pct

        # ── 10. Periodic Logging ───────────────────────────────────────────
        if self._tick_count % 20 == 0:
            self._log_status(game_state, decision)

    # ── Game Over Handling ─────────────────────────────────────────────────

    async def _handle_game_over(self, game_state: GameState):
        """Handle end-of-game: store final transition, update epsilon, restart."""
        logger.info(
            f"Game over! Episode {self._episode_count} | "
            f"Best territory: {self._best_territory:.1%} | "
            f"Episode reward: {self._episode_rewards:.2f}"
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

    # ── Shutdown ───────────────────────────────────────────────────────────

    async def _shutdown(self):
        """Clean up all resources."""
        logger.info("Shutting down bot...")

        if self.enable_training:
            self.brain.save_model()
            logger.info("Final model saved.")

        if self.debug_overlay:
            cv2.destroyAllWindows()

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

    def _log_status(self, game_state: GameState, decision: Decision):
        """Log current bot status."""
        summary = self.brain.get_strategy_summary(game_state)
        logger.info(
            f"Tick {self._tick_count:5d} | "
            f"Phase: {summary['game_phase']:5s} | "
            f"Own: {summary['own_territory']:6s} | "
            f"Enemy: {summary['enemy_territory']:6s} | "
            f"Action: {decision.action_name:12s} | "
            f"ε: {summary['epsilon']}"
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
    )

    bot.start()
