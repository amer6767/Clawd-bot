"""
Action System for Territorial.io Bot
======================================
Controls the browser using Playwright to send mouse movements,
clicks, and drags into the live game at https://territorial.io
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Tuple

try:
    from playwright.async_api import (
        async_playwright, Browser, BrowserContext, Page, Playwright
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")

from config import BROWSER_CONFIG, ACTION_CONFIG, GAME_URL
from brain_system import Decision, Action

logger = logging.getLogger(__name__)


class ActionSystem:
    """
    Browser automation layer for Territorial.io.

    Responsibilities:
      - Launch and manage a Playwright browser session
      - Navigate to https://territorial.io
      - Capture screenshots for the vision system
      - Execute mouse actions (move, click, drag) based on brain decisions
      - Handle game UI interactions (join game, restart, etc.)
    """

    def __init__(self, browser_cfg: dict = None, action_cfg: dict = None):
        self.browser_cfg = browser_cfg or BROWSER_CONFIG
        self.action_cfg = action_cfg or ACTION_CONFIG

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        self._canvas_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_action_time = 0.0
        self._is_running = False

        logger.info("ActionSystem initialized.")

    # ── Browser Lifecycle ──────────────────────────────────────────────────

    async def launch(self):
        """Launch browser and navigate to Territorial.io."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )

        logger.info("Launching browser...")
        self._playwright = await async_playwright().start()

        browser_type = self.browser_cfg.get("browser_type", "chromium")
        launcher = getattr(self._playwright, browser_type)

        self._browser = await launcher.launch(
            headless=self.browser_cfg.get("headless", False),
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                f"--window-size={self.browser_cfg['window_width']},"
                f"{self.browser_cfg['window_height']}",
            ],
        )

        self._context = await self._browser.new_context(
            viewport={
                "width": self.browser_cfg["window_width"],
                "height": self.browser_cfg["window_height"],
            },
            user_agent=self.browser_cfg.get("user_agent", ""),
        )

        self._page = await self._context.new_page()

        # Mask automation detection
        await self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        logger.info(f"Navigating to {GAME_URL}...")
        try:
            await self._page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            logger.warning(f"Page load warning (may be normal for SPA): {e}")
        await asyncio.sleep(3)

        # Locate the game canvas
        await self._locate_canvas()
        self._is_running = True
        logger.info("Browser launched and game loaded.")

    async def close(self):
        """Close browser and clean up resources."""
        self._is_running = False
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed.")

    # ── Canvas Detection ───────────────────────────────────────────────────

    async def _locate_canvas(self):
        """Find the game canvas element and store its bounding box."""
        try:
            canvas_selector = self.action_cfg.get("canvas_selector", "canvas")
            canvas = await self._page.query_selector(canvas_selector)
            if canvas:
                bbox = await canvas.bounding_box()
                if bbox:
                    self._canvas_bbox = (
                        int(bbox["x"]),
                        int(bbox["y"]),
                        int(bbox["width"]),
                        int(bbox["height"]),
                    )
                    logger.info(f"Canvas found: {self._canvas_bbox}")
                    return
        except Exception as e:
            logger.warning(f"Canvas detection error: {e}")

        # Fallback: use full viewport
        vp = self._page.viewport_size
        if vp:
            self._canvas_bbox = (0, 0, vp["width"], vp["height"])
        logger.warning("Using full viewport as canvas fallback.")

    # ── Screenshot Capture ─────────────────────────────────────────────────

    async def capture_screenshot(self) -> Optional[np.ndarray]:
        """
        Capture the current browser screenshot as a numpy BGR array.
        Returns None on failure.
        """
        if not self._page:
            return None
        try:
            png_bytes = await self._page.screenshot(type="png")
            import cv2
            nparr = np.frombuffer(png_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None

    def get_canvas_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the cached canvas bounding box (x, y, w, h)."""
        return self._canvas_bbox

    # ── Action Execution ───────────────────────────────────────────────────

    async def execute_decision(self, decision: Decision):
        """
        Execute a brain Decision by sending appropriate browser input.
        """
        if not self._page or not self._is_running:
            return

        # Enforce action cooldown
        now = time.time()
        cooldown = self.action_cfg["action_cooldown_ms"] / 1000.0
        elapsed = now - self._last_action_time
        if elapsed < cooldown:
            await asyncio.sleep(cooldown - elapsed)

        if decision.action_id == Action.STAY:
            logger.debug("Action: STAY (no movement)")
            return

        if decision.target_pixel is None:
            logger.debug(f"Action {decision.action_name}: no target pixel, skipping.")
            return

        tx, ty = decision.target_pixel
        logger.debug(f"Executing {decision.action_name} → pixel ({tx}, {ty})")

        await self._attack_move(tx, ty)
        self._last_action_time = time.time()

    async def _attack_move(self, target_x: int, target_y: int):
        """
        Perform an attack move: click and drag from own territory toward target.
        In Territorial.io, you click on your territory and drag toward the enemy.
        """
        if not self._page:
            return

        # Find a source point (center of own territory / canvas center as fallback)
        src_x, src_y = self._get_own_territory_center()

        steps = self.action_cfg.get("mouse_move_steps", 5)
        delay = self.action_cfg.get("mouse_move_delay_ms", 10) / 1000.0
        hold = self.action_cfg.get("click_hold_ms", 50) / 1000.0

        try:
            # Move to source
            await self._page.mouse.move(src_x, src_y)
            await asyncio.sleep(delay)

            # Press mouse button
            await self._page.mouse.down()
            await asyncio.sleep(hold)

            # Smooth drag to target
            for i in range(1, steps + 1):
                ix = int(src_x + (target_x - src_x) * i / steps)
                iy = int(src_y + (target_y - src_y) * i / steps)
                await self._page.mouse.move(ix, iy)
                await asyncio.sleep(delay)

            # Release
            await self._page.mouse.up()
            await asyncio.sleep(delay)

        except Exception as e:
            logger.error(f"Attack move failed: {e}")
            try:
                await self._page.mouse.up()
            except Exception:
                pass

    async def click_at(self, x: int, y: int):
        """Simple click at a canvas coordinate."""
        if not self._page:
            return
        try:
            await self._page.mouse.click(x, y)
        except Exception as e:
            logger.error(f"Click failed: {e}")

    async def move_mouse(self, x: int, y: int):
        """Move mouse to a canvas coordinate without clicking."""
        if not self._page:
            return
        try:
            await self._page.mouse.move(x, y)
        except Exception as e:
            logger.error(f"Mouse move failed: {e}")

    def _get_own_territory_center(self) -> Tuple[int, int]:
        """
        Return the pixel center of own territory.
        Falls back to canvas center if no territory info available.
        """
        if self._canvas_bbox:
            cx, cy, cw, ch = self._canvas_bbox
            return (cx + cw // 2, cy + ch // 2)
        vp = self._page.viewport_size if self._page else None
        if vp:
            return (vp["width"] // 2, vp["height"] // 2)
        return (640, 400)

    # ── Game UI Interactions ───────────────────────────────────────────────

    async def join_game(self, player_name: str = "AIBot"):
        """
        Attempt to join a game session on Territorial.io.
        Territorial.io uses a canvas-based UI; we interact via keyboard/mouse.
        """
        if not self._page:
            return

        logger.info(f"Attempting to join game as '{player_name}'...")

        # Wait for page to be fully loaded
        await asyncio.sleep(2)

        try:
            # territorial.io has a name input field - try multiple selectors
            name_selectors = [
                "input[type='text']",
                "input[placeholder*='name' i]",
                "input[placeholder*='Name' i]",
                "#name-input",
                ".name-input",
                "input[maxlength]",
            ]
            name_input = None
            for selector in name_selectors:
                try:
                    name_input = await self._page.wait_for_selector(
                        selector, timeout=2000
                    )
                    if name_input:
                        break
                except Exception:
                    continue

            if name_input:
                await name_input.triple_click()
                await name_input.fill(player_name)
                logger.info(f"Entered player name: {player_name}")
            else:
                logger.warning("Name input not found; skipping name entry.")
        except Exception as e:
            logger.warning(f"Name entry failed: {e}")

        try:
            # Try multiple play button selectors for territorial.io
            play_selectors = [
                "button:has-text('Play')",
                "button:has-text('Start')",
                "button:has-text('Join')",
                "#play-btn",
                ".play-button",
                ".btn-play",
                "input[type='submit']",
                "button[type='submit']",
            ]
            play_btn = None
            for selector in play_selectors:
                try:
                    play_btn = await self._page.wait_for_selector(
                        selector, timeout=2000
                    )
                    if play_btn:
                        break
                except Exception:
                    continue

            if play_btn:
                await play_btn.click()
                logger.info("Clicked play button.")
                await asyncio.sleep(3)
            else:
                # Fallback: try pressing Enter to start
                logger.info("Play button not found; trying Enter key to start game.")
                await self._page.keyboard.press("Enter")
                await asyncio.sleep(3)
        except Exception as e:
            logger.warning(f"Play button interaction failed: {e}")

        # Re-locate canvas after joining
        await self._locate_canvas()

    async def restart_game(self):
        """Restart the game after dying or game over."""
        if not self._page:
            return

        logger.info("Restarting game...")
        try:
            # Look for restart/respawn button
            restart_btn = await self._page.wait_for_selector(
                "button:has-text('Restart'), button:has-text('Respawn'), "
                "button:has-text('Play Again'), #restart-btn",
                timeout=5000
            )
            if restart_btn:
                await restart_btn.click()
                await asyncio.sleep(2)
                await self._locate_canvas()
                logger.info("Game restarted.")
                return
        except Exception:
            pass

        # Fallback: reload page
        logger.info("Reloading page for restart...")
        try:
            await self._page.reload(wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            logger.warning(f"Page reload warning: {e}")
        await asyncio.sleep(3)
        await self._locate_canvas()

    async def is_game_over(self) -> bool:
        """
        Check if the game is over (player died or game ended).
        Looks for game-over UI elements.
        """
        if not self._page:
            return False
        try:
            # Check for common game-over indicators
            game_over = await self._page.query_selector(
                ".game-over, #game-over, "
                "[class*='gameover'], [class*='game_over'], "
                "button:has-text('Play Again'), button:has-text('Respawn')"
            )
            return game_over is not None
        except Exception:
            return False

    async def set_troop_ratio(self, ratio: float):
        """
        Set the troop allocation ratio (0.0 to 1.0).
        Interacts with the troop slider if present.
        """
        if not self._page:
            return
        ratio = max(0.0, min(1.0, ratio))
        try:
            slider = await self._page.query_selector(
                "input[type='range'], .troop-slider, #troop-slider"
            )
            if slider:
                bbox = await slider.bounding_box()
                if bbox:
                    x = bbox["x"] + bbox["width"] * ratio
                    y = bbox["y"] + bbox["height"] / 2
                    await self._page.mouse.click(x, y)
        except Exception as e:
            logger.debug(f"Troop slider interaction failed: {e}")

    # ── Keyboard Controls ──────────────────────────────────────────────────

    async def press_key(self, key: str):
        """Press a keyboard key."""
        if not self._page:
            return
        try:
            await self._page.keyboard.press(key)
        except Exception as e:
            logger.error(f"Key press failed: {e}")

    async def pause_game(self):
        """Press the pause key."""
        await self._page.keyboard.press(
            self.action_cfg.get("pause_key", "Escape")
        )

    # ── Status ─────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def get_page_title(self) -> str:
        """Return the current page title."""
        if not self._page:
            return ""
        return await self._page.title()

    async def wait_for_game_load(self, timeout_ms: int = 15000):
        """Wait until the game canvas is visible and ready."""
        if not self._page:
            return
        try:
            await self._page.wait_for_selector(
                self.action_cfg.get("canvas_selector", "canvas"),
                timeout=timeout_ms
            )
            await asyncio.sleep(1)
            await self._locate_canvas()
            logger.info("Game canvas is ready.")
        except Exception as e:
            logger.warning(f"Game load wait timed out: {e}")
