"""
Vision System for Territorial.io Bot
=====================================
Handles real-time screen capture, image processing, and territory detection.
Uses OpenCV for image processing and a trained CNN for region classification.
"""

import cv2
import numpy as np
import time
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List, NamedTuple
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Vision model inference disabled.")

from territorial_bot.config import VISION_CONFIG, PLAYER_COLORS, COLOR_MATCH_TOLERANCE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TerritoryCell:
    """Represents one grid cell of the game map."""
    row: int
    col: int
    owner: str          # "own", "enemy", "neutral", "border", "unknown"
    color_rgb: Tuple[int, int, int] = (0, 0, 0)
    troop_count: int = 0
    confidence: float = 0.0


@dataclass
class GameState:
    """Full parsed game state from a single frame."""
    timestamp: float = 0.0
    frame: Optional[np.ndarray] = None
    grid: List[List[TerritoryCell]] = field(default_factory=list)

    own_territory_pct: float = 0.0
    enemy_territory_pct: float = 0.0
    neutral_territory_pct: float = 0.0

    own_cell_count: int = 0
    enemy_cell_count: int = 0
    neutral_cell_count: int = 0

    # Detected border regions (list of (row, col) tuples)
    own_border_cells: List[Tuple[int, int]] = field(default_factory=list)
    enemy_border_cells: List[Tuple[int, int]] = field(default_factory=list)

    # Weakest enemy border (best attack target)
    best_attack_target: Optional[Tuple[int, int]] = None

    # Canvas bounding box in browser window
    canvas_x: int = 0
    canvas_y: int = 0
    canvas_w: int = 1280
    canvas_h: int = 800

    # Game phase
    game_phase: str = "early"  # "early", "mid", "late"


# ─────────────────────────────────────────────
# CNN MODEL DEFINITION (must match Kaggle training)
# ─────────────────────────────────────────────

class TerritoryClassifierCNN(nn.Module):
    """
    Lightweight CNN for classifying game map patches.
    Input: 64x64 RGB patch
    Output: 4-class softmax (own, enemy, neutral, border)
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 32x32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 16x16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─────────────────────────────────────────────
# VISION SYSTEM CLASS
# ─────────────────────────────────────────────

class VisionSystem:
    """
    Real-time vision system for Territorial.io.

    Workflow:
      1. Receive a screenshot (numpy array) from the browser
      2. Locate the game canvas within the screenshot
      3. Divide canvas into a grid
      4. Classify each cell using color heuristics + CNN model
      5. Return a structured GameState object
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or VISION_CONFIG
        self.model: Optional[TerritoryClassifierCNN] = None
        self.device = "cpu"
        self.transform = None
        self.label_map = {0: "own", 1: "enemy", 2: "neutral", 3: "border"}

        self._last_frame_time = 0.0
        self._frame_count = 0

        self._load_model()
        self._setup_transform()

        logger.info("VisionSystem initialized.")

    # ── Model Loading ──────────────────────────────────────────────────────

    def _load_model(self):
        """Load trained CNN model from disk if available."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable; using color-only detection.")
            return

        model_path = Path(self.cfg["vision_model_path"])
        if not model_path.exists():
            logger.warning(
                f"Vision model not found at {model_path}. "
                "Using color heuristics only. Train on Kaggle first."
            )
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = TerritoryClassifierCNN(
                num_classes=self.cfg.get("vision_num_classes", 4)
            )
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Vision model loaded from {model_path} on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            self.model = None

        # Load label encoder if present
        label_path = Path(self.cfg.get("label_encoder_path", ""))
        if label_path.exists():
            with open(label_path, "rb") as f:
                encoder = pickle.load(f)
            self.label_map = {i: cls for i, cls in enumerate(encoder.classes_)}

    def _setup_transform(self):
        """Set up image preprocessing pipeline for CNN input."""
        if not TORCH_AVAILABLE:
            return
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # ── Main Processing Pipeline ───────────────────────────────────────────

    def process_frame(self, screenshot: np.ndarray,
                      canvas_bbox: Tuple[int, int, int, int] = None) -> GameState:
        """
        Process a single screenshot and return a GameState.

        Args:
            screenshot: Full browser screenshot as BGR numpy array (H x W x 3)
            canvas_bbox: (x, y, w, h) of game canvas in screenshot.
                         If None, uses full screenshot.

        Returns:
            GameState with all territory information populated.
        """
        now = time.time()
        state = GameState(timestamp=now)

        # Extract canvas region
        if canvas_bbox:
            cx, cy, cw, ch = canvas_bbox
            state.canvas_x, state.canvas_y = cx, cy
            state.canvas_w, state.canvas_h = cw, ch
            canvas = screenshot[cy:cy+ch, cx:cx+cw]
        else:
            canvas = screenshot
            state.canvas_w = screenshot.shape[1]
            state.canvas_h = screenshot.shape[0]

        # Resize for consistent processing
        target_w = self.cfg["frame_width"]
        target_h = self.cfg["frame_height"]
        canvas_resized = cv2.resize(canvas, (target_w, target_h))
        state.frame = canvas_resized

        # Build grid
        grid = self._build_grid(canvas_resized)
        state.grid = grid

        # Compute statistics
        self._compute_stats(state)

        # Identify borders and attack targets
        self._find_borders(state)
        self._find_best_attack_target(state)

        # Determine game phase
        state.game_phase = self._determine_game_phase(state)

        self._frame_count += 1
        self._last_frame_time = now

        return state

    # ── Grid Building ──────────────────────────────────────────────────────

    def _build_grid(self, canvas: np.ndarray) -> List[List[TerritoryCell]]:
        """Divide canvas into grid cells and classify each."""
        rows = self.cfg["grid_rows"]
        cols = self.cfg["grid_cols"]
        h, w = canvas.shape[:2]
        cell_h = h // rows
        cell_w = w // cols

        grid = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                y1 = r * cell_h
                y2 = y1 + cell_h
                x1 = c * cell_w
                x2 = x1 + cell_w
                patch = canvas[y1:y2, x1:x2]

                cell = self._classify_cell(patch, r, c)
                row_cells.append(cell)
            grid.append(row_cells)

        return grid

    def _classify_cell(self, patch: np.ndarray, row: int, col: int) -> TerritoryCell:
        """Classify a single grid cell patch."""
        # Get dominant color
        avg_bgr = patch.mean(axis=(0, 1)).astype(int)
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))

        # Try CNN first if available
        if self.model is not None and TORCH_AVAILABLE:
            owner, confidence = self._cnn_classify(patch)
        else:
            owner, confidence = self._color_classify(avg_rgb)

        return TerritoryCell(
            row=row,
            col=col,
            owner=owner,
            color_rgb=avg_rgb,
            confidence=confidence,
        )

    def _cnn_classify(self, patch: np.ndarray) -> Tuple[str, float]:
        """Use CNN model to classify a patch."""
        try:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(patch_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)

            label = self.label_map.get(pred.item(), "unknown")
            confidence = conf.item()

            if confidence < self.cfg["confidence_threshold"]:
                # Fall back to color heuristic
                avg_bgr = patch.mean(axis=(0, 1)).astype(int)
                avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
                return self._color_classify(avg_rgb)

            return label, confidence
        except Exception as e:
            logger.debug(f"CNN classify error: {e}")
            avg_bgr = patch.mean(axis=(0, 1)).astype(int)
            avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
            return self._color_classify(avg_rgb)

    def _color_classify(self, rgb: Tuple[int, int, int]) -> Tuple[str, float]:
        """
        Classify a cell by its dominant color using HSV thresholds.
        Returns (owner_label, confidence).
        """
        # Convert to HSV for better color matching
        bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        cfg = self.cfg

        # Check own territory
        lo = cfg["own_color_hsv_lower"]
        hi = cfg["own_color_hsv_upper"]
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            return "own", 0.85

        # Check enemy territory
        lo = cfg["enemy_color_hsv_lower"]
        hi = cfg["enemy_color_hsv_upper"]
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            return "enemy", 0.80

        # Check neutral (low saturation)
        lo = cfg["neutral_color_hsv_lower"]
        hi = cfg["neutral_color_hsv_upper"]
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            return "neutral", 0.75

        # Very dark = border/wall
        if v < 40:
            return "border", 0.90

        return "unknown", 0.3

    # ── Statistics ─────────────────────────────────────────────────────────

    def _compute_stats(self, state: GameState):
        """Compute territory ownership percentages."""
        total = 0
        own = enemy = neutral = 0

        for row in state.grid:
            for cell in row:
                if cell.owner == "border":
                    continue
                total += 1
                if cell.owner == "own":
                    own += 1
                elif cell.owner == "enemy":
                    enemy += 1
                elif cell.owner == "neutral":
                    neutral += 1

        if total > 0:
            state.own_territory_pct = own / total
            state.enemy_territory_pct = enemy / total
            state.neutral_territory_pct = neutral / total

        state.own_cell_count = own
        state.enemy_cell_count = enemy
        state.neutral_cell_count = neutral

    # ── Border Detection ───────────────────────────────────────────────────

    def _find_borders(self, state: GameState):
        """Find cells on the border between own and enemy/neutral territory."""
        rows = len(state.grid)
        cols = len(state.grid[0]) if rows > 0 else 0

        own_borders = []
        enemy_borders = []

        for r in range(rows):
            for c in range(cols):
                cell = state.grid[r][c]
                if cell.owner not in ("own", "enemy"):
                    continue

                neighbors = self._get_neighbors(state.grid, r, c, rows, cols)
                neighbor_owners = {n.owner for n in neighbors}

                if cell.owner == "own" and (
                    "enemy" in neighbor_owners or "neutral" in neighbor_owners
                ):
                    own_borders.append((r, c))

                elif cell.owner == "enemy" and "own" in neighbor_owners:
                    enemy_borders.append((r, c))

        state.own_border_cells = own_borders
        state.enemy_border_cells = enemy_borders

    def _get_neighbors(self, grid, r, c, rows, cols) -> List[TerritoryCell]:
        """Return the 4-connected neighbors of a cell."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append(grid[nr][nc])
        return neighbors

    # ── Attack Target Selection ────────────────────────────────────────────

    def _find_best_attack_target(self, state: GameState):
        """
        Find the best cell to attack.
        Prefers enemy border cells adjacent to own territory.
        Falls back to neutral cells if no enemy borders found.
        """
        if state.enemy_border_cells:
            # Pick the enemy border cell closest to the center of own territory
            own_center = self._compute_centroid(state.own_border_cells)
            if own_center:
                best = min(
                    state.enemy_border_cells,
                    key=lambda rc: abs(rc[0] - own_center[0]) + abs(rc[1] - own_center[1])
                )
                state.best_attack_target = best
                return

        # Fall back: expand into neutral territory
        neutral_cells = [
            (r, c)
            for row in state.grid
            for cell in row
            if cell.owner == "neutral"
            for r, c in [(cell.row, cell.col)]
        ]
        if neutral_cells and state.own_border_cells:
            own_center = self._compute_centroid(state.own_border_cells)
            if own_center:
                best = min(
                    neutral_cells,
                    key=lambda rc: abs(rc[0] - own_center[0]) + abs(rc[1] - own_center[1])
                )
                state.best_attack_target = best

    def _compute_centroid(self, cells: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        """Compute the centroid of a list of (row, col) tuples."""
        if not cells:
            return None
        avg_r = sum(c[0] for c in cells) / len(cells)
        avg_c = sum(c[1] for c in cells) / len(cells)
        return (avg_r, avg_c)

    # ── Game Phase ─────────────────────────────────────────────────────────

    def _determine_game_phase(self, state: GameState) -> str:
        """Determine early/mid/late game based on territory ownership."""
        from territorial_bot.config import BRAIN_CONFIG
        pct = state.own_territory_pct
        if pct < BRAIN_CONFIG["early_game_threshold"]:
            return "early"
        elif pct < BRAIN_CONFIG["mid_game_threshold"]:
            return "mid"
        else:
            return "late"

    # ── Canvas Detection ───────────────────────────────────────────────────

    def find_canvas_bbox(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Attempt to locate the game canvas within a full browser screenshot.
        Returns (x, y, w, h) or None if not found.

        Strategy: look for the largest dark rectangular region
        (the game map background is typically very dark).
        """
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Sanity check: canvas should be reasonably large
        sh, sw = screenshot.shape[:2]
        if w < sw * 0.3 or h < sh * 0.3:
            return None

        return (x, y, w, h)

    # ── Debug Visualization ────────────────────────────────────────────────

    def draw_debug_overlay(self, state: GameState) -> np.ndarray:
        """
        Draw a debug overlay on the game frame showing detected territories.
        Returns annotated BGR image.
        """
        if state.frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        overlay = state.frame.copy()
        rows = len(state.grid)
        cols = len(state.grid[0]) if rows > 0 else 0
        h, w = overlay.shape[:2]
        cell_h = h // rows if rows > 0 else h
        cell_w = w // cols if cols > 0 else w

        color_map = {
            "own":     (255, 100, 0),    # Blue
            "enemy":   (0,   50,  255),  # Red
            "neutral": (100, 100, 100),  # Gray
            "border":  (0,   0,   0),    # Black
            "unknown": (0,   255, 255),  # Yellow
        }

        for row in state.grid:
            for cell in row:
                y1 = cell.row * cell_h
                x1 = cell.col * cell_w
                color = color_map.get(cell.owner, (255, 255, 255))
                cv2.rectangle(overlay, (x1, y1),
                              (x1 + cell_w, y1 + cell_h), color, 1)

        # Mark attack target
        if state.best_attack_target:
            tr, tc = state.best_attack_target
            ty1 = tr * cell_h
            tx1 = tc * cell_w
            cv2.rectangle(overlay, (tx1, ty1),
                          (tx1 + cell_w, ty1 + cell_h), (0, 255, 0), 3)

        # Stats text
        cv2.putText(
            overlay,
            f"Own:{state.own_territory_pct:.1%} "
            f"Enemy:{state.enemy_territory_pct:.1%} "
            f"Phase:{state.game_phase}",
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        return overlay

    # ── Troop Number OCR (simple) ──────────────────────────────────────────

    def extract_troop_numbers(self, frame: np.ndarray) -> Dict[Tuple[int, int], int]:
        """
        Attempt to read troop numbers from the game frame using simple
        digit detection. Returns dict of {(x, y): count}.
        Note: Full OCR requires Tesseract; this is a placeholder.
        """
        # TODO: Integrate pytesseract for full OCR support
        # For now, return empty dict
        return {}
