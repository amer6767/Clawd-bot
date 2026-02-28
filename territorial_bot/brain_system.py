"""
Brain System for Territorial.io Bot
=====================================
Decision-making AI using Deep Q-Network (DQN) reinforcement learning.
Analyzes GameState from the vision system and outputs strategic actions.
"""

import random
import logging
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Brain system will use rule-based fallback.")

from config import BRAIN_CONFIG
from vision_system import GameState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ACTION DEFINITIONS
# ─────────────────────────────────────────────

class Action:
    """Discrete action space for the bot."""
    STAY        = 0
    ATTACK_N    = 1   # Attack North
    ATTACK_NE   = 2
    ATTACK_E    = 3
    ATTACK_SE   = 4
    ATTACK_S    = 5
    ATTACK_SW   = 6
    ATTACK_W    = 7
    ATTACK_NW   = 8

    # Direction vectors (row_delta, col_delta)
    DIRECTION_MAP = {
        STAY:     (0,  0),
        ATTACK_N: (-1, 0),
        ATTACK_NE:(-1, 1),
        ATTACK_E: (0,  1),
        ATTACK_SE:(1,  1),
        ATTACK_S: (1,  0),
        ATTACK_SW:(1, -1),
        ATTACK_W: (0, -1),
        ATTACK_NW:(-1,-1),
    }

    NAMES = {
        STAY: "STAY", ATTACK_N: "ATTACK_N", ATTACK_NE: "ATTACK_NE",
        ATTACK_E: "ATTACK_E", ATTACK_SE: "ATTACK_SE", ATTACK_S: "ATTACK_S",
        ATTACK_SW: "ATTACK_SW", ATTACK_W: "ATTACK_W", ATTACK_NW: "ATTACK_NW",
    }

    @staticmethod
    def name(action_id: int) -> str:
        return Action.NAMES.get(action_id, f"ACTION_{action_id}")


@dataclass
class Decision:
    """Output of the brain system for one tick."""
    action_id: int
    action_name: str
    target_cell: Optional[Tuple[int, int]]   # (row, col) in grid
    target_pixel: Optional[Tuple[int, int]]  # (x, y) in canvas pixels
    confidence: float
    reasoning: str


# ─────────────────────────────────────────────
# DQN NETWORK
# ─────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for game decision making.
    Input: state vector (flattened game state features)
    Output: Q-values for each action
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# REPLAY MEMORY
# ─────────────────────────────────────────────

class ReplayMemory:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ─────────────────────────────────────────────
# BRAIN SYSTEM CLASS
# ─────────────────────────────────────────────

class BrainSystem:
    """
    Strategic decision-making system for Territorial.io.

    Uses a DQN trained on Kaggle to select actions based on game state.
    Falls back to rule-based heuristics if no model is loaded.
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or BRAIN_CONFIG
        self.state_size = self.cfg["state_size"]
        self.action_size = self.cfg["action_size"]

        self.policy_net: Optional[DQNNetwork] = None
        self.target_net: Optional[DQNNetwork] = None
        self.optimizer = None
        self.memory = ReplayMemory(self.cfg["memory_size"])
        self.device = "cpu"

        self.epsilon = self.cfg["epsilon_start"]
        self.steps_done = 0
        self.episode = 0

        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

        self._load_model()
        logger.info("BrainSystem initialized.")

    # ── Model Loading ──────────────────────────────────────────────────────

    def _load_model(self):
        """Load trained DQN model from disk."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable; using rule-based decisions only.")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.cfg["learning_rate"]
        )

        model_path = Path(self.cfg["brain_model_path"])
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                self.epsilon = checkpoint.get("epsilon", self.cfg["epsilon_end"])
                self.episode = checkpoint.get("episode", 0)
                logger.info(
                    f"Brain model loaded from {model_path} "
                    f"(episode {self.episode}, epsilon {self.epsilon:.3f})"
                )
            except Exception as e:
                logger.error(f"Failed to load brain model: {e}")
        else:
            logger.warning(
                f"Brain model not found at {model_path}. "
                "Using rule-based fallback. Train on Kaggle first."
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── State Encoding ─────────────────────────────────────────────────────

    def encode_state(self, game_state: GameState) -> np.ndarray:
        """
        Convert a GameState into a fixed-size feature vector for the DQN.

        Feature vector layout (64 features total):
          [0]    own_territory_pct
          [1]    enemy_territory_pct
          [2]    neutral_territory_pct
          [3]    own_cell_count (normalized)
          [4]    enemy_cell_count (normalized)
          [5]    neutral_cell_count (normalized)
          [6]    own_border_count (normalized)
          [7]    enemy_border_count (normalized)
          [8]    game_phase_early (one-hot)
          [9]    game_phase_mid (one-hot)
          [10]   game_phase_late (one-hot)
          [11]   has_attack_target
          [12]   attack_target_row (normalized)
          [13]   attack_target_col (normalized)
          [14-63] flattened 5x10 grid ownership map (own=1, enemy=-1, neutral=0)
        """
        vec = np.zeros(self.state_size, dtype=np.float32)

        # Territory percentages
        vec[0] = game_state.own_territory_pct
        vec[1] = game_state.enemy_territory_pct
        vec[2] = game_state.neutral_territory_pct

        # Cell counts (normalize by total cells)
        total = max(1, game_state.own_cell_count +
                    game_state.enemy_cell_count +
                    game_state.neutral_cell_count)
        vec[3] = game_state.own_cell_count / total
        vec[4] = game_state.enemy_cell_count / total
        vec[5] = game_state.neutral_cell_count / total

        # Border counts
        max_border = 100.0
        vec[6] = len(game_state.own_border_cells) / max_border
        vec[7] = len(game_state.enemy_border_cells) / max_border

        # Game phase (one-hot)
        phase_map = {"early": 8, "mid": 9, "late": 10}
        vec[phase_map.get(game_state.game_phase, 8)] = 1.0

        # Attack target
        if game_state.best_attack_target:
            vec[11] = 1.0
            rows = len(game_state.grid) or 1
            cols = len(game_state.grid[0]) if game_state.grid else 1
            vec[12] = game_state.best_attack_target[0] / rows
            vec[13] = game_state.best_attack_target[1] / cols

        # Compact grid map (5x10 = 50 features, starting at index 14)
        if game_state.grid:
            rows = len(game_state.grid)
            cols = len(game_state.grid[0])
            # Sample a 5x10 sub-grid
            sample_rows = np.linspace(0, rows - 1, 5, dtype=int)
            sample_cols = np.linspace(0, cols - 1, 10, dtype=int)
            idx = 14
            for r in sample_rows:
                for c in sample_cols:
                    if idx >= self.state_size:
                        break
                    cell = game_state.grid[r][c]
                    if cell.owner == "own":
                        vec[idx] = 1.0
                    elif cell.owner == "enemy":
                        vec[idx] = -1.0
                    else:
                        vec[idx] = 0.0
                    idx += 1

        return vec

    # ── Decision Making ────────────────────────────────────────────────────

    def decide(self, game_state: GameState) -> Decision:
        """
        Main decision function. Returns a Decision for the current game state.
        Uses epsilon-greedy policy: explore randomly or exploit DQN.
        """
        state_vec = self.encode_state(game_state)
        self._last_state = state_vec

        # Epsilon-greedy action selection
        if (self.policy_net is None or
                random.random() < self.epsilon):
            action_id = self._rule_based_action(game_state)
            reasoning = f"Rule-based (ε={self.epsilon:.3f})"
        else:
            action_id = self._dqn_action(state_vec)
            reasoning = f"DQN (ε={self.epsilon:.3f})"

        self._last_action = action_id

        # Convert grid action to pixel target
        target_cell, target_pixel = self._action_to_target(action_id, game_state)

        decision = Decision(
            action_id=action_id,
            action_name=Action.name(action_id),
            target_cell=target_cell,
            target_pixel=target_pixel,
            confidence=1.0 - self.epsilon,
            reasoning=reasoning,
        )

        logger.debug(
            f"Decision: {decision.action_name} → cell={target_cell} "
            f"pixel={target_pixel} [{reasoning}]"
        )
        return decision

    def _dqn_action(self, state_vec: np.ndarray) -> int:
        """Select action using DQN policy network."""
        try:
            tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(tensor)
            return q_values.argmax(dim=1).item()
        except Exception as e:
            logger.error(f"DQN inference error: {e}")
            return Action.STAY

    def _rule_based_action(self, game_state: GameState) -> int:
        """
        Heuristic rule-based fallback strategy.
        Used during exploration or when no model is loaded.
        """
        aggression = self.cfg["aggression"]
        phase = game_state.game_phase

        # Early game: always expand into neutral
        if phase == "early":
            if game_state.best_attack_target:
                return self._direction_to_target(game_state)
            return random.choice([Action.ATTACK_N, Action.ATTACK_E,
                                  Action.ATTACK_S, Action.ATTACK_W])

        # Mid game: balance attack and defense
        if phase == "mid":
            if game_state.enemy_territory_pct > self.cfg["defense_threshold"]:
                # Defensive: attack nearest enemy border
                if game_state.enemy_border_cells:
                    return self._direction_to_target(game_state)
            if random.random() < aggression and game_state.best_attack_target:
                return self._direction_to_target(game_state)
            return Action.STAY

        # Late game: aggressive expansion
        if game_state.best_attack_target:
            return self._direction_to_target(game_state)

        return Action.STAY

    def _direction_to_target(self, game_state: GameState) -> int:
        """
        Compute the best directional action to move toward the attack target.
        """
        if not game_state.best_attack_target or not game_state.own_border_cells:
            return Action.STAY

        target_r, target_c = game_state.best_attack_target
        own_center = self._centroid(game_state.own_border_cells)
        if own_center is None:
            return Action.STAY

        dr = target_r - own_center[0]
        dc = target_c - own_center[1]

        # Map to 8-directional action
        if abs(dr) < 0.5 and abs(dc) < 0.5:
            return Action.STAY

        angle = np.arctan2(dc, -dr)  # angle from north
        # Quantize to 8 directions
        sector = int((angle + np.pi) / (np.pi / 4)) % 8
        direction_actions = [
            Action.ATTACK_S, Action.ATTACK_SW, Action.ATTACK_W, Action.ATTACK_NW,
            Action.ATTACK_N, Action.ATTACK_NE, Action.ATTACK_E, Action.ATTACK_SE,
        ]
        return direction_actions[sector]

    def _centroid(self, cells: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        if not cells:
            return None
        return (sum(c[0] for c in cells) / len(cells),
                sum(c[1] for c in cells) / len(cells))

    # ── Action → Pixel Mapping ─────────────────────────────────────────────

    def _action_to_target(
        self, action_id: int, game_state: GameState
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Convert an action ID to a (grid_cell, canvas_pixel) target.
        """
        if action_id == Action.STAY:
            return None, None

        # Use the best_attack_target if available
        if game_state.best_attack_target:
            target_cell = game_state.best_attack_target
        else:
            # Compute target cell from direction + own border centroid
            own_center = self._centroid(game_state.own_border_cells)
            if own_center is None:
                return None, None
            dr, dc = Action.DIRECTION_MAP.get(action_id, (0, 0))
            rows = len(game_state.grid) or 1
            cols = len(game_state.grid[0]) if game_state.grid else 1
            target_r = int(np.clip(own_center[0] + dr * 2, 0, rows - 1))
            target_c = int(np.clip(own_center[1] + dc * 2, 0, cols - 1))
            target_cell = (target_r, target_c)

        # Convert grid cell to canvas pixel
        target_pixel = self._cell_to_pixel(target_cell, game_state)
        return target_cell, target_pixel

    def _cell_to_pixel(
        self, cell: Tuple[int, int], game_state: GameState
    ) -> Tuple[int, int]:
        """Convert a grid (row, col) to canvas pixel (x, y)."""
        rows = len(game_state.grid) or 1
        cols = len(game_state.grid[0]) if game_state.grid else 1
        cell_h = game_state.canvas_h / rows
        cell_w = game_state.canvas_w / cols
        px = int(cell[1] * cell_w + cell_w / 2) + game_state.canvas_x
        py = int(cell[0] * cell_h + cell_h / 2) + game_state.canvas_y
        return (px, py)

    # ── Reward Computation ─────────────────────────────────────────────────

    def compute_reward(
        self, prev_state: GameState, curr_state: GameState, done: bool
    ) -> float:
        """
        Compute reward signal for reinforcement learning.

        Reward structure:
          +territory gained (own_pct increase)
          -territory lost (own_pct decrease)
          +bonus for eliminating enemies
          -penalty for dying (own_pct → 0)
          +large bonus for winning
        """
        if done:
            if curr_state.own_territory_pct < 0.01:
                return -10.0   # Died
            else:
                return +20.0   # Won

        territory_delta = (
            curr_state.own_territory_pct - prev_state.own_territory_pct
        )
        enemy_delta = (
            prev_state.enemy_territory_pct - curr_state.enemy_territory_pct
        )

        reward = (
            territory_delta * 10.0   # Reward territory gain
            + enemy_delta * 5.0      # Reward reducing enemy territory
            - 0.01                   # Small time penalty (encourages speed)
        )

        return float(np.clip(reward, -5.0, 5.0))

    # ── Training Step ──────────────────────────────────────────────────────

    def store_transition(
        self,
        prev_game_state: GameState,
        action_id: int,
        reward: float,
        curr_game_state: GameState,
        done: bool,
    ):
        """Store a transition in replay memory."""
        prev_vec = self.encode_state(prev_game_state)
        curr_vec = self.encode_state(curr_game_state)
        self.memory.push(prev_vec, action_id, reward, curr_vec, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one DQN training step.
        Returns loss value or None if not enough samples.
        """
        if not TORCH_AVAILABLE or self.policy_net is None:
            return None
        if len(self.memory) < self.cfg["batch_size"]:
            return None

        batch = self.memory.sample(self.cfg["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            next_q[dones_t] = 0.0
            target_q = rewards_t + self.cfg["gamma"] * next_q

        loss = nn.MSELoss()(q_values.squeeze(1), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.cfg["epsilon_end"],
            self.epsilon * self.cfg["epsilon_decay"]
        )

    def update_target_network(self):
        """Copy policy network weights to target network."""
        if self.target_net and self.policy_net:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Model Saving ───────────────────────────────────────────────────────

    def save_model(self, path: str = None):
        """Save model checkpoint to disk."""
        if not TORCH_AVAILABLE or self.policy_net is None:
            return
        path = path or self.cfg["brain_model_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode,
        }, path)
        logger.info(f"Brain model saved to {path}")

    # ── Strategy Summary ───────────────────────────────────────────────────

    def get_strategy_summary(self, game_state: GameState) -> Dict:
        """Return a human-readable summary of current strategy."""
        return {
            "game_phase": game_state.game_phase,
            "own_territory": f"{game_state.own_territory_pct:.1%}",
            "enemy_territory": f"{game_state.enemy_territory_pct:.1%}",
            "neutral_territory": f"{game_state.neutral_territory_pct:.1%}",
            "own_borders": len(game_state.own_border_cells),
            "enemy_borders": len(game_state.enemy_border_cells),
            "attack_target": game_state.best_attack_target,
            "epsilon": f"{self.epsilon:.3f}",
            "memory_size": len(self.memory),
            "aggression": self.cfg["aggression"],
        }
