"""
Smart Strategy System for Territorial.io Bot
==============================================
Advanced strategic decision making including auto-defense, corner strategies, and neutral coordination.
"""

import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of strategic behaviors."""
    EXPAND = "expand"
    ATTACK = "attack" 
    DEFEND = "defend"
    CONSOLIDATE = "consolidate"
    CORNER = "corner"
    NEUTRAL_ALLIANCE = "neutral_alliance"


@dataclass 
class StrategicSituation:
    """Current strategic situation assessment."""
    is_being_attacked: bool = False
    attack_source: Optional[Tuple[int, int]] = None
    attack_severity: float = 0.0
    is_cornered: bool = False
    corner_position: Optional[str] = None
    vulnerable_border_count: int = 0
    neutral_neighbors: List[Tuple[int, int]] = None
    enemy_pressure: float = 0.0
    
    def __post_init__(self):
        if self.neutral_neighbors is None:
            self.neutral_neighbors = []


class SmartStrategy:
    """
    Advanced strategic decision making system.
    
    Features:
    - Auto-defense when under attack
    - Corner strategies for map edges
    - Neutral territory alliance strategy
    - Dynamic strategy selection based on game state
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Strategy weights
        self.defense_weight = 0.4
        self.expansion_weight = 0.3
        self.aggression_weight = 0.3
        
        # Corner detection settings
        self.corner_threshold = 0.15  # % of map size from corner
        
        # Defense settings  
        self.defense_activate_threshold = 0.3  # Enemy border ratio to activate defense
        self.defense_aggression = 0.8
        
        # Current strategy state
        self.current_strategy = StrategyType.EXPAND
        self.strategy_timer = 0
        self.strategy_duration = 10  # Minimum ticks per strategy
        
        # Tracking
        self.attack_warnings = []
        self.defense_count = 0
        self.corner_strategies_used = 0
        
        logger.info("SmartStrategy initialized")
    
    def analyze_situation(self, game_state, enemy_tracker=None) -> StrategicSituation:
        """
        Analyze current game situation and return strategic assessment.
        """
        situation = StrategicSituation()
        
        if not game_state.grid:
            return situation
        
        rows = len(game_state.grid)
        cols = len(game_state.grid[0]) if rows > 0 else 0
        
        # Check if being attacked
        situation = self._detect_attacks(game_state, situation)
        
        # Check corner position
        situation = self._detect_corner_position(game_state, situation, rows, cols)
        
        # Identify vulnerable borders
        situation = self._assess_vulnerable_borders(game_state, situation)
        
        # Find neutral neighbors for alliance strategy
        situation = self._find_neutral_neighbors(game_state, situation)
        
        # Calculate enemy pressure
        if game_state.own_border_cells and game_state.enemy_border_cells:
            situation.enemy_pressure = (
                len(game_state.enemy_border_cells) / 
                max(1, len(game_state.own_border_cells))
            )
        
        return situation
    
    def _detect_attacks(self, game_state, situation: StrategicSituation) -> StrategicSituation:
        """Detect if bot is under attack."""
        if not game_state.own_border_cells or not game_state.enemy_border_cells:
            return situation
        
        # Check if enemies are close to our borders
        for enemy_pos in game_state.enemy_border_cells:
            # Check adjacent cells for our territory
            for row in game_state.grid:
                for cell in row:
                    if cell.owner == "own":
                        # Check if enemy is adjacent
                        distance = abs(cell.row - enemy_pos[0]) + abs(cell.col - enemy_pos[1])
                        if distance <= 1:  # Adjacent
                            situation.is_being_attacked = True
                            situation.attack_source = enemy_pos
                            situation.attack_severity = min(1.0, situation.attack_severity + 0.2)
        
        return situation
    
    def _detect_corner_position(self, game_state, situation: StrategicSituation, 
                               rows: int, cols: int) -> StrategicSituation:
        """Detect if bot is in a corner position."""
        if not game_state.own_border_cells:
            return situation
        
        # Find our territory extent
        min_row = min(c[0] for c in game_state.own_border_cells)
        max_row = max(c[0] for c in game_state.own_border_cells)
        min_col = min(c[1] for c in game_state.own_border_cells)
        max_col = max(c[1] for c in game_state.own_border_cells)
        
        # Calculate distance from each corner
        corners = {
            "top_left": (min_row, min_col),
            "top_right": (min_row, max_col),
            "bottom_left": (max_row, min_col),
            "bottom_right": (max_row, max_col)
        }
        
        corner_threshold_rows = rows * self.corner_threshold
        corner_threshold_cols = cols * self.corner_threshold
        
        for corner_name, (corner_row, corner_col) in corners.items():
            # Check if we're near this corner
            if corner_row < corner_threshold_rows and corner_col < corner_threshold_cols:
                situation.is_cornered = True
                situation.corner_position = corner_name
                break
            elif corner_row < corner_threshold_rows and corner_col > cols - corner_threshold_cols:
                situation.is_cornered = True
                situation.corner_position = corner_name  
                break
            elif corner_row > rows - corner_threshold_rows and corner_col < corner_threshold_cols:
                situation.is_cornered = True
                situation.corner_position = corner_name
                break
            elif corner_row > rows - corner_threshold_rows and corner_col > cols - corner_threshold_cols:
                situation.is_cornered = True
                situation.corner_position = corner_name
                break
        
        return situation
    
    def _assess_vulnerable_borders(self, game_state, situation: StrategicSituation) -> StrategicSituation:
        """Assess which borders are most vulnerable."""
        vulnerable = 0
        
        if not game_state.own_border_cells:
            return situation
        
        # Count borders adjacent to enemies
        for own_border in game_state.own_border_cells:
            for enemy_border in game_state.enemy_border_cells:
                distance = abs(own_border[0] - enemy_border[0]) + abs(own_border[1] - enemy_border[1])
                if distance <= 2:
                    vulnerable += 1
                    break
        
        situation.vulnerable_border_count = vulnerable
        return situation
    
    def _find_neutral_neighbors(self, game_state, situation: StrategicSituation) -> StrategicSituation:
        """Find neutral territories that could be allies."""
        neutral_neighbors = []
        
        if not game_state.own_border_cells or not game_state.grid:
            return situation
        
        # Find neutral cells adjacent to our territory
        for own_border in game_state.own_border_cells:
            row, col = own_border
            # Check neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < len(game_state.grid) and 0 <= nc < len(game_state.grid[0]):
                    if game_state.grid[nr][nc].owner == "neutral":
                        if (nr, nc) not in neutral_neighbors:
                            neutral_neighbors.append((nr, nc))
        
        situation.neutral_neighbors = neutral_neighbors
        return situation
    
    def determine_strategy(self, situation: StrategicSituation, 
                         game_state, learning_params: Optional[Dict] = None) -> StrategyType:
        """
        Determine the best strategy based on current situation.
        """
        # Defense takes highest priority if being attacked
        if situation.is_being_attacked and situation.attack_severity > 0.3:
            self.current_strategy = StrategyType.DEFEND
            self.strategy_timer = 0
            return StrategyType.DEFEND
        
        # Corner strategy if cornered
        if situation.is_cornered:
            self.current_strategy = StrategyType.CORNER
            self.strategy_timer = 0
            return StrategyType.CORNER
        
        # Neutral alliance if we have neutral neighbors and low territory
        if (situation.neutral_neighbors and 
            game_state.own_territory_pct < 0.3 and
            len(situation.neutral_neighbors) > 3):
            self.current_strategy = StrategyType.NEUTRAL_ALLIANCE
            self.strategy_timer = 0
            return StrategyType.NEUTRAL_ALLIANCE
        
        # Otherwise use learning-based strategy
        if learning_params:
            aggression = learning_params.get("aggression", 0.5)
            
            if aggression > 0.7:
                self.current_strategy = StrategyType.ATTACK
            elif aggression < 0.3:
                self.current_strategy = StrategyType.CONSOLIDATE
            else:
                self.current_strategy = StrategyType.EXPAND
        else:
            # Default expansion strategy
            self.current_strategy = StrategyType.EXPAND
        
        # Enforce minimum strategy duration
        if self.strategy_timer < self.strategy_duration:
            return self.current_strategy
        
        self.strategy_timer = 0
        return self.current_strategy
    
    def get_defense_target(self, situation: StrategicSituation, 
                          game_state) -> Optional[Tuple[int, int]]:
        """Get the best defense target when under attack."""
        if not situation.is_being_attacked or not situation.attack_source:
            return None
        
        # Defend by attacking the attacker
        attack_row, attack_col = situation.attack_source
        
        # Find our border cell closest to the attack
        if game_state.own_border_cells:
            closest = min(
                game_state.own_border_cells,
                key=lambda c: abs(c[0] - attack_row) + abs(c[1] - attack_col)
            )
            return closest
        
        return situation.attack_source
    
    def get_corner_strategy_target(self, situation: StrategicSituation,
                                   game_state) -> Optional[Tuple[int, int]]:
        """Get target for corner-based expansion strategy."""
        if not situation.corner_position or not game_state.grid:
            return None
        
        rows = len(game_state.grid)
        cols = len(game_state.grid[0])
        
        # Different expansion directions based on corner
        if situation.corner_position == "top_left":
            # Expand down and right
            target = (rows // 4, cols // 4)
        elif situation.corner_position == "top_right":
            target = (rows // 4, 3 * cols // 4)
        elif situation.corner_position == "bottom_left":
            target = (3 * rows // 4, cols // 4)
        elif situation.corner_position == "bottom_right":
            target = (3 * rows // 4, 3 * cols // 4)
        else:
            target = None
        
        # Find closest neutral cell to target
        if target and situation.neutral_neighbors:
            closest = min(
                situation.neutral_neighbors,
                key=lambda c: abs(c[0] - target[0]) + abs(c[1] - target[1])
            )
            return closest
        
        return target if target else situation.neutral_neighbors[0] if situation.neutral_neighbors else None
    
    def get_neutral_alliance_target(self, situation: StrategicSituation,
                                   game_state) -> Optional[Tuple[int, int]]:
        """Get target for neutral territory alliance strategy."""
        if not situation.neutral_neighbors:
            return None
        
        # Prioritize neutral cells that are:
        # 1. Adjacent to our territory (already have border)
        # 2. Away from enemies
        # 3. Provide good expansion potential
        
        if not game_state.enemy_border_cells:
            # No enemies, pick any neutral
            return random.choice(situation.neutral_neighbors)
        
        # Find neutral cell farthest from enemies
        best_neutral = None
        best_score = float('-inf')
        
        for neutral in situation.neutral_neighbors:
            # Calculate distance to nearest enemy
            min_enemy_dist = float('inf')
            for enemy in game_state.enemy_border_cells:
                dist = abs(neutral[0] - enemy[0]) + abs(neutral[1] - enemy[1])
                min_enemy_dist = min(min_enemy_dist, dist)
            
            # Score: prefer cells far from enemies
            score = min_enemy_dist
            
            if score > best_score:
                best_score = score
                best_neutral = neutral
        
        return best_neutral
    
    def get_strategy_action_params(self, strategy: StrategyType) -> Dict:
        """Get action parameters for a specific strategy."""
        params = {
            "aggression": 0.5,
            "troop_ratio": 0.5,
            "target_type": "auto"
        }
        
        if strategy == StrategyType.DEFEND:
            params["aggression"] = self.defense_aggression
            params["troop_ratio"] = 0.7  # Send more troops for defense
            params["target_type"] = "defense"
            
        elif strategy == StrategyType.ATTACK:
            params["aggression"] = 0.9
            params["troop_ratio"] = 0.8
            params["target_type"] = "enemy"
            
        elif strategy == StrategyType.EXPAND:
            params["aggression"] = 0.5
            params["troop_ratio"] = 0.5
            params["target_type"] = "neutral"
            
        elif strategy == StrategyType.CORNER:
            params["aggression"] = 0.6
            params["troop_ratio"] = 0.6
            params["target_type"] = "corner"
            
        elif strategy == StrategyType.NEUTRAL_ALLIANCE:
            params["aggression"] = 0.4
            params["troop_ratio"] = 0.4
            params["target_type"] = "neutral_alliance"
            
        elif strategy == StrategyType.CONSOLIDATE:
            params["aggression"] = 0.2
            params["troop_ratio"] = 0.3
            params["target_type"] = "consolidate"
        
        return params
    
    def tick(self):
        """Update strategy timer."""
        self.strategy_timer += 1
    
    def get_strategy_summary(self) -> Dict:
        """Get summary of current strategy state."""
        return {
            "current_strategy": self.current_strategy.value,
            "strategy_timer": self.strategy_timer,
            "defense_count": self.defense_count,
            "corner_strategies_used": self.corner_strategies_used
        }
