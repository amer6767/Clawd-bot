"""
Enemy Tracking System for Territorial.io Bot
==============================================
Tracks individual enemy players, identifies patterns, and adapts targeting strategy.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class EnemyPlayer:
    """Represents a tracked enemy player."""
    player_id: str
    first_seen: float
    last_seen: float
    color_detected: Tuple[int, int, int]  # RGB color in game
    
    # Territory tracking
    territory_history: List[float] = field(default_factory=list)
    peak_territory: float = 0.0
    current_territory: float = 0.0
    
    # Behavior analysis
    attack_frequency: float = 0.0  # Attacks per minute
    expansion_rate: float = 0.0    # Territory gained per minute
    aggression_score: float = 0.0  # 0-1 scale
    
    # Combat tracking
    times_attacked_us: int = 0
    times_we_attacked: int = 0
    successful_defenses: int = 0
    
    # Spatial tracking
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    current_position: Optional[Tuple[int, int]] = None
    
    # Strategy classification
    behavior_type: str = "unknown"  # "aggressive", "passive", "defensive", "expander"
    
    def update_territory(self, territory_pct: float, timestamp: float):
        """Update territory information."""
        self.territory_history.append((territory_pct, timestamp))
        self.current_territory = territory_pct
        self.peak_territory = max(self.peak_territory, territory_pct)
        self.last_seen = timestamp
        
        # Keep only recent history (last 2 minutes)
        cutoff = timestamp - 120
        self.territory_history = [
            (p, t) for p, t in self.territory_history if t > cutoff
        ]
    
    def update_position(self, grid_pos: Tuple[int, int], timestamp: float):
        """Update position in grid coordinates."""
        self.position_history.append((grid_pos, timestamp))
        self.current_position = grid_pos
        self.last_seen = timestamp
        
        # Keep only recent history
        cutoff = timestamp - 60
        self.position_history = [
            (pos, t) for pos, t in self.position_history if t > cutoff
        ]
    
    def analyze_behavior(self):
        """Analyze player behavior patterns."""
        if not self.territory_history or len(self.territory_history) < 2:
            return
        
        # Calculate expansion rate
        if len(self.territory_history) >= 2:
            first_terr, first_time = self.territory_history[0]
            last_terr, last_time = self.territory_history[-1]
            time_diff = max(1, last_time - first_time)
            self.expansion_rate = (last_terr - first_terr) / (time_diff / 60)
        
        # Calculate aggression based on attack frequency
        if self.times_attacked_us > 0:
            game_duration = max(1, self.last_seen - self.first_seen) / 60
            self.attack_frequency = self.times_attacked_us / game_duration
        
        # Combined aggression score
        self.aggression_score = np.clip(
            (self.attack_frequency * 0.5 + max(0, self.expansion_rate) * 0.5),
            0, 1
        )
        
        # Classify behavior type
        if self.aggression_score > 0.7:
            self.behavior_type = "aggressive"
        elif self.aggression_score < 0.3:
            if self.expansion_rate > 0.1:
                self.behavior_type = "expander"
            else:
                self.behavior_type = "passive"
        elif self.successful_defenses > self.times_attacked_us:
            self.behavior_type = "defensive"
        else:
            self.behavior_type = "balanced"
    
    def to_dict(self):
        return asdict(self)


class EnemyTracker:
    """
    Tracks enemy players across the game, identifies patterns, and recommends targeting.
    
    Features:
    - Identifies individual enemies by color
    - Tracks behavior patterns (aggressive vs passive)
    - Recommends targeting weakest/most vulnerable enemies
    - Remembers enemy strategies for adaptive play
    """
    
    def __init__(self, save_path: str = "models/enemy_tracking.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Active enemy tracking
        self.enemies: Dict[str, EnemyPlayer] = {}

        # Color to player ID mapping (for Territorial.io, each player has unique color)
        # Keys stored as tuples; JSON serialization converts to lists so we normalize on load
        self.color_map: Dict[Tuple[int, int, int], str] = {}

        # Historical data
        self.total_enemies_encountered = 0

        # Load previous data
        self._load_data()

        logger.info(f"EnemyTracker initialized with {len(self.enemies)} tracked enemies")
    
    def detect_enemies(self, game_state) -> List[EnemyPlayer]:
        """
        Detect enemies from current game state.
        Returns list of detected EnemyPlayer objects.
        """
        current_time = time.time()
        detected_enemies = []
        
        if not game_state.grid:
            return detected_enemies
        
        # Analyze grid for enemy territories
        enemy_positions = []
        
        for row in game_state.grid:
            for cell in row:
                if cell.owner == "enemy":
                    enemy_positions.append((cell.row, cell.col, cell.color_rgb))
        
        # Group by color to identify individual players
        color_groups = defaultdict(list)
        for r, c, color in enemy_positions:
            if color != (0, 0, 0):  # Ignore black/border
                color_groups[color].append((r, c))
        
        # Process each color group as a potential enemy
        for color, positions in color_groups.items():
            if color in self.color_map:
                # Known enemy
                player_id = self.color_map[color]
            else:
                # New enemy - assign ID
                player_id = f"enemy_{len(self.enemies)}_{int(current_time)}"
                self.color_map[color] = player_id
                self.total_enemies_encountered += 1
                
                # Create new player record
                self.enemies[player_id] = EnemyPlayer(
                    player_id=player_id,
                    first_seen=current_time,
                    last_seen=current_time,
                    color_detected=color
                )
                logger.info(f"New enemy detected: {player_id} (color: {color})")
            
            # Update enemy info
            if player_id in self.enemies:
                enemy = self.enemies[player_id]
                
                # Calculate territory percentage from positions
                grid_rows = len(game_state.grid) if game_state.grid else 1
                grid_cols = len(game_state.grid[0]) if game_state.grid else 1
                territory_pct = len(positions) / max(1, grid_rows * grid_cols)
                enemy.update_territory(territory_pct, current_time)
                
                # Estimate position (centroid of territory)
                if positions:
                    avg_row = sum(p[0] for p in positions) / len(positions)
                    avg_col = sum(p[1] for p in positions) / len(positions)
                    enemy.update_position((int(avg_row), int(avg_col)), current_time)
                
                detected_enemies.append(enemy)
        
        # Analyze behaviors
        for enemy in self.enemies.values():
            enemy.analyze_behavior()
        
        return detected_enemies
    
    def record_combat_event(self, player_id: str, event_type: str):
        """Record combat interactions with a specific enemy."""
        if player_id not in self.enemies:
            logger.warning(f"Combat event for unknown enemy: {player_id}")
            return
        
        enemy = self.enemies[player_id]
        
        if event_type == "attacked_us":
            enemy.times_attacked_us += 1
        elif event_type == "we_attacked":
            enemy.times_we_attacked += 1
        elif event_type == "defense_success":
            enemy.successful_defenses += 1
        
        logger.debug(f"Combat event: {player_id} - {event_type}")
    
    def get_weakest_enemy(self, detected_enemies: List[EnemyPlayer]) -> Optional[EnemyPlayer]:
        """Get the weakest enemy to target."""
        if not detected_enemies:
            return None
        
        # Rank by multiple factors
        def weakness_score(enemy: EnemyPlayer) -> float:
            # Lower territory = weaker
            territory_score = 1 - enemy.current_territory
            
            # Lower aggression = easier target
            aggression_score = 1 - enemy.aggression_score
            
            # Recent activity (don't target inactive players)
            activity_score = 1.0 if (time.time() - enemy.last_seen) < 30 else 0.3
            
            return territory_score * 0.5 + aggression_score * 0.3 + activity_score * 0.2
        
        weakest = min(detected_enemies, key=weakness_score)
        
        logger.debug(f"Targeting weakest enemy: {weakest.player_id} "
                    f"(territory: {weakest.current_territory:.1%}, "
                    f"aggression: {weakest.aggression_score:.2f})")
        
        return weakest
    
    def get_most_aggressive_enemy(self, detected_enemies: List[EnemyPlayer]) -> Optional[EnemyPlayer]:
        """Get the most aggressive enemy (for defensive prioritization)."""
        if not detected_enemies:
            return None
        
        most_aggressive = max(detected_enemies, key=lambda e: e.aggression_score)
        
        logger.debug(f"Most aggressive enemy: {most_aggressive.player_id} "
                    f"(aggression: {most_aggressive.aggression_score:.2f})")
        
        return most_aggressive
    
    def get_best_target_for_strategy(self, detected_enemies: List[EnemyPlayer],
                                    bot_territory: float,
                                    strategy: str = "balanced") -> Optional[EnemyPlayer]:
        """
        Get the best enemy target based on current strategy.
        
        Strategies:
        - "aggressive": Target weakest to expand quickly
        - "defensive": Target most aggressive to neutralize threats  
        - "strategic": Target enemies in good strategic positions
        - "balanced": Mix of factors
        """
        if not detected_enemies:
            return None
        
        if strategy == "aggressive":
            return self.get_weakest_enemy(detected_enemies)
        elif strategy == "defensive":
            return self.get_most_aggressive_enemy(detected_enemies)
        else:
            # Balanced: prefer weaker enemies but not inactive ones
            current_time = time.time()
            
            def balanced_score(enemy: EnemyPlayer) -> float:
                # Don't target inactive enemies
                if current_time - enemy.last_seen > 60:
                    return float('inf')
                
                # Balance between weakness and strategic value
                weakness = 1 - enemy.current_territory
                strategic_value = enemy.current_territory * 0.5 + enemy.expansion_rate * 0.5
                
                return weakness * 0.6 + strategic_value * 0.4
            
            return min(detected_enemies, key=balanced_score)
    
    def get_threat_assessment(self, detected_enemies: List[EnemyPlayer]) -> Dict:
        """Get overall threat assessment from all detected enemies."""
        if not detected_enemies:
            return {"overall_threat": 0.0, "immediate_threats": [], "passive_enemies": []}
        
        # Calculate overall threat level
        threat_scores = []
        immediate_threats = []
        passive_enemies = []
        
        for enemy in detected_enemies:
            # Threat based on aggression and territory
            threat = enemy.aggression_score * (0.5 + enemy.current_territory)
            threat_scores.append(threat)
            
            if enemy.aggression_score > 0.6 and enemy.current_territory > 0.1:
                immediate_threats.append(enemy.player_id)
            elif enemy.aggression_score < 0.3:
                passive_enemies.append(enemy.player_id)
        
        return {
            "overall_threat": np.mean(threat_scores) if threat_scores else 0.0,
            "immediate_threats": immediate_threats,
            "passive_enemies": passive_enemies,
            "enemy_count": len(detected_enemies)
        }
    
    def cleanup_stale_enemies(self, max_age_seconds: int = 180):
        """Remove enemies not seen recently."""
        current_time = time.time()
        stale = []
        
        for player_id, enemy in self.enemies.items():
            if current_time - enemy.last_seen > max_age_seconds:
                stale.append(player_id)
        
        for player_id in stale:
            del self.enemies[player_id]
        
        if stale:
            logger.debug(f"Cleaned up {len(stale)} stale enemy tracks")
    
    def _save_data(self):
        """Save enemy tracking data."""
        try:
            data = {
                "total_enemies_encountered": self.total_enemies_encountered,
                "enemies": {
                    pid: enemy.to_dict() 
                    for pid, enemy in self.enemies.items()
                }
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save enemy data: {e}")
    
    def _load_data(self):
        """Load enemy tracking data."""
        if not self.save_path.exists():
            return

        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)

            self.total_enemies_encountered = data.get("total_enemies_encountered", 0)

            # Restore enemy data (without history to save space)
            for pid, enemy_data in data.get("enemies", {}).items():
                try:
                    # JSON converts tuples to lists; convert color_detected back to tuple
                    if "color_detected" in enemy_data and isinstance(
                        enemy_data["color_detected"], list
                    ):
                        enemy_data["color_detected"] = tuple(enemy_data["color_detected"])

                    # Convert position_history entries from lists to tuples
                    if "position_history" in enemy_data:
                        enemy_data["position_history"] = [
                            (tuple(pos) if isinstance(pos, list) else pos, t)
                            for pos, t in enemy_data["position_history"]
                        ]

                    # Convert territory_history entries (list of [pct, time])
                    if "territory_history" in enemy_data:
                        enemy_data["territory_history"] = [
                            tuple(entry) if isinstance(entry, list) else entry
                            for entry in enemy_data["territory_history"]
                        ]

                    enemy = EnemyPlayer(**enemy_data)
                    # Clear old history to start fresh
                    enemy.territory_history = []
                    enemy.position_history = []
                    self.enemies[pid] = enemy

                    if enemy.color_detected:
                        color_key = tuple(enemy.color_detected)
                        self.color_map[color_key] = pid

                except Exception as inner_e:
                    logger.warning(f"Skipping corrupt enemy record {pid}: {inner_e}")

            logger.info(f"Loaded {len(self.enemies)} tracked enemies from history")

        except Exception as e:
            logger.error(f"Failed to load enemy data: {e}")
