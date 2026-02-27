"""
Learning System for Territorial.io Bot
======================================
Tracks game performance, learns from wins/losses, and persists learning progress.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """Record of a single game session."""
    game_id: str
    timestamp: float
    won: bool
    final_territory_pct: float
    duration_seconds: float
    enemies_encountered: int
    total_ticks: int
    territory_history: List[float]  # Territory % at each tick
    action_history: List[str]       # Actions taken during game
    final_enemy_count: int
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StrategyPerformance:
    """Performance metrics for different strategies."""
    strategy_name: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    avg_territory: float = 0.0
    avg_duration: float = 0.0
    aggression_score: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)
    
    @property
    def success_score(self) -> float:
        """Combined success metric."""
        return (self.win_rate * 0.4 + 
                min(1.0, self.avg_territory) * 0.4 + 
                min(1.0, self.avg_duration / 180) * 0.2)


class LearningSystem:
    """
    Self-learning system that tracks performance and improves strategy.
    
    Features:
    - Tracks wins/losses and territory progression
    - Saves/loads learning progress between sessions
    - Analyzes which strategies work best
    - Adapts behavior based on historical performance
    """
    
    def __init__(self, save_path: str = "models/learning_progress.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Game history
        self.game_history: List[GameRecord] = []
        self.total_games = 0
        self.total_wins = 0
        self.total_losses = 0
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = defaultdict(
            lambda: StrategyPerformance(strategy_name="default")
        )
        
        # Current game tracking
        self.current_game_id: Optional[str] = None
        self.current_game_start: Optional[float] = None
        self.current_game_territory_history: List[float] = []
        self.current_game_action_history: List[str] = []
        
        # Load previous progress
        self._load_progress()
        
        logger.info(f"LearningSystem initialized with {len(self.game_history)} previous games")
    
    # ── Game Lifecycle ───────────────────────────────────────────────────────
    
    def start_new_game(self, game_id: Optional[str] = None) -> str:
        """Start tracking a new game session."""
        self.current_game_id = game_id or f"game_{int(time.time())}"
        self.current_game_start = time.time()
        self.current_game_territory_history = []
        self.current_game_action_history = []
        
        logger.debug(f"Started tracking game: {self.current_game_id}")
        return self.current_game_id
    
    def record_game_tick(self, territory_pct: float, action: str):
        """Record progress during an active game."""
        if self.current_game_id:
            self.current_game_territory_history.append(territory_pct)
            self.current_game_action_history.append(action)
    
    def end_game(self, won: bool, final_territory_pct: float, 
                 enemies_encountered: int, final_enemy_count: int) -> GameRecord:
        """End game tracking and store results."""
        if not self.current_game_id or self.current_game_start is None:
            logger.warning("No active game to end")
            return None
        
        duration = time.time() - self.current_game_start
        
        record = GameRecord(
            game_id=self.current_game_id,
            timestamp=self.current_game_start,
            won=won,
            final_territory_pct=final_territory_pct,
            duration_seconds=duration,
            enemies_encountered=enemies_encountered,
            total_ticks=len(self.current_game_territory_history),
            territory_history=self.current_game_territory_history.copy(),
            action_history=self.current_game_action_history.copy(),
            final_enemy_count=final_enemy_count
        )
        
        self.game_history.append(record)
        self.total_games += 1
        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
        
        # Update strategy performance
        self._update_strategy_performance(record)
        
        # Save progress
        self._save_progress()
        
        logger.info(f"Game {self.current_game_id} ended: {'WIN' if won else 'LOSS'} | "
                   f"Territory: {final_territory_pct:.1%} | Duration: {duration:.0f}s")
        
        # Reset current game
        self.current_game_id = None
        self.current_game_start = None
        
        return record
    
    # ── Strategy Analysis ────────────────────────────────────────────────────
    
    def _update_strategy_performance(self, record: GameRecord):
        """Update performance metrics for the current strategy."""
        # Determine strategy based on game characteristics
        strategy = self._classify_game_strategy(record)
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = StrategyPerformance(strategy_name=strategy)
        
        perf = self.strategy_performance[strategy]
        
        # Update running averages
        n = perf.games_played
        perf.games_played += 1
        perf.wins += 1 if record.won else 0
        
        # Rolling average updates
        perf.avg_territory = (perf.avg_territory * n + record.final_territory_pct) / (n + 1)
        perf.avg_duration = (perf.avg_duration * n + record.duration_seconds) / (n + 1)
        
        # Calculate aggression based on action history
        if record.action_history:
            attack_actions = sum(1 for a in record.action_history if 'ATTACK' in a)
            perf.aggression_score = attack_actions / len(record.action_history)
    
    def _classify_game_strategy(self, record: GameRecord) -> str:
        """Classify the strategy used in a game based on characteristics."""
        if not record.action_history:
            return "conservative"
        
        attack_count = sum(1 for a in record.action_history if 'ATTACK' in a)
        attack_ratio = attack_count / len(record.action_history) if record.action_history else 0
        
        # Analyze territory progression
        if len(record.territory_history) > 10:
            early_avg = np.mean(record.territory_history[:len(record.territory_history)//3])
            late_avg = np.mean(record.territory_history[-len(record.territory_history)//3:])
            growth_rate = (late_avg - early_avg) / max(0.01, early_avg)
        else:
            growth_rate = 0
        
        # Classify strategy
        if attack_ratio > 0.7 and growth_rate > 0.5:
            return "aggressive_expander"
        elif attack_ratio > 0.5:
            return "aggressive"
        elif record.final_territory_pct > 0.5:
            return "consolidator"
        elif growth_rate > 0.3:
            return "expander"
        else:
            return "conservative"
    
    def get_best_strategy(self) -> str:
        """Get the strategy with the highest success score."""
        if not self.strategy_performance:
            return "conservative"
        
        best = max(
            self.strategy_performance.values(),
            key=lambda s: s.success_score
        )
        
        logger.info(f"Best strategy: {best.strategy_name} "
                   f"(win_rate={best.win_rate:.1%}, success={best.success_score:.2f})")
        
        return best.strategy_name
    
    def get_recommended_aggression(self) -> float:
        """Get recommended aggression level based on historical performance."""
        if not self.strategy_performance:
            return 0.5  # Default moderate aggression
        
        best = max(
            self.strategy_performance.values(),
            key=lambda s: s.success_score
        )
        
        # Convert aggression score to aggression level
        return min(1.0, max(0.0, best.aggression_score))
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        return {
            "total_games": self.total_games,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": self.total_wins / max(1, self.total_games),
            "best_strategy": self.get_best_strategy(),
            "recommended_aggression": self.get_recommended_aggression(),
            "strategy_count": len(self.strategy_performance)
        }
    
    # ── Persistence ──────────────────────────────────────────────────────────
    
    def _save_progress(self):
        """Save learning progress to disk."""
        try:
            data = {
                "game_history": [r.to_dict() for r in self.game_history[-100:]],  # Keep last 100
                "total_games": self.total_games,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "strategy_performance": {
                    name: asdict(perf) 
                    for name, perf in self.strategy_performance.items()
                }
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Learning progress saved to {self.save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save learning progress: {e}")
    
    def _load_progress(self):
        """Load learning progress from disk."""
        if not self.save_path.exists():
            logger.info("No previous learning progress found, starting fresh")
            return
        
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            self.total_games = data.get("total_games", 0)
            self.total_wins = data.get("total_wins", 0)
            self.total_losses = data.get("total_losses", 0)
            
            # Restore game history
            for game_data in data.get("game_history", []):
                self.game_history.append(GameRecord(**game_data))
            
            # Restore strategy performance
            for name, perf_data in data.get("strategy_performance", {}):
                self.strategy_performance[name] = StrategyPerformance(**perf_data)
            
            logger.info(f"Loaded {len(self.game_history)} previous games and "
                       f"{len(self.strategy_performance)} strategy profiles")
            
        except Exception as e:
            logger.error(f"Failed to load learning progress: {e}")
    
    # ── Adaptive Parameters ─────────────────────────────────────────────────
    
    def get_adaptive_parameters(self) -> Dict:
        """
        Get adaptive parameters based on learned experience.
        Returns dict with recommended parameter adjustments.
        """
        summary = self.get_performance_summary()
        
        params = {
            "aggression": summary["recommended_aggression"],
            "defense_threshold": 0.4,
            "expansion_drive": 0.7,
        }
        
        # Adjust based on recent performance
        recent_games = self.game_history[-10:] if self.game_history else []
        
        if recent_games:
            recent_wins = sum(1 for g in recent_games if g.won)
            recent_win_rate = recent_wins / len(recent_games)
            
            # If losing too often, become more defensive
            if recent_win_rate < 0.3:
                params["aggression"] = max(0.2, params["aggression"] * 0.7)
                params["defense_threshold"] = 0.3  # Defend more readily
                
            # If winning easily, become more aggressive
            elif recent_win_rate > 0.7:
                params["aggression"] = min(1.0, params["aggression"] * 1.2)
                params["expansion_drive"] = min(1.0, params["expansion_drive"] * 1.1)
        
        return params
