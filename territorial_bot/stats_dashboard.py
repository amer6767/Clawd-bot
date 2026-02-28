"""
Stats Dashboard for Territorial.io Bot
========================================
Real-time statistics aggregation and reporting system.

Collects metrics from all bot subsystems (Vision, Brain, Action, EnemyTracker,
LearningSystem, PerformanceOptimizer, MapAnalyzer) and provides:
- Live console dashboard
- Periodic log summaries
- Session statistics export (JSON)
- Performance trend analysis

Usage:
    dashboard = StatsDashboard()
    dashboard.update(game_state, decision, analysis, perf_metrics)
    dashboard.print_summary()
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TickStats:
    """Statistics for a single bot tick."""
    tick: int
    timestamp: float
    own_territory_pct: float
    enemy_territory_pct: float
    neutral_territory_pct: float
    game_phase: str
    action_name: str
    decision_confidence: float
    vulnerability_score: float
    expansion_opportunity: float
    fps: float
    epsilon: float


@dataclass
class EpisodeStats:
    """Statistics for a complete game episode."""
    episode_id: int
    start_time: float
    end_time: float = 0.0
    won: bool = False
    peak_territory: float = 0.0
    final_territory: float = 0.0
    total_ticks: int = 0
    total_reward: float = 0.0
    avg_fps: float = 0.0
    actions_taken: Dict[str, int] = field(default_factory=dict)
    territory_timeline: List[float] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["duration_seconds"] = self.duration_seconds
        return d


@dataclass
class SessionStats:
    """Statistics for the entire bot session."""
    session_start: float = field(default_factory=time.time)
    total_ticks: int = 0
    total_episodes: int = 0
    total_wins: int = 0
    total_losses: int = 0
    peak_territory_ever: float = 0.0
    total_reward: float = 0.0
    avg_episode_duration: float = 0.0
    action_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.total_wins / max(1, self.total_episodes)

    @property
    def session_duration(self) -> float:
        return time.time() - self.session_start


# ─────────────────────────────────────────────
# STATS DASHBOARD CLASS
# ─────────────────────────────────────────────

class StatsDashboard:
    """
    Centralized statistics collection and reporting for the Territorial.io bot.

    Aggregates data from all subsystems and provides:
    - Per-tick metric tracking
    - Per-episode summaries
    - Session-level statistics
    - Trend analysis (moving averages)
    - JSON export for post-session analysis
    """

    def __init__(
        self,
        save_path: str = "logs/session_stats.json",
        history_window: int = 100,
        log_interval_ticks: int = 50,
    ):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.history_window = history_window
        self.log_interval_ticks = log_interval_ticks

        # Rolling history (bounded deque for memory efficiency)
        self.tick_history: deque = deque(maxlen=history_window)
        self.fps_history: deque = deque(maxlen=30)
        self.territory_history: deque = deque(maxlen=history_window)
        self.reward_history: deque = deque(maxlen=history_window)

        # Episode tracking
        self.current_episode: Optional[EpisodeStats] = None
        self.episode_history: List[EpisodeStats] = []

        # Session-level stats
        self.session = SessionStats()

        # Trend tracking
        self._territory_trend: float = 0.0  # Positive = growing, negative = shrinking
        self._last_territory: float = 0.0

        logger.info("StatsDashboard initialized.")

    # ── Episode Lifecycle ──────────────────────────────────────────────────

    def start_episode(self, episode_id: int):
        """Begin tracking a new game episode."""
        self.current_episode = EpisodeStats(
            episode_id=episode_id,
            start_time=time.time(),
        )
        logger.debug(f"Episode {episode_id} started.")

    def end_episode(self, won: bool, final_territory: float, total_reward: float):
        """Finalize the current episode and store its stats."""
        if self.current_episode is None:
            return

        ep = self.current_episode
        ep.end_time = time.time()
        ep.won = won
        ep.final_territory = final_territory
        ep.total_reward = total_reward

        # Compute average FPS for episode
        if self.fps_history:
            ep.avg_fps = float(np.mean(list(self.fps_history)))

        # Update session stats
        self.session.total_episodes += 1
        if won:
            self.session.total_wins += 1
        else:
            self.session.total_losses += 1
        self.session.total_reward += total_reward
        self.session.peak_territory_ever = max(
            self.session.peak_territory_ever, ep.peak_territory
        )

        # Update average episode duration
        n = self.session.total_episodes
        self.session.avg_episode_duration = (
            (self.session.avg_episode_duration * (n - 1) + ep.duration_seconds) / n
        )

        self.episode_history.append(ep)
        self.current_episode = None

        logger.info(
            f"Episode {ep.episode_id} ended | "
            f"{'WIN' if won else 'LOSS'} | "
            f"Territory: {final_territory:.1%} | "
            f"Duration: {ep.duration_seconds:.0f}s | "
            f"Reward: {total_reward:.2f}"
        )

        # Auto-save after each episode
        self._save_stats()

    # ── Per-Tick Update ────────────────────────────────────────────────────

    def update(
        self,
        tick: int,
        game_state,
        decision,
        fps: float = 0.0,
        epsilon: float = 0.0,
        vulnerability: float = 0.0,
        expansion_opportunity: float = 0.0,
        reward: float = 0.0,
    ):
        """
        Update dashboard with data from the current tick.

        Args:
            tick: Current tick number.
            game_state: GameState from VisionSystem.
            decision: Decision from BrainSystem.
            fps: Current frames per second.
            epsilon: Current exploration rate.
            vulnerability: Map vulnerability score.
            expansion_opportunity: Map expansion opportunity score.
            reward: Reward received this tick.
        """
        # Build tick stats
        tick_stat = TickStats(
            tick=tick,
            timestamp=time.time(),
            own_territory_pct=game_state.own_territory_pct,
            enemy_territory_pct=game_state.enemy_territory_pct,
            neutral_territory_pct=game_state.neutral_territory_pct,
            game_phase=game_state.game_phase,
            action_name=decision.action_name if decision else "NONE",
            decision_confidence=decision.confidence if decision else 0.0,
            vulnerability_score=vulnerability,
            expansion_opportunity=expansion_opportunity,
            fps=fps,
            epsilon=epsilon,
        )

        self.tick_history.append(tick_stat)
        self.fps_history.append(fps)
        self.territory_history.append(game_state.own_territory_pct)
        self.reward_history.append(reward)

        # Update session totals
        self.session.total_ticks += 1
        action_name = decision.action_name if decision else "NONE"
        self.session.action_distribution[action_name] = (
            self.session.action_distribution.get(action_name, 0) + 1
        )

        # Update current episode
        if self.current_episode is not None:
            self.current_episode.total_ticks += 1
            self.current_episode.territory_timeline.append(
                game_state.own_territory_pct
            )
            self.current_episode.peak_territory = max(
                self.current_episode.peak_territory,
                game_state.own_territory_pct,
            )
            self.current_episode.actions_taken[action_name] = (
                self.current_episode.actions_taken.get(action_name, 0) + 1
            )

        # Compute territory trend
        self._update_territory_trend(game_state.own_territory_pct)

        # Periodic logging
        if tick % self.log_interval_ticks == 0:
            self._log_periodic_summary(tick)

    # ── Trend Analysis ─────────────────────────────────────────────────────

    def _update_territory_trend(self, current_territory: float):
        """Update territory trend (positive = growing, negative = shrinking)."""
        if len(self.territory_history) >= 10:
            recent = list(self.territory_history)[-10:]
            if len(recent) >= 2:
                self._territory_trend = recent[-1] - recent[0]
        self._last_territory = current_territory

    def get_territory_trend(self) -> str:
        """Get human-readable territory trend."""
        if self._territory_trend > 0.02:
            return "↑ Growing"
        elif self._territory_trend < -0.02:
            return "↓ Shrinking"
        else:
            return "→ Stable"

    def get_moving_avg_territory(self, window: int = 20) -> float:
        """Get moving average of own territory percentage."""
        if not self.territory_history:
            return 0.0
        recent = list(self.territory_history)[-window:]
        return float(np.mean(recent))

    def get_moving_avg_fps(self) -> float:
        """Get moving average FPS."""
        if not self.fps_history:
            return 0.0
        return float(np.mean(list(self.fps_history)))

    def get_moving_avg_reward(self, window: int = 20) -> float:
        """Get moving average reward per tick."""
        if not self.reward_history:
            return 0.0
        recent = list(self.reward_history)[-window:]
        return float(np.mean(recent))

    # ── Reporting ──────────────────────────────────────────────────────────

    def _log_periodic_summary(self, tick: int):
        """Log a periodic summary to the logger."""
        avg_territory = self.get_moving_avg_territory()
        avg_fps = self.get_moving_avg_fps()
        trend = self.get_territory_trend()
        win_rate = self.session.win_rate

        logger.info(
            f"[Dashboard] Tick {tick:6d} | "
            f"Avg Territory: {avg_territory:.1%} {trend} | "
            f"FPS: {avg_fps:.1f} | "
            f"Win Rate: {win_rate:.1%} | "
            f"Episodes: {self.session.total_episodes}"
        )

    def print_summary(self):
        """Print a formatted summary to stdout."""
        lines = [
            "=" * 60,
            "  TERRITORIAL.IO BOT - SESSION SUMMARY",
            "=" * 60,
            f"  Session Duration:   {self.session.session_duration:.0f}s",
            f"  Total Ticks:        {self.session.total_ticks}",
            f"  Total Episodes:     {self.session.total_episodes}",
            f"  Win Rate:           {self.session.win_rate:.1%}",
            f"  Peak Territory:     {self.session.peak_territory_ever:.1%}",
            f"  Total Reward:       {self.session.total_reward:.2f}",
            f"  Avg Episode Length: {self.session.avg_episode_duration:.0f}s",
            f"  Avg FPS:            {self.get_moving_avg_fps():.1f}",
            "",
            "  Action Distribution:",
        ]

        total_actions = sum(self.session.action_distribution.values())
        for action, count in sorted(
            self.session.action_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = count / max(1, total_actions)
            lines.append(f"    {action:<15s} {count:5d} ({pct:.1%})")

        if self.episode_history:
            lines.append("")
            lines.append("  Recent Episodes:")
            for ep in self.episode_history[-5:]:
                result = "WIN " if ep.won else "LOSS"
                lines.append(
                    f"    Ep {ep.episode_id:3d}: {result} | "
                    f"Territory: {ep.final_territory:.1%} | "
                    f"Duration: {ep.duration_seconds:.0f}s"
                )

        lines.append("=" * 60)
        print("\n".join(lines))

    def get_live_status_line(self) -> str:
        """
        Get a compact single-line status string for real-time display.
        Suitable for embedding in log messages or terminal output.
        """
        avg_terr = self.get_moving_avg_territory()
        avg_fps = self.get_moving_avg_fps()
        trend = self.get_territory_trend()
        ep_id = (
            self.current_episode.episode_id
            if self.current_episode
            else self.session.total_episodes
        )

        return (
            f"Ep:{ep_id} | "
            f"Terr:{avg_terr:.1%} {trend} | "
            f"FPS:{avg_fps:.1f} | "
            f"W/L:{self.session.total_wins}/{self.session.total_losses}"
        )

    # ── Performance Analysis ───────────────────────────────────────────────

    def get_action_efficiency(self) -> Dict[str, float]:
        """
        Analyze which actions correlate with territory growth.

        Returns dict mapping action names to their average territory delta
        when that action was taken.
        """
        if len(self.tick_history) < 2:
            return {}

        ticks = list(self.tick_history)
        action_deltas: Dict[str, List[float]] = {}

        for i in range(1, len(ticks)):
            prev = ticks[i - 1]
            curr = ticks[i]
            delta = curr.own_territory_pct - prev.own_territory_pct
            action = prev.action_name

            if action not in action_deltas:
                action_deltas[action] = []
            action_deltas[action].append(delta)

        return {
            action: float(np.mean(deltas))
            for action, deltas in action_deltas.items()
            if deltas
        }

    def get_phase_performance(self) -> Dict[str, Dict]:
        """
        Analyze performance by game phase (early/mid/late).

        Returns dict with per-phase statistics.
        """
        phase_data: Dict[str, List[float]] = {
            "early": [], "mid": [], "late": []
        }

        for tick in self.tick_history:
            phase = tick.game_phase
            if phase in phase_data:
                phase_data[phase].append(tick.own_territory_pct)

        result = {}
        for phase, territories in phase_data.items():
            if territories:
                result[phase] = {
                    "avg_territory": float(np.mean(territories)),
                    "max_territory": float(np.max(territories)),
                    "tick_count": len(territories),
                }
        return result

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_stats(self):
        """Save session statistics to JSON file."""
        try:
            data = {
                "session": {
                    "start_time": self.session.session_start,
                    "duration_seconds": self.session.session_duration,
                    "total_ticks": self.session.total_ticks,
                    "total_episodes": self.session.total_episodes,
                    "total_wins": self.session.total_wins,
                    "total_losses": self.session.total_losses,
                    "win_rate": self.session.win_rate,
                    "peak_territory_ever": self.session.peak_territory_ever,
                    "total_reward": self.session.total_reward,
                    "avg_episode_duration": self.session.avg_episode_duration,
                    "action_distribution": self.session.action_distribution,
                },
                "episodes": [ep.to_dict() for ep in self.episode_history[-50:]],
                "action_efficiency": self.get_action_efficiency(),
                "phase_performance": self.get_phase_performance(),
            }

            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Stats saved to {self.save_path}")

        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def load_stats(self):
        """Load previous session statistics from JSON file."""
        if not self.save_path.exists():
            return

        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)

            session_data = data.get("session", {})
            self.session.total_episodes = session_data.get("total_episodes", 0)
            self.session.total_wins = session_data.get("total_wins", 0)
            self.session.total_losses = session_data.get("total_losses", 0)
            self.session.peak_territory_ever = session_data.get("peak_territory_ever", 0.0)
            self.session.total_reward = session_data.get("total_reward", 0.0)
            self.session.avg_episode_duration = session_data.get("avg_episode_duration", 0.0)
            self.session.action_distribution = session_data.get("action_distribution", {})

            logger.info(
                f"Loaded previous stats: {self.session.total_episodes} episodes, "
                f"win rate {self.session.win_rate:.1%}"
            )

        except Exception as e:
            logger.error(f"Failed to load stats: {e}")

    def export_full_report(self, path: str = None) -> str:
        """
        Export a full human-readable report to a text file.

        Returns the path of the exported file.
        """
        report_path = Path(path or "logs/bot_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("TERRITORIAL.IO BOT - FULL SESSION REPORT")
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        # Session overview
        lines.append("SESSION OVERVIEW")
        lines.append(f"  Duration:         {self.session.session_duration:.0f}s")
        lines.append(f"  Total Ticks:      {self.session.total_ticks}")
        lines.append(f"  Total Episodes:   {self.session.total_episodes}")
        lines.append(f"  Win Rate:         {self.session.win_rate:.1%}")
        lines.append(f"  Peak Territory:   {self.session.peak_territory_ever:.1%}")
        lines.append(f"  Total Reward:     {self.session.total_reward:.2f}")
        lines.append("")

        # Action efficiency
        lines.append("ACTION EFFICIENCY (avg territory delta per action)")
        efficiency = self.get_action_efficiency()
        for action, delta in sorted(efficiency.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {action:<15s} {delta:+.4f}")
        lines.append("")

        # Phase performance
        lines.append("PERFORMANCE BY GAME PHASE")
        phase_perf = self.get_phase_performance()
        for phase, stats in phase_perf.items():
            lines.append(
                f"  {phase.upper():<8s}: "
                f"avg={stats['avg_territory']:.1%}, "
                f"max={stats['max_territory']:.1%}, "
                f"ticks={stats['tick_count']}"
            )
        lines.append("")

        # Episode history
        lines.append("EPISODE HISTORY (last 20)")
        for ep in self.episode_history[-20:]:
            result = "WIN " if ep.won else "LOSS"
            lines.append(
                f"  Ep {ep.episode_id:3d}: {result} | "
                f"Territory: {ep.final_territory:.1%} | "
                f"Peak: {ep.peak_territory:.1%} | "
                f"Duration: {ep.duration_seconds:.0f}s | "
                f"Reward: {ep.total_reward:.2f}"
            )

        report_text = "\n".join(lines)

        try:
            with open(report_path, "w") as f:
                f.write(report_text)
            logger.info(f"Full report exported to {report_path}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")

        return str(report_path)
