"""
Map Analyzer for Territorial.io Bot
=====================================
Advanced map analysis including territory clustering, chokepoint detection,
strategic position scoring, and expansion path planning.

Integrates with VisionSystem's GameState to provide deeper spatial intelligence
for the BrainSystem and SmartStrategy modules.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

from vision_system import GameState, TerritoryCell

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TerritoryCluster:
    """A contiguous group of cells with the same owner."""
    cluster_id: int
    owner: str
    cells: List[Tuple[int, int]] = field(default_factory=list)
    centroid: Optional[Tuple[float, float]] = None
    size: int = 0
    perimeter: int = 0
    compactness: float = 0.0  # size / perimeter ratio (higher = more compact)

    def compute_properties(self):
        """Compute derived properties from cell list."""
        self.size = len(self.cells)
        if self.cells:
            self.centroid = (
                sum(c[0] for c in self.cells) / self.size,
                sum(c[1] for c in self.cells) / self.size,
            )
        # Perimeter is approximated by counting border cells
        self.compactness = self.size / max(1, self.perimeter)


@dataclass
class Chokepoint:
    """A narrow passage between territory regions."""
    position: Tuple[int, int]
    width: int          # Number of passable cells at narrowest point
    connects: List[str] = field(default_factory=list)  # Owners on each side
    strategic_value: float = 0.0


@dataclass
class ExpansionPath:
    """A planned expansion route from own territory into neutral/enemy."""
    start: Tuple[int, int]
    end: Tuple[int, int]
    waypoints: List[Tuple[int, int]] = field(default_factory=list)
    path_length: int = 0
    obstacle_count: int = 0
    score: float = 0.0  # Higher = better expansion path


@dataclass
class MapAnalysis:
    """Complete map analysis result for one game frame."""
    timestamp: float = 0.0

    # Territory clusters
    own_clusters: List[TerritoryCluster] = field(default_factory=list)
    enemy_clusters: List[TerritoryCluster] = field(default_factory=list)
    neutral_clusters: List[TerritoryCluster] = field(default_factory=list)

    # Chokepoints
    chokepoints: List[Chokepoint] = field(default_factory=list)

    # Expansion paths
    best_expansion_paths: List[ExpansionPath] = field(default_factory=list)

    # Strategic scores per grid cell (higher = more strategically valuable)
    strategic_heatmap: Optional[np.ndarray] = None

    # Frontier cells (own cells adjacent to non-own territory)
    frontier_cells: List[Tuple[int, int]] = field(default_factory=list)

    # Vulnerability score (0-1, higher = more vulnerable)
    vulnerability_score: float = 0.0

    # Expansion opportunity score (0-1, higher = more room to grow)
    expansion_opportunity: float = 0.0

    # Dominant direction of enemy pressure (row_delta, col_delta)
    enemy_pressure_direction: Optional[Tuple[float, float]] = None


# ─────────────────────────────────────────────
# MAP ANALYZER CLASS
# ─────────────────────────────────────────────

class MapAnalyzer:
    """
    Advanced spatial analysis of the Territorial.io game map.

    Provides:
    - Territory cluster detection (connected components)
    - Chokepoint identification (narrow passages)
    - Strategic heatmap generation
    - Expansion path planning (BFS-based)
    - Vulnerability and opportunity scoring

    Usage:
        analyzer = MapAnalyzer()
        analysis = analyzer.analyze(game_state)
        best_target = analysis.best_expansion_paths[0].end
    """

    def __init__(self):
        self._cluster_id_counter = 0
        logger.info("MapAnalyzer initialized.")

    # ── Main Analysis Entry Point ──────────────────────────────────────────

    def analyze(self, game_state: GameState) -> MapAnalysis:
        """
        Perform full map analysis on the current game state.

        Args:
            game_state: Current GameState from VisionSystem.

        Returns:
            MapAnalysis with all spatial intelligence populated.
        """
        import time
        analysis = MapAnalysis(timestamp=time.time())

        if not game_state.grid:
            return analysis

        rows = len(game_state.grid)
        cols = len(game_state.grid[0]) if rows > 0 else 0

        # 1. Find territory clusters
        analysis.own_clusters = self._find_clusters(game_state.grid, "own", rows, cols)
        analysis.enemy_clusters = self._find_clusters(game_state.grid, "enemy", rows, cols)
        analysis.neutral_clusters = self._find_clusters(game_state.grid, "neutral", rows, cols)

        # 2. Find frontier cells
        analysis.frontier_cells = self._find_frontier(game_state.grid, rows, cols)

        # 3. Detect chokepoints
        analysis.chokepoints = self._detect_chokepoints(game_state.grid, rows, cols)

        # 4. Build strategic heatmap
        analysis.strategic_heatmap = self._build_strategic_heatmap(
            game_state, rows, cols
        )

        # 5. Plan expansion paths
        analysis.best_expansion_paths = self._plan_expansion_paths(
            game_state, analysis, rows, cols
        )

        # 6. Compute vulnerability and opportunity scores
        analysis.vulnerability_score = self._compute_vulnerability(game_state, analysis)
        analysis.expansion_opportunity = self._compute_expansion_opportunity(
            game_state, analysis
        )

        # 7. Compute enemy pressure direction
        analysis.enemy_pressure_direction = self._compute_enemy_pressure_direction(
            game_state
        )

        logger.debug(
            f"MapAnalysis: {len(analysis.own_clusters)} own clusters, "
            f"{len(analysis.enemy_clusters)} enemy clusters, "
            f"{len(analysis.chokepoints)} chokepoints, "
            f"vuln={analysis.vulnerability_score:.2f}, "
            f"opp={analysis.expansion_opportunity:.2f}"
        )

        return analysis

    # ── Cluster Detection (Connected Components) ───────────────────────────

    def _find_clusters(
        self, grid: List[List[TerritoryCell]], owner: str, rows: int, cols: int
    ) -> List[TerritoryCluster]:
        """
        Find all contiguous clusters of cells with the given owner.
        Uses BFS flood-fill for connected component labeling.
        """
        visited: Set[Tuple[int, int]] = set()
        clusters: List[TerritoryCluster] = []

        for r in range(rows):
            for c in range(cols):
                if (r, c) in visited:
                    continue
                if grid[r][c].owner != owner:
                    continue

                # BFS to find all connected cells
                cluster_cells = []
                queue = deque([(r, c)])
                visited.add((r, c))

                while queue:
                    cr, cc = queue.popleft()
                    cluster_cells.append((cr, cc))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and (nr, nc) not in visited
                            and grid[nr][nc].owner == owner
                        ):
                            visited.add((nr, nc))
                            queue.append((nr, nc))

                # Build cluster object
                self._cluster_id_counter += 1
                cluster = TerritoryCluster(
                    cluster_id=self._cluster_id_counter,
                    owner=owner,
                    cells=cluster_cells,
                )

                # Compute perimeter (cells adjacent to non-owner cells)
                perimeter = 0
                for cr, cc in cluster_cells:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            perimeter += 1  # Map edge counts as perimeter
                        elif grid[nr][nc].owner != owner:
                            perimeter += 1
                cluster.perimeter = perimeter
                cluster.compute_properties()

                clusters.append(cluster)

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters

    # ── Frontier Detection ─────────────────────────────────────────────────

    def _find_frontier(
        self, grid: List[List[TerritoryCell]], rows: int, cols: int
    ) -> List[Tuple[int, int]]:
        """
        Find all 'own' cells that are adjacent to non-own territory.
        These are the frontier cells from which expansion can occur.
        """
        frontier = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c].owner != "own":
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr][nc].owner in ("neutral", "enemy"):
                            frontier.append((r, c))
                            break
        return frontier

    # ── Chokepoint Detection ───────────────────────────────────────────────

    def _detect_chokepoints(
        self, grid: List[List[TerritoryCell]], rows: int, cols: int
    ) -> List[Chokepoint]:
        """
        Detect narrow passages (chokepoints) in the map.

        A chokepoint is a cell (or small group of cells) where the passable
        width between two regions is minimal. We approximate this by finding
        cells where the number of passable neighbors is <= 2.
        """
        chokepoints = []

        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                if cell.owner == "border":
                    continue

                # Count passable neighbors
                passable_neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr][nc].owner != "border":
                            passable_neighbors.append((nr, nc))

                # A chokepoint has exactly 2 passable neighbors (corridor)
                if len(passable_neighbors) == 2:
                    # Check if the two neighbors are from different "sides"
                    n1_owner = grid[passable_neighbors[0][0]][passable_neighbors[0][1]].owner
                    n2_owner = grid[passable_neighbors[1][0]][passable_neighbors[1][1]].owner

                    if n1_owner != n2_owner:
                        strategic_value = 0.8  # High value: connects different territories
                    else:
                        strategic_value = 0.3  # Lower value: same territory corridor

                    chokepoints.append(Chokepoint(
                        position=(r, c),
                        width=1,
                        connects=[n1_owner, n2_owner],
                        strategic_value=strategic_value,
                    ))

        # Limit to most strategically valuable chokepoints
        chokepoints.sort(key=lambda cp: cp.strategic_value, reverse=True)
        return chokepoints[:20]

    # ── Strategic Heatmap ──────────────────────────────────────────────────

    def _build_strategic_heatmap(
        self, game_state: GameState, rows: int, cols: int
    ) -> np.ndarray:
        """
        Build a strategic value heatmap for the entire grid.

        Higher values indicate more strategically important cells.
        Factors:
        - Proximity to own territory (expansion value)
        - Proximity to enemy territory (threat/opportunity)
        - Distance from map center (central positions are valuable)
        - Neutral territory density (expansion potential)
        """
        heatmap = np.zeros((rows, cols), dtype=np.float32)

        center_r, center_c = rows / 2.0, cols / 2.0
        max_dist = np.sqrt(center_r**2 + center_c**2)

        for r in range(rows):
            for c in range(cols):
                cell = game_state.grid[r][c]
                score = 0.0

                # Skip border cells
                if cell.owner == "border":
                    heatmap[r][c] = 0.0
                    continue

                # Centrality bonus (central positions are more valuable)
                dist_from_center = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                centrality = 1.0 - (dist_from_center / max(1, max_dist))
                score += centrality * 0.3

                # Neutral territory is expansion opportunity
                if cell.owner == "neutral":
                    score += 0.4

                    # Bonus if adjacent to own territory
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if game_state.grid[nr][nc].owner == "own":
                                score += 0.3
                                break

                # Enemy territory adjacent to own = high priority target
                elif cell.owner == "enemy":
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if game_state.grid[nr][nc].owner == "own":
                                score += 0.5
                                break

                # Own territory near enemy = defensive priority
                elif cell.owner == "own":
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if game_state.grid[nr][nc].owner == "enemy":
                                score += 0.2
                                break

                heatmap[r][c] = np.clip(score, 0.0, 1.0)

        return heatmap

    # ── Expansion Path Planning ────────────────────────────────────────────

    def _plan_expansion_paths(
        self,
        game_state: GameState,
        analysis: MapAnalysis,
        rows: int,
        cols: int,
        max_paths: int = 5,
    ) -> List[ExpansionPath]:
        """
        Plan the best expansion paths from own territory into neutral/enemy.

        Uses BFS from frontier cells to find shortest paths to high-value targets.
        Returns up to max_paths ranked expansion paths.
        """
        if not analysis.frontier_cells:
            return []

        # Find high-value target cells (neutral or enemy with high strategic value)
        target_cells = []
        if analysis.strategic_heatmap is not None:
            for r in range(rows):
                for c in range(cols):
                    cell = game_state.grid[r][c]
                    if cell.owner in ("neutral", "enemy"):
                        value = analysis.strategic_heatmap[r][c]
                        if value > 0.5:
                            target_cells.append((r, c, value))

        if not target_cells:
            # Fall back to any neutral/enemy cell
            for r in range(rows):
                for c in range(cols):
                    if game_state.grid[r][c].owner in ("neutral", "enemy"):
                        target_cells.append((r, c, 0.3))

        if not target_cells:
            return []

        # Sort targets by strategic value
        target_cells.sort(key=lambda t: t[2], reverse=True)
        top_targets = target_cells[:10]

        # BFS from each frontier cell to find paths to top targets
        paths = []
        target_set = {(r, c) for r, c, _ in top_targets}
        target_values = {(r, c): v for r, c, v in top_targets}

        # Use a single BFS from all frontier cells simultaneously
        # (multi-source BFS)
        dist = {}
        parent = {}
        queue = deque()

        for fr, fc in analysis.frontier_cells:
            dist[(fr, fc)] = 0
            parent[(fr, fc)] = None
            queue.append((fr, fc))

        while queue:
            r, c = queue.popleft()
            current_dist = dist[(r, c)]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in dist:
                    continue
                neighbor = game_state.grid[nr][nc]
                if neighbor.owner == "border":
                    continue

                dist[(nr, nc)] = current_dist + 1
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

        # Reconstruct paths to top targets
        for tr, tc, value in top_targets:
            if (tr, tc) not in dist:
                continue

            # Reconstruct path
            waypoints = []
            current = (tr, tc)
            while current is not None:
                waypoints.append(current)
                current = parent.get(current)
            waypoints.reverse()

            if len(waypoints) < 2:
                continue

            path = ExpansionPath(
                start=waypoints[0],
                end=(tr, tc),
                waypoints=waypoints,
                path_length=dist[(tr, tc)],
                obstacle_count=sum(
                    1 for r, c in waypoints
                    if game_state.grid[r][c].owner == "enemy"
                ),
                score=value / max(1, dist[(tr, tc)]),  # Value per step
            )
            paths.append(path)

        # Sort by score descending
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:max_paths]

    # ── Vulnerability Scoring ──────────────────────────────────────────────

    def _compute_vulnerability(
        self, game_state: GameState, analysis: MapAnalysis
    ) -> float:
        """
        Compute overall vulnerability score (0-1).

        Higher score = more vulnerable to enemy attack.
        Factors:
        - Ratio of enemy border cells to own border cells
        - Number of chokepoints controlled by enemy
        - Fragmentation of own territory (multiple clusters = more vulnerable)
        """
        score = 0.0

        # Enemy border pressure
        own_borders = len(game_state.own_border_cells)
        enemy_borders = len(game_state.enemy_border_cells)
        if own_borders > 0:
            border_pressure = min(1.0, enemy_borders / own_borders)
            score += border_pressure * 0.5

        # Territory fragmentation (multiple own clusters = vulnerable)
        if len(analysis.own_clusters) > 1:
            fragmentation = min(1.0, (len(analysis.own_clusters) - 1) * 0.2)
            score += fragmentation * 0.3

        # Chokepoints controlled by enemy
        enemy_chokepoints = sum(
            1 for cp in analysis.chokepoints
            if "enemy" in cp.connects
        )
        if analysis.chokepoints:
            choke_ratio = enemy_chokepoints / len(analysis.chokepoints)
            score += choke_ratio * 0.2

        return float(np.clip(score, 0.0, 1.0))

    # ── Expansion Opportunity Scoring ─────────────────────────────────────

    def _compute_expansion_opportunity(
        self, game_state: GameState, analysis: MapAnalysis
    ) -> float:
        """
        Compute expansion opportunity score (0-1).

        Higher score = more room and opportunity to expand.
        Factors:
        - Amount of neutral territory available
        - Number of frontier cells
        - Quality of expansion paths
        """
        score = 0.0

        # Neutral territory availability
        neutral_pct = game_state.neutral_territory_pct
        score += neutral_pct * 0.4

        # Frontier size (more frontier = more expansion options)
        total_cells = max(1, (
            game_state.own_cell_count
            + game_state.enemy_cell_count
            + game_state.neutral_cell_count
        ))
        frontier_ratio = len(analysis.frontier_cells) / total_cells
        score += min(1.0, frontier_ratio * 5) * 0.3

        # Quality of best expansion path
        if analysis.best_expansion_paths:
            best_path_score = analysis.best_expansion_paths[0].score
            score += min(1.0, best_path_score) * 0.3

        return float(np.clip(score, 0.0, 1.0))

    # ── Enemy Pressure Direction ───────────────────────────────────────────

    def _compute_enemy_pressure_direction(
        self, game_state: GameState
    ) -> Optional[Tuple[float, float]]:
        """
        Compute the dominant direction from which enemy pressure is coming.

        Returns a normalized (row_delta, col_delta) vector pointing toward
        the main enemy threat, or None if no enemy borders detected.
        """
        if not game_state.enemy_border_cells or not game_state.own_border_cells:
            return None

        # Compute centroid of own territory
        own_r = sum(c[0] for c in game_state.own_border_cells) / len(game_state.own_border_cells)
        own_c = sum(c[1] for c in game_state.own_border_cells) / len(game_state.own_border_cells)

        # Compute centroid of enemy borders
        enemy_r = sum(c[0] for c in game_state.enemy_border_cells) / len(game_state.enemy_border_cells)
        enemy_c = sum(c[1] for c in game_state.enemy_border_cells) / len(game_state.enemy_border_cells)

        # Direction from own to enemy
        dr = enemy_r - own_r
        dc = enemy_c - own_c
        magnitude = np.sqrt(dr**2 + dc**2)

        if magnitude < 0.001:
            return None

        return (dr / magnitude, dc / magnitude)

    # ── Utility Methods ────────────────────────────────────────────────────

    def get_best_attack_target_from_analysis(
        self, analysis: MapAnalysis, game_state: GameState
    ) -> Optional[Tuple[int, int]]:
        """
        Get the best attack target cell using full map analysis.

        Combines expansion path planning with strategic heatmap to find
        the optimal target, considering both opportunity and threat.
        """
        # Prefer the end of the best expansion path
        if analysis.best_expansion_paths:
            return analysis.best_expansion_paths[0].end

        # Fall back to highest-value cell on strategic heatmap
        if analysis.strategic_heatmap is not None:
            rows, cols = analysis.strategic_heatmap.shape
            best_val = -1.0
            best_cell = None
            for r in range(rows):
                for c in range(cols):
                    if game_state.grid[r][c].owner in ("neutral", "enemy"):
                        val = analysis.strategic_heatmap[r][c]
                        if val > best_val:
                            best_val = val
                            best_cell = (r, c)
            return best_cell

        return game_state.best_attack_target

    def get_defensive_priority_cells(
        self, analysis: MapAnalysis, game_state: GameState
    ) -> List[Tuple[int, int]]:
        """
        Get cells that should be prioritized for defense.

        Returns own border cells sorted by vulnerability (most vulnerable first).
        """
        if not game_state.own_border_cells:
            return []

        # Score each own border cell by proximity to enemy borders
        scored = []
        for own_cell in game_state.own_border_cells:
            if not game_state.enemy_border_cells:
                scored.append((own_cell, 0.0))
                continue

            min_enemy_dist = min(
                abs(own_cell[0] - ec[0]) + abs(own_cell[1] - ec[1])
                for ec in game_state.enemy_border_cells
            )
            # Lower distance = higher vulnerability
            vulnerability = 1.0 / max(1, min_enemy_dist)
            scored.append((own_cell, vulnerability))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cell for cell, _ in scored]

    def summarize(self, analysis: MapAnalysis) -> Dict:
        """Return a human-readable summary of the map analysis."""
        return {
            "own_clusters": len(analysis.own_clusters),
            "enemy_clusters": len(analysis.enemy_clusters),
            "neutral_clusters": len(analysis.neutral_clusters),
            "chokepoints": len(analysis.chokepoints),
            "frontier_cells": len(analysis.frontier_cells),
            "expansion_paths": len(analysis.best_expansion_paths),
            "vulnerability": f"{analysis.vulnerability_score:.2f}",
            "expansion_opportunity": f"{analysis.expansion_opportunity:.2f}",
            "enemy_pressure_dir": analysis.enemy_pressure_direction,
        }
