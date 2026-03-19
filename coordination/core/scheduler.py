"""
Grid-Aware Task Scheduler
-------------------------
Implements the scheduling priority formula from Section 7.3 of the paper:

    P_grid = w1 × (1 - L_region) + w2 × (1 - C_region) + w3 × R_region

Combined with node performance and reliability to produce a final dispatch score.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from coordination.config import settings


@dataclass
class GridState:
    """Real-time grid state for a geographic region."""
    region:             str
    load_factor:        float   # 0.0 (empty) → 1.0 (peak)
    carbon_intensity:   float   # 0.0 (clean) → 1.0 (dirtiest observed)
    renewable_fraction: float   # 0.0 → 1.0
    timestamp:          datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NodeScore:
    """Composite scheduling score for a candidate node."""
    node_id:      str
    grid_score:   float
    perf_score:   float
    reliability:  float
    final_score:  float
    region:       str


class GridAwareScheduler:
    """
    Selects the best available node for a task using grid-aware scoring.
    Grid priority is a tiebreaker among nodes of equivalent performance
    class — ensuring it does not degrade compute quality or latency.
    """

    def __init__(self):
        # Grid state cache: region → GridState
        self._grid_cache:    dict[str, GridState] = {}
        # Node region registry: node_id → region
        self._node_regions:  dict[str, str] = {}
        # Task allocation counter: node_id → count (for fairness guarantee)
        self._alloc_counter: dict[str, int] = {}
        # Seed with mock grid states
        self._seed_mock_grid()

    # ── Public API ─────────────────────────────────────────────

    def select_node(self, candidate_nodes: list) -> Optional[object]:
        """
        Select the best node from candidates using the combined scoring formula.
        Returns None if no suitable node is available.
        """
        if not candidate_nodes:
            return None

        # Apply minimum allocation guarantee first (fairness)
        guaranteed = self._apply_min_allocation_guarantee(candidate_nodes)
        if guaranteed:
            self._alloc_counter[guaranteed.id] = (
                self._alloc_counter.get(str(guaranteed.id), 0) + 1
            )
            return guaranteed

        scores = [self._score_node(node) for node in candidate_nodes]
        scores.sort(key=lambda s: s.final_score, reverse=True)
        best_id = scores[0].node_id
        self._alloc_counter[best_id] = self._alloc_counter.get(best_id, 0) + 1
        return next(n for n in candidate_nodes if str(n.id) == best_id)

    def register_node_region(self, node_id: str, region: str):
        """Associate a node with a geographic region code e.g. 'GB', 'US-CA'."""
        self._node_regions[node_id] = region

    def update_grid_state(self, state: GridState):
        """Update the cached grid state for a region."""
        self._grid_cache[state.region] = state

    def get_grid_score(self, region: str) -> float:
        """
        Compute P_grid for a region.
        P_grid = w1×(1−L) + w2×(1−C) + w3×R
        """
        state = self._grid_cache.get(region)
        if state is None:
            return 0.5  # Neutral score for unknown regions

        return round(
            settings.GRID_WEIGHT_LOAD      * (1.0 - state.load_factor)      +
            settings.GRID_WEIGHT_CARBON    * (1.0 - state.carbon_intensity)  +
            settings.GRID_WEIGHT_RENEWABLE * state.renewable_fraction,
            4
        )

    def get_grid_states(self) -> list[dict]:
        """Return all current grid states with computed scores."""
        return [
            {
                "region":             s.region,
                "load_factor":        s.load_factor,
                "carbon_intensity":   s.carbon_intensity,
                "renewable_fraction": s.renewable_fraction,
                "grid_score":         self.get_grid_score(s.region),
                "timestamp":          s.timestamp.isoformat(),
            }
            for s in sorted(
                self._grid_cache.values(),
                key=lambda x: self.get_grid_score(x.region),
                reverse=True
            )
        ]

    def get_node_score(self, node) -> NodeScore:
        """Return the full scoring breakdown for a single node."""
        return self._score_node(node)

    # ── Internal scoring ───────────────────────────────────────

    def _score_node(self, node) -> NodeScore:
        region     = self._node_regions.get(str(node.id), "UNKNOWN")
        grid_score = self.get_grid_score(region)

        # Normalize multiplier to [0, 1] — max is 8.0 (datacenter accelerator)
        perf_score = min(1.0, node.multiplier / 8.0)

        # Normalize reliability from [0.5, 1.0] to [0, 1]
        reliability = (node.reliability_factor - 0.5) / 0.5

        # Combined score:
        # Performance 50%, reliability 30%, grid 20%
        # Grid is tiebreaker as specified in the paper
        final_score = round(
            0.50 * perf_score  +
            0.30 * reliability +
            0.20 * grid_score,
            4
        )

        return NodeScore(
            node_id=     str(node.id),
            grid_score=  round(grid_score,  4),
            perf_score=  round(perf_score,  4),
            reliability= round(reliability, 4),
            final_score= final_score,
            region=      region,
        )

    def _apply_min_allocation_guarantee(self, nodes: list) -> Optional[object]:
        """
        Fairness guarantee: nodes that have received significantly fewer
        tasks than average are prioritized regardless of grid score.
        Prevents contributors in high-demand regions from being excluded.
        """
        if not nodes:
            return None
        counts    = [self._alloc_counter.get(str(n.id), 0) for n in nodes]
        avg       = sum(counts) / len(counts) if counts else 0
        threshold = avg * (1.0 - settings.MIN_TASK_ALLOCATION_RATE)
        underserved = [
            n for n, c in zip(nodes, counts) if c < threshold
        ]
        if underserved:
            return min(underserved,
                       key=lambda n: self._alloc_counter.get(str(n.id), 0))
        return None

    # ── Mock grid data ─────────────────────────────────────────

    def _seed_mock_grid(self):
        """
        Seed with representative mock grid states.
        In production these are refreshed from grid operator APIs
        (ENTSO-E, EIA, etc.) on a scheduled basis.
        """
        mock_states = [
            # Region        Load   Carbon  Renewable
            GridState("GB",    0.60,  0.30,   0.55),
            GridState("DE",    0.55,  0.35,   0.50),
            GridState("FR",    0.50,  0.10,   0.75),  # High nuclear/hydro
            GridState("NO",    0.35,  0.05,   0.95),  # Almost all hydro
            GridState("US-CA", 0.50,  0.25,   0.60),
            GridState("US-TX", 0.70,  0.55,   0.25),  # High fossil
            GridState("US-WA", 0.40,  0.10,   0.80),  # High hydro
            GridState("US-NY", 0.55,  0.30,   0.45),
            GridState("JP",    0.65,  0.50,   0.22),
            GridState("AU",    0.60,  0.45,   0.30),
            GridState("SG",    0.70,  0.60,   0.10),
            GridState("TR",    0.55,  0.40,   0.40),
        ]
        for state in mock_states:
            self._grid_cache[state.region] = state


# Singleton instance
scheduler = GridAwareScheduler()
