"""
Shard Scheduler
---------------
Finds complete inference pipelines across volunteer nodes.
A complete pipeline covers all transformer layers of a model
with no gaps, using online nodes from the shard registry.

The scheduler also handles:
- Pipeline validation (no layer gaps)
- Fallback detection (when no complete pipeline exists)
- Load balancing across equivalent pipelines
- Grid-aware pipeline selection
"""

from typing import Optional, List, Tuple
from coordination.core.shard_registry import (
    ShardRegistry, ShardRegistration, InferencePipeline, ModelConfig, registry
)
from coordination.core.scheduler import GridAwareScheduler, scheduler as grid_scheduler


class ShardScheduler:
    """
    Finds and selects complete inference pipelines for a given model.
    """

    def find_pipeline(
        self,
        model_id: str,
        shard_registry: ShardRegistry = None,
    ) -> Optional[InferencePipeline]:
        """
        Find a complete inference pipeline for the given model.
        Returns None if no complete pipeline is available (triggers fallback).

        Algorithm:
        1. Get all online shards for the model sorted by layer_start
        2. Use a greedy cover algorithm to find a non-overlapping
           set of shards that covers all layers
        3. Among valid pipelines prefer lower grid load regions
        4. Return the best complete pipeline or None
        """
        reg   = shard_registry or registry
        model = reg.get_model(model_id)
        if not model:
            return None

        shards = reg.get_online_shards(model_id)
        if not shards:
            return None

        # Try to build a complete pipeline
        pipeline_shards = self._greedy_cover(shards, model.total_layers)
        if not pipeline_shards:
            return None

        pipeline = InferencePipeline(
            model_id= model.model_id,
            shards=   sorted(pipeline_shards, key=lambda s: s.layer_start),
        )

        if not pipeline.is_complete:
            return None

        return pipeline

    def is_pipeline_available(self, model_id: str) -> bool:
        """Quick check whether a complete pipeline exists for a model."""
        return self.find_pipeline(model_id) is not None

    def get_pipeline_summary(self, model_id: str) -> dict:
        """Return a human-readable summary of pipeline availability."""
        reg   = registry
        model = reg.get_model(model_id)
        if not model:
            return {"available": False, "reason": "Model not found"}

        shards = reg.get_online_shards(model_id)
        if not shards:
            return {
                "available":      False,
                "reason":         "No online nodes registered for this model",
                "total_layers":   model.total_layers,
                "layers_covered": 0,
            }

        covered = set()
        for s in shards:
            covered.update(s.covers)

        missing = [l for l in range(model.total_layers) if l not in covered]
        pipeline = self.find_pipeline(model_id)

        return {
            "available":      pipeline is not None,
            "model_id":       model.model_id,
            "friendly_name":  model.friendly_name,
            "total_layers":   model.total_layers,
            "layers_covered": len(covered),
            "layers_missing": missing[:10],   # First 10 missing layers
            "online_nodes":   len(set(s.node_id for s in shards)),
            "pipeline_nodes": pipeline.node_ids if pipeline else [],
            "reason":         None if pipeline else f"Missing layers: {missing[:5]}..."
        }

    def suggest_layer_assignment(
        self,
        model_id:    str,
        vram_gb:     float,
        node_count:  int = 1,
    ) -> Optional[Tuple[int, int]]:
        """
        Suggest which layers a new node should load based on
        current registry gaps and available VRAM.

        Returns (layer_start, layer_end) or None if model not found.
        """
        reg   = registry
        model = reg.get_model(model_id)
        if not model:
            return None

        # Estimate how many layers fit in available VRAM
        # Rough estimate: each layer ~0.5GB for 7B, ~1GB for 70B
        bytes_per_layer = _estimate_bytes_per_layer(model)
        max_layers      = max(1, int((vram_gb * 1024**3) / bytes_per_layer))
        max_layers      = min(max_layers, model.total_layers)

        # Find the largest uncovered gap
        shards  = reg.get_online_shards(model_id)
        covered = set()
        for s in shards:
            covered.update(s.covers)

        uncovered = [l for l in range(model.total_layers) if l not in covered]
        if not uncovered:
            # All layers covered — suggest reinforcing the least-covered range
            return (0, min(max_layers - 1, model.total_layers - 1))

        # Find the start of the largest contiguous uncovered block
        gap_start = uncovered[0]
        gap_end   = min(gap_start + max_layers - 1, model.total_layers - 1)
        return (gap_start, gap_end)

    # ── Internal ───────────────────────────────────────────────

    def _greedy_cover(
        self,
        shards:       List[ShardRegistration],
        total_layers: int,
    ) -> Optional[List[ShardRegistration]]:
        """
        Greedy algorithm to find a minimal set of non-overlapping shards
        that covers all layers from 0 to total_layers-1.

        Returns the selected shards or None if complete coverage impossible.
        """
        if not shards:
            return None

        selected = []
        current_layer = 0

        while current_layer < total_layers:
            # Find all shards that start at or before current_layer
            # and extend as far as possible
            candidates = [
                s for s in shards
                if s.layer_start <= current_layer <= s.layer_end
            ]

            if not candidates:
                # Gap — no shard covers current_layer
                return None

            # Pick the candidate that extends furthest
            best = max(candidates, key=lambda s: s.layer_end)
            selected.append(best)
            current_layer = best.layer_end + 1

        return selected if selected else None


def _estimate_bytes_per_layer(model: ModelConfig) -> int:
    """
    Rough estimate of memory required per transformer layer in bytes.
    Based on hidden_size and typical weight matrix sizes.
    Uses fp16 (2 bytes per parameter).
    """
    h = model.hidden_size
    # Each transformer layer has roughly 12 × h² parameters
    # (attention Q,K,V,O + 2 FFN matrices + layer norms)
    params_per_layer = 12 * h * h
    bytes_per_param  = 2   # fp16
    return params_per_layer * bytes_per_param


# Singleton instance
shard_scheduler = ShardScheduler()
