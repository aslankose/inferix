"""
Shard Registry
--------------
Maintains the registry of which contributor nodes are serving
which layers of which models. Enables the shard scheduler to
find complete inference pipelines across volunteer nodes.

A shard is defined as a contiguous range of transformer layers
for a specific model hosted by a specific node.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Set, Dict
import threading


# ── Data structures ────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Configuration for a supported model.
    Model-agnostic — works with any HuggingFace transformer.
    """
    model_id:         str         # e.g. "meta-llama/Meta-Llama-3-70B"
    friendly_name:    str         # e.g. "llama3-70b"
    total_layers:     int         # Total transformer layers
    hidden_size:      int         # Hidden dimension size
    flops_per_layer:  float       # GFLOPs per layer per token (estimated)
    min_nodes:        int = 2     # Minimum nodes to form a pipeline
    optimal_nodes:    int = 4     # Recommended node count


@dataclass
class ShardRegistration:
    """
    A single node's registration for serving a contiguous
    range of layers for a specific model.
    """
    node_id:        str
    model_id:       str
    layer_start:    int           # First layer this node serves (inclusive)
    layer_end:      int           # Last layer this node serves (inclusive)
    vram_gb:        float         # Available VRAM on this node
    region:         str           # Geographic region for grid-aware routing
    host:           str           # IP/hostname for direct P2P activation passing
    port:           int           # Port for activation server
    registered_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_online:      bool = True
    requests_served: int = 0
    flops_delivered: float = 0.0

    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start + 1

    @property
    def covers(self) -> range:
        return range(self.layer_start, self.layer_end + 1)


@dataclass
class InferencePipeline:
    """
    A complete ordered sequence of shards that covers all layers
    of a model — ready to serve an inference request.
    """
    model_id:   str
    shards:     List[ShardRegistration]   # Ordered by layer_start
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_complete(self) -> bool:
        """True if shards cover all layers with no gaps."""
        if not self.shards:
            return False
        sorted_shards = sorted(self.shards, key=lambda s: s.layer_start)
        # First shard must start at layer 0
        if sorted_shards[0].layer_start != 0:
            return False
        # Each shard must connect to the next with no gap
        for i in range(len(sorted_shards) - 1):
            if sorted_shards[i].layer_end + 1 != sorted_shards[i+1].layer_start:
                return False
        return True

    @property
    def node_ids(self) -> List[str]:
        return [s.node_id for s in self.shards]

    @property
    def total_flops_per_token(self) -> float:
        """Estimated total GFLOPs per output token across all shards."""
        if not self.shards:
            return 0.0
        # All shards share the same model config via registry lookup
        return sum(s.layer_count for s in self.shards)  # Normalized per layer


# ── Registry ───────────────────────────────────────────────────

class ShardRegistry:
    """
    Central registry of all active shard registrations.
    Thread-safe — multiple requests may query simultaneously.
    """

    # Shard is considered offline after this many seconds without heartbeat
    HEARTBEAT_TIMEOUT = 120

    def __init__(self):
        self._shards:  Dict[str, ShardRegistration] = {}  # shard_id → shard
        self._models:  Dict[str, ModelConfig]        = {}  # model_id → config
        self._lock     = threading.RLock()

        # Register built-in supported models
        self._register_default_models()

    # ── Model registration ─────────────────────────────────────

    def register_model(self, config: ModelConfig):
        """Register a model configuration."""
        with self._lock:
            self._models[config.model_id]       = config
            self._models[config.friendly_name]  = config

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> List[ModelConfig]:
        with self._lock:
            # Deduplicate (model registered under both id and friendly name)
            seen = set()
            result = []
            for m in self._models.values():
                if m.model_id not in seen:
                    seen.add(m.model_id)
                    result.append(m)
            return result

    # ── Shard registration ─────────────────────────────────────

    def register_shard(self, shard: ShardRegistration) -> str:
        """
        Register a node's shard. Returns the shard ID.
        A node may register multiple shards for different models.
        """
        shard_id = f"{shard.node_id}:{shard.model_id}:{shard.layer_start}-{shard.layer_end}"
        with self._lock:
            # Validate model exists
            if shard.model_id not in self._models:
                raise ValueError(f"Unknown model: {shard.model_id}")

            model = self._models[shard.model_id]

            # Validate layer range
            if shard.layer_start < 0 or shard.layer_end >= model.total_layers:
                raise ValueError(
                    f"Invalid layer range [{shard.layer_start}, {shard.layer_end}] "
                    f"for model with {model.total_layers} layers."
                )
            if shard.layer_start > shard.layer_end:
                raise ValueError("layer_start must be <= layer_end.")

            self._shards[shard_id] = shard
        return shard_id

    def deregister_shard(self, node_id: str, model_id: str):
        """Remove all shards for a node+model combination."""
        with self._lock:
            keys = [k for k, s in self._shards.items()
                    if s.node_id == node_id and s.model_id == model_id]
            for k in keys:
                del self._shards[k]

    def heartbeat(self, node_id: str):
        """Update heartbeat timestamp for all shards of a node."""
        now = datetime.now(timezone.utc)
        with self._lock:
            for shard in self._shards.values():
                if shard.node_id == node_id:
                    shard.last_heartbeat = now
                    shard.is_online      = True

    def mark_offline(self, node_id: str):
        """Mark all shards of a node as offline."""
        with self._lock:
            for shard in self._shards.values():
                if shard.node_id == node_id:
                    shard.is_online = False

    def record_request(self, node_id: str, model_id: str, flops: float):
        """Record a completed inference request for a node's shard."""
        with self._lock:
            for shard in self._shards.values():
                if shard.node_id == node_id and shard.model_id == model_id:
                    shard.requests_served += 1
                    shard.flops_delivered += flops

    # ── Query ──────────────────────────────────────────────────

    def get_online_shards(self, model_id: str) -> List[ShardRegistration]:
        """Return all online shards for a model, sorted by layer_start."""
        timeout = timedelta(seconds=self.HEARTBEAT_TIMEOUT)
        now     = datetime.now(timezone.utc)

        # Resolve both the full model_id and friendly_name
        model        = self._models.get(model_id)
        full_id      = model.model_id      if model else model_id
        friendly     = model.friendly_name if model else model_id

        with self._lock:
            shards = [
                s for s in self._shards.values()
                if s.model_id in (full_id, friendly)
                and s.is_online
                and (now - s.last_heartbeat) < timeout
            ]
            return sorted(shards, key=lambda s: s.layer_start)

    def get_node_shards(self, node_id: str) -> List[ShardRegistration]:
        """Return all shards registered by a specific node."""
        with self._lock:
            return [s for s in self._shards.values() if s.node_id == node_id]

    def get_registry_summary(self) -> dict:
        """Return a summary of the current registry state."""
        with self._lock:
            summary = {}
            for shard in self._shards.values():
                mid = shard.model_id
                if mid not in summary:
                    summary[mid] = {"online": 0, "offline": 0, "layers_covered": set()}
                if shard.is_online:
                    summary[mid]["online"] += 1
                else:
                    summary[mid]["offline"] += 1
                for l in shard.covers:
                    summary[mid]["layers_covered"].add(l)

            # Convert sets to counts
            for mid in summary:
                summary[mid]["layers_covered"] = len(summary[mid]["layers_covered"])
                model = self._models.get(mid)
                summary[mid]["total_layers"] = model.total_layers if model else "unknown"

            return summary

    def check_stale_shards(self):
        """Mark shards as offline if heartbeat has timed out."""
        timeout = timedelta(seconds=self.HEARTBEAT_TIMEOUT)
        now     = datetime.now(timezone.utc)
        with self._lock:
            for shard in self._shards.values():
                if shard.is_online and (now - shard.last_heartbeat) > timeout:
                    shard.is_online = False

    # ── Default models ─────────────────────────────────────────

    def _register_default_models(self):
        """Register supported models with their configurations."""
        defaults = [
            ModelConfig(
                model_id=        "meta-llama/Meta-Llama-3-8B",
                friendly_name=   "llama3-8b",
                total_layers=    32,
                hidden_size=     4096,
                flops_per_layer= 0.42,
                min_nodes=       1,
                optimal_nodes=   2,
            ),
            ModelConfig(
                model_id=        "meta-llama/Meta-Llama-3-70B",
                friendly_name=   "llama3-70b",
                total_layers=    80,
                hidden_size=     8192,
                flops_per_layer= 1.68,
                min_nodes=       2,
                optimal_nodes=   4,
            ),
            ModelConfig(
                model_id=        "mistralai/Mistral-7B-v0.1",
                friendly_name=   "mistral-7b",
                total_layers=    32,
                hidden_size=     4096,
                flops_per_layer= 0.37,
                min_nodes=       1,
                optimal_nodes=   2,
            ),
            ModelConfig(
                model_id=        "mistralai/Mixtral-8x7B-v0.1",
                friendly_name=   "mixtral-8x7b",
                total_layers=    32,
                hidden_size=     4096,
                flops_per_layer= 0.74,
                min_nodes=       2,
                optimal_nodes=   4,
            ),
        ]
        for model in defaults:
            self.register_model(model)


# Singleton instance
registry = ShardRegistry()
