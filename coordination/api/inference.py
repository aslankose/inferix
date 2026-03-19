"""
Inference API
-------------
Handles inference requests from token-holding contributors.

Routing logic:
1. Check if a complete shard pipeline exists for the requested model
2. If yes → route to distributed shard pipeline (earn tokens per node)
3. If no  → route to centralized fallback inference service
4. Deduct GFT tokens from requester's balance either way
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from typing import Optional, List

from coordination.core.shard_registry import registry
from coordination.core.shard_scheduler import shard_scheduler
from coordination.core.pipeline_executor import pipeline_executor, PipelineExecutionError
from coordination.ledger.ledger import ledger
from coordination.db import get_db
from coordination.config import settings

router = APIRouter(prefix="/inference", tags=["Inference"])


# ── Request / Response models ──────────────────────────────────

class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    node_id:        str
    model_id:       str
    prompt:         str
    max_new_tokens: int   = 256
    temperature:    float = 0.7
    precision:      str   = "fp16"


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    text:           str
    tokens:         int
    input_tokens:   int
    gft_deducted:   float
    new_balance:    float
    backend:        str       # "distributed" or "fallback"
    pipeline:       List[str] # Node IDs used (empty for fallback)
    latency_ms:     float
    model_id:       str


# ── Routes ─────────────────────────────────────────────────────

@router.post("/generate", response_model=InferenceResponse)
def generate(req: InferenceRequest, db: Session = Depends(get_db)):
    """
    Generate text using the distributed shard pipeline or fallback.
    Deducts GFT tokens from the requester's balance.
    """
    # Estimate token cost upfront
    cost = _estimate_cost(req.model_id, req.max_new_tokens, req.precision)

    # Check balance
    balance = ledger.get_balance(db, req.node_id)
    if balance < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance: {balance:.4f} GFT available, "
                   f"{cost:.4f} GFT required."
        )

    # Try distributed pipeline first
    pipeline = shard_scheduler.find_pipeline(req.model_id)

    if pipeline:
        try:
            result  = pipeline_executor.execute(
                pipeline=       pipeline,
                prompt=         req.prompt,
                max_new_tokens= req.max_new_tokens,
                temperature=    req.temperature,
            )
            backend = "distributed"

            # Issue tokens to each contributing node
            _issue_node_tokens(db, result["flops_per_node"], req.model_id)

            actual_cost = _calculate_actual_cost(
                result["total_flops"], req.precision
            )

        except PipelineExecutionError as e:
            # Pipeline failed — fall through to fallback
            pipeline = None
            result   = None

    if not pipeline or result is None:
        # Centralized fallback
        result  = _fallback_inference(req)
        backend = "fallback"
        actual_cost = cost

    # Deduct tokens from requester
    inference_id = f"inf-{req.node_id[:8]}-{int(__import__('time').time())}"
    try:
        ledger.redeem_tokens(
            db=                   db,
            node_id=              req.node_id,
            inference_request_id= inference_id,
            flops_consumed=       actual_cost,
        )
    except ValueError as e:
        raise HTTPException(status_code=402, detail=str(e))

    new_balance = ledger.get_balance(db, req.node_id)

    return InferenceResponse(
        text=         result["text"],
        tokens=       result.get("tokens", 0),
        input_tokens= result.get("input_tokens", 0),
        gft_deducted= actual_cost,
        new_balance=  new_balance,
        backend=      backend,
        pipeline=     result.get("pipeline", []),
        latency_ms=   result.get("latency_ms", 0.0),
        model_id=     req.model_id,
    )


@router.get("/models")
def list_models():
    """List all supported models and their pipeline availability."""
    models  = registry.list_models()
    summary = registry.get_registry_summary()

    return {
        "models": [
            {
                "model_id":       m.model_id,
                "friendly_name":  m.friendly_name,
                "total_layers":   m.total_layers,
                "min_nodes":      m.min_nodes,
                "optimal_nodes":  m.optimal_nodes,
                "pipeline_available": shard_scheduler.is_pipeline_available(m.model_id),
                "registry":       summary.get(m.model_id, {
                    "online": 0, "offline": 0, "layers_covered": 0
                }),
            }
            for m in models
        ]
    }


@router.get("/pipeline/{model_id:path}")
def get_pipeline_status(model_id: str):
    """Check pipeline availability and coverage for a model."""
    return shard_scheduler.get_pipeline_summary(model_id)


@router.get("/suggest/{model_id:path}")
def suggest_layers(model_id: str, vram_gb: float = 8.0):
    """
    Suggest which layers a new node should load to fill gaps
    in the current pipeline.
    """
    suggestion = shard_scheduler.suggest_layer_assignment(model_id, vram_gb)
    if not suggestion:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    layer_start, layer_end = suggestion
    model = registry.get_model(model_id)

    return {
        "model_id":    model_id,
        "layer_start": layer_start,
        "layer_end":   layer_end,
        "layer_count": layer_end - layer_start + 1,
        "total_layers": model.total_layers if model else None,
        "vram_required_gb": _estimate_vram(model_id, layer_start, layer_end),
    }


@router.get("/cost/estimate")
def estimate_cost(model_id: str, max_new_tokens: int, precision: str = "fp16"):
    """Estimate GFT cost for an inference request."""
    cost = _estimate_cost(model_id, max_new_tokens, precision)
    return {
        "model_id":       model_id,
        "max_new_tokens": max_new_tokens,
        "precision":      precision,
        "estimated_gft":  cost,
    }


# ── Shard registration endpoints ──────────────────────────────

class ShardRegistrationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    node_id:     str
    model_id:    str
    layer_start: int
    layer_end:   int
    vram_gb:     float
    region:      str
    host:        str
    port:        int


@router.post("/shards/register")
def register_shard(req: ShardRegistrationRequest):
    """Register a node as serving a layer range for a model."""
    from coordination.core.shard_registry import ShardRegistration
    shard = ShardRegistration(
        node_id=     req.node_id,
        model_id=    req.model_id,
        layer_start= req.layer_start,
        layer_end=   req.layer_end,
        vram_gb=     req.vram_gb,
        region=      req.region,
        host=        req.host,
        port=        req.port,
    )
    try:
        shard_id = registry.register_shard(shard)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "shard_id":    shard_id,
        "node_id":     req.node_id,
        "model_id":    req.model_id,
        "layers":      f"{req.layer_start}-{req.layer_end}",
        "layer_count": req.layer_end - req.layer_start + 1,
    }


@router.post("/shards/heartbeat/{node_id}")
def shard_heartbeat(node_id: str):
    """Update heartbeat for all shards of a node."""
    shards = registry.get_node_shards(node_id)
    if not shards:
        raise HTTPException(
            status_code=404,
            detail="Node not registered in shard registry."
        )
    registry.heartbeat(node_id)
    return {"status": "ok", "node_id": node_id, "shards": len(shards)}


@router.delete("/shards/{node_id}")
def deregister_shard(node_id: str, model_id: Optional[str] = None):
    """
    Remove all shard registrations for a node.
    Optionally filter by model_id to remove only a specific model's shards.
    """
    # Get all shards first regardless of model
    all_shards = registry.get_node_shards(node_id)

    if not all_shards:
        # Show what's currently in registry to help debug
        summary = registry.get_registry_summary()
        all_node_ids = list({s.node_id for shards in
                            [registry.get_node_shards(k) for k in
                             list(registry._shards.keys())]
                            for s in shards})
        raise HTTPException(
            status_code=404,
            detail={
                "message":          f"No shards found for node {node_id}.",
                "hint":             "Shard registry resets on server restart. "
                                    "Restart your client to re-register.",
                "registry_summary": summary,
            }
        )

    if model_id:
        registry.deregister_shard(node_id, model_id)
        removed = [s for s in all_shards if s.model_id == model_id]
    else:
        # Remove all shards for this node across all models
        for shard in all_shards:
            registry.deregister_shard(node_id, shard.model_id)
        removed = all_shards

    return {
        "status":         "deregistered",
        "node_id":        node_id,
        "shards_removed": len(removed),
        "layers":         [f"{s.layer_start}-{s.layer_end}" for s in removed],
        "models":         list({s.model_id for s in removed}),
    }


@router.get("/shards/node/{node_id}")
def get_node_shards(node_id: str):
    """List all shards registered by a specific node."""
    shards = registry.get_node_shards(node_id)
    if not shards:
        raise HTTPException(
            status_code=404,
            detail=f"No shards found for node {node_id}. "
                   f"Registry resets on server restart — restart your client."
        )
    return {
        "node_id": node_id,
        "shards": [
            {
                "model_id":    s.model_id,
                "layer_start": s.layer_start,
                "layer_end":   s.layer_end,
                "layer_count": s.layer_count,
                "is_online":   s.is_online,
                "region":      s.region,
                "host":        s.host,
                "port":        s.port,
            }
            for s in shards
        ]
    }


@router.get("/shards/registry")
def get_registry():
    """Return the full shard registry summary."""
    all_shards = []
    seen_ids   = set()
    for shard in registry._shards.values():
        if shard.node_id not in seen_ids:
            seen_ids.add(shard.node_id)
        all_shards.append({
                "node_id":     shard.node_id,
                "model_id":    shard.model_id,
                "layer_start": shard.layer_start,
                "layer_end":   shard.layer_end,
                "is_online":   shard.is_online,
                "region":      shard.region,
                "host":        shard.host,
                "port":        shard.port,
            })
    return {
        "summary":      registry.get_registry_summary(),
        "models":       [m.friendly_name for m in registry.list_models()],
        "shards":       all_shards,
        "total_nodes":  len(seen_ids),
        "total_shards": len(all_shards),
    }


# ── Internal helpers ───────────────────────────────────────────

def _estimate_cost(model_id: str, max_new_tokens: int, precision: str) -> float:
    """Estimate GFT cost for an inference request."""
    # Use friendly name lookup first, then full model ID
    model            = registry.get_model(model_id)
    friendly         = model.friendly_name if model else model_id
    flops_per_token  = settings.MODEL_FLOPS.get(friendly,
                       settings.MODEL_FLOPS.get(model_id, 0.5))
    precision_factor = settings.PRECISION_FACTORS.get(precision, 0.5)
    return round(flops_per_token * max_new_tokens * precision_factor, 6)


def _calculate_actual_cost(total_flops: float, precision: str) -> float:
    """Calculate actual GFT cost based on measured FLOPs."""
    precision_factor = settings.PRECISION_FACTORS.get(precision, 0.5)
    return round(total_flops * precision_factor, 6)


def _issue_node_tokens(db, flops_per_node: dict, model_id: str):
    """Issue tokens to each node that contributed to a distributed inference."""
    from coordination.api.nodes import get_node_by_id
    for node_id, flops in flops_per_node.items():
        node = get_node_by_id(db, node_id)
        if node:
            ledger.issue_tokens(
                db=                  db,
                node_id=             node_id,
                task_id=             None,
                flops_delivered=     flops,
                hardware_multiplier= node.multiplier,
                reliability_factor=  node.reliability_factor,
            )
            node.token_balance = ledger.get_balance(db, node_id)
            db.commit()


def _fallback_inference(req: InferenceRequest) -> dict:
    """
    Centralized fallback inference.
    In production this calls a self-hosted model service
    or a third-party inference provider.
    Currently returns a stub response for development.
    """
    return {
        "text":         f"[Fallback inference for: {req.prompt[:50]}...]",
        "tokens":       req.max_new_tokens,
        "input_tokens": len(req.prompt.split()),
        "pipeline":     [],
        "latency_ms":   0.0,
    }


def _estimate_vram(model_id: str, layer_start: int, layer_end: int) -> float:
    """Estimate VRAM required for a layer range."""
    from coordination.core.shard_scheduler import _estimate_bytes_per_layer
    model = registry.get_model(model_id)
    if not model:
        return 0.0
    layer_count      = layer_end - layer_start + 1
    bytes_per_layer  = _estimate_bytes_per_layer(model)
    total_bytes      = layer_count * bytes_per_layer
    return round(total_bytes / (1024 ** 3), 2)
