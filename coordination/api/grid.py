from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from coordination.core.scheduler import scheduler, GridState
from coordination.api.nodes import get_node_by_id, get_active_nodes
from coordination.db import get_db

router = APIRouter(prefix="/grid", tags=["Grid Scheduler"])


@router.get("/states")
def get_grid_states():
    """Return all current grid states ranked by grid score."""
    return {
        "regions": scheduler.get_grid_states(),
        "weights": {
            "load":      0.4,
            "carbon":    0.4,
            "renewable": 0.2,
        }
    }


@router.get("/score/{region}")
def get_region_score(region: str):
    """Return the grid priority score for a specific region."""
    score = scheduler.get_grid_score(region.upper())
    state = scheduler._grid_cache.get(region.upper())
    if not state:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found.")
    return {
        "region":             state.region,
        "grid_score":         score,
        "load_factor":        state.load_factor,
        "carbon_intensity":   state.carbon_intensity,
        "renewable_fraction": state.renewable_fraction,
    }


@router.post("/states/update")
def update_grid_state(
    region:             str,
    load_factor:        float,
    carbon_intensity:   float,
    renewable_fraction: float,
):
    """Update the grid state for a region (for testing or manual override)."""
    if not (0.0 <= load_factor <= 1.0):
        raise HTTPException(status_code=400, detail="load_factor must be between 0 and 1.")
    if not (0.0 <= carbon_intensity <= 1.0):
        raise HTTPException(status_code=400, detail="carbon_intensity must be between 0 and 1.")
    if not (0.0 <= renewable_fraction <= 1.0):
        raise HTTPException(status_code=400, detail="renewable_fraction must be between 0 and 1.")

    state = GridState(
        region=             region.upper(),
        load_factor=        load_factor,
        carbon_intensity=   carbon_intensity,
        renewable_fraction= renewable_fraction,
    )
    scheduler.update_grid_state(state)
    return {
        "updated": region.upper(),
        "grid_score": scheduler.get_grid_score(region.upper()),
    }


@router.post("/nodes/{node_id}/region")
def assign_node_region(node_id: str, region: str, db: Session = Depends(get_db)):
    """Assign a geographic region to a contributor node."""
    node = get_node_by_id(db, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found.")
    scheduler.register_node_region(node_id, region.upper())
    return {
        "node_id":    node_id,
        "region":     region.upper(),
        "grid_score": scheduler.get_grid_score(region.upper()),
    }


@router.get("/nodes/{node_id}/score")
def get_node_score(node_id: str, db: Session = Depends(get_db)):
    """Return the full scheduling score breakdown for a node."""
    node = get_node_by_id(db, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found.")
    score = scheduler.get_node_score(node)
    return {
        "node_id":     node_id,
        "region":      score.region,
        "grid_score":  score.grid_score,
        "perf_score":  score.perf_score,
        "reliability": score.reliability,
        "final_score": score.final_score,
    }


@router.get("/select")
def select_best_node(db: Session = Depends(get_db)):
    """Select the best available node for a task based on grid-aware scoring."""
    active_nodes = get_active_nodes(db)
    if not active_nodes:
        raise HTTPException(status_code=404, detail="No active nodes available.")

    selected = scheduler.select_node(active_nodes)
    if not selected:
        raise HTTPException(status_code=404, detail="No suitable node found.")

    score = scheduler.get_node_score(selected)
    return {
        "selected_node_id": str(selected.id),
        "hardware_class":   selected.hardware_class,
        "multiplier":       selected.multiplier,
        "region":           score.region,
        "grid_score":       score.grid_score,
        "perf_score":       score.perf_score,
        "reliability":      score.reliability,
        "final_score":      score.final_score,
    }
