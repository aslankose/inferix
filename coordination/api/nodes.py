from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from coordination.models.node import (
    Node, NodeRegistrationRequest, NodeResponse, NodeStatus
)
from coordination.models.db_models import NodeDB
from coordination.core.multiplier import assign_multiplier
from coordination.core.quarantine import (
    compute_quarantine_end, is_in_quarantine, quarantine_summary
)
from coordination.config import settings
from coordination.db import get_db

router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.post("/register", response_model=NodeResponse, status_code=201)
def register_node(req: NodeRegistrationRequest, db: Session = Depends(get_db)):
    fingerprint = req.hardware_profile.fingerprint

    # Sybil detection
    if fingerprint:
        existing = db.query(NodeDB).filter(
            NodeDB.public_key == req.public_key
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="Public key already registered.")

    hw_class, multiplier = assign_multiplier(req.hardware_profile)

    node = NodeDB(
        public_key=          req.public_key,
        hardware_class=      hw_class.value if hw_class else None,
        multiplier=          multiplier,
        status=              NodeStatus.ACTIVE.value,
        reliability_factor=  1.0,
        availability_start=  req.availability_start,
        availability_end=    req.availability_end,
        max_gpu_utilization= req.max_gpu_utilization,
    )
    db.add(node)
    db.commit()
    db.refresh(node)

    return NodeResponse(
        id=                str(node.id),
        hardware_class=    node.hardware_class,
        multiplier=        node.multiplier,
        status=            node.status,
        reliability_factor=node.reliability_factor,
        token_balance=     node.token_balance,
        registered_at=     node.registered_at,
    )


@router.get("/{node_id}", response_model=NodeResponse)
def get_node(node_id: str, db: Session = Depends(get_db)):
    node = _get_node_or_404(db, node_id)
    return NodeResponse(
        id=                str(node.id),
        hardware_class=    node.hardware_class,
        multiplier=        node.multiplier,
        status=            node.status,
        reliability_factor=node.reliability_factor,
        token_balance=     node.token_balance,
        registered_at=     node.registered_at,
    )


@router.post("/{node_id}/heartbeat")
def heartbeat(node_id: str, db: Session = Depends(get_db)):
    node = _get_node_or_404(db, node_id)
    node.last_seen = datetime.now(timezone.utc)
    if node.status == NodeStatus.OFFLINE.value:
        if not is_in_quarantine(node.quarantine_until):
            node.status = NodeStatus.ACTIVE.value
    db.commit()
    return {"status": node.status, "last_seen": node.last_seen}


@router.post("/{node_id}/hardware-update")
def hardware_update(node_id: str, new_profile: NodeRegistrationRequest, db: Session = Depends(get_db)):
    node = _get_node_or_404(db, node_id)
    old_multiplier = node.multiplier
    new_hw_class, new_multiplier = assign_multiplier(new_profile.hardware_profile)
    summary = quarantine_summary(old_multiplier, new_multiplier)

    if summary["quarantine_required"]:
        node.status = NodeStatus.QUARANTINE.value
        node.quarantine_until = compute_quarantine_end(old_multiplier, new_multiplier)

    node.hardware_class   = new_hw_class.value if new_hw_class else None
    node.multiplier       = new_multiplier
    node.hardware_changes += 1

    if node.hardware_changes >= settings.MAX_HARDWARE_CHANGES_BEFORE_REVIEW:
        node.status = NodeStatus.SUSPENDED.value

    db.commit()
    return {
        "node_id":        node_id,
        "new_multiplier": new_multiplier,
        "quarantine":     summary,
        "status":         node.status,
    }


@router.post("/{node_id}/unsuspend")
def unsuspend_node(node_id: str, db: Session = Depends(get_db)):
    """Reactivate a suspended node and reset reliability factor."""
    node = _get_node_or_404(db, node_id)
    if node.status != NodeStatus.SUSPENDED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Node is not suspended. Current status: {node.status}"
        )
    node.status             = NodeStatus.ACTIVE.value
    node.reliability_factor = 1.0
    node.tasks_assigned     = 0
    node.tasks_completed    = 0
    db.commit()
    return {
        "status":           "active",
        "reliability":      node.reliability_factor,
        "message":          "Node reactivated successfully.",
    }
def list_nodes(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(NodeDB)
    if status:
        query = query.filter(NodeDB.status == status)
    nodes = query.all()
    return [
        NodeResponse(
            id=                str(n.id),
            hardware_class=    n.hardware_class,
            multiplier=        n.multiplier,
            status=            n.status,
            reliability_factor=n.reliability_factor,
            token_balance=     n.token_balance,
            registered_at=     n.registered_at,
        )
        for n in nodes
    ]


def _get_node_or_404(db: Session, node_id: str) -> NodeDB:
    node = db.query(NodeDB).filter(NodeDB.id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found.")
    return node


def get_active_nodes(db: Session) -> list[NodeDB]:
    return db.query(NodeDB).filter(NodeDB.status == NodeStatus.ACTIVE.value).all()


def get_node_by_id(db: Session, node_id: str) -> Optional[NodeDB]:
    return db.query(NodeDB).filter(NodeDB.id == node_id).first()
