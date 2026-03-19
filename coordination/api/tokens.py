from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from coordination.models.token import (
    TokenBalance, TokenRedemptionRequest, InferenceTokenCost
)
from coordination.ledger.ledger import ledger
from coordination.config import settings
from coordination.db import get_db

router = APIRouter(prefix="/tokens", tags=["Tokens"])


@router.get("/{node_id}/balance", response_model=TokenBalance)
def get_balance(node_id: str, db: Session = Depends(get_db)):
    return ledger.get_token_balance(db, node_id)


@router.get("/{node_id}/history")
def get_history(node_id: str, limit: int = 50, db: Session = Depends(get_db)):
    entries = ledger.get_entries(db, node_id=node_id)
    return {
        "node_id":      node_id,
        "entries":      [e.model_dump() for e in entries[-limit:]],
        "total_entries": len(entries),
    }


@router.post("/redeem")
def redeem_tokens(req: TokenRedemptionRequest, db: Session = Depends(get_db)):
    cost = calculate_inference_cost(req.model_id, req.output_tokens, req.precision)
    try:
        entry = ledger.redeem_tokens(
            db=                   db,
            node_id=              req.node_id,
            inference_request_id= req.inference_request_id,
            flops_consumed=       cost.total_gft_cost,
        )
    except ValueError as e:
        raise HTTPException(status_code=402, detail=str(e))
    return {
        "success":         True,
        "gft_deducted":    cost.total_gft_cost,
        "new_balance":     ledger.get_balance(db, req.node_id),
        "ledger_entry_id": entry.id,
    }


@router.get("/cost/estimate")
def estimate_inference_cost(
    model_id: str,
    output_tokens: int,
    precision: str = "fp16"
) -> InferenceTokenCost:
    return calculate_inference_cost(model_id, output_tokens, precision)


@router.get("/ledger/verify")
def verify_ledger(db: Session = Depends(get_db)):
    return {
        "chain_valid":   ledger.verify_chain(db),
        "total_entries": ledger.total_entries(db),
        "total_supply":  round(ledger.total_supply(db), 6),
    }


@router.get("/ledger/audit")
def audit_ledger(limit: int = 100, db: Session = Depends(get_db)):
    entries = ledger.get_entries(db)
    return {
        "entries":       [e.model_dump() for e in entries[-limit:]],
        "total_entries": ledger.total_entries(db),
        "total_supply":  round(ledger.total_supply(db), 6),
        "chain_valid":   ledger.verify_chain(db),
    }


def calculate_inference_cost(model_id, output_tokens, precision) -> InferenceTokenCost:
    flops_per_token = settings.MODEL_FLOPS.get(model_id)
    if flops_per_token is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    precision_factor = settings.PRECISION_FACTORS.get(precision)
    if precision_factor is None:
        raise HTTPException(status_code=400, detail=f"Precision '{precision}' not supported.")
    return InferenceTokenCost(
        model_id=        model_id,
        output_tokens=   output_tokens,
        precision=       precision,
        flops_per_token= flops_per_token,
        precision_factor=precision_factor,
        total_gft_cost=  round(flops_per_token * output_tokens * precision_factor, 6),
    )
