from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum
from datetime import datetime
import uuid


class LedgerEventType(str, Enum):
    ISSUANCE    = "issuance"     # Tokens earned from task completion
    REDEMPTION  = "redemption"   # Tokens spent on inference
    PURCHASE    = "purchase"     # Tokens bought directly
    EXPIRY      = "expiry"       # Tokens expired due to inactivity
    PENALTY     = "penalty"      # Tokens deducted for misconduct


class LedgerEntry(BaseModel):
    """A single entry in the append-only token ledger."""
    id:              str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type:      LedgerEventType
    node_id:         str
    amount:          float = Field(..., description="GFT amount (positive=credit, negative=debit)")
    # Issuance details
    task_id:         Optional[str] = None
    flops_delivered: Optional[float] = None
    hardware_multiplier: Optional[float] = None
    reliability_factor:  Optional[float] = None
    # Redemption details
    inference_request_id: Optional[str] = None
    flops_consumed:       Optional[float] = None
    # Chain integrity
    previous_hash:   Optional[str] = None   # Hash of previous ledger entry
    entry_hash:      Optional[str] = None   # SHA-256 of this entry's content
    timestamp:       datetime = Field(default_factory=datetime.utcnow)
    # Token expiry
    expires_at:      Optional[datetime] = None


class TokenBalance(BaseModel):
    node_id:         str
    balance:         float
    total_earned:    float
    total_spent:     float
    last_updated:    datetime


class TokenIssuanceRequest(BaseModel):
    node_id:         str
    task_id:         str
    flops_delivered: float
    hardware_multiplier: float
    reliability_factor:  float


class TokenRedemptionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    node_id:              str
    inference_request_id: str
    model_id:             str
    output_tokens:        int
    precision:            str = "fp16"


class InferenceTokenCost(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id:        str
    output_tokens:   int
    precision:       str
    flops_per_token: float
    precision_factor:float
    total_gft_cost:  float
