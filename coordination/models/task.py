from pydantic import BaseModel, Field
from typing import Optional, Any, List
from enum import Enum
from datetime import datetime
import uuid


class TaskType(str, Enum):
    INFERENCE_CHALLENGE = "inference_challenge"  # Known prompt → verify activations
    INFERENCE_SERVING   = "inference_serving"    # Real inference request via pipeline


class TaskStatus(str, Enum):
    PENDING    = "pending"     # Created, not yet dispatched
    DISPATCHED = "dispatched"  # Sent to a node
    VERIFIED   = "verified"    # Result confirmed correct
    FAILED     = "failed"      # Node failed or returned wrong result
    EXPIRED    = "expired"     # Node did not respond in time


class Task(BaseModel):
    id:               str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type:        TaskType
    # For inference_challenge: known prompt + expected activation hash
    # For inference_serving:   real user prompt
    payload:          dict = Field(default_factory=dict)
    flops_estimated:  float = Field(..., description="Estimated GFLOPs required")
    assigned_node_id: Optional[str] = None
    model_id:         Optional[str] = None      # Which model this task targets
    layer_start:      Optional[int] = None      # Which layers this node serves
    layer_end:        Optional[int] = None
    # Challenge verification
    is_challenge:     bool = False
    challenge_answer: Optional[str] = None      # Expected activation hash (coordinator only)
    # Result
    status:           TaskStatus = TaskStatus.PENDING
    result_hash:      Optional[str] = None
    flops_delivered:  Optional[float] = None
    created_at:       datetime = Field(default_factory=datetime.utcnow)
    dispatched_at:    Optional[datetime] = None
    completed_at:     Optional[datetime] = None
    expires_at:       Optional[datetime] = None


class TaskDispatchResponse(BaseModel):
    task_id:         str
    task_type:       TaskType
    payload:         dict
    model_id:        Optional[str]
    layer_start:     Optional[int]
    layer_end:       Optional[int]
    flops_estimated: float
    is_challenge:    bool
    expires_at:      Optional[datetime]


class TaskResultRequest(BaseModel):
    task_id:           str
    node_id:           str
    result_hash:       str    # SHA-256 of activations or output
    flops_delivered:   float
    execution_time_ms: float
