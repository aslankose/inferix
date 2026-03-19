from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime
import uuid


class HardwareClass(str, Enum):
    MOBILE_EDGE          = "mobile_edge"
    CPU_ONLY             = "cpu_only"
    ENTRY_CONSUMER_GPU   = "entry_consumer_gpu"
    MID_CONSUMER_GPU     = "mid_consumer_gpu"
    HIGH_CONSUMER_GPU    = "high_consumer_gpu"
    PROSUMER_GPU         = "prosumer_gpu"
    PROFESSIONAL_ACCEL   = "professional_accel"
    DATACENTER_ACCEL     = "datacenter_accel"


class NodeStatus(str, Enum):
    PENDING      = "pending"       # Registered, awaiting benchmark
    ACTIVE       = "active"        # Contributing, earning tokens
    QUARANTINE   = "quarantine"    # Hardware change detected
    SUSPENDED    = "suspended"     # Anomalous behaviour flagged
    OFFLINE      = "offline"       # Not seen recently


class HardwareProfile(BaseModel):
    gpu_model:          Optional[str] = None
    gpu_memory_gb:      Optional[float] = None
    cpu_model:          Optional[str] = None
    cpu_cores:          Optional[int] = None
    ram_gb:             Optional[float] = None
    # Benchmark scores (set after benchmarking)
    matmul_score:       Optional[float] = None   # Normalized 0-1
    memory_score:       Optional[float] = None   # Normalized 0-1
    latency_score:      Optional[float] = None   # Normalized 0-1
    api_score:          Optional[float] = None   # Composite AI Performance Index
    # Fingerprint hash for Sybil detection
    fingerprint:        Optional[str] = None


class NodeRegistrationRequest(BaseModel):
    public_key:         str  = Field(..., description="Node's public key for auth")
    hardware_profile:   HardwareProfile
    availability_start: Optional[str] = Field(None, description="HH:MM UTC")
    availability_end:   Optional[str] = Field(None, description="HH:MM UTC")
    max_gpu_utilization:float = Field(0.8, ge=0.0, le=1.0)


class Node(BaseModel):
    id:                 str = Field(default_factory=lambda: str(uuid.uuid4()))
    public_key:         str
    hardware_profile:   HardwareProfile
    hardware_class:     Optional[HardwareClass] = None
    multiplier:         float = 1.0
    status:             NodeStatus = NodeStatus.PENDING
    reliability_factor: float = Field(1.0, ge=0.5, le=1.0)
    tasks_assigned:     int = 0
    tasks_completed:    int = 0
    token_balance:      float = 0.0
    registered_at:      datetime = Field(default_factory=datetime.utcnow)
    last_seen:          Optional[datetime] = None
    quarantine_until:   Optional[datetime] = None
    # Hardware change tracking
    previous_multiplier:Optional[float] = None
    hardware_changes:   int = 0
    availability_start: Optional[str] = None
    availability_end:   Optional[str] = None
    max_gpu_utilization:float = 0.8

    @property
    def completion_rate(self) -> float:
        if self.tasks_assigned == 0:
            return 1.0
        return self.tasks_completed / self.tasks_assigned


class NodeResponse(BaseModel):
    id:                 str
    hardware_class:     Optional[HardwareClass]
    multiplier:         float
    status:             NodeStatus
    reliability_factor: float
    token_balance:      float
    registered_at:      datetime
