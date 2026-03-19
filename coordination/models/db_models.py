from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from coordination.db import Base
import uuid


class NodeDB(Base):
    __tablename__ = "nodes"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    public_key          = Column(Text, nullable=False, unique=True)
    hardware_class      = Column(String(64))
    multiplier          = Column(Float, nullable=False, default=1.0)
    status              = Column(String(32), nullable=False, default="pending")
    reliability_factor  = Column(Float, nullable=False, default=1.0)
    tasks_assigned      = Column(Integer, nullable=False, default=0)
    tasks_completed     = Column(Integer, nullable=False, default=0)
    token_balance       = Column(Float, nullable=False, default=0.0)
    registered_at       = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen           = Column(DateTime)
    quarantine_until    = Column(DateTime)
    hardware_changes    = Column(Integer, nullable=False, default=0)

    availability_start  = Column(String(10))
    availability_end    = Column(String(10))
    max_gpu_utilization = Column(Float, nullable=False, default=0.8)

    tasks               = relationship("TaskDB", back_populates="node")
    ledger_entries      = relationship("LedgerEntryDB", back_populates="node")


class TaskDB(Base):
    __tablename__ = "tasks"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type           = Column(String(32), nullable=False)
    payload             = Column(JSONB, nullable=False, default=dict)
    model_id            = Column(String(128))
    layer_start         = Column(Integer)
    layer_end           = Column(Integer)
    assigned_node_id    = Column(UUID(as_uuid=True), ForeignKey("nodes.id"))
    flops_estimated     = Column(Float, nullable=False)
    flops_delivered     = Column(Float)
    status              = Column(String(32), nullable=False, default="pending")
    is_challenge        = Column(Boolean, nullable=False, default=False)
    challenge_answer    = Column(Text)
    result_hash         = Column(Text)
    created_at          = Column(DateTime, nullable=False, default=datetime.utcnow)
    dispatched_at       = Column(DateTime)
    completed_at        = Column(DateTime)
    expires_at          = Column(DateTime)

    node                = relationship("NodeDB", back_populates="tasks")


class LedgerEntryDB(Base):
    __tablename__ = "ledger_entries"

    id                      = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type              = Column(String(32), nullable=False)
    node_id                 = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    amount                  = Column(Float, nullable=False)
    task_id                 = Column(UUID(as_uuid=True), ForeignKey("tasks.id"))
    flops_delivered         = Column(Float)
    hardware_multiplier     = Column(Float)
    reliability_factor      = Column(Float)
    inference_request_id    = Column(Text)
    flops_consumed          = Column(Float)
    previous_hash           = Column(Text)
    entry_hash              = Column(Text, nullable=False)
    timestamp               = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at              = Column(DateTime)

    node                    = relationship("NodeDB", back_populates="ledger_entries")
