from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional
import math


class Settings(BaseSettings):

    # ── App ────────────────────────────────────────────────────
    APP_NAME:    str = "Inferix — Coordination Layer"
    VERSION:     str = "0.1.0"
    DEBUG:       bool = False
    SECRET_KEY:  str = "change-me-in-production"

    # ── Database ───────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./inferix.db"

    # ── Token Economics ────────────────────────────────────────
    # Token expiry: 1 year in seconds
    TOKEN_EXPIRY_SECONDS:    int = 365 * 24 * 3600
    # Warning sent 90 days before expiry
    TOKEN_EXPIRY_WARNING_DAYS: int = 90
    # Minimum token lifetime guarantee (seconds)
    TOKEN_MIN_LIFETIME:      int = 365 * 24 * 3600

    # ── Hardware Multipliers ───────────────────────────────────
    MULTIPLIER_MOBILE_EDGE:         float = 0.05
    MULTIPLIER_CPU_ONLY:            float = 0.10
    MULTIPLIER_ENTRY_CONSUMER_GPU:  float = 0.50
    MULTIPLIER_MID_CONSUMER_GPU:    float = 1.00   # Baseline
    MULTIPLIER_HIGH_CONSUMER_GPU:   float = 2.00
    MULTIPLIER_PROSUMER_GPU:        float = 3.00
    MULTIPLIER_PROFESSIONAL_ACCEL:  float = 5.00
    MULTIPLIER_DATACENTER_ACCEL:    float = 8.00

    # ── API Benchmark Weights ──────────────────────────────────
    API_WEIGHT_MATMUL:   float = 0.5
    API_WEIGHT_MEMORY:   float = 0.3
    API_WEIGHT_LATENCY:  float = 0.2

    # ── Quarantine ─────────────────────────────────────────────
    # Q_duration = Q_base × log2(M_new / M_old)  in hours
    QUARANTINE_BASE_HOURS: float = 24.0

    # ── Reliability Factor ─────────────────────────────────────
    RELIABILITY_MIN:         float = 0.5
    RELIABILITY_MAX:         float = 1.0
    RELIABILITY_WINDOW_TASKS: int  = 100   # Rolling window size
    RELIABILITY_SUSPEND_THRESHOLD: float = 0.52
    # Minimum tasks before suspension can trigger
    RELIABILITY_MIN_TASKS_BEFORE_SUSPEND: int = 10

    # ── Task Verification ──────────────────────────────────────
    # Fraction of tasks assigned for redundant verification
    REDUNDANCY_RATE_NEW_NODE:      float = 0.5    # 50% for new nodes
    REDUNDANCY_RATE_TRUSTED_NODE:  float = 0.05   # 5% for trusted nodes
    TASK_TIMEOUT_SECONDS:          int   = 300    # 5 minutes

    # ── Grid-Aware Scheduling ──────────────────────────────────
    GRID_WEIGHT_LOAD:      float = 0.4
    GRID_WEIGHT_CARBON:    float = 0.4
    GRID_WEIGHT_RENEWABLE: float = 0.2
    # Minimum task allocation regardless of grid score (fairness guarantee)
    MIN_TASK_ALLOCATION_RATE: float = 0.1

    # ── Inference Models ───────────────────────────────────────
    # FLOPs per output token for each hosted model (GFLOPs)
    MODEL_FLOPS: dict = {
        "llama3-8b":  0.5,
        "llama3-70b": 4.0,
        "mistral-7b": 0.45,
    }
    PRECISION_FACTORS: dict = {
        "fp32": 1.0,
        "fp16": 0.5,
        "int8": 0.25,
    }

    # ── Node Management ────────────────────────────────────────
    NODE_OFFLINE_TIMEOUT_SECONDS: int = 300
    MAX_HARDWARE_CHANGES_BEFORE_REVIEW: int = 3

    def quarantine_duration_hours(self, old_multiplier: float, new_multiplier: float) -> float:
        """Q_duration = Q_base × log2(M_new / M_old)"""
        if new_multiplier <= old_multiplier:
            return 0.0
        return self.QUARANTINE_BASE_HOURS * math.log2(new_multiplier / old_multiplier)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
