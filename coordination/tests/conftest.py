"""
Shared test fixtures and configuration.
"""
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone


@pytest.fixture
def mock_node():
    """A mock active contributor node."""
    node = MagicMock()
    node.id                = "test-node-001"
    node.public_key        = "test-public-key"
    node.hardware_class    = "mid_consumer_gpu"
    node.multiplier        = 1.325
    node.status            = "active"
    node.reliability_factor= 1.0
    node.tasks_assigned    = 10
    node.tasks_completed   = 10
    node.token_balance     = 0.0
    node.registered_at     = datetime.now(timezone.utc)
    node.quarantine_until  = None
    return node


@pytest.fixture
def mock_db():
    """A mock database session."""
    return MagicMock()
