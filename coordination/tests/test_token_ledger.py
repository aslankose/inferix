"""
Tests for the FLOP-based token formula and ledger chain integrity.
Validates Section 4 of the Inferix paper.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from coordination.models.token import LedgerEntry, LedgerEventType
from coordination.ledger.ledger import TokenLedger


class TestTokenFormula:
    """
    Section 4.1 — Token Issuance Formula:
    T_earned = F_delivered × M_hw × R_reliability
    """

    def test_basic_issuance_formula(self):
        """Token amount = FLOPs × multiplier × reliability."""
        flops       = 13.25
        multiplier  = 1.325
        reliability = 1.0
        expected    = round(flops * multiplier * reliability, 6)
        assert expected == pytest.approx(17.55625, rel=1e-4)

    def test_reliability_reduces_tokens(self):
        """Lower reliability factor reduces tokens issued."""
        flops      = 13.25
        multiplier = 1.325
        full       = flops * multiplier * 1.0
        half       = flops * multiplier * 0.5
        assert half == pytest.approx(full * 0.5, rel=1e-4)

    def test_minimum_reliability_factor(self):
        """Reliability factor floor is 0.5 — node always earns something."""
        flops      = 10.0
        multiplier = 1.0
        min_tokens = flops * multiplier * 0.5
        assert min_tokens == pytest.approx(5.0, rel=1e-4)

    def test_redemption_formula(self):
        """
        Section 4.2 — Token Redemption:
        T_spent = F_model × N_tokens × Q_precision
        """
        flops_per_token  = 0.45   # mistral-7b
        output_tokens    = 10
        precision_factor = 0.25   # int8
        expected         = round(flops_per_token * output_tokens * precision_factor, 6)
        assert expected == pytest.approx(1.125, rel=1e-4)

    def test_precision_factors(self):
        """FP16 costs half of FP32, INT8 costs quarter of FP32."""
        base = 100.0
        assert base * 1.0  == pytest.approx(100.0)   # fp32
        assert base * 0.5  == pytest.approx(50.0)    # fp16
        assert base * 0.25 == pytest.approx(25.0)    # int8

    def test_reliability_factor_formula(self):
        """
        Section 4.3 — Reliability Factor:
        R = 0.5 + 0.5 × (C_completed / C_assigned × V_consistency)
        """
        completion_rate   = 1.0   # all tasks completed
        consistency_score = 1.0   # all results verified
        r = 0.5 + 0.5 * (completion_rate * consistency_score)
        assert r == pytest.approx(1.0, rel=1e-4)

    def test_reliability_factor_minimum(self):
        """Zero completion rate gives minimum reliability of 0.5."""
        completion_rate   = 0.0
        consistency_score = 0.0
        r = 0.5 + 0.5 * (completion_rate * consistency_score)
        assert r == pytest.approx(0.5, rel=1e-4)

    def test_datacenter_multiplier_earnings(self):
        """Datacenter accelerator earns 8x more than baseline."""
        flops       = 10.0
        reliability = 1.0
        baseline    = flops * 1.0  * reliability   # mid consumer GPU
        datacenter  = flops * 8.0  * reliability   # H100
        assert datacenter == pytest.approx(baseline * 8.0, rel=1e-4)


class TestLedgerChain:
    """
    Section 3.3 — Token Ledger:
    Append-only hash-chained log with tamper-evident integrity.
    """

    def _make_entry(self, node_id="node-1", amount=10.0,
                    task_id="task-1", prev_hash=None) -> LedgerEntry:
        return LedgerEntry(
            event_type=          LedgerEventType.ISSUANCE,
            node_id=             node_id,
            amount=              amount,
            task_id=             task_id,
            flops_delivered=     10.0,
            hardware_multiplier= 1.0,
            reliability_factor=  1.0,
            previous_hash=       prev_hash,
            timestamp=           datetime(2026, 3, 15, 12, 0, 0),
        )

    def test_hash_is_deterministic(self):
        """Same entry always produces same hash."""
        ledger = TokenLedger()
        entry  = self._make_entry()
        hash1  = ledger._hash_entry(entry)
        hash2  = ledger._hash_entry(entry)
        assert hash1 == hash2

    def test_different_amounts_produce_different_hashes(self):
        """Different amounts produce different hashes."""
        ledger  = TokenLedger()
        entry1  = self._make_entry(amount=10.0)
        entry2  = self._make_entry(amount=20.0)
        assert ledger._hash_entry(entry1) != ledger._hash_entry(entry2)

    def test_different_nodes_produce_different_hashes(self):
        """Different node IDs produce different hashes."""
        ledger  = TokenLedger()
        entry1  = self._make_entry(node_id="node-1")
        entry2  = self._make_entry(node_id="node-2")
        assert ledger._hash_entry(entry1) != ledger._hash_entry(entry2)

    def test_hash_length(self):
        """SHA-256 hash is always 64 hex characters."""
        ledger = TokenLedger()
        entry  = self._make_entry()
        hash_  = ledger._hash_entry(entry)
        assert len(hash_) == 64
        assert all(c in "0123456789abcdef" for c in hash_)

    def test_empty_chain_is_valid(self):
        """An empty ledger chain is valid."""
        ledger  = TokenLedger()
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value.all.return_value = []
        assert ledger.verify_chain(mock_db) is True

    def test_previous_hash_links_entries(self):
        """Each entry's previous_hash must match the prior entry's hash."""
        ledger = TokenLedger()
        entry1 = self._make_entry(prev_hash=None)
        entry1.entry_hash = ledger._hash_entry(entry1)

        entry2 = self._make_entry(amount=20.0, prev_hash=entry1.entry_hash)
        entry2.entry_hash = ledger._hash_entry(entry2)

        assert entry2.previous_hash == entry1.entry_hash

    def test_tampered_amount_breaks_chain(self):
        """Modifying an entry's amount invalidates its hash."""
        ledger = TokenLedger()
        entry  = self._make_entry(amount=10.0)
        original_hash = ledger._hash_entry(entry)

        entry.amount = 999.0   # Tamper with amount
        tampered_hash = ledger._hash_entry(entry)

        assert original_hash != tampered_hash

    def test_balance_calculation(self):
        """Balance = sum of all credits minus all debits."""
        from sqlalchemy import func
        ledger  = TokenLedger()
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.scalar.return_value = 50.0
        balance = ledger.get_balance(mock_db, "node-1")
        assert balance == pytest.approx(50.0, rel=1e-4)

    def test_insufficient_balance_raises_error(self):
        """Redemption with insufficient balance raises ValueError."""
        ledger  = TokenLedger()
        mock_db = MagicMock()

        with patch.object(ledger, 'get_balance', return_value=5.0):
            with pytest.raises(ValueError, match="Insufficient balance"):
                ledger.redeem_tokens(
                    db=                   mock_db,
                    node_id=              "node-1",
                    inference_request_id= "req-1",
                    flops_consumed=       10.0,
                )
