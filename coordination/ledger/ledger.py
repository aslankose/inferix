import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy.orm import Session
from coordination.models.token import LedgerEntry, LedgerEventType, TokenBalance
from coordination.models.db_models import LedgerEntryDB
from coordination.config import settings


class TokenLedger:
    """
    Append-only hash-chained token ledger backed by PostgreSQL.
    """

    def _hash_entry(self, entry: LedgerEntry) -> str:
        # Normalize timestamp to UTC ISO format without timezone suffix
        # to ensure consistent hashing regardless of DB timezone storage
        ts = entry.timestamp
        if hasattr(ts, 'utcoffset') and ts.utcoffset() is not None:
            ts = ts.replace(tzinfo=None)
        content = {
            "id":            str(entry.id),
            "event_type":    str(entry.event_type),
            "node_id":       str(entry.node_id),
            "amount":        round(float(entry.amount), 6),
            "task_id":       str(entry.task_id) if entry.task_id else None,
            "timestamp":     ts.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "previous_hash": entry.previous_hash,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def _get_last_hash(self, db: Session) -> Optional[str]:
        last = db.query(LedgerEntryDB)\
                 .order_by(LedgerEntryDB.timestamp.desc())\
                 .first()
        return last.entry_hash if last else None

    def _append(self, db: Session, entry: LedgerEntry) -> LedgerEntry:
        entry.previous_hash = self._get_last_hash(db)
        entry.entry_hash    = self._hash_entry(entry)
        db_entry = LedgerEntryDB(
            id=                   entry.id,
            event_type=           entry.event_type,
            node_id=              entry.node_id,
            amount=               entry.amount,
            task_id=              entry.task_id,
            flops_delivered=      entry.flops_delivered,
            hardware_multiplier=  entry.hardware_multiplier,
            reliability_factor=   entry.reliability_factor,
            inference_request_id= entry.inference_request_id,
            flops_consumed=       entry.flops_consumed,
            previous_hash=        entry.previous_hash,
            entry_hash=           entry.entry_hash,
            timestamp=            entry.timestamp,
            expires_at=           entry.expires_at,
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        return entry

    def issue_tokens(self, db, node_id, task_id, flops_delivered, hardware_multiplier, reliability_factor):
        amount     = flops_delivered * hardware_multiplier * reliability_factor
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.TOKEN_EXPIRY_SECONDS)
        entry = LedgerEntry(
            event_type=          LedgerEventType.ISSUANCE,
            node_id=             node_id,
            amount=              round(amount, 6),
            task_id=             task_id,
            flops_delivered=     flops_delivered,
            hardware_multiplier= hardware_multiplier,
            reliability_factor=  reliability_factor,
            expires_at=          expires_at,
        )
        return self._append(db, entry)

    def redeem_tokens(self, db, node_id, inference_request_id, flops_consumed):
        balance = self.get_balance(db, node_id)
        if balance < flops_consumed:
            raise ValueError(
                f"Insufficient balance: {balance:.4f} GFT available, "
                f"{flops_consumed:.4f} GFT required."
            )
        entry = LedgerEntry(
            event_type=           LedgerEventType.REDEMPTION,
            node_id=              node_id,
            amount=               -round(flops_consumed, 6),
            inference_request_id= inference_request_id,
            flops_consumed=       flops_consumed,
        )
        return self._append(db, entry)

    def get_balance(self, db, node_id):
        from sqlalchemy import func
        result = db.query(func.sum(LedgerEntryDB.amount))\
                   .filter(LedgerEntryDB.node_id == node_id)\
                   .scalar()
        return max(0.0, float(result or 0.0))

    def get_token_balance(self, db, node_id):
        from sqlalchemy import func
        earned = db.query(func.sum(LedgerEntryDB.amount))\
                   .filter(LedgerEntryDB.node_id == node_id,
                           LedgerEntryDB.amount > 0).scalar() or 0.0
        spent  = db.query(func.sum(LedgerEntryDB.amount))\
                   .filter(LedgerEntryDB.node_id == node_id,
                           LedgerEntryDB.amount < 0).scalar() or 0.0
        return TokenBalance(
            node_id=      node_id,
            balance=      self.get_balance(db, node_id),
            total_earned= float(earned),
            total_spent=  abs(float(spent)),
            last_updated= datetime.now(timezone.utc),
        )

    def get_entries(self, db, node_id=None):
        query = db.query(LedgerEntryDB)
        if node_id:
            query = query.filter(LedgerEntryDB.node_id == node_id)
        rows = query.order_by(LedgerEntryDB.timestamp.asc()).all()
        return [self._db_to_entry(r) for r in rows]

    def verify_chain(self, db):
        entries = db.query(LedgerEntryDB)\
                    .order_by(LedgerEntryDB.timestamp.asc()).all()
        prev_hash = None
        for row in entries:
            if row.previous_hash != prev_hash:
                return False
            entry = self._db_to_entry(row)
            if self._hash_entry(entry) != row.entry_hash:
                return False
            prev_hash = row.entry_hash
        return True

    def total_entries(self, db):
        return db.query(LedgerEntryDB).count()

    def total_supply(self, db):
        from sqlalchemy import func
        result = db.query(func.sum(LedgerEntryDB.amount)).scalar()
        return max(0.0, float(result or 0.0))

    def _db_to_entry(self, row):
        return LedgerEntry(
            id=                   str(row.id),
            event_type=           row.event_type,
            node_id=              str(row.node_id),
            amount=               row.amount,
            task_id=              str(row.task_id) if row.task_id else None,
            flops_delivered=      row.flops_delivered,
            hardware_multiplier=  row.hardware_multiplier,
            reliability_factor=   row.reliability_factor,
            inference_request_id= row.inference_request_id,
            flops_consumed=       row.flops_consumed,
            previous_hash=        row.previous_hash,
            entry_hash=           row.entry_hash,
            timestamp=            row.timestamp,
            expires_at=           row.expires_at,
        )


ledger = TokenLedger()
