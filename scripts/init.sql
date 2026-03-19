-- Inferix — Complete Database Schema
-- Single file — no separate migrations needed

CREATE TABLE IF NOT EXISTS nodes (
    id                  UUID PRIMARY KEY,
    public_key          TEXT NOT NULL UNIQUE,
    hardware_class      TEXT,
    multiplier          FLOAT NOT NULL DEFAULT 1.0,
    status              TEXT NOT NULL DEFAULT 'pending',
    reliability_factor  FLOAT NOT NULL DEFAULT 1.0,
    tasks_assigned      INT NOT NULL DEFAULT 0,
    tasks_completed     INT NOT NULL DEFAULT 0,
    token_balance       FLOAT NOT NULL DEFAULT 0.0,
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen           TIMESTAMPTZ,
    quarantine_until    TIMESTAMPTZ,
    hardware_changes    INT NOT NULL DEFAULT 0,
    availability_start  VARCHAR(10),
    availability_end    VARCHAR(10),
    max_gpu_utilization FLOAT NOT NULL DEFAULT 0.8
);

CREATE TABLE IF NOT EXISTS tasks (
    id                  UUID PRIMARY KEY,
    task_type           TEXT NOT NULL,
    payload             JSONB NOT NULL DEFAULT '{}',
    model_id            VARCHAR(128),
    layer_start         INTEGER,
    layer_end           INTEGER,
    assigned_node_id    UUID REFERENCES nodes(id),
    flops_estimated     FLOAT NOT NULL,
    flops_delivered     FLOAT,
    status              TEXT NOT NULL DEFAULT 'pending',
    is_challenge        BOOLEAN NOT NULL DEFAULT FALSE,
    challenge_answer    TEXT,
    result_hash         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dispatched_at       TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    expires_at          TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ledger_entries (
    id                      UUID PRIMARY KEY,
    event_type              TEXT NOT NULL,
    node_id                 UUID NOT NULL REFERENCES nodes(id),
    amount                  FLOAT NOT NULL,
    task_id                 UUID REFERENCES tasks(id),
    flops_delivered         FLOAT,
    hardware_multiplier     FLOAT,
    reliability_factor      FLOAT,
    inference_request_id    TEXT,
    flops_consumed          FLOAT,
    previous_hash           TEXT,
    entry_hash              TEXT NOT NULL,
    timestamp               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at              TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_status        ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_tasks_status        ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_node          ON tasks(assigned_node_id);
CREATE INDEX IF NOT EXISTS idx_ledger_node         ON ledger_entries(node_id);
CREATE INDEX IF NOT EXISTS idx_ledger_timestamp    ON ledger_entries(timestamp);
