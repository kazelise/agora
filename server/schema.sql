-- Idempotent Phase 0/1 schema. last_seq lives on rooms so concurrent
-- inserts in one room serialize on a single row and stay gapless.

CREATE TABLE IF NOT EXISTS rooms (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    last_seq BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS computers (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS participants (
    id UUID PRIMARY KEY,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('human', 'agent')),
    name TEXT NOT NULL,
    persona TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    computer_id UUID REFERENCES computers (id)
);

-- Existing Phase 0–3 databases already have participants; add the FK if missing.
ALTER TABLE participants
    ADD COLUMN IF NOT EXISTS computer_id UUID REFERENCES computers (id);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    author_id UUID NOT NULL REFERENCES participants (id) ON DELETE RESTRICT,
    body TEXT NOT NULL,
    seq BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (room_id, seq)
);

CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    task_key TEXT NOT NULL,
    claimed_by UUID NOT NULL REFERENCES participants (id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (room_id, task_key)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES participants (id) ON DELETE CASCADE,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    prompt_tokens INT NOT NULL,
    completion_tokens INT NOT NULL,
    purpose TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Inbox cursor for the Phase 1 turn stub. This is NOT the Redis
-- seen-cursor used later for freshness HOLD — do not merge the two.
CREATE TABLE IF NOT EXISTS conversation_reads (
    agent_id UUID NOT NULL REFERENCES participants (id) ON DELETE CASCADE,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    last_read_seq BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, room_id)
);

CREATE INDEX IF NOT EXISTS messages_room_seq_idx ON messages (room_id, seq);
CREATE INDEX IF NOT EXISTS participants_room_idx ON participants (room_id);
CREATE INDEX IF NOT EXISTS participants_computer_idx ON participants (computer_id);

-- Phase 4a: GitHub users. created_by on rooms/computers is NULL when
-- admission is off (dev/tests) or for rows from before this phase.
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    github_id BIGINT NOT NULL UNIQUE,
    login TEXT NOT NULL,
    name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE rooms
    ADD COLUMN IF NOT EXISTS created_by UUID REFERENCES users (id);

ALTER TABLE computers
    ADD COLUMN IF NOT EXISTS created_by UUID REFERENCES users (id);

CREATE INDEX IF NOT EXISTS rooms_created_by_idx ON rooms (created_by);
CREATE INDEX IF NOT EXISTS computers_created_by_idx ON computers (created_by);
