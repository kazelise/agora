-- Idempotent Phase 0/1 schema. last_seq lives on rooms so concurrent
-- inserts in one room serialize on a single row and stay gapless.

CREATE TABLE IF NOT EXISTS rooms (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    last_seq BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    mode TEXT NOT NULL DEFAULT 'open' CHECK (mode IN ('open', 'moderated'))
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
    computer_id UUID REFERENCES computers (id),
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('member', 'moderator'))
);

-- Existing Phase 0–3 databases already have participants; add the FK if missing.
ALTER TABLE participants
    ADD COLUMN IF NOT EXISTS computer_id UUID REFERENCES computers (id);

-- Phase 7: room mode + participant role on existing DBs. New installs
-- already have the columns from CREATE TABLE; ADD COLUMN IF NOT EXISTS
-- is a no-op there. CHECKs are added separately so a pre-Phase-7 table
-- still gets the invariant after the column appears.
ALTER TABLE rooms
    ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'open';

ALTER TABLE participants
    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'member';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rooms_mode_check' AND conrelid = 'rooms'::regclass
    ) THEN
        ALTER TABLE rooms
            ADD CONSTRAINT rooms_mode_check
            CHECK (mode IN ('open', 'moderated'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'participants_role_check'
          AND conrelid = 'participants'::regclass
    ) THEN
        ALTER TABLE participants
            ADD CONSTRAINT participants_role_check
            CHECK (role IN ('member', 'moderator'));
    END IF;
END $$;

-- At most one moderator seat per room (API also 409s the second).
CREATE UNIQUE INDEX IF NOT EXISTS participants_one_moderator_per_room
    ON participants (room_id)
    WHERE role = 'moderator';

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

-- Inbox cursor. Freshness compares in-state seen_seq to rooms.last_seq;
-- do not reuse this column as that high-water (it would drain the inbox).
CREATE TABLE IF NOT EXISTS conversation_reads (
    agent_id UUID NOT NULL REFERENCES participants (id) ON DELETE CASCADE,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    last_read_seq BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, room_id)
);

-- Phase 7: one decision per (room, trigger_seq). A moderator rerun for
-- the same trigger is "already decided", not an error — mirrors
-- yuanzhuo's (meeting_id, trigger_key) uniqueness.
CREATE TABLE IF NOT EXISTS moderator_decisions (
    id UUID PRIMARY KEY,
    room_id UUID NOT NULL REFERENCES rooms (id) ON DELETE CASCADE,
    moderator_id UUID NOT NULL REFERENCES participants (id) ON DELETE CASCADE,
    trigger_seq BIGINT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('call_on', 'say', 'silence')),
    target_id UUID REFERENCES participants (id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (room_id, trigger_seq)
);

-- Existing DBs created the FK as RESTRICT; room-scoped lifetime is CASCADE.
ALTER TABLE moderator_decisions
    DROP CONSTRAINT IF EXISTS moderator_decisions_moderator_id_fkey;
ALTER TABLE moderator_decisions
    ADD CONSTRAINT moderator_decisions_moderator_id_fkey
    FOREIGN KEY (moderator_id) REFERENCES participants (id) ON DELETE CASCADE;

-- Names are the addressing surface for @-mentions and call_on targets.
CREATE UNIQUE INDEX IF NOT EXISTS participants_room_name_idx
    ON participants (room_id, name);

CREATE INDEX IF NOT EXISTS messages_room_seq_idx ON messages (room_id, seq);
CREATE INDEX IF NOT EXISTS participants_room_idx ON participants (room_id);
CREATE INDEX IF NOT EXISTS participants_computer_idx ON participants (computer_id);
DROP INDEX IF EXISTS moderator_decisions_room_idx;
