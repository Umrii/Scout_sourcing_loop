-- ─────────────────────────────────────────────────────────────────────────
-- Scout — canonical Postgres / Supabase schema.
--
-- This is the source of truth for the production database. Paste it into the
-- Supabase SQL editor (or run via `psql`). For local development the app can
-- fall back to SQLite, where the ORM (app/db.py) creates an equivalent schema
-- automatically — but JSONB + GIN indexing below is the real target.
-- ─────────────────────────────────────────────────────────────────────────

create extension if not exists "pgcrypto";  -- for gen_random_uuid()

create table if not exists experts (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    current_title text,
    company text,
    domains jsonb default '[]',
    seniority text,
    years_experience int,
    location text,
    notable_topics jsonb default '[]',
    extraction_confidence real,
    raw_bio text not null,
    source text not null,
    created_at timestamptz default now()
);
create index if not exists idx_experts_seniority on experts (seniority);
create index if not exists idx_experts_domains on experts using gin (domains);

create table if not exists projects (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    description text not null,
    required_domains jsonb default '[]',
    min_seniority text,
    num_experts_needed int default 3,
    created_at timestamptz default now()
);

create table if not exists matches (
    id uuid primary key default gen_random_uuid(),
    project_id uuid references projects(id) on delete cascade,
    expert_id uuid references experts(id) on delete cascade,
    relevance text not null,
    domain_match_score real,
    seniority_fit real,
    overall_score real,
    rationale text,
    outreach_draft text,
    created_at timestamptz default now()
);
create index if not exists idx_matches_project on matches (project_id, overall_score desc);

-- observability: one row per agent call
create table if not exists agent_runs (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null,
    stage text not null,           -- extract | classify | enrich | outreach
    expert_id uuid,
    project_id uuid,
    model text,
    prompt_version text,
    latency_ms int,
    input_tokens int,
    output_tokens int,
    status text,                   -- ok | error
    error text,
    created_at timestamptz default now()
);
create index if not exists idx_agent_runs_run on agent_runs (run_id);
create index if not exists idx_agent_runs_stage on agent_runs (stage, status);

-- eval tracking: one row per eval run (a prompt version scored over the set)
create table if not exists eval_results (
    id uuid primary key default gen_random_uuid(),
    eval_run_id uuid not null,
    prompt_version text not null,
    model text,
    dataset_size int,
    extraction_accuracy real,      -- overall field-level accuracy
    hallucination_rate real,       -- value invented where expected null
    field_accuracies jsonb,        -- per-field breakdown
    edge_case_accuracies jsonb,    -- per edge-case-tag breakdown
    created_at timestamptz default now()
);
