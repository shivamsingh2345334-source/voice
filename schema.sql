-- Run this once in Supabase -> SQL Editor
create extension if not exists vector;

create table if not exists public.memories (
    id          uuid primary key default gen_random_uuid(),
    user_id     text        not null default 'demo_user',
    raw_text    text        not null,                 -- Whisper transcript
    tags        jsonb       not null,                 -- {Place_to_go, Time, Event, Priority}
    priority    text generated always as (tags->>'Priority') stored,
    embedding   vector(384),                          -- must match the embedding model dimension
    created_at  timestamptz not null default now()
);

create index if not exists memories_user_created_idx on public.memories (user_id, created_at desc);
create index if not exists memories_user_priority_idx on public.memories (user_id, priority);
create index if not exists memories_tags_gin_idx      on public.memories using gin (tags);
create index if not exists memories_embedding_hnsw_idx
    on public.memories using hnsw (embedding vector_cosine_ops);

-- Semantic search RPC (called by app/memory_db.py)
create or replace function public.match_memories(
    query_embedding vector(384),
    p_user_id       text    default 'demo_user',
    match_count     int     default 3,
    min_similarity  float   default 0.2
)
returns table (id uuid, raw_text text, tags jsonb, created_at timestamptz, similarity float)
language sql stable
as $$
    select m.id, m.raw_text, m.tags, m.created_at,
           1 - (m.embedding <=> query_embedding) as similarity
    from public.memories m
    where m.user_id = p_user_id
      and m.embedding is not null
      and 1 - (m.embedding <=> query_embedding) >= min_similarity
    order by m.embedding <=> query_embedding
    limit match_count;
$$;

-- The backend uses the service_role key (bypasses RLS). Enable RLS so the anon key can never read this table.
alter table public.memories enable row level security;
