-- ============================================================
-- EdinTech RAG — industrial document migration
-- ============================================================

-- 1. Enable pgvector
create extension if not exists vector;

-- 2. Enum types
create type file_type as enum ('pdf', 'xlsx', 'csv', 'docx');

create type document_category as enum (
    'manual',
    'datasheet',
    'maintenance_record',
    'procedure',
    'report',
    'specification',
    'log',
    'other'
);

-- 3. documents table
create table if not exists documents (
    id            bigserial primary key,
    filename      text not null,
    file_type     file_type not null,
    document_category document_category not null,
    title         text,
    equipment_id  bigint,
    location      text,
    revision      text,
    document_date date,
    markdown_content text,
    source_path   text,
    ingested_at   timestamptz default now(),
    metadata      jsonb
);

-- 4. chunks table
create table if not exists chunks (
    id            bigserial primary key,
    document_id   bigint not null references documents(id) on delete cascade,
    content       text not null,
    metadata      jsonb,
    embedding     vector(1024),
    fts           tsvector generated always as (to_tsvector('english', content)) stored
);

-- 5. Indexes

-- documents
create index on documents (equipment_id);
create index on documents (document_category);
create index on documents using gin (metadata);
create index on documents (file_type);

-- chunks
create index on chunks (document_id);
create index on chunks using hnsw (embedding vector_cosine_ops);
create index on chunks using gin (fts);
create index on chunks using gin (metadata);

-- 6. Hybrid search (RRF) with optional filters
create or replace function hybrid_search(
    query_text            text,
    query_embedding       vector(1024),
    match_count           int,
    rrf_k                 int default 60,
    filter_equipment_id   bigint default null,
    filter_document_category document_category default null,
    filter_file_type       file_type default null,
    filter_location        text default null
)
returns table (
    id                  bigint,
    document_id         bigint,
    content             text,
    metadata            jsonb,
    filename            text,
    document_category   document_category,
    equipment_id        bigint,
    location            text,
    score               float
)
language plpgsql
as $$
declare
    v_limit int;
begin
    v_limit := least(match_count * 3, 500);

    return query
    with text_matches as (
        select
            c.id,
            c.document_id,
            c.content,
            c.metadata,
            d.document_category,
            d.equipment_id,
            d.location,
            1.0 / (rrf_k + gi) as text_rank
        from chunks c
        inner join documents d on d.id = c.document_id
        where c.fts @@ to_tsquery('english', query_text)
          and (filter_equipment_id   is null or d.equipment_id   = filter_equipment_id)
          and (filter_document_category is null or d.document_category = filter_document_category)
          and (filter_file_type     is null or d.file_type     = filter_file_type)
          and (filter_location      is null or d.location      = filter_location)
        cross join lateral generate_series(1, 100) as gi
        order by c.fts @@ to_tsquery('english', query_text) desc
        limit v_limit
    ),
    vector_matches as (
        select
            c.id,
            c.document_id,
            c.content,
            c.metadata,
            d.document_category,
            d.equipment_id,
            d.location,
            1.0 / (rrf_k + gi) as vector_rank
        from chunks c
        inner join documents d on d.id = c.document_id
        where c.embedding is not null
          and (filter_equipment_id   is null or d.equipment_id   = filter_equipment_id)
          and (filter_document_category is null or d.document_category = filter_document_category)
          and (filter_file_type     is null or d.file_type     = filter_file_type)
          and (filter_location      is null or d.location      = filter_location)
        order by c.embedding <#> query_embedding
        limit v_limit
    ),
    combined as (
        select
            id,
            document_id,
            content,
            metadata,
            document_category,
            equipment_id,
            location,
            sum(rank) as score
        from (
            select id, document_id, content, metadata, document_category, equipment_id, location, text_rank as rank from text_matches
            union all
            select id, document_id, content, metadata, document_category, equipment_id, location, vector_rank as rank from vector_matches
        ) all_matches
        group by id, document_id, content, metadata, document_category, equipment_id, location
        order by score desc
        limit match_count
    )
    select
        c.id,
        c.document_id,
        c.content,
        c.metadata,
        d.filename,
        c.document_category,
        c.equipment_id,
        c.location,
        c.score::float
    from combined c
    inner join documents d on d.id = c.document_id;
end;
$$;
