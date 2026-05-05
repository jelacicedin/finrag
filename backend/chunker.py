"""Chunk structured markdown for embedding, then insert into the database.

Uses content-aware strategies: prose chunks by heading boundaries with overlap,
tables kept together or split by row groups.  Each chunk is embedded via Ollama
(qwen3-embedding:0.6b) and batch-inserted into the chunks table.

Usage
-----
    # Library
    from chunker import chunk_and_insert

    # CLI
    python chunker.py --document-id 1 --file path/to/document.pdf
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ------ token counting (approximate, no extra deps) ------

def _count_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token, plus punctuation words."""
    words = len(re.findall(r"\S+", text))
    chars = len(text)
    return max(1, chars // 4 + words // 2)


# ------ data classes ------

@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ------ markdown parsing ------

# Detect a markdown heading line (## or ### at start of line)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Detect a markdown table block: starts with `| ... |`, contains alternating
# | header | ... | and `| --- | ... |` separator lines, ends at a blank line.
_TABLE_RE = re.compile(
    r"^(\|[^\n]+\n(\|[-| :]+\|\n)*(?:\|[^\n]+\n)+)(?=\n|$)",
    re.MULTILINE,
)


def _is_heading(line: str) -> bool:
    return bool(re.match(r"^(#{1,3})\s+", line))


def _is_table_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and not re.match(r"^(\|[\s:-]+\|)$", stripped)


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^(\|[\s:-]+\|)$", stripped))


def _extract_tables(lines: list[str], start: int) -> tuple[list[tuple[int, int]], int]:
    """Extract all table blocks starting at or after *start*.

    Returns ((start_line, end_line), ...) and the index after the last table.
    """
    tables: list[tuple[int, int]] = []
    i = start
    while i < len(lines):
        if _is_table_header(lines[i]):
            # Find the separator row
            sep_idx = None
            for j in range(i + 1, min(i + 5, len(lines))):
                if _is_table_separator(lines[j]):
                    sep_idx = j
                    break
            if sep_idx is not None:
                # Find the end: consecutive non-table lines
                end = sep_idx + 1
                while end < len(lines):
                    if lines[end].strip() == "":
                        break
                    if _is_table_header(lines[end]) or _is_table_separator(lines[end]):
                        end += 1
                    else:
                        break
                tables.append((i, end - 1))
                i = end
                continue
        i += 1
    return tables, i


def _split_by_headings(markdown: str) -> list[tuple[str, list[tuple[int, int]], list[str]]]:
    """Split markdown into sections by heading lines.

    Each section returns (heading_text, [(table_start, table_end), ...], lines_without_tables).
    """
    sections: list[tuple[str, list[tuple[int, int]], list[str]]] = []
    lines = markdown.split("\n")

    heading_starts = [m.start() for m in _HEADING_RE.finditer(markdown)]
    if not heading_starts:
        return [("Document", [], [(0, len(lines) - 1)])]

    for idx, start in enumerate(heading_starts):
        end = heading_starts[idx + 1] if idx + 1 < len(heading_starts) else len(lines)
        heading_line = lines[start] if start < len(lines) else ""
        heading_text = re.sub(r"^#{1,3}\s+", "", heading_line).strip()

        # Extract tables in this section
        section_lines = lines[start:end]
        tables, _ = _extract_tables(section_lines, 0)
        # Convert to absolute indices
        abs_tables = [(s + start, e + start) for s, e in tables]

        # Lines without table content
        table_lines = set()
        for s, e in abs_tables:
            table_lines.update(range(s, e + 1))
        prose_lines = [(i, i) for i in range(start, end) if i not in table_lines]

        sections.append((heading_text, abs_tables, prose_lines))

    return sections


# ------ chunking strategies ------

def _chunk_prose(prose_lines: list[tuple[int, int]], markdown: str,
                 target: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split prose lines into chunks of ~target tokens with overlap."""
    if not prose_lines:
        return []

    # Build full prose text with positions
    lines = markdown.split("\n")
    text_lines = [lines[i[0]] for i in prose_lines]
    full_text = "\n".join(text_lines)

    # Split into paragraphs on blank lines
    paragraphs = re.split(r"\n\n+", full_text)

    chunks: list[Chunk] = []
    buf = ""
    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if _count_tokens(candidate) > target and buf:
            # Flush buffer
            chunks.append(Chunk(content=buf))
            # Keep overlap from the end
            overlap_text = "\n".join(buf.split("\n")[-(max(1, overlap // 4)):])
            buf = overlap_text
        buf = candidate

    if buf:
        chunks.append(Chunk(content=buf))

    # Merge adjacent small chunks to avoid tiny fragments
    merged: list[Chunk] = []
    for c in chunks:
        if merged and _count_tokens(merged[-1].content) + _count_tokens(c.content) <= target + overlap:
            merged[-1].content = merged[-1].content + "\n\n" + c.content
        else:
            merged.append(c)
    return merged


def _chunk_table(lines: list[str], start: int, end: int,
                 max_rows: int = 20, max_tokens: int = 800) -> list[Chunk]:
    """Chunk a markdown table. Keep whole if under max_tokens; otherwise split."""
    table_lines = lines[start:end + 1]
    if _count_tokens("\n".join(table_lines)) <= max_tokens:
        return [Chunk(content="\n".join(table_lines))]

    header = table_lines[0]
    separator = table_lines[1]
    data_rows = table_lines[2:]

    chunks: list[Chunk] = []
    for i in range(0, len(data_rows), max_rows):
        group = data_rows[i : i + max_rows]
        chunk_lines = [header, separator] + group
        chunks.append(Chunk(content="\n".join(chunk_lines)))
    return chunks


# ------ embedding ------

def _embed_text(text: str) -> list[float]:
    """Embed text using Ollama qwen3-embedding:0.6b."""
    import ollama
    resp = ollama.embeddings(model="qwen3-embedding:0.6b", text=text)
    return resp["embedding"]


class ChunkingError(Exception):
    """Raised when chunking or database insertion fails."""


# ------ DB insertion ------

def _upsert_document(doc_id: int, file_path: str, file_type: str,
                     category: str, title: str | None, revision: str | None,
                     doc_date: str | None, markdown: str, equipment_id: int | None,
                     location: str | None, db_conn) -> None:
    """Insert or update the document record."""
    cur = db_conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (id, filename, file_type, document_category,
                               title, revision, document_date, markdown_content,
                               source_path, equipment_id, location, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            filename = EXCLUDED.filename,
            file_type = EXCLUDED.file_type,
            document_category = EXCLUDED.document_category,
            title = EXCLUDED.title,
            revision = EXCLUDED.revision,
            document_date = EXCLUDED.document_date,
            markdown_content = EXCLUDED.markdown_content,
            source_path = EXCLUDED.source_path,
            equipment_id = EXCLUDED.equipment_id,
            location = EXCLUDED.location,
            metadata = EXCLUDED.metadata,
            ingested_at = now()
        """,
        (doc_id, file_path, file_type, category, title, revision,
         doc_date, markdown, file_path, equipment_id, location,
         {"title": title, "file_type": file_type, "category": category}),
    )
    db_conn.commit()


def chunk_and_insert(document_id: int, markdown: str, file_type: str,
                     db_conn) -> int:
    """Chunk markdown content and insert into the chunks table.

    Parameters
    ----------
    document_id : int
        Primary key of the document in the documents table.
    markdown : str
        Structured markdown from converter.py.
    file_type : str
        One of 'pdf', 'xlsx', 'csv'.
    db_conn : psycopg connection
        Active DB connection (will be committed).

    Returns
    -------
    int
        Number of chunks created.
    """
    lines = markdown.split("\n")
    try:
        sections = _split_by_headings(markdown)
    except Exception as e:
        raise ChunkingError(f"Failed to split markdown into sections: {e}") from e

    all_chunks: list[Chunk] = []

    for sec_idx, (heading, tables, prose_lines) in enumerate(sections):
        section_meta = {"section_heading": heading, "chunk_index": 0, "total_chunks": 0}

        # Process prose
        try:
            prose_chunks = _chunk_prose(prose_lines, markdown)
        except Exception as e:
            raise ChunkingError(f"Failed to chunk prose section '{heading}': {e}") from e
        for c in prose_chunks:
            c.metadata = {**section_meta, "content_type": "prose"}
            all_chunks.append(c)

        # Process tables
        for table_start, table_end in tables:
            # Extract table info for metadata
            table_meta = {**section_meta, "content_type": "table"}
            # Try to get sheet name or page number from heading
            sheet_match = re.match(r"## Sheet:\s*(.+)", heading)
            page_match = re.match(r"## Page\s+(\d+)", heading)
            if sheet_match:
                table_meta["sheet_name"] = sheet_match.group(1).strip()
            if page_match:
                table_meta["page_number"] = int(page_match.group(1))

            try:
                table_chunks = _chunk_table(lines, table_start, table_end)
            except Exception as e:
                raise ChunkingError(f"Failed to chunk table: {e}") from e
            for c in table_chunks:
                c.metadata = {**table_meta}
                all_chunks.append(c)

    total = len(all_chunks)
    if total == 0:
        return 0

    # Assign chunk indices and update metadata
    for i, c in enumerate(all_chunks):
        c.metadata["chunk_index"] = i + 1
        c.metadata["total_chunks"] = total

    # Embed and insert in batches of 20
    batch_size = 20
    cur = db_conn.cursor()

    for batch_start in range(0, total, batch_size):
        batch = all_chunks[batch_start : batch_start + batch_size]
        for c in batch:
            try:
                embedding = _embed_text(c.content)
            except Exception as e:
                raise ChunkingError(f"Failed to embed chunk {i + 1}: {e}") from e
            meta_json = c.metadata

            try:
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, c.content, meta_json, embedding),
                )
            except Exception as e:
                raise ChunkingError(f"Failed to insert chunk {i + 1}: {e}") from e
        db_conn.commit()
    return total


# ------ CLI ------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk markdown and insert into the RAG database"
    )
    parser.add_argument("--document-id", type=int, required=True,
                        help="Document ID in the DB (from documents table)")
    parser.add_argument("--file", required=True, help="Path to the source file")
    args = parser.parse_args()

    # Convert the file
    from converter import convert_file
    markdown, meta = convert_file(args.file)

    file_type = Path(args.file).suffix.lstrip(".").lower()

    db = _get_conn()

    # Insert document
    _upsert_document(
        doc_id=args.document_id,
        file_path=args.file,
        file_type=file_type,
        category="other",
        title=meta.get("title"),
        revision=meta.get("detected_revision"),
        doc_date=meta.get("detected_date"),
        markdown=markdown,
        equipment_id=None,
        location=None,
        db_conn=db,
    )

    # Chunk and insert
    chunks_count = chunk_and_insert(args.document_id, markdown, file_type, db)
    print(f"Inserted {chunks_count} chunks for document {args.document_id}")


def _get_conn():
    import os

    url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    try:
        return psycopg.connect(url)
    except Exception as e:
        raise ChunkingError(f"Failed to connect to database: {e}") from e


if __name__ == "__main__":
    main()
