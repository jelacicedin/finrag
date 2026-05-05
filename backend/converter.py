"""File converter for industrial documents into structured markdown.

Supported formats: PDF, XLSX, CSV.

Usage
-----
    python converter.py --file path/to/document.pdf

Returns: (markdown_string, metadata_dict)
"""

from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"
)
_REVISION_RE = re.compile(
    r"\b(v?\d+\.\d+(?:\.\d+)?)\b"
)
_SECTION_RE = re.compile(
    r"^(#{1,3})\s+(.+)$", re.MULTILINE
)


def _find_first_dates(text: str) -> list[datetime.date]:
    """Return dates found in text, sorted earliest-first."""
    dates: list[datetime.date] = []
    for match in _DATE_RE.finditer(text):
        raw = match.group(1)
        for fmt in ("%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
                    "%m/%d/%y", "%d-%m-%y"):
            try:
                dates.append(datetime.datetime.strptime(raw, fmt).date())
                break
            except ValueError:
                continue
    return sorted(dates)


def _find_first_revision(text: str) -> str | None:
    """Return the first version-like string, e.g. 'v1.2.3' or '3.1.0'."""
    match = _REVISION_RE.search(text)
    return match.group(0) if match else None


def _escape_md_cell(val: Any) -> str:
    """Sanitize a cell value for markdown tables."""
    s = str(val).replace("|", "\\|").replace("\n", "\\n")
    return s.strip()


def _dataframe_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table, preserving headers."""
    lines: list[str] = []

    # Build the header row (first column names as header)
    headers = df.columns.astype(str).tolist()
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for _, row in df.iterrows():
        cells = [_escape_md_cell(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _infer_page_count(filename: str) -> int | None:
    """Heuristic page count from filename patterns like 'page_5.pdf'."""
    m = re.search(r"page[_ ](\d+)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _convert_pdf(file_path: str) -> tuple[str, dict[str, Any]]:
    """Extract structured markdown from a PDF.

    Uses pdfplumber for text + table extraction per page.
    """
    import pdfplumber  # type: ignore

    path = Path(file_path)
    md_parts: list[str] = []
    page_count = 0

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)

        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Try to detect a page heading from filename or first line
            title = path.stem
            if page_count > 1:
                heading_text = f"## Page {page_idx}"
            else:
                heading_text = "# " + (title or "Document")

            # Try to detect table
            tables = page.extract_tables()
            table_md = ""
            if tables:
                lines: list[str] = []
                for row_idx, row in enumerate(tables):
                    clean_row = [str(cell or "").strip() for cell in row]
                    if row_idx == 0:
                        lines.append("| " + " | ".join(clean_row) + " |")
                        lines.append(
                            "| " + " | ".join("---" for _ in clean_row) + " |"
                        )
                    else:
                        lines.append("| " + " | ".join(clean_row) + " |")
                table_md = "\n\n".join(lines)

            # Assemble section for this page
            section_parts: list[str] = [heading_text]
            if text:
                section_parts.append(f"### Text\n\n{text}")
            if table_md:
                section_parts.append(f"### Table\n\n{table_md}")
            md_parts.append("\n\n".join(section_parts))
            md_parts.append("---")

    first_dates = _find_first_dates("\n".join(md_parts))

    metadata: dict[str, Any] = {
        "title": title,
        "detected_date": first_dates[0].isoformat() if first_dates else None,
        "detected_revision": _find_first_revision("\n".join(md_parts)),
        "page_count": page_count,
    }

    return "\n\n".join(md_parts), metadata


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------

def _convert_xlsx(file_path: str) -> tuple[str, dict[str, Any]]:
    """Convert XLSX workbook to structured markdown.

    Each sheet becomes a ``## Sheet: <name>`` section.
    Merged cells are handled by fill-down.
    """
    try:
        import openpyxl  # type: ignore
    except ImportError:
        raise ImportError("Install openpyxl for XLSX support: pip install openpyxl")

    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    md_parts: list[str] = []
    sheet_names: list[str] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        sheet_names.append(sheet_name)

        # Build a normalised grid with column-wise fill-down for merged cells
        num_rows = ws.max_row
        num_cols = ws.max_column
        if num_rows == 0 or num_cols == 0:
            continue

        raw: dict[tuple[int, int], Any] = {}
        for row in ws.iter_rows():
            for cell in row:
                raw[(cell.row - 1, cell.column - 1)] = cell.value

        # Fill down within each column
        fill_down: dict[int, Any] = {}
        for col in range(num_cols):
            for row in range(num_rows):
                val = raw.get((row, col))
                if val is not None:
                    fill_down[col] = val
                elif col in fill_down:
                    raw[(row, col)] = fill_down[col]

        # Convert to list of tuples; skip fully-empty rows
        rows_list: list[tuple[Any, ...]] = [
            tuple(raw[(row, col)] for col in range(num_cols))
            for row in range(num_rows)
            if any(raw.get((row, col)) is not None for col in range(num_cols))
        ]

        if not rows_list:
            continue

        heading = f"## Sheet: {sheet_name}"
        md_parts.append(heading)

        # Treat first row as header
        header = [_escape_md_cell(c) for c in rows_list[0]]
        md_parts.append("| " + " | ".join(header) + " |")
        md_parts.append("| " + " | ".join("---" for _ in header) + " |")

        for data_row in rows_list[1:]:
            cols = len(header)
            cells = [_escape_md_cell(c) for c in list(data_row)[:cols]]
            # Pad if row is shorter than header
            cells += [""] * (cols - len(cells))
            md_parts.append("| " + " | ".join(cells) + " |")

        md_parts.append("")  # blank separator

    first_dates = _find_first_dates("\n".join(md_parts))

    metadata: dict[str, Any] = {
        "title": Path(file_path).stem,
        "detected_date": first_dates[0].isoformat() if first_dates else None,
        "detected_revision": _find_first_revision("\n".join(md_parts)),
        "sheet_names": sheet_names,
    }

    wb.close()
    return "\n".join(md_parts), metadata


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _convert_csv(file_path: str) -> tuple[str, dict[str, Any]]:
    """Convert CSV to markdown table.

    If >500 rows, split into chunks of ~100.
    """
    df = pd.read_csv(file_path)
    chunk_size = 100
    md_parts: list[str] = []

    if len(df) <= chunk_size:
        md_parts.append(f"# {Path(file_path).stem}")
        md_parts.append(_dataframe_to_md(df))
    else:
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            md_parts.append(f"## Chunk {i + 1}/{n_chunks} (rows {i * chunk_size + 1}-{min((i + 1) * chunk_size, len(df))})")
            md_parts.append(_dataframe_to_md(chunk_df))
            md_parts.append("")  # separator

    first_dates = _find_first_dates("\n".join(md_parts))

    metadata: dict[str, Any] = {
        "title": Path(file_path).stem,
        "detected_date": first_dates[0].isoformat() if first_dates else None,
        "detected_revision": _find_first_revision("\n".join(md_parts)),
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    return "\n\n".join(md_parts), metadata


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def convert_file(file_path: str) -> tuple[str, dict[str, Any]]:
    """Convert a file to structured markdown suitable for chunking.

    Parameters
    ----------
    file_path : str
        Path to the file. Supported: .pdf, .xlsx, .csv, .md, .txt.

    Returns
    -------
    tuple[str, dict]
        (markdown_string, metadata_dict)
        Metadata keys vary by file type; always includes:
        *title*, *detected_date*, *detected_revision*.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _convert_pdf(file_path)
    elif ext in (".xlsx", ".xls"):
        return _convert_xlsx(file_path)
    elif ext in (".csv",):
        return _convert_csv(file_path)
    elif ext in (".md", ".txt"):
        # Plain text/markdown files pass through as-is
        content = path.read_text(encoding="utf-8", errors="replace")
        metadata = {
            "title": path.stem,
            "detected_date": None,
            "detected_revision": None,
            "file_type": ext.lstrip("."),
        }
        return content, metadata
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert industrial documents to structured markdown"
    )
    parser.add_argument("--file", required=True, help="Path to the input file")
    args = parser.parse_args()

    markdown, metadata = convert_file(args.file)

    # Print metadata as JSON
    import json
    print("---METADATA---")
    print(json.dumps(metadata, indent=2, default=str))
    print("---MARKDOWN---")
    print(markdown)


if __name__ == "__main__":
    main()
