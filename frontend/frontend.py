"""EdinTech-RAG — Streamlit frontend for industrial document Q&A.

Provides a chat-based interface to query ingested documents, upload files
for ingestion, and manage the document corpus via the FastAPI backend.

Run:
    streamlit run frontend.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd  # noqa: E402
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")

CATEGORIES = [
    "manual",
    "datasheet",
    "maintenance_record",
    "procedure",
    "report",
    "specification",
    "log",
    "other",
]


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EdinTech-RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EdinTech-RAG")
st.caption("Industrial document Q&A — ask questions or upload documents below.")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # type: ignore[assignment]


def _init_session() -> None:
    """Ensure all session-state keys exist."""
    defaults = {
        "upload_category": "other",
        "upload_equipment_id": "",
        "upload_location": "",
        "upload_revision": "",
        "filter_equipment_id": "",
        "filter_file_type": "",
        "filter_document_category": "",
        "filter_location": "",
        "filter_top_k": 5,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ---------------------------------------------------------------------------
# Helper functions (defined before use)
# ---------------------------------------------------------------------------


def _extract_error_detail(text: str) -> str:
    """Try to parse JSON error detail from FastAPI responses."""
    try:
        data = json.loads(text)
        return data.get("detail", text[:200])
    except Exception:
        return text[:200]


def _handle_ingestion(files: list) -> None:
    """Upload files one-by-one to POST /ingest and poll for progress.

    Uses session_state to track in-progress jobs so the UI can update
    without blocking Streamlit's event loop.
    """
    import time

    # Initialize session state for tracking ingestion jobs if not present
    if "ingestion_jobs" not in st.session_state:
        st.session_state.ingestion_jobs = {}  # type: ignore[attr-defined]

    for f in files:
        resp_data = f.read()

        # Start ingestion job
        resp = httpx.post(
            f"{BACKEND_URL}/ingest",
            files={"file": (f.name, resp_data, f.type)},
            data={
                "category": upload_category,
                "equipment_id": upload_equipment_id or None,
                "location": upload_location or None,
                "revision": upload_revision or None,
            },
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

        # Store job info in session state for polling
        st.session_state.ingestion_jobs[job_id] = {  # type: ignore[attr-defined]
            "filename": f.name,
            "status": "queued",
            "message": f"Queued **{f.name}**",
            "result": None,
            "error": None,
        }

    # Trigger rerun so the polling loop picks up new jobs
    st.rerun()


def _delete_document(doc_id: int) -> None:
    """Delete a document by ID and refresh."""
    try:
        resp = httpx.delete(f"{BACKEND_URL}/documents/{doc_id}", timeout=10)
        if resp.status_code == 204:
            st.toast(f"Document {doc_id} deleted", icon="🗑️")
        else:
            st.toast(f"Failed to delete document {doc_id}", icon="❌")
    except Exception as exc:
        st.toast(f"Error deleting document: {exc}", icon="❌")
    st.rerun()


def _render_documents_table() -> None:
    """Fetch and display all documents with delete buttons."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/documents", timeout=10)
        resp.raise_for_status()
        docs = resp.json()
    except Exception as exc:
        st.caption(f"Could not fetch documents: {exc}")
        return

    if not docs:
        st.caption("No documents ingested yet.")
        return

    # Render delete buttons per row alongside a simple list
    for d in docs:
        col1, col2 = st.columns([6, 1])
        with col1:
            st.caption(
                f"{d['filename']} "
                f"({d['document_category']}) — {d['chunk_count']} chunks"
            )
        with col2:
            if st.button("Delete", key=f"del_{d['id']}", use_container_width=True):
                _delete_document(d["id"])


# ---------------------------------------------------------------------------
# Sidebar — file upload & document management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Select files to ingest",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "csv"],
        key="file_uploader",
    )

    upload_category = st.selectbox(
        "Document Category",
        options=CATEGORIES,
        index=CATEGORIES.index("other"),
        key="upload_category",
    )
    upload_equipment_id = st.text_input(
        "Equipment ID (optional)", key="upload_equipment_id"
    )
    upload_location = st.text_input("Location (optional)", key="upload_location")
    upload_revision = st.text_input("Revision (optional)", key="upload_revision")

    ingest_btn = st.button("Ingest Files", type="primary", use_container_width=True)

    if ingest_btn and uploaded_files:
        _handle_ingestion(uploaded_files)
        st.rerun()

    st.divider()

    with st.expander("View Ingested Documents", expanded=False):
        _render_documents_table()

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []  # type: ignore[assignment]
        st.rerun()


# ---------------------------------------------------------------------------
# Ingestion job polling — runs on every rerun to show progress
# ---------------------------------------------------------------------------
import time as _time

if "ingestion_jobs" in st.session_state and st.session_state.ingestion_jobs:  # type: ignore[attr-defined]
    jobs = st.session_state.ingestion_jobs.copy()  # type: ignore[attr-defined]
    for job_id, job_info in list(jobs.items()):
        if job_info["status"] in ("complete", "error"):
            continue
        try:
            status_resp = httpx.get(
                f"{BACKEND_URL}/ingest/status/{job_id}", timeout=10
            )
            if status_resp.status_code == 404:
                job_info["status"] = "error"
                job_info["message"] = "Job not found (expired?)"
                continue
            job_data = status_resp.json()
            job_info["status"] = job_data.get("status", "unknown")
            job_info["message"] = job_data.get("message", "")
            if "result" in job_data:
                job_info["result"] = job_data["result"]
        except Exception as exc:
            job_info["error"] = str(exc)

    # Show progress for active jobs
    for job_id, job_info in jobs.items():
        if job_info["status"] in ("complete", "error"):
            with st.status(
                job_info["message"],
                state="complete" if job_info["status"] == "complete" else "error",
            ) as status:
                pass  # Status already collapsed
        else:
            with st.status(
                job_info["message"], expanded=True
            ) as status:
                status.update(label=job_info["message"])
            _time.sleep(0.5)  # Brief pause to avoid tight polling loop
            st.rerun()  # Rerun to update UI with new progress

    # Clean up completed jobs after a delay
    if all(
        j["status"] in ("complete", "error") for j in jobs.values()
    ):
        _time.sleep(2)  # Let user see final state briefly
        st.session_state.ingestion_jobs = {}  # type: ignore[attr-defined]
        st.toast("All documents ingested successfully", icon="✅")
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

# Render existing messages
for msg in st.session_state.messages:  # type: ignore[attr-defined]
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "thinking" in msg and msg["thinking"]:
            with st.expander("Reasoning"):
                st.markdown(msg["thinking"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['filename']}** — {src['section']} "
                        f"({src['document_category']}) | score: {src['score']:.4f}"
                    )

# Filter panel (collapsible) above chat input
with st.expander("Query Filters", expanded=False):
    with st.form("filter_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_equipment_id = st.text_input(
                "Equipment ID", key="filter_equipment_id"
            )
            filter_file_type = st.text_input("File Type", key="filter_file_type")
        with col_f2:
            filter_document_category = st.selectbox(
                "Document Category",
                options=[""] + CATEGORIES,
                index=0,
                key="filter_document_category",
            )
            filter_location = st.text_input("Location", key="filter_location")
        filter_top_k = st.number_input(
            "Top K results", min_value=1, max_value=50, value=5, key="filter_top_k"
        )
        st.form_submit_button("Apply Filters", use_container_width=True)

# Chat input at the bottom
if prompt := st.chat_input("Ask a question about your documents…"):
    # Build filters dict from form values
    def _non_empty(val: str | None) -> str | None:
        return val if val and val.strip() else None

    filters = {
        "equipment_id": int(filter_equipment_id) if filter_equipment_id and filter_equipment_id.strip() else None,
        "file_type": _non_empty(filter_file_type),
        "document_category": _non_empty(filter_document_category),
        "location": _non_empty(filter_location),
    }
    top_k = int(st.session_state.filter_top_k)

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})  # type: ignore[operator]

    # Call the backend
    with st.chat_message("assistant"):
        container = st.empty()
        thinking_container = None

        try:
            resp = httpx.post(
                f"{BACKEND_URL}/query",
                json={
                    "question": prompt,
                    "filters": filters,
                    "top_k": top_k,
                    "show_thinking": True,
                },
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()

            # Show answer with streaming-like character-by-character feel
            answer = result.get("answer", "")
            thinking = result.get("thinking")
            sources = result.get("sources", [])

            container.markdown(answer)

            # Thinking expander (collapsed by default, shown if present)
            if thinking:
                with st.expander("Reasoning"):
                    st.markdown(thinking)

            # Sources expander
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(
                            f"- **{src['filename']}** — {src['section']} "
                            f"({src['document_category']}) | score: {src['score']:.4f}"
                        )

            # Store assistant message
            st.session_state.messages.append(  # type: ignore[operator]
                {
                    "role": "assistant",
                    "content": answer,
                    "thinking": thinking,
                    "sources": sources,
                }
            )

        except httpx.HTTPStatusError as exc:
            container.error(f"Backend error ({exc.response.status_code}):")
            detail = _extract_error_detail(exc.response.text)
            st.markdown(detail)
        except Exception as exc:
            container.error(f"Request failed:")
            st.markdown(str(exc))
