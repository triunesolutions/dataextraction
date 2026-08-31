"""SQLite storage: normalized projects + team + equipment, keyed by file hash."""
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

from schemas import Equipment, ProjectMetadata

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_hash    TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    filename     TEXT NOT NULL,
    status       TEXT NOT NULL,          -- 'ok' | 'error' | 'needs_ocr'
    error        TEXT,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    file_hash      TEXT PRIMARY KEY REFERENCES files(file_hash) ON DELETE CASCADE,
    project_name   TEXT,
    location       TEXT,
    address        TEXT,
    project_number TEXT,
    drawing_date   TEXT,
    revision_date  TEXT,
    revision       TEXT,
    issue_status   TEXT,
    engineer       TEXT,   -- derived: mechanical/MEP firm (for the flat CSV)
    architect      TEXT,   -- derived: architect firm (for the flat CSV)
    source_file    TEXT
);

CREATE TABLE IF NOT EXISTS project_team (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash      TEXT NOT NULL REFERENCES files(file_hash) ON DELETE CASCADE,
    role           TEXT,
    firm           TEXT,
    address        TEXT,
    city_state_zip TEXT,
    phone          TEXT,
    contact        TEXT,
    email          TEXT
);

CREATE TABLE IF NOT EXISTS equipment (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash     TEXT NOT NULL REFERENCES files(file_hash) ON DELETE CASCADE,
    schedule      TEXT,
    tag           TEXT,
    manufacturer  TEXT,
    model         TEXT,
    size_capacity TEXT
);

CREATE INDEX IF NOT EXISTS idx_equipment_file ON equipment(file_hash);
CREATE INDEX IF NOT EXISTS idx_team_file ON project_team(file_hash);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def already_done(conn: sqlite3.Connection, file_hash: str) -> bool:
    row = conn.execute(
        "SELECT status FROM files WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    return row is not None and row[0] == "ok"


def _role_firm(metadata: ProjectMetadata, *keywords: str) -> Optional[str]:
    """Firm name of the first team member whose role matches any keyword."""
    for member in metadata.team:
        role = (member.role or "").upper()
        if any(k in role for k in keywords):
            return member.firm
    return None


def _clear(conn: sqlite3.Connection, file_hash: str) -> None:
    conn.execute("DELETE FROM equipment WHERE file_hash = ?", (file_hash,))
    conn.execute("DELETE FROM project_team WHERE file_hash = ?", (file_hash,))
    conn.execute("DELETE FROM projects WHERE file_hash = ?", (file_hash,))
    conn.execute("DELETE FROM files WHERE file_hash = ?", (file_hash,))


def record_result(
    conn: sqlite3.Connection,
    *,
    file_hash: str,
    path: str,
    filename: str,
    status: str,
    processed_at: str,
    error: Optional[str] = None,
    metadata: Optional[ProjectMetadata] = None,
    equipment: Optional[Iterable[Equipment]] = None,
) -> None:
    """Idempotent write: replaces any prior rows for this file hash."""
    _clear(conn, file_hash)
    conn.execute(
        "INSERT INTO files (file_hash, path, filename, status, error, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_hash, path, filename, status, error, processed_at),
    )
    if metadata is not None:
        engineer = _role_firm(metadata, "MECHAN", "MEP", "M/E/P", "M.E.P")
        architect = _role_firm(metadata, "ARCHITECT")
        conn.execute(
            "INSERT INTO projects (file_hash, project_name, location, address, project_number, "
            "drawing_date, revision_date, revision, issue_status, engineer, architect, source_file) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_hash,
                metadata.project_name,
                metadata.location,
                metadata.address,
                metadata.project_number,
                metadata.drawing_date,
                metadata.revision_date,
                metadata.revision,
                metadata.issue_status,
                engineer,
                architect,
                filename,
            ),
        )
        for m in metadata.team:
            conn.execute(
                "INSERT INTO project_team (file_hash, role, firm, address, city_state_zip, "
                "phone, contact, email) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (file_hash, m.role, m.firm, m.address, m.city_state_zip, m.phone, m.contact, m.email),
            )
    for eq in equipment or []:
        conn.execute(
            "INSERT INTO equipment (file_hash, schedule, tag, manufacturer, model, size_capacity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (file_hash, eq.schedule, eq.tag, eq.manufacturer, eq.model, eq.size_capacity),
        )
    conn.commit()
