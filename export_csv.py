"""Export to flat CSVs: the equipment list (main) and the project team directory."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

# --- Main flat list: one row per equipment item, with core project columns ---
EQUIP_COLUMNS = [
    "Project", "Location", "Engineer", "Architect", "Address", "Project Number",
    "Drawing Date", "Revision Date", "Revision", "Issue Status", "Source File",
    "Schedule", "Tag", "Manufacturer", "Model", "Size/Capacity",
]

EQUIP_QUERY = """
SELECT
    p.project_name, p.location, p.engineer, p.architect, p.address, p.project_number,
    p.drawing_date, p.revision_date, p.revision, p.issue_status, p.source_file,
    e.schedule, e.tag, e.manufacturer, e.model, e.size_capacity
FROM equipment e
JOIN projects p ON p.file_hash = e.file_hash
{where}
ORDER BY p.project_name, e.schedule, e.tag
"""

# --- Project team directory: one row per team member per project ---
TEAM_COLUMNS = [
    "Project", "Source File", "Role", "Firm", "Address", "City/State/Zip",
    "Phone", "Contact", "Email",
]

TEAM_QUERY = """
SELECT
    p.project_name, p.source_file,
    t.role, t.firm, t.address, t.city_state_zip, t.phone, t.contact, t.email
FROM project_team t
JOIN projects p ON p.file_hash = t.file_hash
{where}
ORDER BY p.project_name, t.role
"""


def _write(
    db_path: str,
    out_path: str,
    columns,
    query: str,
    hash_column: str,
    file_hashes: Optional[Iterable[str]],
) -> int:
    conn = sqlite3.connect(db_path)
    try:
        if file_hashes is None:
            # No scope given: export everything in the database (existing behavior).
            rows = conn.execute(query.format(where="")).fetchall()
        else:
            hashes = list(file_hashes)
            if not hashes:
                rows = []
            else:
                placeholders = ",".join("?" for _ in hashes)
                where = f"WHERE {hash_column} IN ({placeholders})"
                rows = conn.execute(query.format(where=where), hashes).fetchall()
    finally:
        conn.close()
    # utf-8-sig (BOM) so Excel reads UTF-8 cleanly — no em-dash / degree-sign mojibake.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])
    return len(rows)


def export(
    db_path: str,
    out_path: str,
    file_hashes: Optional[Iterable[str]] = None,
) -> tuple[int, int, str]:
    """
    Write the main equipment CSV to out_path and the team directory CSV to
    '<stem>_team.csv' beside it. Returns (equipment_rows, team_rows, team_path).

    file_hashes: if given, restrict the export to only these files (e.g. the
    PDFs discovered in the current run), so a fresh project doesn't pick up
    rows left over from unrelated projects processed earlier into the same
    database. If omitted (the --export-only path), exports the whole database
    unchanged.
    """
    n_equip = _write(db_path, out_path, EQUIP_COLUMNS, EQUIP_QUERY, "e.file_hash", file_hashes)
    p = Path(out_path)
    team_path = str(p.with_name(f"{p.stem}_team{p.suffix or '.csv'}"))
    n_team = _write(db_path, team_path, TEAM_COLUMNS, TEAM_QUERY, "t.file_hash", file_hashes)
    return n_equip, n_team, team_path
