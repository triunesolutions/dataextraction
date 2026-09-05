"""
Batch HVAC schedule extractor (Groq Cloud by default, local Ollama optional).

Walks one or more folders for PDFs, extracts project metadata + equipment
schedules with an LLM, stores everything in SQLite, and exports one flat CSV.

Examples
--------
    # Process every PDF under two folders and export the CSV
    python run.py "D:/Drawings/2025" "D:/Drawings/2026" --export out.csv

    # Re-run everything from scratch (ignore the "already done" cache)
    python run.py "D:/Drawings" --force --export out.csv

    # Just rebuild the CSV from what's already in the database
    python run.py --export-only out.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import config
import db
import export_csv
from extract import extract_pdf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(roots: List[str]) -> List[Path]:
    found: List[Path] = []
    for root in roots:
        p = Path(root)
        if p.is_file() and p.suffix.lower() == ".pdf":
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("*.pdf")))
        else:
            print(f"! skipping (not found): {root}", file=sys.stderr)
    return sorted(set(found))


def _check_backend() -> bool:
    """Fail fast with a friendly message if the backend isn't configured."""
    if config.MODEL_BACKEND == "groq" and not config.GROQ_API_KEY:
        print(
            "ERROR: MODEL_BACKEND is 'groq' but no GROQ_API_KEY is set.\n"
            "  -> Run setup.bat and paste your key, or add GROQ_API_KEY to the .env file.\n"
            "  -> Get a free key at https://console.groq.com/keys",
            file=sys.stderr,
        )
        return False
    if config.MODEL_BACKEND not in ("groq", "ollama"):
        print(f"ERROR: MODEL_BACKEND must be 'groq' or 'ollama' (got '{config.MODEL_BACKEND}').",
              file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch HVAC schedule extractor (Groq / Ollama).")
    ap.add_argument("roots", nargs="*", help="Folders and/or PDF files to process.")
    ap.add_argument("--db", default=config.DB_PATH, help="SQLite database path.")
    ap.add_argument("--export", metavar="CSV", help="Write the flat CSV after processing.")
    ap.add_argument("--export-only", metavar="CSV", help="Skip processing; only (re)write the CSV.")
    ap.add_argument("--force", action="store_true", help="Reprocess PDFs already marked done.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N new PDFs (0 = all).")
    args = ap.parse_args()

    if args.export_only:
        n_eq, n_team, team_path = export_csv.export(args.db, args.export_only)
        print(f"Wrote {n_eq} equipment rows -> {args.export_only}")
        print(f"Wrote {n_team} team rows      -> {team_path}")
        return 0

    if not args.roots:
        ap.error("give at least one folder/PDF, or use --export-only")

    if not _check_backend():
        return 2

    conn = db.connect(args.db)
    pdfs = discover(args.roots)
    print(f"Found {len(pdfs)} PDF(s). Backend: {config.MODEL_BACKEND} / {config.active_model()}\n")

    counts = {"ok": 0, "needs_ocr": 0, "no_schedule_pages": 0, "error": 0, "skipped": 0}
    processed = 0
    all_hashes: set[str] = set()
    # Files the pipeline called a success but got nothing out of. Tracked so a
    # batch can never again report a clean run over files it silently dropped.
    empty_ok: List[str] = []

    for i, path in enumerate(pdfs, 1):
        file_hash = sha256(path)
        all_hashes.add(file_hash)
        if not args.force and db.already_done(conn, file_hash):
            counts["skipped"] += 1
            continue
        if args.limit and processed >= args.limit:
            break

        print(f"[{i}/{len(pdfs)}] {path.name} ...", flush=True)
        result = extract_pdf(str(path))
        n_rows = len(result.equipment or [])
        db.record_result(
            conn,
            file_hash=file_hash,
            path=str(path),
            filename=path.name,
            status=result.status,
            processed_at=datetime.now().isoformat(timespec="seconds"),
            error=result.error,
            metadata=result.metadata,
            equipment=result.equipment,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status == "ok" and n_rows == 0:
            empty_ok.append(path.name)
        processed += 1
        tail = f"{n_rows} rows" if result.status == "ok" else result.status.upper()
        if result.error:
            tail += f"  (note: {result.error[:120]})"
        print(f"    -> {tail}")

    conn.close()

    print(
        "\nDone. "
        f"ok={counts['ok']}  needs_ocr={counts['needs_ocr']}  "
        f"no_schedule_pages={counts['no_schedule_pages']}  "
        f"error={counts['error']}  skipped={counts['skipped']}"
    )
    if counts["needs_ocr"]:
        print("  (needs_ocr = scanned/image-only PDFs with no text layer — see README.)")
    if counts["no_schedule_pages"]:
        print("  (no_schedule_pages = readable PDFs where no page matched SCHEDULE_KEYWORDS —")
        print("   usually a CAD sheet whose schedule table is vector line-art. These used to")
        print("   be reported as 'ok' with zero rows.)")
    if empty_ok:
        print(f"  WARNING: {len(empty_ok)} file(s) reported 'ok' but produced 0 equipment rows:")
        for name in empty_ok[:10]:
            print(f"    - {name}")
        if len(empty_ok) > 10:
            print(f"    ... and {len(empty_ok) - 10} more")

    if args.export:
        n_eq, n_team, team_path = export_csv.export(args.db, args.export, file_hashes=all_hashes)
        print(f"Wrote {n_eq} equipment rows -> {args.export}")
        print(f"Wrote {n_team} team rows      -> {team_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
