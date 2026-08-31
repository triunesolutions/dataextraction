# HVAC Schedule Extractor (Groq Cloud)

Batch-reads HVAC construction PDFs, pulls out the **project details** (name,
location, engineer, architect, dates…) and every **equipment schedule row**
(schedule / tag / manufacturer / model / size-capacity), stores it in SQLite,
and exports flat CSVs.

The AI structuring step runs on **Groq Cloud** — so there's **no multi-GB model
download**. (A fully-local Ollama mode is still available; see *Local mode* below.)

> ⚠️ **Privacy note:** in Groq mode, the text of the drawing pages is sent to
> Groq's API for structuring. If drawings must never leave the machine, use the
> local Ollama mode instead.

---

## 1. Get a free Groq API key

1. Go to **https://console.groq.com/keys**
2. Sign in (Google / GitHub / email).
3. Click **Create API Key**, give it any name.
4. Copy the key — it starts with **`gsk_`**. (You won't be able to see it again,
   so paste it into setup right away.)

Groq has a free tier that's plenty for this tool. Keep the key private — anyone
with it can use your account.

## 2. Setup (one time)

1. **Install Python 3.10+** — https://www.python.org/downloads/
   (tick **“Add Python to PATH”** on the first install screen).
2. **Double-click `setup.bat`.** It installs the Python packages and asks you to
   paste your `gsk_...` key, which it saves to a local `.env` file.

That's it — no model download.

## 3. Run

- **Drag a folder of PDFs onto `run.bat`** (or double-click it and paste a path).
- Results appear in this folder:
  - **`hvac.csv`** — one row per equipment item
  - **`hvac_team.csv`** — the project team directory

### Command line (optional)

```bash
# Process every PDF under one or more folders, then write the CSV
python run.py "D:/Drawings/2025" "D:/Drawings/2026" --export hvac.csv

# Try just 5 PDFs first to sanity-check quality
python run.py "D:/Drawings" --limit 5 --export sample.csv

# Redo everything from scratch
python run.py "D:/Drawings" --force --export hvac.csv

# Rebuild the CSV from the database without re-processing
python run.py --export-only hvac.csv
```

---

## How it works

```
folders ─▶ walk for *.pdf ─▶ hash (skip if already done)
       ─▶ pypdf text layer ─▶ pick MECHANICAL (M-series) schedule pages
       ─▶ per page: Groq structures it into fields
       ─▶ SQLite (projects + team + equipment) ─▶ flat CSV export
```

- **Idempotent** — each PDF is fingerprinted by content hash; re-runs skip files
  already processed. Use `--force` to redo them.
- **Error-isolated** — one bad PDF never stops the batch.
- **Scanned PDFs** (no text layer) are flagged `needs_ocr` instead of producing
  garbage rows.

## Output columns

**`hvac.csv`** — the main equipment list:
```
Project | Location | Engineer | Architect | Address | Project Number |
Drawing Date | Revision Date | Revision | Issue Status | Source File |
Schedule | Tag | Manufacturer | Model | Size/Capacity
```

**`hvac_team.csv`** — the project directory:
```
Project | Source File | Role | Firm | Address | City/State/Zip | Phone | Contact | Email
```

Both are UTF-8 with a BOM so Excel opens them cleanly.

---

## Choosing the model

Set `GROQ_MODEL` in `.env`. Good choices on Groq:

- `openai/gpt-oss-120b` — **default**, most accurate for table extraction.
- `openai/gpt-oss-20b` — faster / lighter.
- `qwen/qwen3.8-27b` — strong alternative.

See the current list at https://console.groq.com/docs/models .

## Local mode (fully offline, optional)

To keep everything on the machine (nothing sent to the cloud):

1. Install Ollama (https://ollama.com/download) and pull a model:
   `ollama pull qwen2.5:7b`
2. `pip install ollama`
3. In `.env`, set `MODEL_BACKEND=ollama`.

---

## Known limitations

- **Bluebeam “AIO”/takeoff-export PDFs** sometimes have a *scrambled text layer*
  (the embedded font has no proper Unicode mapping), so pypdf reads gibberish.
  These have plenty of characters, so they pass the `needs_ocr` check but yield
  no usable data — they'll show `ok` with empty results. Use the original CAD
  drawing set for those projects, not the Bluebeam summary export.
- **Vendor quote/submittal PDFs** (e.g. a Greenheck or Qmark quote) contain the
  words SCHEDULE/MANUFACTURER/MODEL and may be picked up as if they were project
  schedules. Point the tool at drawing sets, not vendor packets, for cleanest output.
- The `Engineer` column is derived from a team member whose role contains
  “MECHANICAL”/“MEP”. On mechanical-only sets whose title block belongs to the
  architect, it may be blank — the full team lives on the cover sheet (`G0.00`/`T-1`).

## Files

| File            | Purpose                                             |
|-----------------|-----------------------------------------------------|
| `run.py`        | CLI orchestrator (walk, hash, skip, process, export)|
| `extract.py`    | PDF text extraction + LLM structuring (Groq/Ollama) |
| `schemas.py`    | Data shape (also the JSON schema given to the model)|
| `db.py`         | SQLite schema + idempotent writes                   |
| `export_csv.py` | Join projects + equipment → flat CSV                |
| `config.py`     | Settings from `.env`                                |
| `setup.bat`     | One-time: install deps + save Groq key              |
| `run.bat`       | Drag-and-drop runner                                |
