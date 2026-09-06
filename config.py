"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env sitting next to this file (if present).
load_dotenv(Path(__file__).with_name(".env"))

# Which backend structures the extracted text into fields:
#   "groq"   -> Groq Cloud API (default; no local model download)
#   "ollama" -> a local Ollama model (fully offline, nothing leaves the PC)
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "groq").strip().lower()

# --- Groq Cloud (default backend) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# --- Local Ollama (optional alternative backend) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

DB_PATH = os.getenv("DB_PATH", "hvac.db")

SCHEDULE_KEYWORDS = [
    kw.strip().upper()
    for kw in os.getenv("SCHEDULE_KEYWORDS", "SCHEDULE,MANUFACTURER,MODEL NO,MODEL #").split(",")
    if kw.strip()
]

MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "100"))

# How many pages (cover + page 1 + first mechanical sheet) to feed the metadata pass.
METADATA_PAGES = int(os.getenv("METADATA_PAGES", "4"))

# How many times the metadata pass may halve an oversized request before giving
# up. Deliberately lower than the equipment pass's depth: each level doubles the
# number of API calls, and a title block holds a handful of fields, so paying up
# to 31 rate-limited calls to recover them is a bad trade. Measured on Kaggle,
# the metadata pass reached 72.6s on one file while every pdfplumber table scan
# on it finished in under 2s. Depth 2 caps it at 7 calls; anything still too
# large returns partial metadata rather than burning the rate limit.
METADATA_MAX_SPLIT_DEPTH = int(os.getenv("METADATA_MAX_SPLIT_DEPTH", "2"))

# Longest single rate-limit sleep to accept. Groq answers 429 with a Retry-After
# that, once the tokens-per-minute budget is exhausted, can be many minutes; the
# old code slept for whatever it said, silently, up to five times per call. A
# run doing that is indistinguishable from a hung one -- it was reported as the
# metadata call "getting stuck". Waits under this are taken (and logged); a
# longer one means the tier is exhausted rather than momentarily busy, which is
# worth surfacing immediately instead of sleeping through.
GROQ_MAX_BACKOFF = float(os.getenv("GROQ_MAX_BACKOFF", "90"))

# Markers (case-insensitive) that identify a MECHANICAL sheet — the M-series /
# "Schedules & Notes – Mechanical" pages where HVAC schedules live.
MECH_MARKERS = [
    m.strip().upper()
    for m in os.getenv("MECH_MARKERS", "MECHANICAL,HVAC").split(",")
    if m.strip()
]

# When true, restrict schedule extraction to mechanical sheets so plumbing /
# electrical schedules are excluded. Falls back to all schedule pages if no
# mechanical sheet is detected.
RESTRICT_TO_MECHANICAL = os.getenv("RESTRICT_TO_MECHANICAL", "true").lower() != "false"

# Prefer pages carrying an actual schedule TITLE line over any page merely
# containing a SCHEDULE_KEYWORDS substring. Measured over 8 drawing sets, the
# substring rule qualified 51 of 103 pages against 13 with a real title -- most
# of the difference being title blocks, sheet indexes and "SEE SCHEDULE" notes,
# each costing a pdfplumber pass and an API call for nothing. Falls back to the
# wider set on any file where no title matches, so no file loses coverage.
PREFER_SCHEDULE_TITLES = os.getenv("PREFER_SCHEDULE_TITLES", "true").lower() != "false"

# Print a line per schedule page as it is processed. A single file can take
# minutes (pdfplumber table detection plus a rate-limited API call per page),
# and without this a long run is indistinguishable from a hung one.
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "true").lower() != "false"

# The full project team directory lives on the COVER / TITLE sheet. These markers
# (case-insensitive) find that sheet so the metadata pass can read the whole team.
COVER_MARKERS = [
    m.strip().upper()
    for m in os.getenv(
        "COVER_MARKERS",
        "COVER SHEET,TITLE SHEET,PROJECT TEAM,PROJECT DIRECTORY,CONSULTANTS,"
        "ARCHITECT OF RECORD,ENGINEER OF RECORD,SHEET INDEX",
    ).split(",")
    if m.strip()
]


def active_model() -> str:
    """The model name for the currently selected backend (for status output)."""
    return GROQ_MODEL if MODEL_BACKEND == "groq" else OLLAMA_MODEL
