"""PDF text extraction and LLM structuring (Groq Cloud by default, Ollama optional)."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader

import config
from schemas import Equipment, EquipmentPage, ProjectMetadata

_METADATA_SYSTEM = (
    "You read the cover/title sheet and title block of a construction drawing set and "
    "extract project metadata plus the full project team directory.\n"
    "\n"
    "Field rules:\n"
    "- project_name: the building/project name (e.g. 'Many Farms High School Dormitory').\n"
    "- project_number: the design firm's project or job number from the title block "
    "(e.g. '221244', '480.02.02'). This is NOT the sheet number — sheet numbers look "
    "like 'M102', 'M600', 'M0.10', 'A101' and must NEVER be used as project_number.\n"
    "- location: the city and state of the project.\n"
    "- team: EVERY firm listed (owner, architect of record, civil engineer, structural "
    "engineer, mechanical/MEP engineer, etc.). For each, capture role, firm, address, "
    "city/state/zip, phone, contact person, email when shown. The 'firm' is the company "
    "name only — do not append stray words like 'KEYPLAN' or 'SHEET INDEX'.\n"
    "\n"
    "Only use information present in the text. Use null for anything not stated. Do not guess."
)

_EQUIPMENT_SYSTEM = (
    "You read ONE page of a mechanical (HVAC) construction drawing that contains equipment "
    "schedules (tables of equipment). Extract every equipment row.\n"
    "\n"
    "For each row, fill these fields EXACTLY as defined — do not mix them up:\n"
    "- schedule: the TABLE TITLE only, e.g. 'FAN COIL UNIT SCHEDULE' or "
    "'GRILLES, REGISTERS AND DIFFUSERS SCHEDULE'. NEVER put a room name, a location, "
    "or the row's own data in this field.\n"
    "- tag: the mark/tag in the first column, e.g. 'FC-1', 'CL-1K', 'R6L1'. "
    "Keep ranges as one row (e.g. 'RG-1 THRU RG-7').\n"
    "- manufacturer: the BRAND only, e.g. 'DAIKIN', 'TITUS', 'GREENHECK'. "
    "Do NOT include the model here.\n"
    "- model: the model/part number ONLY, e.g. 'FXAQ07PVIU', '50F', 'TDC'. "
    "Do NOT repeat the manufacturer inside this field.\n"
    "- size_capacity: size or capacity, e.g. '0.6 ton', '260 CFM', '24x24', '7.5 ton'. "
    "Combine multiple with commas.\n"
    "\n"
    "Example — table titled 'FAN COIL UNIT SCHEDULE' with row "
    "'CL-1K  MECH RM 107  DAIKIN  FXAQ07PVIU  0.6 ton  260 CFM' becomes:\n"
    '{"schedule":"FAN COIL UNIT SCHEDULE","tag":"CL-1K","manufacturer":"DAIKIN",'
    '"model":"FXAQ07PVIU","size_capacity":"0.6 ton, 260 CFM"}\n'
    "\n"
    "Copy values verbatim. Use null for a blank cell. Return an empty list if this page "
    "has no equipment schedule."
)


@dataclass
class ExtractionResult:
    status: str  # 'ok' | 'needs_ocr' | 'error'
    metadata: Optional[ProjectMetadata] = None
    equipment: Optional[List[Equipment]] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Model backends
# --------------------------------------------------------------------------- #
_ollama_client = None


def _get_ollama_client():
    """Lazily create the Ollama client (only needed for the local backend)."""
    global _ollama_client
    if _ollama_client is None:
        from ollama import Client  # imported here so Groq users don't need the package
        _ollama_client = Client(host=config.OLLAMA_HOST)
    return _ollama_client


def _ollama_chat_json(system: str, user: str, schema: dict) -> dict:
    resp = _get_ollama_client().chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema,
        options={"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
    )
    return json.loads(resp["message"]["content"])


def _groq_chat_json(system: str, user: str, schema: dict) -> dict:
    """Call Groq's OpenAI-compatible API and return parsed JSON.

    Uses JSON-object mode (works on every Groq chat model) with the exact JSON
    schema appended to the system prompt so the model returns the right shape.
    Retries on rate limits (HTTP 429).
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file (run setup.bat, or see README)."
        )

    sys_full = (
        system
        + "\n\nReturn ONLY a single JSON object that conforms to this JSON Schema "
        "(no markdown, no prose):\n" + json.dumps(schema)
    )
    body = json.dumps({
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": sys_full},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()

    url = config.GROQ_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
        # A real User-Agent is required — Groq's Cloudflare blocks the default
        # Python-urllib agent with HTTP 403 (error 1010).
        "User-Agent": "hvac-extractor/1.0",
    }

    for attempt in range(5):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read())
            return json.loads(resp["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:  # rate limited -> back off and retry
                wait = float(e.headers.get("retry-after", 4)) + 1
                time.sleep(wait)
                continue
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 401:
                raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY in .env.")
            raise RuntimeError(f"Groq API error {e.code}: {detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Groq ({e.reason}). Check your internet connection."
            )
    raise RuntimeError("Groq API kept rate-limiting after several retries.")


def _chat_json(system: str, user: str, schema: dict) -> dict:
    """Structure text into JSON using the configured backend."""
    if config.MODEL_BACKEND == "groq":
        return _groq_chat_json(system, user, schema)
    if config.MODEL_BACKEND == "ollama":
        return _ollama_chat_json(system, user, schema)
    raise RuntimeError(
        f"Unknown MODEL_BACKEND '{config.MODEL_BACKEND}'. Use 'groq' or 'ollama' in .env."
    )


# --------------------------------------------------------------------------- #
#  PDF reading + page selection
# --------------------------------------------------------------------------- #
def read_pages(path: str) -> List[str]:
    """Return the extracted text of each page (empty string if a page has none)."""
    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence "Rotated text discovered"
            text = page.extract_text() or ""
        pages.append(text)
    return pages


def _is_schedule_page(text: str) -> bool:
    upper = text.upper()
    return any(kw in upper for kw in config.SCHEDULE_KEYWORDS)


def _is_mechanical_page(text: str) -> bool:
    upper = text.upper()
    return any(m in upper for m in config.MECH_MARKERS)


def _is_cover_page(text: str) -> bool:
    upper = text.upper()
    return any(m in upper for m in config.COVER_MARKERS)


def _select_pages(pages: List[str]) -> tuple[List[int], List[int]]:
    """Returns (schedule_page_indices, metadata_page_indices)."""
    sched = [i for i, t in enumerate(pages) if _is_schedule_page(t)]
    mech = [i for i, t in enumerate(pages) if _is_mechanical_page(t)]

    if config.RESTRICT_TO_MECHANICAL and mech:
        mech_set = set(mech)
        targeted = [i for i in sched if i in mech_set]
        sched_targets = targeted or sched  # never silently drop everything
    else:
        sched_targets = sched

    cover = [i for i, t in enumerate(pages) if _is_cover_page(t)]
    ordered: List[int] = []
    for i in cover + ([0] if pages else []) + (mech[:1] if mech else []):
        if i not in ordered:
            ordered.append(i)
    meta = ordered[: config.METADATA_PAGES]

    return sched_targets, meta


def _extract_metadata(head: str) -> ProjectMetadata:
    head = head.strip()
    if not head:
        return ProjectMetadata()
    prompt = (
        "Extract the project metadata from this mechanical drawing-set title block "
        "/ notes. The 'engineer' should be the mechanical / MEP engineer of record:\n\n"
        f"{head}"
    )
    data = _chat_json(_METADATA_SYSTEM, prompt, ProjectMetadata.model_json_schema())
    return ProjectMetadata.model_validate(data)


def _extract_equipment_page(page_text: str) -> List[Equipment]:
    text = page_text
    if len(text) > 10000:
        upper = text.upper()
        first_pos = -1
        for kw in config.SCHEDULE_KEYWORDS:
            pos = upper.find(kw)
            if pos != -1 and (first_pos == -1 or pos < first_pos):
                first_pos = pos
        if first_pos != -1:
            start = max(0, first_pos - 500)
            text = text[start:start + 10000]
        else:
            text = text[:10000]

    prompt = "Extract every equipment schedule row from this page:\n\n" + text
    data = _chat_json(_EQUIPMENT_SYSTEM, prompt, EquipmentPage.model_json_schema())
    items: List[Equipment] = []
    for raw in data.get("equipment", []):
        try:
            if isinstance(raw, dict):
                items.append(Equipment.model_validate(raw))
        except Exception:
            continue
    return items


def _dedupe(rows: List[Equipment]) -> List[Equipment]:
    seen = set()
    out: List[Equipment] = []
    for r in rows:
        key = (r.schedule, r.tag, r.manufacturer, r.model, r.size_capacity)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def extract_pdf(path: str) -> ExtractionResult:
    """Full extraction for one PDF. Never raises for content problems."""
    try:
        pages = read_pages(path)
    except Exception as exc:  # corrupt / unreadable PDF
        return ExtractionResult(status="error", error=f"read failed: {exc}")

    total_chars = sum(len(p) for p in pages)
    if total_chars < config.MIN_TEXT_CHARS:
        return ExtractionResult(status="needs_ocr", error="no extractable text layer")

    sched_idx, meta_idx = _select_pages(pages)

    try:
        metadata = _extract_metadata("\n\n".join(pages[i] for i in meta_idx))
    except Exception as exc:
        metadata = ProjectMetadata()
        meta_err = f"metadata: {exc}"
    else:
        meta_err = None

    rows: List[Equipment] = []
    errors: List[str] = [meta_err] if meta_err else []
    for i in sched_idx:
        try:
            rows.extend(_extract_equipment_page(pages[i]))
        except Exception as exc:
            errors.append(f"page {i + 1}: {exc}")

    return ExtractionResult(
        status="ok",
        metadata=metadata,
        equipment=_dedupe(rows),
        error="; ".join(errors) if errors else None,
    )
