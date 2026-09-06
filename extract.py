"""PDF text extraction and LLM structuring (Groq Cloud by default, Ollama optional)."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import pdfplumber
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
    "Only use information present in the text. Use null for anything not stated. Do not guess. "
    "Ignore equipment schedule rows when determining project metadata; schedule titles, equipment "
    "tags, model numbers, and capacities are not project metadata."
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
    "The page text below may end with a second section titled 'STRUCTURED TABLE "
    "DATA'. That holds the same table(s) read directly from the PDF's table grid, "
    "so each row's columns (e.g. manufacturer and model) are correctly paired to "
    "that row and only that row. Where a row there already starts with "
    "'Tag: ...', that tag was matched to this exact row from the table's own "
    "layout -- use it directly as this row's tag; do NOT look up a tag for it "
    "in the raw text above, and do NOT pair it with any other row's tag. Only "
    "when a structured row has no 'Tag: ...' should you look for that row's "
    "tag in the raw text, matching by a shared value (e.g. the same CFM, size, "
    "or model number) -- never by position or order, and never a tag already "
    "used for a different row. A row visible in only one section should still "
    "be extracted from whichever section has it. Where a cell is already "
    "labeled 'Manufacturer: ...' or 'Model: ...', that split was verified from "
    "the table's own layout -- use it directly rather than re-guessing it, and "
    "never swap in a manufacturer or model value from a different row.\n"
    "\n"
    "Copy values verbatim. Use null for a blank cell. Return an empty list if this page "
    "has no equipment schedule."
)


def _progress(msg: str, started: Optional[float] = None) -> float:
    """Print a phase marker, optionally with how long the previous one took.

    extract_pdf spends minutes in three places that produced no output at all --
    page text extraction, pdfplumber table detection, and a rate-limited
    metadata call -- which made a working run indistinguishable from a hung one.
    Returns a timestamp so the caller can close the phase off.
    """
    if config.SHOW_PROGRESS:
        took = f"  ({time.time() - started:.1f}s)" if started else ""
        print(f"      {msg}{took}", flush=True)
    return time.time()


@dataclass
class ExtractionResult:
    status: str  # 'ok' | 'needs_ocr' | 'no_schedule_pages' | 'error'
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


class GroqPayloadTooLarge(RuntimeError):
    """The request alone exceeds Groq's tokens-per-minute limit (HTTP 413).

    Waiting and retrying the same request cannot help here — the request has
    to be made smaller. Raised so callers that can shrink their input (e.g.
    by splitting a page in half) get a chance to do so instead of just
    losing the whole call.
    """


class GroqRateLimited(RuntimeError):
    """Groq's Retry-After exceeded config.GROQ_MAX_BACKOFF (HTTP 429).

    Distinct from a brief 429 the request loop absorbs: this means the account's
    budget is spent, not that the endpoint is momentarily busy. Sleeping it off
    can take many minutes per call with nothing to show, so it is raised and
    aborts the batch -- every later file would hit the same wall.
    """


class GroqInvalidJSON(RuntimeError):
    """Groq rejected its own generation as malformed JSON (HTTP 400,
    code 'json_validate_failed') -- typically a truncated or malformed
    completion. Distinct from GroqPayloadTooLarge / plain rate-limit errors
    so the caller knows a repair retry (not a resend) already happened.
    """


_JSON_REPAIR_NOTE = (
    "\n\nIMPORTANT: your previous response was not valid JSON — it may have "
    "been truncated or malformed. Return ONLY one complete, syntactically "
    "valid JSON object conforming to the schema above. Do not truncate it."
)


def _groq_chat_json(system: str, user: str, schema: dict) -> dict:
    """Call Groq's OpenAI-compatible API and return parsed JSON.

    Uses JSON-object mode (works on every Groq chat model) with the exact JSON
    schema appended to the system prompt so the model returns the right shape.
    Retries on rate limits (HTTP 429). Raises GroqPayloadTooLarge on HTTP 413
    (request too large for the account's tokens-per-minute limit) instead of
    retrying, since retrying an oversized request just fails again.

    If Groq's own generation is malformed or truncated JSON -- either a local
    parse failure on an otherwise-successful response, or Groq's HTTP 400
    'json_validate_failed' -- one repair attempt is made with a stronger
    instruction before giving up.
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

    url = config.GROQ_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
        # A real User-Agent is required — Groq's Cloudflare blocks the default
        # Python-urllib agent with HTTP 403 (error 1010).
        "User-Agent": "hvac-extractor/1.0",
    }

    def request_once(sys_content: str) -> str:
        """One round trip (with its own 429 backoff). Returns the raw
        assistant message content string, unparsed, or raises."""
        body = json.dumps({
            "model": config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode()

        for attempt in range(5):
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.loads(r.read())
                return resp["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:  # rate limited -> back off and retry
                    try:
                        wait = float(e.headers.get("retry-after", 4)) + 1
                    except (TypeError, ValueError):
                        wait = 5.0
                    if wait > config.GROQ_MAX_BACKOFF:
                        raise GroqRateLimited(
                            f"Groq asked for a {wait:.0f}s wait (429), over the "
                            f"{config.GROQ_MAX_BACKOFF:.0f}s cap. The account's rate "
                            f"limit is exhausted rather than busy."
                        )
                    _progress(f"rate limited, waiting {wait:.0f}s "
                              f"(attempt {attempt + 1}/5)")
                    time.sleep(wait)
                    continue
                raw_detail = e.read().decode(errors="replace")
                if e.code == 401:
                    raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY in .env.")
                if e.code == 413:
                    raise GroqPayloadTooLarge(f"Groq API error 413: {raw_detail[:300]}")
                if e.code == 400 and "json_validate_failed" in raw_detail:
                    raise GroqInvalidJSON(f"Groq API error 400: {raw_detail[:300]}")
                raise RuntimeError(f"Groq API error {e.code}: {raw_detail[:300]}")
            except urllib.error.URLError as e:
                raise RuntimeError(
                    f"Could not reach Groq ({e.reason}). Check your internet connection."
                )
        raise RuntimeError("Groq API kept rate-limiting after several retries.")

    try:
        return json.loads(request_once(sys_full))
    except (GroqInvalidJSON, json.JSONDecodeError):
        # Safe response validation failed: either Groq itself rejected the
        # generation, or it returned 200 with content that isn't valid JSON
        # (truncation is the usual cause). One repair retry before giving up.
        try:
            return json.loads(request_once(sys_full + _JSON_REPAIR_NOTE))
        except json.JSONDecodeError as e:
            raise GroqInvalidJSON(f"Groq returned invalid JSON even after a repair retry: {e}") from e


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


def _detect_mfr_model_columns(table: List[List[Optional[str]]]) -> Optional[dict]:
    """Best-effort, structure-driven detection of which column(s) hold
    manufacturer/model in one pdfplumber-extracted table. Looks only at
    header keywords (when pdfplumber captured them) and each cell's own
    line-break structure -- never at manufacturer/model VALUES -- so it
    generalizes across PDFs instead of assuming a fixed column layout.

    Returns {"mode": "separate", "mfr_col": i, "model_col": j} when a row
    (almost certainly a header) has "MANUFACTURER"/"MANUF" in one cell and
    "MODEL" in a different cell; {"mode": "combined", "col": i} when no such
    header row exists but some column's cells are consistently two lines
    (manufacturer stacked over model in one ruled cell, as these schedules
    render a single "MANUF. & MODEL" column when it isn't split by a rule);
    or None if neither is supported by this table's structure.
    """
    if not table:
        return None
    ncols = max(len(row) for row in table)

    for row in table:
        mfr_col = model_col = None
        for c, cell in enumerate(row):
            text = (cell or "").upper()
            if mfr_col is None and ("MANUFACTURER" in text or "MANUF" in text):
                mfr_col = c
            if model_col is None and "MODEL" in text:
                model_col = c
        if mfr_col is not None and model_col is not None and mfr_col != model_col:
            return {"mode": "separate", "mfr_col": mfr_col, "model_col": model_col}

    # A combined column stacks manufacturer over model within one ruled
    # cell, so most of its non-empty cells are exactly two lines -- but a
    # size/style column (e.g. '24"x24"\nLAY-IN') can look the same way.
    # What tells them apart structurally: a manufacturer repeats across many
    # rows while a model number mostly doesn't, so the "spread" between how
    # repetitive each stacked line is is far larger for a real mfr/model
    # column than for a column where both lines are drawn from a small,
    # equally-repeated set of standard terms. Scoring every candidate this
    # way (instead of just taking the first/leftmost match) keeps the
    # detection tied to the data's own structure, not column position.
    best_col, best_score = None, -1.0
    for c in range(ncols):
        values = [row[c] for row in table if c < len(row) and (row[c] or "").strip()]
        if len(values) < 2:
            continue
        two_line = [v.split("\n") for v in values if len(v.split("\n")) == 2]
        if len(two_line) / len(values) < 0.6:
            continue
        firsts = [p[0] for p in two_line]
        seconds = [p[1] for p in two_line]
        spread = abs(_distinct_ratio(seconds) - _distinct_ratio(firsts))
        if spread > best_score:
            best_col, best_score = c, spread

    if best_col is not None:
        return {"mode": "combined", "col": best_col}
    return None


def _distinct_ratio(values: List[str]) -> float:
    """Fraction of values that are distinct -- low for a value that repeats
    a lot (a manufacturer name across many rows), high for one that's
    different almost every row (a model number)."""
    return len(set(values)) / len(values) if values else 0.0


def _combined_column_roles(pairs: List[tuple]) -> List[tuple]:
    """Given each row's two stacked values from a combined column, decide
    per row which is the manufacturer: whichever value recurs more often
    across the column as a whole. Manufacturers repeat across rows; model
    numbers mostly don't -- and unlike which text line a value happens to
    render on (not always consistent row to row in these PDFs), how often
    a value repeats is a property of the data itself. Returns (mfr, model)
    per row, defaulting to the original order on an exact tie.
    """
    freq: Dict[str, int] = {}
    for a, b in pairs:
        freq[a] = freq.get(a, 0) + 1
        freq[b] = freq.get(b, 0) + 1
    return [(b, a) if freq[b] > freq[a] else (a, b) for a, b in pairs]


_TAG_HEADER_RE = re.compile(r"\bSYMBOL\b|\bTAG\b|\bMARK\b")


def _detect_tag_column(table: List[List[Optional[str]]]) -> Optional[int]:
    """Header-keyword detection of a Symbol/Tag/Mark column already inside
    the ruled table. Word-boundary matched so it doesn't false-positive on
    unrelated text containing the substring (e.g. 'VOLTAGE', 'STAGE')."""
    for row in table:
        for c, cell in enumerate(row):
            if _TAG_HEADER_RE.search((cell or "").upper()):
                return c
    return None


# How far left of a table's own ruled boundary to look for a tag/symbol
# column that sits outside it (a common CAD-export pattern -- confirmed on
# the real page-6 diffuser schedule, whose Symbol column has no vertical
# rule separating it from the table). Bounded rather than searching the
# whole page so unrelated far-left content on a dense drawing sheet can't
# be picked up.
_TAG_SEARCH_MARGIN = 200


_ANNOTATION_RE = re.compile(r"^\(.*\)$")  # e.g. "(EXISTING)" -- a status note, not a tag


def _rows_are_non_overlapping(table_obj) -> bool:
    """True only if every row's vertical span is disjoint from the next.

    Confirmed on the real file: a table whose cells span multiple visual
    rows (the VAV/AH schedule's COOLING/HEATING sub-rows per unit) gets
    OVERLAPPING row bboxes from pdfplumber, unlike a plain one-row-per-item
    table (the diffuser schedule). With overlapping rows, one tag word can
    fall inside more than one row's range at once, so there's no reliable
    way to say which row it belongs to -- matching it to any of them would
    be a guess, not something the structure actually supports.
    """
    tops_and_bottoms = sorted((r.bbox[1], r.bbox[3]) for r in table_obj.rows)
    return all(
        tops_and_bottoms[i][1] <= tops_and_bottoms[i + 1][0]
        for i in range(len(tops_and_bottoms) - 1)
    )


def _find_row_tags(page, table_obj) -> Dict[int, str]:
    """Recover a tag/symbol column that lives outside a table's ruled grid
    by matching text geometrically: only text whose vertical position falls
    inside one specific row's own bounding box is attributed to that row.
    This is what keeps Tag, Manufacturer, Model, and the rest of the row
    tied to the same physical table row -- never inferred from order,
    never borrowed from a neighboring row or another schedule.

    Returns nothing at all for a table whose rows overlap (see
    _rows_are_non_overlapping) -- guessing there would risk exactly the
    cross-row mismatch this function exists to prevent.
    """
    if not _rows_are_non_overlapping(table_obj):
        return {}
    x0, top, _x1, bottom = table_obj.bbox
    left_edge = max(0, x0 - _TAG_SEARCH_MARGIN)
    if left_edge >= x0:
        return {}
    try:
        words = page.crop((left_edge, top, x0, bottom)).extract_words()
    except Exception:
        return {}
    words = [w for w in words if not _ANNOTATION_RE.match(w["text"])]
    tags: Dict[int, str] = {}
    for i, row in enumerate(table_obj.rows):
        rtop, rbottom = row.bbox[1], row.bbox[3]
        matched = sorted(
            (w for w in words if rtop <= w["top"] < rbottom),
            key=lambda w: w["x0"],
        )
        if matched:
            tags[i] = " ".join(w["text"] for w in matched)
    return tags


def _format_table(
    table: List[List[Optional[str]]],
    row_tags: Optional[Dict[int, str]] = None,
    tag_col: Optional[int] = None,
) -> str:
    """One row per line, columns joined with ' | '. Where the manufacturer
    and model can be identified (see _detect_mfr_model_columns), they're
    labeled explicitly and normalized into the correct two fields; every
    other cell is passed through as-is, with a cell's own internal line
    breaks becoming ' / ' so multi-line values stay visibly distinct.

    row_tags/tag_col (see _find_row_tags/_detect_tag_column) attach each
    row's own Tag/Symbol -- recovered by the same row's geometry, never by
    position across rows -- so a caller never has to guess which tag a
    Manufacturer/Model pair belongs to from a separately-ordered text block.
    """
    detection = _detect_mfr_model_columns(table)

    roles: Dict[int, tuple] = {}
    if detection and detection["mode"] == "combined":
        col = detection["col"]
        row_idx = [i for i, row in enumerate(table)
                   if col < len(row) and len((row[col] or "").split("\n")) == 2]
        pairs = [tuple((table[i][col] or "").split("\n")) for i in row_idx]
        for i, (mfr, model) in zip(row_idx, _combined_column_roles(pairs)):
            roles[i] = (mfr.strip(), model.strip())

    lines = []
    for i, row in enumerate(table):
        cells = [(cell or "").replace("\n", " / ").strip() for cell in row]
        if not any(cells):
            continue
        if detection and detection["mode"] == "combined" and i in roles:
            mfr, model = roles[i]
            cells[detection["col"]] = f"Manufacturer: {mfr} / Model: {model}"
        elif detection and detection["mode"] == "separate":
            mc, mo = detection["mfr_col"], detection["model_col"]
            if mc < len(cells) and cells[mc]:
                cells[mc] = f"Manufacturer: {cells[mc]}"
            if mo < len(cells) and cells[mo]:
                cells[mo] = f"Model: {cells[mo]}"

        tag_prefix = ""
        if tag_col is not None and tag_col < len(cells) and cells[tag_col]:
            cells[tag_col] = f"Tag: {cells[tag_col]}"
        elif row_tags and i in row_tags:
            tag_prefix = f"Tag: {row_tags[i]} | "

        lines.append(tag_prefix + " | ".join(cells))
    return "\n".join(lines)


def _extract_tables_for_pages(path: str, page_indices: List[int]) -> Dict[int, List[str]]:
    """Best-effort structured-table text for the given pages.

    pypdf's plain text extraction reads CAD-drawn tables in the PDF's
    internal paint order, not visual row order, which scrambles which
    manufacturer/model belongs to which row on dense schedules, and which
    tag a row even belongs to. pdfplumber reconstructs each ruled table from
    its actual grid, keeping a row's columns correctly paired; when the
    table's own Symbol/Tag column sits outside that ruled grid (common in
    these CAD exports), it's recovered geometrically instead (see
    _find_row_tags) so it stays tied to the correct row rather than left for
    a caller to match up against a separately-ordered block of text.

    Returns {page_index: [block, block, ...]} -- one formatted block per
    detected table, kept separate rather than pre-joined, so a caller that
    needs to shrink an oversized request can split BETWEEN whole table
    blocks (never inside one, which would break a Tag from its own
    Manufacturer/Model) instead of being stuck re-sending every table's data
    in full no matter how the surrounding page text is divided. Failures
    (corrupt file, unsupported page, no tables) are swallowed so callers
    always still have the plain text to fall back on.
    """
    if not page_indices:
        return {}
    result: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for n, i in enumerate(page_indices, 1):
                if i >= len(pdf.pages):
                    continue
                page = pdf.pages[i]
                tp = _progress(f"table scan page {i + 1} ({n}/{len(page_indices)}) ...")
                try:
                    table_objs = page.find_tables()
                except Exception:
                    continue
                finally:
                    _progress(f"table scan page {i + 1} done", tp)
                blocks = []
                for t in table_objs:
                    try:
                        values = t.extract()
                    except Exception:
                        continue
                    tag_col = _detect_tag_column(values)
                    row_tags = None if tag_col is not None else _find_row_tags(page, t)
                    block = _format_table(values, row_tags=row_tags, tag_col=tag_col)
                    if block:
                        blocks.append(block)
                if blocks:
                    result[i] = blocks
    except Exception:
        return {}
    return result


def _is_schedule_page(text: str) -> bool:
    upper = text.upper()
    return any(kw in upper for kw in config.SCHEDULE_KEYWORDS)


# A line that *is* a schedule title ("FAN COIL UNIT SCHEDULE"), as opposed to
# one merely mentioning the word ("SEE SCHEDULE ON SHEET M601", a sheet-index
# row, a general note). Anchored to end-of-line so only the former matches.
_SCHEDULE_TITLE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,&/'\-\.]{2,60}SCHEDULE\s*$", re.M)


def _is_schedule_title_page(text: str) -> bool:
    return bool(_SCHEDULE_TITLE_RE.search(text.upper()))


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

    mech_set = set(mech)

    def _narrow_to_mechanical(candidates: List[int]) -> List[int]:
        """Drop plumbing/electrical schedules, but never drop everything."""
        if not (config.RESTRICT_TO_MECHANICAL and mech):
            return candidates
        return [i for i in candidates if i in mech_set] or candidates

    # SCHEDULE_KEYWORDS match a bare substring, so a title block, a sheet index
    # or a "SEE SCHEDULE" note qualifies a page carrying no table at all.
    # Measured over 8 real drawing sets: 51 of 103 pages qualified, against 13
    # holding an actual schedule title -- roughly 4x the pdfplumber passes and
    # 4x the API calls needed, on exactly the rate limit that is the bottleneck
    # at corpus scale.
    #
    # A schedule title is the strongest signal available, so it is applied to
    # the whole document FIRST and the mechanical narrowing applied to what it
    # finds. Doing it the other way round lets the mechanical filter discard a
    # real schedule page whose text simply never says "MECHANICAL" -- the word
    # usually lives in the title block, so this held only by luck.
    titled: List[int] = []
    if config.PREFER_SCHEDULE_TITLES:
        titled = [i for i, t in enumerate(pages) if _is_schedule_title_page(t)]

    if titled:
        sched_targets = _narrow_to_mechanical(titled)
    else:
        # No page states a schedule title in a form we recognise. One of the 8
        # sets is like this, and it also yields the most rows of any of them --
        # so fall back to the wider keyword rule at its old cost rather than
        # dropping the file silently.
        sched_targets = _narrow_to_mechanical(sched)

    cover = [i for i, t in enumerate(pages) if _is_cover_page(t)]

    # Metadata should come primarily from the cover/title sheets. The previous
    # logic could feed a mechanical schedule page into the metadata pass before
    # some title pages, which increased the chance of mixing drawing/schedule
    # values into project metadata.
    ordered: List[int] = []
    for i in cover + list(range(min(len(pages), config.METADATA_PAGES))) + (mech[:1] if not cover else []):
        if i not in ordered:
            ordered.append(i)
    meta = ordered[: config.METADATA_PAGES]

    return sched_targets, meta


def _merge_metadata(base: ProjectMetadata, extra: ProjectMetadata) -> ProjectMetadata:
    """Fill in whatever `base` is still missing from `extra`.

    Used when the metadata pass had to be split across several smaller
    requests: the project name may come back from the cover sheet and the
    revision from the title block, and neither attempt sees the whole thing.
    First non-empty wins for scalars. Team members are unioned and collapsed
    where one entry's role and firm are each either equal to, or a prefix of,
    the other's -- 'JGGM' / 'JGGM Engineering', or 'Architect' /
    'Architect of Record' for the same firm. Halves of a split see the same
    firm written at different lengths, so these near-duplicates are an artifact
    this function creates and should therefore also remove. The longer, more
    specific spelling is the one kept.

    Deliberately no fuzzy or acronym matching: 'GSA' and 'General Services
    Administration' stay as two rows. Collapsing those is a normalization
    judgement about the data itself, not cleanup of a split artifact.
    """
    for field in ProjectMetadata.model_fields:
        if field == "team":
            continue
        if not getattr(base, field, None) and getattr(extra, field, None):
            setattr(base, field, getattr(extra, field))

    for m in extra.team:
        for existing in base.team:
            if (_prefix_compatible(existing.role, m.role)
                    and _prefix_compatible(existing.firm, m.firm)):
                _absorb_member(existing, m)
                break
        else:
            base.team.append(m)
    return base


def _prefix_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """True when a and b are the same value written at different lengths.

    Compared case-insensitively and ignoring punctuation, so 'SHIVE-HATTERY,
    INC.' and 'Shive Hattery Inc' line up. A blank only matches another blank
    -- without that guard every empty firm would prefix-match every real one
    and collapse unrelated people into a single row.
    """
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return na == nb
    return na.startswith(nb) or nb.startswith(na)


def _norm_name(s: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", (s or "").upper()).strip()


def _absorb_member(keep, other) -> None:
    """Merge `other` into `keep`, preferring the longer role/firm spelling and
    filling any field `keep` left blank."""
    for field in ("role", "firm"):
        a, b = getattr(keep, field) or "", getattr(other, field) or ""
        if len(_norm_name(b)) > len(_norm_name(a)):
            setattr(keep, field, b)
    for field in ("address", "city_state_zip", "phone", "contact", "email"):
        if not getattr(keep, field, None) and getattr(other, field, None):
            setattr(keep, field, getattr(other, field))


def _extract_metadata(page_texts: List[str], _depth: int = 0) -> ProjectMetadata:
    """Structure the cover/title pages into ProjectMetadata.

    Takes the pages separately rather than pre-joined so an oversized request
    has something to shed. Without this, a single 413 on a large title block
    loses project_name / location / engineer for the whole file -- silently,
    since the equipment rows still extract fine and the file is still recorded
    'ok'. That was the single biggest source of missing project identity in
    earlier runs.

    Shrinks in the order that costs the least information: first fewer pages
    (each extracted separately, then merged), and only then a halved page,
    split on a line boundary so address and phone lines stay intact.
    """
    texts = [t.strip() for t in page_texts if t and t.strip()]
    if not texts:
        return ProjectMetadata()
    try:
        return _metadata_once("\n\n".join(texts))
    except (GroqPayloadTooLarge, GroqInvalidJSON):
        # Capped lower than the equipment pass on purpose -- see
        # config.METADATA_MAX_SPLIT_DEPTH. Returning what we have beats
        # spending dozens of rate-limited calls on a title block.
        if _depth >= config.METADATA_MAX_SPLIT_DEPTH:
            raise
        if len(texts) > 1:
            mid = len(texts) // 2
            halves = [texts[:mid], texts[mid:]]
        else:
            lines = texts[0].splitlines(keepends=True)
            if len(lines) <= 1:
                raise  # nothing left to divide
            mid = len(lines) // 2
            halves = [["".join(lines[:mid])], ["".join(lines[mid:])]]
        merged = ProjectMetadata()
        failures = 0
        for half in halves:
            try:
                merged = _merge_metadata(merged, _extract_metadata(half, _depth + 1))
            except (GroqPayloadTooLarge, GroqInvalidJSON):
                failures += 1
        if failures == len(halves):
            raise  # both sides still too large -- report it rather than fake success
        return merged


def _metadata_once(head: str) -> ProjectMetadata:
    prompt = (
        "Extract the project metadata from this mechanical drawing-set title block "
        "/ notes. The 'engineer' should be the mechanical / MEP engineer of record:\n\n"
        f"{head}"
    )
    data = _chat_json(_METADATA_SYSTEM, prompt, ProjectMetadata.model_json_schema())
    return ProjectMetadata.model_validate(data)


def _validate_rows(data: dict) -> List[Equipment]:
    """Validate the model's rows one at a time.

    `schedule` and `tag` are required on Equipment, so validating the whole
    EquipmentPage payload at once means a single malformed row raises and
    discards every other row on the page -- and ValidationError isn't one of
    the exceptions the split fallback below catches, so those rows are simply
    lost. Per-row validation keeps the strict schema the model is asked for
    while salvaging every row that did come back well-formed.
    """
    rows: List[Equipment] = []
    for raw in (data or {}).get("equipment", []):
        if not isinstance(raw, dict):
            continue
        try:
            rows.append(Equipment.model_validate(raw))
        except Exception:
            continue
    return rows


# A page that's still too large after this many halvings is given up on (16
# pieces is far more than any real schedule page should ever need) so a
# pathological input can't recurse forever.
_MAX_SPLIT_DEPTH = 4

# Chars from the start of the page carried along with the second half of a
# split, so it keeps seeing the schedule title / column headers that only
# appeared once at the top of the original page text.
_SPLIT_HEADER_CHARS = 500


def _split_text_in_half(text: str) -> tuple[str, str]:
    """Split on a line boundary near the midpoint so no row is cut in half.

    If a schedule title (any line matching the same SCHEDULE_KEYWORDS used
    for page selection) falls near that midpoint, the split moves to just
    before it instead -- otherwise the title lands in one half and its data
    rows in the other, and a row split off from its title has nothing valid
    to put in the required 'schedule' field.
    """
    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        mid = len(text) // 2
        return text[:mid], text[mid:]
    mid = len(lines) // 2

    # Among every schedule-TITLE line in the page (not just any line with a
    # schedule-ish keyword -- a column header like "MODEL NO." would also
    # match _is_schedule_page and can sit closer to the midpoint than the
    # real title), snap to whichever title is nearest the natural midpoint.
    # Scans the whole page rather than a fixed window since a title can end
    # up far from the raw numeric midpoint on an unevenly-sized page.
    title_lines = [i for i, line in enumerate(lines) if i > 0 and "SCHEDULE" in line.upper()]
    if title_lines:
        mid = min(title_lines, key=lambda i: abs(i - mid))

    return "".join(lines[:mid]), "".join(lines[mid:])


def _extract_equipment_page(
    page_text: str,
    table_blocks: Optional[List[str]] = None,
    _depth: int = 0,
) -> List[Equipment]:
    """table_blocks (see _extract_tables_for_pages) are kept separate from
    page_text, and joined in only when building the actual prompt, so that
    if the combined size forces a split, there are TWO independent ways to
    shrink it: page_text splits as before, and -- new -- the table blocks
    themselves split between whole tables (never inside one, which would
    break a Tag from its own Manufacturer/Model). Splitting page_text alone
    isn't always enough: the structured-table section can by itself already
    be a large fraction of the size limit, so re-attaching it whole to every
    chunk would put a floor under how small a chunk can ever get, no matter
    how much page_text shrinks. Only when a table block can't be split
    further (0 or 1 of them) does the same block get duplicated into both
    halves, since a labeled 'Tag: ...' row is self-contained regardless of
    which chunk's page_text it travels with.
    """
    table_blocks = table_blocks or []
    prompt_text = page_text
    if table_blocks:
        prompt_text += "\n\n--- STRUCTURED TABLE DATA ---\n\n" + "\n\n".join(table_blocks)
    prompt = "Extract every equipment schedule row from this page:\n\n" + prompt_text
    try:
        data = _chat_json(_EQUIPMENT_SYSTEM, prompt, EquipmentPage.model_json_schema())
        return _validate_rows(data)
    except (GroqPayloadTooLarge, GroqInvalidJSON):
        # Either the page alone is bigger than Groq's per-minute token budget
        # (retrying the same request would just fail again), or the model's
        # JSON kept coming back malformed/truncated even after one repair
        # retry -- which a smaller, less complex request is also less likely
        # to trigger, since there's less to enumerate and less room for the
        # completion to run out of budget partway through. Split whichever
        # of page_text / table_blocks can still be divided, and extract each
        # half separately instead of losing the whole page.
        if _depth >= _MAX_SPLIT_DEPTH:
            raise

        first_text, second_text = _split_text_in_half(page_text)
        text_splittable = bool(first_text.strip()) and bool(second_text.strip())
        if not text_splittable:
            first_text = second_text = page_text

        blocks_splittable = len(table_blocks) > 1
        if blocks_splittable:
            mid = len(table_blocks) // 2
            first_blocks, second_blocks = table_blocks[:mid], table_blocks[mid:]
        else:
            first_blocks = second_blocks = table_blocks

        if not text_splittable and not blocks_splittable:
            raise  # nothing left to split — give up as before

        if text_splittable and len(page_text) > _SPLIT_HEADER_CHARS:
            # Only worth it once the header is a small fraction of the page
            # -- otherwise (a short page, or what's left after several
            # rounds of splitting) the "header" is close to the whole text,
            # and prepending it stops the second half from actually
            # shrinking, defeating the point of splitting at all.
            header = page_text[:_SPLIT_HEADER_CHARS]
            second_text = header + "\n...\n" + second_text

        rows = _extract_equipment_page(first_text, first_blocks, _depth + 1)
        rows += _extract_equipment_page(second_text, second_blocks, _depth + 1)
        # A duplicated (not split) half -- page_text or table_blocks, when
        # only one of the two could be divided this round -- can produce the
        # same row from both branches; dedupe so those don't come back twice.
        return _dedupe(rows)


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
    t = _progress("reading page text ...")
    try:
        pages = read_pages(path)
    except Exception as exc:  # corrupt / unreadable PDF
        return ExtractionResult(status="error", error=f"read failed: {exc}")
    t = _progress(f"read {len(pages)} page(s)", t)

    total_chars = sum(len(p) for p in pages)
    if total_chars < config.MIN_TEXT_CHARS:
        return ExtractionResult(status="needs_ocr", error="no extractable text layer")

    sched_idx, meta_idx = _select_pages(pages)
    _progress(f"selected schedule pages {[i + 1 for i in sched_idx]}, "
              f"metadata pages {[i + 1 for i in meta_idx]}")

    table_blocks_by_page = _extract_tables_for_pages(path, sched_idx)

    t = _progress("metadata call ...")
    try:
        metadata = _extract_metadata([pages[i] for i in meta_idx])
    except GroqRateLimited:
        raise          # fatal for the batch, not just this file
    except Exception as exc:
        metadata = ProjectMetadata()
        meta_err = f"metadata: {exc}"
    else:
        meta_err = None
    _progress("metadata done", t)

    rows: List[Equipment] = []
    errors: List[str] = [meta_err] if meta_err else []
    for n, i in enumerate(sched_idx, 1):
        if config.SHOW_PROGRESS:
            print(f"      extract page {i + 1} ({n}/{len(sched_idx)}) ...", flush=True)
        try:
            page_rows = _extract_equipment_page(pages[i], table_blocks_by_page.get(i, []))
        except GroqRateLimited:
            raise      # fatal for the batch, not just this page
        except Exception as exc:
            errors.append(f"page {i + 1}: {exc}")
        else:
            rows.extend(page_rows)

    # A file that yielded nothing must never look like a clean success. Page
    # selection finding no schedule pages at all (a CAD sheet whose table is
    # vector line-art, or one that uses none of SCHEDULE_KEYWORDS) previously
    # returned 'ok' with an empty list, indistinguishable in the output from a
    # file that genuinely has no equipment on it.
    if not sched_idx:
        return ExtractionResult(
            status="no_schedule_pages",
            metadata=metadata,
            equipment=[],
            error="; ".join(errors + ["no page matched SCHEDULE_KEYWORDS"]),
        )

    return ExtractionResult(
        status="ok",
        metadata=metadata,
        equipment=_dedupe(rows),
        error="; ".join(errors) if errors else None,
    )
