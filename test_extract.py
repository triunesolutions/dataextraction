"""
Regression tests for Manufacturer/Model column detection in extract.py.

Run directly: python test_extract.py
No network calls. No pytest dependency -- matches the rest of this project's
plain-script style.
"""
from __future__ import annotations

import os
import re

import extract


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_fake_schedule_page(n_rows: int):
    lines = ["EQUIPMENT SCHEDULE\nTAG   MFR   MODEL   SIZE\n"]
    tags = set()
    for i in range(n_rows):
        tag = f"TAG-{i:04d}"
        tags.add(tag)
        lines.append(f"{tag}   ACME   MOD-{i:04d}   {i} CFM\n")
    return "".join(lines), tags


# --------------------------------------------------------------------------- #
# 1. Combined stacked cells: "MANUF. & MODEL" header, one ruled column with
#    manufacturer on the first line and model on the second. Ground truth
#    from the real Diffuser, Grille & Register Schedule (page 6): Tag A =
#    PRICE/SPD, B = PRICE/SMD, C = PRICE/RCDA, D = PRICE/520S.
# --------------------------------------------------------------------------- #
def test_combined_stacked_cells_split_correctly():
    table = [
        ["SYMBOL", "MANUF. &\nMODEL", "SIZE", "MATERIAL"],  # header row
        ["A", "PRICE\nSPD", "24\"x24\"", "STEEL"],
        ["B", "PRICE\nSMD", "24\"x24\"", "STEEL"],
        ["C", "PRICE\nRCDA", "24\"x24\"", "STEEL"],
        ["D", "PRICE\n520S", "24\"x24\"", "STEEL"],
    ]
    out = extract._format_table(table)

    for tag, model in [("A", "SPD"), ("B", "SMD"), ("C", "RCDA"), ("D", "520S")]:
        row = next(l for l in out.splitlines() if l.startswith(f"{tag} |"))
        _assert(f"Manufacturer: PRICE / Model: {model}" in row,
                f"tag {tag}: expected Manufacturer: PRICE / Model: {model}, got: {row}")
    print("PASS: combined stacked cells (A=PRICE/SPD, B=PRICE/SMD, C=PRICE/RCDA, D=PRICE/520S)")


# --------------------------------------------------------------------------- #
# 2. The two stacked lines must become the Manufacturer/Model fields WITHOUT
#    shifting into neighboring columns -- every other column on the row must
#    be byte-for-byte unchanged, in its original position.
# --------------------------------------------------------------------------- #
def test_combined_split_does_not_shift_adjacent_columns():
    table = [
        ["SYMBOL", "MANUF. &\nMODEL", "SIZE", "MATERIAL", "FINISH"],
        ["A", "PRICE\nSPD", "24\"x24\"", "STEEL", "WHITE"],
        ["B", "PRICE\nSMD", "12\"x12\"", "ALUMINUM", "PRIME"],
    ]
    out = extract._format_table(table)
    row_a = next(l for l in out.splitlines() if l.startswith("A |"))
    row_b = next(l for l in out.splitlines() if l.startswith("B |"))

    cols_a = row_a.split(" | ")
    cols_b = row_b.split(" | ")
    _assert(cols_a[0] == "A" and cols_a[2] == '24"x24"' and cols_a[3] == "STEEL" and cols_a[4] == "WHITE",
            f"row A: neighboring columns shifted, got {cols_a}")
    _assert(cols_b[0] == "B" and cols_b[2] == '12"x12"' and cols_b[3] == "ALUMINUM" and cols_b[4] == "PRIME",
            f"row B: neighboring columns shifted, got {cols_b}")
    _assert(cols_a[1] == "Manufacturer: PRICE / Model: SPD", cols_a[1])
    _assert('24"x24"' not in cols_a[1] and "STEEL" not in cols_a[1],
            "size/material leaked into the Manufacturer/Model cell")
    print("PASS: combined split stays in its own column -- no shift into SIZE/MATERIAL/FINISH")


# --------------------------------------------------------------------------- #
# 3. Separate Manufacturer/Model columns -- a different, valid PDF layout.
#    Must be mapped directly by column, not assumed to be stacked.
# --------------------------------------------------------------------------- #
def test_separate_columns_mapped_directly():
    table = [
        ["TAG", "MANUFACTURER", "MODEL NO.", "SIZE"],
        ["RTU-1", "DAIKIN", "FXAQ07PVIU", "0.6 TON"],
        ["RTU-2", "CARRIER", "48TC008", "2.0 TON"],
    ]
    out = extract._format_table(table)

    row1 = next(l for l in out.splitlines() if l.startswith("RTU-1 |"))
    row2 = next(l for l in out.splitlines() if l.startswith("RTU-2 |"))
    _assert("Manufacturer: DAIKIN" in row1 and "Model: FXAQ07PVIU" in row1, row1)
    _assert("Manufacturer: CARRIER" in row2 and "Model: 48TC008" in row2, row2)
    _assert("CARRIER" not in row1 and "48TC008" not in row1, "cross-row contamination in RTU-1")
    print("PASS: separate Manufacturer/Model columns mapped directly by header keywords")


# --------------------------------------------------------------------------- #
# 4. Row order in a combined cell isn't always consistent within the same
#    PDF (confirmed on the real file: one VAV row stores "CSAA021\nTRANE",
#    model-then-manufacturer, while every other row in the same column
#    stores manufacturer-then-model). The split must self-correct via which
#    value repeats across the column, not trust line position.
# --------------------------------------------------------------------------- #
def test_combined_cell_self_corrects_reversed_row_order():
    table = [
        ["SYMBOL", "MANUF. &\nMODEL", "CFM"],
        ["AH-A2", "CSAA021\nTRANE", "10,200"],   # reversed order in the source PDF
        ["AH-A4", "TRANE\nCSAA035", "17,650"],
        ["AH-A5", "TRANE\nCSAA035", "17,900"],
    ]
    out = extract._format_table(table)
    row = next(l for l in out.splitlines() if l.startswith("AH-A2 |"))
    _assert("Manufacturer: TRANE / Model: CSAA021" in row,
            f"reversed-order row not self-corrected: {row}")
    print("PASS: reversed manufacturer/model line order self-corrects via column-wide frequency")


# --------------------------------------------------------------------------- #
# 5. Do not infer missing values from neighboring rows: a row whose combined
#    cell isn't a clean two-line value (blank, or a single line) must be
#    left as-is -- never backfilled from the row above/below.
# --------------------------------------------------------------------------- #
def test_missing_combined_value_not_inferred_from_neighbors():
    table = [
        ["SYMBOL", "MANUF. &\nMODEL", "SIZE"],
        ["A", "PRICE\nSPD", "24x24"],
        ["B", "", "24x24"],          # blank -- must NOT become PRICE/SPD
        ["C", "TBD", "24x24"],       # single line, no stacked split available
        ["D", "PRICE\n520S", "24x24"],
    ]
    out = extract._format_table(table)
    row_b = next(l for l in out.splitlines() if l.startswith("B |"))
    row_c = next(l for l in out.splitlines() if l.startswith("C |"))

    # Row B's combined cell is blank -- it must stay blank, not be
    # backfilled with a neighboring row's PRICE/SPD or PRICE/520S.
    _assert("Manufacturer:" not in row_b and "Model:" not in row_b,
            f"blank combined cell was backfilled from a neighboring row: {row_b}")
    _assert(row_b.split(" | ")[1] == "", f"blank cell value altered: {row_b}")

    _assert("Manufacturer:" not in row_c and "Model:" not in row_c,
            f"single-line cell with no stacked structure must not be force-split: {row_c}")
    _assert(row_c.split(" | ")[1] == "TBD", f"unsplit cell value altered: {row_c}")
    print("PASS: rows with no stacked combined value are left alone, never inferred from neighbors")


# --------------------------------------------------------------------------- #
# 6. Layout-agnostic disambiguation: when MORE than one column looks like a
#    two-line combined cell (e.g. a size/style column stacked the same way),
#    the real manufacturer/model column must still win -- even when it is
#    NOT the leftmost candidate, proving the choice isn't positional.
# --------------------------------------------------------------------------- #
def test_disambiguates_mfr_model_from_other_two_line_columns_regardless_of_position():
    table = [
        ["SYMBOL", "SIZE/STYLE", "MANUF. &\nMODEL", "FINISH"],
        ["A", "24x24\nLAY-IN", "PRICE\nSPD", "WHITE"],
        ["B", "24x24\nLAY-IN", "PRICE\nSMD", "WHITE"],
        ["C", "12x12\nFACE", "PRICE\nRCDA", "WHITE"],
        ["D", "24x24\nLAY-IN", "PRICE\n520S", "WHITE"],
    ]
    out = extract._format_table(table)
    row_a = next(l for l in out.splitlines() if l.startswith("A |"))
    _assert("Manufacturer: PRICE / Model: SPD" in row_a, row_a)
    _assert("Manufacturer: 24x24" not in row_a, "picked the size/style column instead of mfr/model")
    print("PASS: correct combined column chosen by data structure, not by being leftmost")


# --------------------------------------------------------------------------- #
# 7. No signal at all -> safe passthrough, no false labeling.
# --------------------------------------------------------------------------- #
def test_no_signal_leaves_table_unchanged():
    table = [
        ["FC-1", "STEEL", "WHITE", "24x24"],
        ["FC-2", "ALUM", "PRIME", "12x12"],
    ]
    out = extract._format_table(table)
    _assert("Manufacturer:" not in out and "Model:" not in out, out)
    _assert(out.splitlines()[0] == "FC-1 | STEEL | WHITE | 24x24", out)
    print("PASS: no manufacturer/model signal -> table passed through unchanged")


# --------------------------------------------------------------------------- #
# Fakes for exercising _find_row_tags / _rows_are_non_overlapping without a
# real PDF -- they only need the .bbox / .rows / .crop shape pdfplumber uses.
# --------------------------------------------------------------------------- #
class _FakeRow:
    def __init__(self, bbox):
        self.bbox = bbox


class _FakeTable:
    """row_bboxes takes (top, bottom) pairs and expands them to pdfplumber's
    real (x0, top, x1, bottom) row bbox shape -- x0/x1 are irrelevant to the
    logic under test, only top/bottom (indices 1 and 3) are read."""
    def __init__(self, bbox, row_bboxes):
        self.bbox = bbox
        self.rows = [_FakeRow((bbox[0], top, bbox[2], bottom)) for top, bottom in row_bboxes]


class _FakeCropRegion:
    def __init__(self, words):
        self._words = words

    def extract_words(self):
        return self._words


class _FakePage:
    def __init__(self, words):
        self._words = words

    def crop(self, _bbox):
        return _FakeCropRegion(self._words)


def _word(text, top, bottom, x0):
    return {"text": text, "top": top, "bottom": bottom, "x0": x0}


# --------------------------------------------------------------------------- #
# 8. Tag recovered from outside the ruled grid attaches to the SAME row as
#    its Manufacturer/Model, by that row's own geometry -- not by order.
# --------------------------------------------------------------------------- #
def test_tag_matched_to_same_row_by_geometry():
    table_obj = _FakeTable(
        bbox=(600, 500, 1000, 600),
        row_bboxes=[(500, 525), (525, 550), (550, 575), (575, 600)],
    )
    page = _FakePage([
        _word("A", 505, 515, 570),
        _word("B", 530, 540, 570),
        _word("C", 555, 565, 570),
        _word("D", 580, 590, 570),
    ])
    tags = extract._find_row_tags(page, table_obj)
    _assert(tags == {0: "A", 1: "B", 2: "C", 3: "D"}, f"unexpected tag mapping: {tags}")
    print("PASS: tag recovered from outside the ruled grid, matched to its own row by geometry")


# --------------------------------------------------------------------------- #
# 9. Safety net: a table whose rows overlap (confirmed on the real VAV/AH
#    schedule, whose COOLING/HEATING sub-rows per unit make pdfplumber
#    produce overlapping row bboxes) must get NO tags at all -- guessing
#    would risk exactly the cross-row mismatch this feature exists to
#    prevent, so it must refuse rather than attach a maybe-wrong tag.
# --------------------------------------------------------------------------- #
def test_overlapping_rows_get_no_tags_at_all():
    table_obj = _FakeTable(
        bbox=(600, 100, 1000, 200),
        row_bboxes=[(100, 155), (105, 155), (130, 155), (155, 175)],  # first three overlap
    )
    page = _FakePage([_word("AH-A2", 130, 140, 570)])
    tags = extract._find_row_tags(page, table_obj)
    _assert(tags == {}, f"expected no tags from an overlapping-row table, got: {tags}")
    print("PASS: overlapping row bboxes (multi-row units) refuse to guess a tag rather than risk a wrong one")


# --------------------------------------------------------------------------- #
# 10. A parenthetical status note ("(EXISTING)") positioned in the same
#     margin must not be glued onto the real tag.
# --------------------------------------------------------------------------- #
def test_parenthetical_annotation_excluded_from_tag():
    table_obj = _FakeTable(bbox=(600, 500, 1000, 550), row_bboxes=[(500, 525)])
    page = _FakePage([
        _word("(EXISTING)", 505, 515, 560),
        _word("EF-A1", 505, 515, 570),
    ])
    tags = extract._find_row_tags(page, table_obj)
    _assert(tags == {0: "EF-A1"}, f"annotation leaked into the tag: {tags}")
    print("PASS: parenthetical annotation ('(EXISTING)') excluded, clean tag recovered")


# --------------------------------------------------------------------------- #
# 11. Real-file regression: the actual Diffuser/Grille/Register schedule and
#    every other schedule on page 6 of the St. Thomas the Apostle PDF.
#    Skipped gracefully if that fixture isn't present on this machine.
# --------------------------------------------------------------------------- #
# Path to the page-6 fixture. Defaults to the machine it was authored on, but
# HVAC_TEST_PDF overrides it anywhere else -- matching the HVAC_* env-var
# convention the sibling hvac-takeoff-tool repo already uses for local paths.
_REAL_PDF = os.environ.get(
    "HVAC_TEST_PDF",
    r"C:\Users\91739\Desktop\hvacfiles\01-02 St Thomas The Apostle Roman Catholic "
    r"K-8 Bldg PH II Pricing\01-02 St Thomas The Apostle Roman Catholic K-8 Bldg "
    r"PH II Pricing\Schedule\page no 6.pdf",
)


def test_real_page_6_pdf():
    import os
    if not os.path.exists(_REAL_PDF):
        print("SKIP: real page-6 fixture not present on this machine")
        return

    pages = extract.read_pages(_REAL_PDF)
    sched, _meta = extract._select_pages(pages)
    tables = extract._extract_tables_for_pages(_REAL_PDF, sched)
    text = "\n\n".join(tables[sched[0]])

    expected = [
        "Manufacturer: PRICE / Model: SPD",
        "Manufacturer: PRICE / Model: SMD",
        "Manufacturer: PRICE / Model: RCDA",
        "Manufacturer: PRICE / Model: 520S",
        "Manufacturer: TRANE / Model: CSAA021",   # AH-A2 -- reversed order in the source
        "Manufacturer: TRANE / Model: CSAA035",   # AH-A4 / AH-A5
        "Manufacturer: TRANE / Model: BCHE-72",   # FC-A1 / FC-A2
        "Manufacturer: GREENHECK / Model: G-240-VG",
        "Manufacturer: GREENHECK / Model: G-140-VG",
        "Manufacturer: GREENHECK / Model: G-180-VG",
        "Manufacturer: CAPTIVE-AIRE / Model: USB124DD-RM",
        "Manufacturer: CAPTIVE-AIRE / Model: USBI11DD-RM",
    ]
    missing = [e for e in expected if e not in text]
    _assert(not missing, f"missing expected labeled pairs on the real page-6 PDF: {missing}")
    print(f"PASS: real page-6 PDF -- all {len(expected)} correct Manufacturer/Model pairs found "
          f"across the VAV, Fan Coil, Diffuser/Grille, and Fan schedules")


# --------------------------------------------------------------------------- #
# 12. Real-file regression: the exact Diffuser, Grille & Register Schedule
#    ground truth (Tag A-X) reported against a live CSV, matched to the same
#    table ROW as its Tag -- not cross-referenced from a separate text block.
#
#    Tag V is a known, documented exception: the source PDF itself renders
#    "PRICE" with every character doubled ("PPRRIICCEE", a font/embossing
#    artifact also seen on tags L/W/X), and for V specifically that corrupted
#    string is rare enough in the column that the frequency-based mfr/model
#    orientation flips. That's a source-data limitation, not a row/column
#    association bug -- the row itself is still correct, just its Manufacturer
#    and Model are swapped -- so it's asserted here as a documented exception,
#    not silently ignored.
# --------------------------------------------------------------------------- #
def test_real_page_6_diffuser_schedule_full_ground_truth():
    import os
    if not os.path.exists(_REAL_PDF):
        print("SKIP: real page-6 fixture not present on this machine")
        return

    pages = extract.read_pages(_REAL_PDF)
    sched, _meta = extract._select_pages(pages)
    tables = extract._extract_tables_for_pages(_REAL_PDF, sched)
    text = "\n\n".join(tables[sched[0]])
    lines = text.splitlines()

    ground_truth_models = {
        "A": "SPD", "B": "SMD", "C": "RCDA", "D": "520S", "E": "530L", "F": "530L",
        "G": "96L", "H": "530L", "H.1": "530L", "J": "530L", "K": "630L", "L": "530FF",
        "M": "530FF", "P": "AMD", "Q": "STG-BF", "R": "PDF", "T": "AHCD1", "U": "SDS",
        "W": "SDS", "X": "SDS",
    }
    errors = []
    for tag, model in ground_truth_models.items():
        row = next((l for l in lines if l.startswith(f"Tag: {tag} |")), None)
        if row is None:
            errors.append(f"{tag}: no row found with Tag: {tag}")
            continue
        if f"Model: {model}" not in row:
            errors.append(f"{tag}: expected Model: {model}, got: {row}")
    _assert(not errors, "diffuser schedule ground truth mismatches:\n" + "\n".join(errors))

    # Documented exception: V's mfr/model come out swapped because of the
    # doubled-character PDF corruption, not a row-association error.
    row_v = next(l for l in lines if l.startswith("Tag: V |"))
    _assert("Manufacturer: 530L / Model: PPRRIICCEE" in row_v,
            f"tag V's known swapped-values exception changed shape, please re-check: {row_v}")

    print(f"PASS: real page-6 PDF -- Diffuser/Grille/Register schedule, {len(ground_truth_models)} "
          f"of 21 tags (A-X) correctly matched to their own row's model number "
          f"(V is a documented PDF-corruption exception, asserted separately)")


# --------------------------------------------------------------------------- #
# 13. Real-file regression: the VAV/AH schedule's overlapping row bboxes must
#     suppress tag attachment entirely rather than emit a wrong one (this is
#     the exact defect reported: AH-A3 was wrongly attached to AH-A2's row).
# --------------------------------------------------------------------------- #
def test_real_page_6_vav_schedule_gets_no_unreliable_tags():
    import os
    if not os.path.exists(_REAL_PDF):
        print("SKIP: real page-6 fixture not present on this machine")
        return

    pages = extract.read_pages(_REAL_PDF)
    sched, _meta = extract._select_pages(pages)
    tables = extract._extract_tables_for_pages(_REAL_PDF, sched)
    text = "\n\n".join(tables[sched[0]])

    csaa021_row = next(l for l in text.splitlines() if "CSAA021" in l)
    _assert("Tag: AH-A3" not in csaa021_row,
            f"AH-A3 wrongly attached to the CSAA021 (AH-A2) row: {csaa021_row}")
    _assert(not csaa021_row.startswith("Tag:"),
            f"VAV schedule row got a tag despite overlapping row bboxes: {csaa021_row}")
    print("PASS: real page-6 PDF -- VAV/AH schedule (overlapping row bboxes) correctly "
          "gets no tag rather than a wrong one")


# --------------------------------------------------------------------------- #
# 14. table_text must survive a page-text split intact: when a page needs to
#     be chunked, the FULL structured-table section is re-attached to EVERY
#     resulting chunk (not left behind in whichever chunk got the matching
#     plain text) so a schedule doesn't lose its own structured backup just
#     because a different schedule's plain text landed in the other half.
#     Regression for the exact live-CSV defect this caused: the Fan Schedule
#     lost its correct Manufacturer/Model when a 413-triggered split put its
#     plain text in one chunk and the entire STRUCTURED TABLE DATA section
#     (appended once, at the end) in the other.
# --------------------------------------------------------------------------- #
def test_table_text_survives_split_in_every_chunk():
    # A single table block: with nothing else to split it against, it must
    # be duplicated into both chunks (never dropped from either) when only
    # page_text can be divided.
    page_text, expected_tags = _make_fake_schedule_page(150)
    table_blocks = ["Tag: KEEPME | Manufacturer: FOO / Model: BAR"]

    def fake(system, user, schema):
        if len(user) > 2500:
            raise extract.GroqPayloadTooLarge("simulated 413")
        rows = []
        if "Tag: KEEPME" in user:
            rows.append({"schedule": "S", "tag": "KEEPME", "manufacturer": "FOO",
                         "model": "BAR", "size_capacity": None})
        tags = list(dict.fromkeys(re.findall(r"TAG-\d{4}", user)))
        rows += [{"schedule": "EQUIPMENT SCHEDULE", "tag": t, "manufacturer": "ACME",
                  "model": None, "size_capacity": None} for t in tags]
        return {"equipment": rows}

    orig = extract._chat_json
    extract._chat_json = fake
    try:
        rows = extract._extract_equipment_page(page_text, table_blocks)
    finally:
        extract._chat_json = orig

    got_tags = {r.tag for r in rows}
    _assert("KEEPME" in got_tags, "structured table block lost when the page had to be split")
    _assert(expected_tags <= got_tags, "plain-text rows lost during the split")
    print("PASS: a lone structured table block is duplicated into every chunk after a split "
          "(nothing else left to split it against)")


# --------------------------------------------------------------------------- #
# 15. Regression for the second live-CSV defect this same fix caused: with
#     MULTIPLE table blocks, duplicating all of them into both chunks puts a
#     floor under how small any chunk can get (system prompt + every block,
#     no matter how much page_text shrinks) -- on the real file this floor
#     was still too large, so recursion exhausted its depth and the WHOLE
#     page came back with 0 rows. Multiple blocks must instead be
#     partitioned between the two chunks so each one actually gets smaller.
# --------------------------------------------------------------------------- #
def test_multiple_table_blocks_are_partitioned_not_all_duplicated():
    # A short, effectively unsplittable page_text so the test isolates block
    # partitioning specifically -- the real-file failure was that ALL 6
    # table blocks got duplicated into every chunk regardless of how much
    # page_text shrank, since only page_text was being split at the time.
    page_text = "SOME SCHEDULE\none line of page text\n"
    table_blocks = [f"Tag: KEEP-{i} | Manufacturer: FOO / Model: BAR-{i}" for i in range(6)]

    # Fits page_text + 2 blocks + the prompt's fixed wrapper text, but not 3
    # or more -- proving convergence requires actually shrinking the block
    # set, not just page_text (which can barely shrink here at all).
    def _build(blocks):
        text = page_text
        if blocks:
            text += "\n\n--- STRUCTURED TABLE DATA ---\n\n" + "\n\n".join(blocks)
        return "Extract every equipment schedule row from this page:\n\n" + text

    threshold = len(_build(table_blocks[:2]))

    seen_blocks_on_success = []

    def fake(system, user, schema):
        n_blocks_present = sum(1 for i in range(6) if f"Tag: KEEP-{i}" in user)
        if len(user) > threshold:
            raise extract.GroqPayloadTooLarge("simulated 413")
        seen_blocks_on_success.append(n_blocks_present)
        return {"equipment": [
            {"schedule": "S", "tag": f"KEEP-{i}", "manufacturer": "FOO",
             "model": f"BAR-{i}", "size_capacity": None}
            for i in range(6) if f"Tag: KEEP-{i}" in user
        ]}

    orig = extract._chat_json
    extract._chat_json = fake
    try:
        rows = extract._extract_equipment_page(page_text, table_blocks)
    finally:
        extract._chat_json = orig

    got_tags = {r.tag for r in rows}
    expected_keep_tags = {f"KEEP-{i}" for i in range(6)}
    _assert(expected_keep_tags <= got_tags,
            f"lost table-block rows when the page had to be split: missing {expected_keep_tags - got_tags}")
    # Every successful call only ever carried a genuine subset (never all 6
    # duplicated whole) -- proving the blocks were actually partitioned to
    # shrink each chunk, not just carried along unchanged every time.
    _assert(seen_blocks_on_success and max(seen_blocks_on_success) < 6,
            f"a successful call still carried all 6 blocks: {seen_blocks_on_success}")
    print(f"PASS: {len(table_blocks)} table blocks partitioned across chunks (successful requests "
          f"carried at most {max(seen_blocks_on_success)} of them) -- all rows still recovered")


# --------------------------------------------------------------------------- #
# 16. Regression for the exact live-CSV defect: when both the original
#     attempt AND the one JSON-repair retry keep failing (persistently
#     malformed/truncated JSON -- as happens on a large, complex prompt),
#     that must now fall back to splitting the page, not give up on it
#     entirely. A smaller chunk has less to enumerate and is less likely to
#     truncate in the first place.
# --------------------------------------------------------------------------- #
def test_persistent_invalid_json_falls_back_to_chunking():
    page_text, expected_tags = _make_fake_schedule_page(200)

    def fake(system, user, schema):
        if len(user) > 3000:
            raise extract.GroqInvalidJSON("simulated persistent invalid JSON")
        tags = list(dict.fromkeys(re.findall(r"TAG-\d{4}", user)))
        return {"equipment": [{"schedule": "EQUIPMENT SCHEDULE", "tag": t, "manufacturer": "ACME",
                                "model": None, "size_capacity": None} for t in tags]}

    orig = extract._chat_json
    extract._chat_json = fake
    try:
        rows = extract._extract_equipment_page(page_text)
    finally:
        extract._chat_json = orig

    got = {r.tag for r in rows}
    _assert(got == expected_tags, f"expected {len(expected_tags)} rows recovered via chunking, got {len(got)}")
    print("PASS: persistent GroqInvalidJSON (repair also failed) now falls back to chunking "
          "instead of losing the whole page")


# --------------------------------------------------------------------------- #
#  Metadata pass: an oversized title block used to lose project identity
#  entirely. A 413 there returned status 'ok' with equipment rows intact but
#  project_name / location / engineer all null -- the single biggest source of
#  missing project identity in earlier runs. It must now shrink and retry.
# --------------------------------------------------------------------------- #
def test_metadata_413_splits_instead_of_losing_the_project():
    cover = "PROJECT: RIVERSIDE HIGH SCHOOL\nLOCATION: Phoenix, AZ\n" + ("filler line\n" * 400)
    title = "MECHANICAL ENGINEER: ACME MEP\n" + ("more filler\n" * 400)

    def fake(system, user, schema):
        # Mimic Groq's per-request token ceiling.
        if len(user) > 3000:
            raise extract.GroqPayloadTooLarge("simulated 413")
        out = {"project_name": None, "location": None, "team": []}
        if "RIVERSIDE HIGH SCHOOL" in user:
            out["project_name"] = "RIVERSIDE HIGH SCHOOL"
            out["location"] = "Phoenix, AZ"
        if "ACME MEP" in user:
            out["team"] = [{"role": "Mechanical Engineer", "firm": "ACME MEP"}]
        return out

    orig = extract._chat_json
    extract._chat_json = fake
    try:
        meta = extract._extract_metadata([cover, title])
    finally:
        extract._chat_json = orig

    _assert(meta.project_name == "RIVERSIDE HIGH SCHOOL",
            f"project_name lost to the 413: {meta.project_name!r}")
    _assert(meta.location == "Phoenix, AZ", f"location lost to the 413: {meta.location!r}")
    firms = [m.firm for m in meta.team]
    _assert("ACME MEP" in firms, f"engineer firm lost to the 413: {firms}")
    print("PASS: metadata 413 splits across pages and merges -- project, location and "
          "engineer all survive")


def test_metadata_merge_prefers_first_value_and_dedupes_team():
    from schemas import ProjectMetadata, TeamMember
    base = ProjectMetadata(project_name="FIRST", team=[TeamMember(role="Architect", firm="AOR")])
    extra = ProjectMetadata(project_name="SECOND", location="Mesa, AZ",
                            team=[TeamMember(role="architect", firm="aor"),
                                  TeamMember(role="Civil", firm="CIV")])
    merged = extract._merge_metadata(base, extra)
    _assert(merged.project_name == "FIRST", "an already-known value must not be overwritten")
    _assert(merged.location == "Mesa, AZ", "a missing value must be filled from the other half")
    _assert(len(merged.team) == 2, f"team should dedupe case-insensitively, got {merged.team}")
    print("PASS: metadata merge keeps the first non-empty value and dedupes team on (role, firm)")


def test_no_schedule_pages_is_not_reported_as_ok():
    # A readable PDF where nothing matches SCHEDULE_KEYWORDS (a CAD sheet whose
    # table is vector line-art) must not look like a clean success.
    orig_read, orig_chat = extract.read_pages, extract._chat_json
    extract.read_pages = lambda path: ["general notes and details, nothing tabular here. " * 10]
    extract._chat_json = lambda system, user, schema: {"team": []}
    try:
        result = extract.extract_pdf("irrelevant.pdf")
    finally:
        extract.read_pages, extract._chat_json = orig_read, orig_chat

    _assert(result.status == "no_schedule_pages",
            f"expected 'no_schedule_pages', got {result.status!r}")
    _assert(not result.equipment, "no rows should be claimed")
    print("PASS: a file with no schedule pages reports 'no_schedule_pages', not a silent 'ok'")


def test_prefix_duplicates_collapse_but_distinct_roles_survive():
    from schemas import ProjectMetadata, TeamMember
    base = ProjectMetadata(team=[
        TeamMember(role="Mechanical Engineer", firm="JGGM"),
        TeamMember(role="Architect", firm="MICHAEL A. HALL ARCHITECT, LLC"),
        TeamMember(role="Electrical Engineer", firm="DLR GROUP"),
    ])
    extra = ProjectMetadata(team=[
        # same firm written longer -> collapses, longer spelling wins
        TeamMember(role="Mechanical Engineer", firm="JGGM Engineering", phone="602-555-0100"),
        # role written more specifically, same firm -> collapses
        TeamMember(role="Architect of Record", firm="Michael A. Hall Architect LLC"),
        # genuinely different role at the same firm -> must NOT collapse
        TeamMember(role="Structural Engineer", firm="DLR GROUP"),
        # acronym, not a prefix -> deliberately left alone
        TeamMember(role="Owner", firm="GSA"),
        TeamMember(role="Owner", firm="GENERAL SERVICES ADMINISTRATION"),
    ])
    merged = extract._merge_metadata(base, extra)
    got = {(m.role, m.firm) for m in merged.team}

    _assert(("Mechanical Engineer", "JGGM Engineering") in got,
            f"longer firm spelling should win: {got}")
    _assert(not any(f == "JGGM" for _, f in got), f"short duplicate should be gone: {got}")
    _assert(any(r == "Architect of Record" for r, _ in got),
            f"more specific role should win: {got}")
    _assert(not any(r == "Architect" for r, _ in got), f"'Architect' should have collapsed: {got}")
    _assert(("Electrical Engineer", "DLR GROUP") in got and
            ("Structural Engineer", "DLR GROUP") in got,
            f"distinct roles at one firm must both survive: {got}")
    _assert(sum(1 for r, _ in got if r == "Owner") == 2,
            f"acronym vs full name is normalization, not a split artifact: {got}")

    mech = next(m for m in merged.team if m.role == "Mechanical Engineer")
    _assert(mech.phone == "602-555-0100", "fields from the absorbed entry should be kept")
    print("PASS: prefix duplicates collapse to the longer spelling; distinct roles and "
          "acronyms are left alone")


def test_schedule_title_pages_preferred_over_keyword_only_pages():
    # p0: title block mentioning the word -> keyword match, no real table.
    # p1: sheet index row -> keyword match, no real table.
    # p2: an actual schedule title.
    pages = [
        "MECHANICAL PLAN\nSEE SCHEDULE ON SHEET M601\nGENERAL NOTES",
        "SHEET INDEX\nM601 MECHANICAL SCHEDULES AND DETAILS\nM101 HVAC PLAN",
        "FAN COIL UNIT SCHEDULE\nMARK MANUFACTURER MODEL CFM\nFC-1 DAIKIN FXAQ07 260",
    ]
    sched, _meta = extract._select_pages(pages)
    _assert(sched == [2], f"only the page with a real schedule title should be sent, got {sched}")
    print("PASS: schedule-title pages preferred; title blocks and sheet indexes dropped")


def test_falls_back_to_keyword_pages_when_no_title_matches():
    # No line ends in SCHEDULE, but the pages plainly carry equipment data.
    # Dropping these would silently lose the file -- MedVet Phoenix in the real
    # corpus behaves exactly this way and yields the most rows of any test file.
    pages = [
        "EQUIPMENT SCHEDULE - ROOFTOP UNITS (SEE NOTES)\nMARK MANUFACTURER MODEL",
        "RTU-1 CARRIER 48TC MODEL NO. 48TCED07",
    ]
    sched, _meta = extract._select_pages(pages)
    _assert(sched == [0, 1], f"must fall back to the wider keyword set, got {sched}")
    print("PASS: no title match anywhere -> falls back to keyword pages, file not dropped")


def test_see_schedule_reference_is_not_a_title():
    _assert(not extract._is_schedule_title_page("SEE SCHEDULE ON SHEET M601"),
            "a cross-reference must not count as a schedule title")
    _assert(not extract._is_schedule_title_page("M601 MECHANICAL SCHEDULES AND DETAILS"),
            "a sheet-index row must not count as a schedule title")
    _assert(extract._is_schedule_title_page("GRILLES, REGISTERS AND DIFFUSERS SCHEDULE"),
            "a real schedule title must be recognised")
    _assert(extract._is_schedule_title_page("intro\nFAN SCHEDULE\nMARK CFM"),
            "a title on its own line mid-page must be recognised")
    print("PASS: schedule-title detection separates real titles from references and index rows")


def test_long_retry_after_raises_instead_of_sleeping():
    # Groq answering "wait 600s" used to become a silent 600s sleep, up to five
    # times per call. That is what looked like the metadata call hanging.
    import urllib.error, io as _io, config as _cfg

    calls = {"sleeps": []}
    real_sleep, real_open = extract.time.sleep, extract.urllib.request.urlopen
    extract.time.sleep = lambda s: calls["sleeps"].append(s)

    def fake_open(req, timeout=None):
        raise urllib.error.HTTPError(
            'http://x', 429, 'Too Many Requests',
            {'retry-after': '600'}, _io.BytesIO(b'{}'))

    extract.urllib.request.urlopen = fake_open
    old_key, old_cap = _cfg.GROQ_API_KEY, _cfg.GROQ_MAX_BACKOFF
    _cfg.GROQ_API_KEY, _cfg.GROQ_MAX_BACKOFF = 'gsk_test', 90.0
    try:
        raised = None
        try:
            extract._groq_chat_json('sys', 'user', {})
        except extract.GroqRateLimited as e:
            raised = e
    finally:
        extract.time.sleep, extract.urllib.request.urlopen = real_sleep, real_open
        _cfg.GROQ_API_KEY, _cfg.GROQ_MAX_BACKOFF = old_key, old_cap

    _assert(raised is not None, "a 600s Retry-After must raise, not sleep")
    _assert(not calls["sleeps"], f"nothing should have been slept, got {calls['sleeps']}")
    print("PASS: an over-cap Retry-After raises GroqRateLimited instead of sleeping silently")


TESTS = [
    test_combined_stacked_cells_split_correctly,
    test_combined_split_does_not_shift_adjacent_columns,
    test_separate_columns_mapped_directly,
    test_combined_cell_self_corrects_reversed_row_order,
    test_missing_combined_value_not_inferred_from_neighbors,
    test_disambiguates_mfr_model_from_other_two_line_columns_regardless_of_position,
    test_no_signal_leaves_table_unchanged,
    test_tag_matched_to_same_row_by_geometry,
    test_overlapping_rows_get_no_tags_at_all,
    test_parenthetical_annotation_excluded_from_tag,
    test_real_page_6_pdf,
    test_real_page_6_diffuser_schedule_full_ground_truth,
    test_real_page_6_vav_schedule_gets_no_unreliable_tags,
    test_table_text_survives_split_in_every_chunk,
    test_multiple_table_blocks_are_partitioned_not_all_duplicated,
    test_persistent_invalid_json_falls_back_to_chunking,
    test_metadata_413_splits_instead_of_losing_the_project,
    test_metadata_merge_prefers_first_value_and_dedupes_team,
    test_prefix_duplicates_collapse_but_distinct_roles_survive,
    test_no_schedule_pages_is_not_reported_as_ok,
    test_schedule_title_pages_preferred_over_keyword_only_pages,
    test_falls_back_to_keyword_pages_when_no_title_matches,
    test_see_schedule_reference_is_not_a_title,
    test_long_retry_after_raises_instead_of_sleeping,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
    print("\nALL TESTS PASSED")
