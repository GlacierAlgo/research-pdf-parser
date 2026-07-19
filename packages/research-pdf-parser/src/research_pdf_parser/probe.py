"""Content probe that routes native-vector PDFs before expensive inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from liteparse import LiteParse

from .contracts import DocumentProbe, PageProbe, sha256_file
from .pdf_utils import resolve_page_numbers, scanned_page_reason


@dataclass(frozen=True)
class ProbeSession:
    """Probe facts plus the reusable LiteParse pass for the native fast path."""

    probe: DocumentProbe
    parsed: Any


def _font_signals(page: pymupdf.Page) -> tuple[int, int, int, int]:
    text = page.get_text("text")
    native_chars = len("".join(text.split()))
    private_use = sum(0xE000 <= ord(char) <= 0xF8FF for char in text)
    replacement = text.count("\ufffd")
    symbol_font_chars = 0
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = str(span.get("font", "")).lower()
                if "symbol" in font or "math" in font or "stix" in font:
                    symbol_font_chars += len(str(span.get("text", "")))
    return native_chars, private_use, replacement, symbol_font_chars


def _page_formula_counts(parsed_page: Any) -> tuple[int, int, int, tuple[str, ...]]:
    candidates = list(getattr(parsed_page, "formula_candidates", ()) or ())
    vision = [candidate for candidate in candidates if str(candidate.route) == "vision"]
    table = [
        candidate
        for candidate in candidates
        if "ruled_table_cell" in tuple(str(reason) for reason in candidate.reasons)
    ]
    reasons = sorted({str(reason) for candidate in candidates for reason in candidate.reasons})
    return len(candidates), len(vision), len(table), tuple(reasons)


def probe_pdf(
    pdf_path: Path,
    *,
    pages: str | None = None,
    image_mode: str = "off",
) -> ProbeSession:
    """Run one OCR-free LiteParse pass and return a deterministic route."""
    pdf_path = Path(pdf_path)
    started = time.perf_counter()
    with pymupdf.open(pdf_path) as document:
        selected_pages = resolve_page_numbers(document, pages)
        page_count = document.page_count
        page_signals = {
            page_number: (
                _font_signals(document[page_number - 1]),
                scanned_page_reason(document[page_number - 1]),
            )
            for page_number in selected_pages
        }

    parser = LiteParse(
        output_format="markdown",
        target_pages=",".join(map(str, selected_pages)),
        quiet=True,
        ocr_enabled=False,
        emit_word_boxes=True,
        image_mode=image_mode,
    )
    parsed = parser.parse(pdf_path)
    parsed_pages = {page.page_num: page for page in parsed.pages}

    page_probes: list[PageProbe] = []
    for page_number in selected_pages:
        (native_chars, private_use, replacement, symbol_chars), scanned_reason = page_signals[page_number]
        parsed_page = parsed_pages[page_number]
        formula_count, vision_count, table_count, formula_reasons = _page_formula_counts(parsed_page)
        reasons = list(formula_reasons)
        if scanned_reason:
            reasons.insert(0, f"scanned:{scanned_reason}")
        if private_use:
            reasons.append("private-use-glyphs")
        if replacement:
            reasons.append("replacement-glyphs")
        complexity = min(
            100,
            (45 if scanned_reason else 0)
            + min(30, vision_count * 6)
            + min(15, table_count * 5)
            + min(10, private_use + replacement),
        )
        largest_image_ratio = 0.0
        if scanned_reason and "largest_image=" in scanned_reason:
            try:
                largest_image_ratio = float(scanned_reason.rsplit("=", 1)[1].rstrip("%")) / 100.0
            except ValueError:
                largest_image_ratio = 0.0
        page_probes.append(
            PageProbe(
                page=page_number,
                native_chars=native_chars,
                largest_image_ratio=round(largest_image_ratio, 4),
                scanned=scanned_reason is not None,
                private_use_chars=private_use,
                replacement_chars=replacement,
                symbol_font_chars=symbol_chars,
                formula_candidates=formula_count,
                vision_formula_candidates=vision_count,
                table_formula_candidates=table_count,
                complexity_score=complexity,
                reasons=tuple(reasons),
            )
        )

    scanned_pages = tuple(page.page for page in page_probes if page.scanned)
    formula_count = sum(page.formula_candidates for page in page_probes)
    vision_count = sum(page.vision_formula_candidates for page in page_probes)
    table_count = sum(page.table_formula_candidates for page in page_probes)
    if len(scanned_pages) == len(selected_pages):
        recommended = "scanned-deferred"
        route_reasons = ("all-selected-pages-scanned", "native-vector-only")
    elif vision_count or table_count:
        recommended = "formula-cpu"
        route_reasons = tuple(
            reason
            for condition, reason in (
                (vision_count > 0, f"vision-formula-candidates={vision_count}"),
                (table_count > 0, f"table-formula-candidates={table_count}"),
                (bool(scanned_pages), f"mixed-scanned-pages={len(scanned_pages)}"),
            )
            if condition
        )
    else:
        recommended = "native-fast"
        route_reasons = tuple(
            ["native-text-usable", "no-complex-formula-candidates"]
            + ([f"mixed-scanned-pages={len(scanned_pages)}"] if scanned_pages else [])
        )

    probe = DocumentProbe(
        source_sha256=sha256_file(pdf_path),
        page_count=page_count,
        selected_pages=tuple(selected_pages),
        recommended_profile=recommended,  # type: ignore[arg-type]
        route_reasons=route_reasons,
        pages=tuple(page_probes),
        formula_candidates=formula_count,
        vision_formula_candidates=vision_count,
        table_formula_candidates=table_count,
        scanned_pages=scanned_pages,
        probe_seconds=time.perf_counter() - started,
    )
    return ProbeSession(probe=probe, parsed=parsed)
