"""Single-pass LiteParse path for native-vector PDFs without complex formulas."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from liteparse import LiteParse

from .assets import materialize_liteparse_images
from .markdown_cleanup import normalize_markdown
from .pdf_utils import normalize_private_use, resolve_page_numbers, scanned_page_reason


@dataclass(frozen=True)
class NativeParseResult:
    markdown_path: Path
    parse_seconds: float
    page_count: int
    skipped_pages: list[int] = field(default_factory=list)


def parse_native_pdf(
    pdf_path: Path,
    output_path: Path,
    *,
    pages: str | None = None,
    image_mode: str = "off",
    parsed: Any | None = None,
) -> NativeParseResult:
    """Parse native-vector pages once and write canonical Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output_path.parent / f"{output_path.stem}_assets"

    with pymupdf.open(pdf_path) as document:
        selected_pages = resolve_page_numbers(document, pages)
        skipped = {
            page_number: reason
            for page_number in selected_pages
            if (reason := scanned_page_reason(document[page_number - 1])) is not None
        }

    active_pages = [page_number for page_number in selected_pages if page_number not in skipped]
    start = time.perf_counter()
    if active_pages and parsed is None:
        parser = LiteParse(
            output_format="markdown",
            target_pages=",".join(map(str, active_pages)),
            quiet=True,
            ocr_enabled=False,
            image_mode=image_mode,
        )
        parsed = parser.parse(pdf_path)
    parse_seconds = time.perf_counter() - start

    parsed_pages = {page.page_num: page for page in parsed.pages} if parsed else {}
    sections: list[str] = []
    for page_number in selected_pages:
        if page_number in skipped:
            sections.append(
                f"<!-- page {page_number} skipped: scanned PDF ({skipped[page_number]}) -->\n\n"
                f"> Page {page_number} looks scanned; the native-vector profile skipped it."
            )
            continue
        page = parsed_pages[page_number]
        body = normalize_private_use(page.markdown or page.text).strip()
        sections.append(f"<!-- page {page_number} -->\n\n{body}".strip())

    markdown = normalize_markdown("\n\n---\n\n".join(sections).strip())
    markdown = materialize_liteparse_images(parsed, markdown, output_path, assets_dir)
    output_path.write_text(markdown + "\n", encoding="utf-8")
    return NativeParseResult(
        markdown_path=output_path,
        parse_seconds=parse_seconds,
        page_count=len(active_pages),
        skipped_pages=sorted(skipped),
    )
