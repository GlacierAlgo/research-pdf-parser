"""Small PDF primitives shared by probe, native, and formula profiles."""

from __future__ import annotations

import re

import pymupdf

PRIVATE_USE_REPLACEMENTS = {
    "\uf02b": "+",
    "\uf02d": "−",
    "\uf03c": "<",
    "\uf03d": "=",
    "\uf03e": ">",
    "\uf061": "α",
    "\uf062": "β",
    "\uf065": "ε",
    "\uf06d": "μ",
    "\uf073": "σ",
    "\uf074": "τ",
    "\uf078": "ξ",
    "\uf0a2": "′",
    "\uf0a3": "≤",
    "\uf0ae": "→",
    "\uf0b1": "±",
    "\uf0b3": "≥",
    "\uf0d7": "⋅",
    "\uf0e5": "∑",
    "\uf0ec": "⎧",
    "\uf0ed": "⎨",
    "\uf0ee": "⎩",
    "\uf0ef": "⎪",
}


def normalize_private_use(text: str) -> str:
    """Map known Symbol-font glyphs and drop unknown private-use codepoints."""
    for source, replacement in PRIVATE_USE_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return "".join(char for char in text if not 0xE000 <= ord(char) <= 0xF8FF)


def resolve_page_numbers(document: pymupdf.Document, pages: str | None) -> list[int]:
    """Resolve the public 1-indexed page selector against an open document."""
    if not pages:
        return list(range(1, document.page_count + 1))
    selected: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"页码范围无效：{part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected or min(selected) < 1 or max(selected) > document.page_count:
        raise ValueError(f"页码必须位于 1-{document.page_count}")
    return sorted(selected)


def scanned_page_reason(page: pymupdf.Page) -> str | None:
    """Return an observable reason when a page is image-only enough to defer."""
    native_chars = len(re.sub(r"\s+", "", page.get_text("text")))
    largest_image_ratio = 0.0
    page_area = max(page.rect.width * page.rect.height, 1.0)
    for image in page.get_images(full=True):
        try:
            rects = page.get_image_rects(image[0])
        except Exception:
            continue
        for rect in rects:
            largest_image_ratio = max(largest_image_ratio, rect.width * rect.height / page_area)
    if native_chars < 20 and largest_image_ratio >= 0.55:
        return f"native_chars={native_chars}, largest_image={largest_image_ratio:.0%}"
    return None
