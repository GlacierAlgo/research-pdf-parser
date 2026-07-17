"""Explicit high-accuracy MinerU profile; never selected by auto routing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf
from liteparse import LiteParse

from .assets import markdown_assets_dir, rebundle_markdown_images
from .markdown_cleanup import normalize_markdown, prefer_table_of_contents
from .mineru_remote import RemoteMineruConfig, convert_pdf_on_dgx


@dataclass(frozen=True)
class HighAccuracyResult:
    markdown_path: Path
    assets_dir: Path
    elapsed_seconds: float
    backend: str
    host: str


def _vector_toc(pdf_path: Path) -> str:
    with pymupdf.open(pdf_path) as document:
        if document.page_count < 2:
            return ""
    parser = LiteParse(
        output_format="markdown",
        target_pages="2",
        quiet=True,
        ocr_enabled=False,
        image_mode="off",
    )
    parsed = parser.parse(pdf_path)
    if not parsed.pages:
        return ""
    page = parsed.pages[0]
    return page.markdown or page.text or ""


def parse_best_pdf(
    pdf_path: Path,
    output_path: Path,
    *,
    config: RemoteMineruConfig | None = None,
) -> HighAccuracyResult:
    """Run MinerU remotely and apply deterministic local Markdown cleanup."""
    config = config or RemoteMineruConfig()
    started = time.perf_counter()
    with TemporaryDirectory(prefix="research-pdf-parser-mineru-") as directory:
        raw_markdown_path = convert_pdf_on_dgx(pdf_path, Path(directory), config)
        markdown = raw_markdown_path.read_text(encoding="utf-8")
        try:
            reference_toc = _vector_toc(pdf_path)
            if reference_toc:
                markdown = prefer_table_of_contents(markdown, reference_toc)
        except Exception:
            # TOC repair is secondary; the remote primary result remains usable.
            pass
        markdown = normalize_markdown(markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            rebundle_markdown_images(markdown, raw_markdown_path, output_path) + "\n",
            encoding="utf-8",
        )
    return HighAccuracyResult(
        markdown_path=output_path,
        assets_dir=markdown_assets_dir(output_path),
        elapsed_seconds=time.perf_counter() - started,
        backend=config.backend,
        host=config.host,
    )
