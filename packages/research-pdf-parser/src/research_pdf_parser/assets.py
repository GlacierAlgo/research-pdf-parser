"""Portable Markdown asset packaging helpers."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^\s)]+)([^)]*\))")


def markdown_assets_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_assets"


def materialize_liteparse_images(
    parsed: Any,
    markdown: str,
    output_path: Path,
    assets_dir: Path,
) -> str:
    """Write LiteParse image bytes and make all Markdown references portable."""
    if not parsed or not parsed.images:
        return markdown
    images_dir = assets_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    references_by_page: dict[int, list[str]] = {}
    for embedded in parsed.images:
        filename = f"image_{embedded.id}.{embedded.format}"
        path = images_dir / filename
        path.write_bytes(embedded.bytes)
        relative = path.relative_to(output_path.parent).as_posix()
        reference = quote(relative, safe="/-._~")
        references_by_page.setdefault(int(embedded.page), []).append(reference)
        markdown = markdown.replace(f"]({filename})", f"]({reference})")

    sections = markdown.split("\n\n---\n\n")
    for index, section in enumerate(sections):
        match = re.search(r"<!-- page (\d+) -->", section)
        if not match:
            continue
        page_number = int(match.group(1))
        missing = [
            f"![]({reference})"
            for reference in references_by_page.get(page_number, [])
            if reference not in section
        ]
        if missing:
            sections[index] = section.rstrip() + "\n\n" + "\n\n".join(missing)
    return "\n\n---\n\n".join(sections)


def rebundle_markdown_images(markdown: str, source_path: Path, output_path: Path) -> str:
    """Copy parser-generated local images beside the canonical Markdown."""
    target_images = markdown_assets_dir(output_path) / "images"

    def replace(match: re.Match[str]) -> str:
        raw_url = match.group(2)
        if raw_url.startswith(("http://", "https://", "data:")):
            return match.group(0)
        decoded = unquote(raw_url)
        source = (source_path.parent / decoded).resolve()
        if not source.is_file():
            return match.group(0)
        target_images.mkdir(parents=True, exist_ok=True)
        target = target_images / source.name
        if source != target.resolve():
            shutil.copy2(source, target)
        relative = target.relative_to(output_path.parent).as_posix()
        return f"{match.group(1)}{quote(relative, safe='/-._~')}{match.group(3)}"

    return MARKDOWN_IMAGE_RE.sub(replace, markdown)
