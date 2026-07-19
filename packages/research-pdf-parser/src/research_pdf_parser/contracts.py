"""Stable, business-neutral result contract for PDF parsing consumers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

RequestedProfile = Literal["auto", "native-fast", "formula-cpu", "formula-best"]
ActualProfile = Literal["native-fast", "formula-cpu", "formula-best", "scanned-deferred"]

RESULT_SCHEMA = "research-pdf-parser.result.v1"
PROBE_SCHEMA = "research-pdf-parser.probe.v1"


@dataclass(frozen=True)
class PageProbe:
    """Cheap routing facts for one selected PDF page."""

    page: int
    native_chars: int
    largest_image_ratio: float
    scanned: bool
    private_use_chars: int
    replacement_chars: int
    symbol_font_chars: int
    formula_candidates: int
    vision_formula_candidates: int
    table_formula_candidates: int
    complexity_score: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentProbe:
    """Content-derived routing decision; source labels never override it."""

    source_sha256: str
    page_count: int
    selected_pages: tuple[int, ...]
    recommended_profile: ActualProfile
    route_reasons: tuple[str, ...]
    pages: tuple[PageProbe, ...]
    formula_candidates: int
    vision_formula_candidates: int
    table_formula_candidates: int
    scanned_pages: tuple[int, ...]
    probe_seconds: float
    schema: str = PROBE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentBlock:
    """Sparse index over canonical Markdown rather than a second document copy."""

    id: str
    kind: Literal["heading", "paragraph", "table", "formula", "image", "notice"]
    page: int | None
    markdown_start: int
    markdown_end: int
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    asset_sha256: str | None = None


@dataclass(frozen=True)
class QualitySummary:
    status: Literal["ok", "partial", "deferred"]
    markdown_chars: int
    block_count: int
    formula_candidates: int
    accepted_latex: int
    image_fallbacks: int
    scanned_pages: tuple[int, ...]


@dataclass(frozen=True)
class ParseResult:
    """Public result returned to personal, AlphaSeeker, and Shadow adapters."""

    source_path: str
    source_sha256: str
    markdown_path: str
    requested_profile: RequestedProfile
    actual_profile: ActualProfile
    route_reasons: tuple[str, ...]
    probe: DocumentProbe
    blocks: tuple[DocumentBlock, ...]
    quality: QualitySummary
    timings: dict[str, float]
    warnings: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    schema: str = RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        """Materialize the optional sparse contract; Markdown remains canonical."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)(?:\s+[^>]*)?-->")
HEADING_RE = re.compile(r"^#{1,6}\s+")
IMAGE_RE = re.compile(r"^\s*!\[[^]]*]\(([^)]+)\)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_id(source_sha256: str, page: int | None, kind: str, start: int, text: str) -> str:
    value = f"{source_sha256}:{page}:{kind}:{start}:{text}".encode()
    return "block_" + hashlib.sha256(value).hexdigest()[:20]


def _classify(lines: list[str]) -> str:
    first = lines[0].lstrip()
    if HEADING_RE.match(first):
        return "heading"
    if first.startswith("|") and sum(line.lstrip().startswith("|") for line in lines) >= 2:
        return "table"
    if first.startswith(("$$", "\\[")):
        return "formula"
    if IMAGE_RE.match(first):
        return "image"
    if first.startswith(">") and ("skipped" in first.lower() or "扫描" in first):
        return "notice"
    return "paragraph"


def project_markdown_blocks(
    markdown: str,
    source_sha256: str,
    markdown_path: Path | None = None,
) -> tuple[DocumentBlock, ...]:
    """Build stable typed spans without duplicating the full Markdown payload."""
    lines = markdown.splitlines(keepends=True)
    blocks: list[DocumentBlock] = []
    page: int | None = None
    offset = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        marker = PAGE_MARKER_RE.search(line)
        if marker:
            page = int(marker.group(1))
            offset += len(line)
            index += 1
            continue
        if not line.strip() or line.strip() == "---" or line.lstrip().startswith("<!--"):
            offset += len(line)
            index += 1
            continue

        start = offset
        group = [line]
        stripped = line.lstrip()
        index += 1
        offset += len(line)

        if stripped.startswith("|"):
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                group.append(lines[index])
                offset += len(lines[index])
                index += 1
        elif stripped.startswith(("$$", "\\[")):
            closing = "$$" if stripped.startswith("$$") else "\\]"
            formula_line = stripped.strip()
            is_single_line = (
                len(formula_line) > len(closing) * 2 and formula_line.endswith(closing)
            )
            if not is_single_line:
                while index < len(lines):
                    group.append(lines[index])
                    offset += len(lines[index])
                    done = lines[index].strip() == closing
                    index += 1
                    if done:
                        break
        elif not HEADING_RE.match(stripped) and not IMAGE_RE.match(stripped):
            while index < len(lines):
                candidate = lines[index]
                candidate_stripped = candidate.lstrip()
                if (
                    not candidate.strip()
                    or candidate.strip() == "---"
                    or candidate_stripped.startswith(("<!--", "|", "$$", "\\["))
                    or HEADING_RE.match(candidate_stripped)
                    or IMAGE_RE.match(candidate_stripped)
                ):
                    break
                group.append(candidate)
                offset += len(candidate)
                index += 1

        text = "".join(group).strip()
        if not text:
            continue
        kind = _classify(group)
        asset_sha256 = None
        if kind == "image" and markdown_path is not None:
            match = IMAGE_RE.match(text)
            if match and not match.group(1).startswith(("http://", "https://", "data:")):
                asset_path = (markdown_path.parent / unquote(match.group(1))).resolve()
                if asset_path.is_file():
                    asset_sha256 = sha256_file(asset_path)
        blocks.append(
            DocumentBlock(
                id=_block_id(source_sha256, page, kind, start, text),
                kind=kind,  # type: ignore[arg-type]
                page=page,
                markdown_start=start,
                markdown_end=start + len("".join(group)),
                text=text,
                asset_sha256=asset_sha256,
            )
        )
    return tuple(blocks)
