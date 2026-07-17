"""Deterministic cleanup shared by every production parse profile."""

from __future__ import annotations

import re

TOC_HEADING_RE = re.compile(r"^(?:#{1,6}\s*)?(?:目\s*录|table\s+of\s+contents|contents)\s*$", re.IGNORECASE)
TOC_LEADER_RE = re.compile(r"[ \t]*(?:(?:\.[ \t]*){4,}|(?:…[ \t]*){2,}|(?:·[ \t]*){4,})[ \t]*(\d+)")
TOC_LEADER_ONLY_RE = re.compile(r"[ \t]*(?:(?:\.[ \t]*){4,}|(?:…[ \t]*){2,}|(?:·[ \t]*){4,})[ \t]*")
TOC_NEXT_ENTRY_RE = re.compile(r"(第\d+页)[ \t]+(?=(?:\d+(?:\.\d+)*\.|附录\s+\d+)\s)")


def compact_markdown_blank_lines(markdown: str) -> str:
    """Keep one blank line outside fenced code blocks."""
    result: list[str] = []
    fence_char: str | None = None
    fence_width = 0

    for line in markdown.splitlines():
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence_char is None:
                fence_char = marker[0]
                fence_width = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_width:
                fence_char = None
                fence_width = 0
            result.append(line.rstrip())
            continue

        if fence_char is None and not line.strip():
            if result and result[-1] != "":
                result.append("")
            continue
        result.append(line.rstrip())

    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


def normalize_table_of_contents(markdown: str) -> str:
    """Normalize leaders only inside an identified table of contents."""
    result: list[str] = []
    in_toc = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if TOC_HEADING_RE.fullmatch(stripped):
            in_toc = True
            result.append(line)
            continue
        is_next_heading = bool(re.match(r"^#{1,6}\s+", stripped))
        if in_toc and (stripped == "---" or stripped.startswith("<!-- page ") or is_next_heading):
            in_toc = False
            result.append(line)
            continue
        if not in_toc:
            result.append(line)
            continue

        normalized = TOC_LEADER_RE.sub(lambda match: f" ... 第{match.group(1)}页", line)
        normalized = TOC_LEADER_ONLY_RE.sub(" ... ", normalized).rstrip()
        normalized = TOC_NEXT_ENTRY_RE.sub(r"\1\n", normalized)
        result.extend(normalized.splitlines() or [""])
    return "\n".join(result)


def clean_extraction_markers(markdown: str) -> str:
    """Remove known parser control labels without rewriting document content."""
    markdown = re.sub(r"(?m)^.*\[Table\\_Title\].*\n?", "", markdown)
    markdown = markdown.replace("[Table\\_Summary] ", "")
    markdown = re.sub(r"(?m)^##\s+\[Table\\_R.*$", "## 相关报告", markdown)
    return markdown.replace(" ", "- ")


def table_of_contents_span(markdown: str) -> tuple[int, int] | None:
    lines = markdown.splitlines(keepends=True)
    offset = 0
    start: int | None = None
    end: int | None = None
    for line in lines:
        stripped = line.strip()
        if start is None:
            if TOC_HEADING_RE.fullmatch(stripped):
                start = offset
        elif (
            stripped == "---"
            or stripped.startswith("<!-- page ")
            or (re.match(r"^#{1,6}\s+", stripped) and not TOC_HEADING_RE.fullmatch(stripped))
        ):
            end = offset
            break
        offset += len(line)
    if start is None:
        return None
    return start, len(markdown) if end is None else end


def prefer_table_of_contents(markdown: str, reference_markdown: str) -> str:
    """Replace a damaged TOC with a better extraction of the same source page."""
    target_span = table_of_contents_span(markdown)
    reference_span = table_of_contents_span(reference_markdown)
    if target_span is None or reference_span is None:
        return markdown
    reference_toc = normalize_table_of_contents(reference_markdown[reference_span[0] : reference_span[1]].strip())
    if not reference_toc:
        return markdown
    target_start, target_end = target_span
    return f"{markdown[:target_start]}{reference_toc}\n\n{markdown[target_end:]}"


def normalize_markdown(markdown: str) -> str:
    """Apply the safe cleanup chain used by all production profiles."""
    return compact_markdown_blank_lines(normalize_table_of_contents(clean_extraction_markers(markdown)))
