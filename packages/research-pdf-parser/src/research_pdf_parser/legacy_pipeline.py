from __future__ import annotations

import json
import os
import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote, unquote

import click
import pymupdf
from liteparse import LiteParse

from .formula_dispatch import FormulaDispatchError, FormulaProcessorRegistry
from .mineru_remote import RemoteMineruConfig, RemoteMineruError, convert_pdf_on_dgx

MATH_FONTS = ("symbol", "math", "cambria", "stix", "times new roman,italic")
MATH_CHARS = set("=+-*/<>^_{}[]()|→←±×÷≤≥≠≈∑∫√∞σμλαβγδεθρπωΣΠ")
FORMULA_WORD_RE = re.compile(
    r"\b(?:alpha|corr|rank|sum|mean|std|sma|delay|close|open|high|low|volume|vwap|ret|tsrank|tsmax|tsmin|abs|log|max|min)\b",
    re.IGNORECASE,
)
NOISE_TEXT_RE = re.compile(r"电话|邮箱|证书|mailto:|@\w|^\d{4}\.\d{1,2}\.\d{1,2}$")
INLINE_PROSE_RE = re.compile(
    r"\b(?:where|when|while|using|used|denotes?|means?|is|are|as|for|with|from|we|the|this|that|measure[sd]?|calculate[sd]?)\b|其中|我们|使用|作为|表示|衡量|定义为",
    re.IGNORECASE,
)
FORMULA_PLACEHOLDER_RE = re.compile(r"\{\{formula:([^}]+)\}\}")
MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^\s)]+)([^)]*\))")
TOC_HEADING_RE = re.compile(r"^(?:#{1,6}\s*)?(?:目\s*录|table\s+of\s+contents|contents)\s*$", re.IGNORECASE)
TOC_LEADER_RE = re.compile(r"[ \t]*(?:(?:\.[ \t]*){4,}|(?:…[ \t]*){2,}|(?:·[ \t]*){4,})[ \t]*(\d+)")
TOC_LEADER_ONLY_RE = re.compile(r"[ \t]*(?:(?:\.[ \t]*){4,}|(?:…[ \t]*){2,}|(?:·[ \t]*){4,})[ \t]*")
TOC_NEXT_ENTRY_RE = re.compile(r"(第\d+页)[ \t]+(?=(?:\d+(?:\.\d+)*\.|附录\s+\d+)\s)")
FORMULA_CROP_STRATEGY = "tight_white_scaled_v1"
PAGE_SNAPSHOT_STRATEGY = "page_snapshot"
CROP_OCR_ENGINES = ("pix2tex", "paddleocr-crop")
PAGE_OCR_ENGINES = ("paddleocr-page",)
FORMULA_OCR_ENGINES = CROP_OCR_ENGINES + PAGE_OCR_ENGINES
FORMULA_RENDER_SCALE = 3.0
FORMULA_INK_THRESHOLD = 12
FORMULA_TARGET_CONTENT_HEIGHT = 48
DEFAULT_FORMULA_PROCESSOR = "pix2tex"
ADOBE_SYMBOL_PRIVATE_USE_REPLACEMENTS = {
    0xF02B: "+",
    0xF02D: "−",
    0xF03C: "<",
    0xF03D: "=",
    0xF03E: ">",
    0xF061: "α",
    0xF062: "β",
    0xF065: "ε",
    0xF06D: "μ",
    0xF073: "σ",
    0xF074: "τ",
    0xF078: "ξ",
    0xF0A2: "′",
    0xF0A3: "≤",
    0xF0AE: "→",
    0xF0B1: "±",
    0xF0B3: "≥",
    0xF0D7: "⋅",
    0xF0E5: "∑",
    0xF0EC: "⎧",
    0xF0ED: "⎨",
    0xF0EE: "⎩",
    0xF0EF: "⎪",
}
WINGDINGS_PRIVATE_USE_REPLACEMENTS = {0xF06C: "•"}

FormulaRecord = dict[str, Any]
FormulaRecognizer = Callable[[Path], str]
PageFormulaResult = dict[str, Any]


class FormulaFillError(ValueError):
    pass


@dataclass(frozen=True)
class TextChar:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str


@dataclass
class FormulaCandidate:
    id: str
    page: int
    bbox: tuple[float, float, float, float]
    preview_text: str
    confidence: float
    kind: str = "display"
    status: str = "needs_formula_ocr"
    route_reasons: list[str] = field(default_factory=list)
    crop_path: Path | None = None
    context_before: str = ""
    context_after: str = ""


@dataclass
class SmartParseResult:
    markdown: str
    formulas: list[FormulaCandidate] = field(default_factory=list)
    images: list[MarkdownImage] = field(default_factory=list)
    manifest_path: Path | None = None
    artifacts_dir: Path | None = None


@dataclass(frozen=True)
class MarkdownImage:
    filename: str
    content: bytes


@dataclass(frozen=True)
class FormulaCropStrategy:
    name: str
    render_scale: float
    target_content_height: int | None
    trim_to_ink: bool = True
    page_padding_x: float = 1.0
    page_padding_y: float = 3.0
    ink_padding: int = 1
    margin_ratio_x: float = 0.12
    margin_ratio_y: float = 0.08


DEFAULT_FORMULA_CROP_STRATEGY = FormulaCropStrategy(
    name=FORMULA_CROP_STRATEGY,
    render_scale=FORMULA_RENDER_SCALE,
    target_content_height=FORMULA_TARGET_CONTENT_HEIGHT,
)
CROP_STRATEGIES: dict[str, FormulaCropStrategy] = {
    DEFAULT_FORMULA_CROP_STRATEGY.name: DEFAULT_FORMULA_CROP_STRATEGY,
    "raw_3x": FormulaCropStrategy(
        name="raw_3x",
        render_scale=3.0,
        target_content_height=None,
        trim_to_ink=False,
        page_padding_x=2.0,
        page_padding_y=2.0,
    ),
    "tight_h40": FormulaCropStrategy(name="tight_h40", render_scale=3.0, target_content_height=40),
    "tight_h48": FormulaCropStrategy(name="tight_h48", render_scale=3.0, target_content_height=48),
    "tight_h56": FormulaCropStrategy(name="tight_h56", render_scale=3.0, target_content_height=56),
}


def cjk_ratio(text: str) -> float:
    visible_count = sum(1 for char in text if not char.isspace())
    if visible_count == 0:
        return 0.0
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return cjk_count / visible_count


def is_private_use(char: str) -> bool:
    return 0xE000 <= ord(char) <= 0xF8FF


def detect_private_use_replacements(pdf_path: Path) -> dict[int, str]:
    """Derive safe PUA replacements from the actual embedded font names."""
    fonts_by_codepoint: dict[int, set[str]] = {}
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font = str(span.get("font", "")).lower()
                        for char in span.get("chars", []):
                            text = char.get("c", "")
                            if len(text) == 1 and is_private_use(text):
                                fonts_by_codepoint.setdefault(ord(text), set()).add(font)

    replacements: dict[int, str] = {}
    for codepoint, fonts in fonts_by_codepoint.items():
        if fonts and all("symbol" in font for font in fonts):
            replacement = ADOBE_SYMBOL_PRIVATE_USE_REPLACEMENTS.get(codepoint)
        elif fonts and all("wingdings" in font for font in fonts):
            replacement = WINGDINGS_PRIVATE_USE_REPLACEMENTS.get(codepoint)
        else:
            replacement = None
        if replacement is not None:
            replacements[codepoint] = replacement
    return replacements


def normalize_private_use_text(text: str, replacements: dict[int, str]) -> str:
    return text.translate(replacements)


def group_preview_text(group: list[TextChar]) -> str:
    chars = sorted(group, key=lambda item: (item.x0, item.y0))
    text = "".join(char.text for char in chars)
    return re.sub(r"\s{2,}", " ", text).strip()


def formula_route_score(group: list[TextChar]) -> tuple[float, list[str]]:
    visible = [char for char in group if char.text.strip()]
    if not visible:
        return 0.0, []

    text = group_preview_text(visible)
    reasons: list[str] = []
    score = 0.0
    fonts = {char.font.lower() for char in visible}
    max_size = max(char.size for char in visible)
    min_size = min(char.size for char in visible)

    if any(any(token in font for token in MATH_FONTS) for font in fonts):
        score += 0.28
        reasons.append("math_font")
    if any(char.text in MATH_CHARS or is_private_use(char.text) for char in visible):
        score += 0.28
        reasons.append("math_glyph")
    if max_size > 0 and min_size / max_size < 0.8:
        score += 0.18
        reasons.append("mixed_baseline_or_script")
    if FORMULA_WORD_RE.search(text):
        score += 0.18
        reasons.append("formula_word")
    if cjk_ratio(text) > 0.35 and not FORMULA_WORD_RE.search(text):
        score -= 0.22
        reasons.append("cjk_dense")
    if NOISE_TEXT_RE.search(text):
        score -= 0.45
        reasons.append("contact_or_date_noise")
    if len(text) < 3:
        score -= 0.2
        reasons.append("too_short")

    return max(0.0, min(score, 1.0)), reasons


def should_externalize_formula(group: list[TextChar]) -> bool:
    score, _ = formula_route_score(group)
    return score >= 0.45 and is_standalone_formula_line(group)


def is_standalone_formula_line(group: list[TextChar]) -> bool:
    text = group_preview_text(group)
    if not text:
        return False
    if cjk_ratio(text) > 0.15:
        return False
    if INLINE_PROSE_RE.search(text):
        return False

    visible = [char for char in group if char.text.strip()]
    if not visible:
        return False

    x0, _, x1, _ = group_bbox(group, padding=0.0)
    width = x1 - x0
    formula_marks = sum(
        1 for char in visible if char.text in MATH_CHARS or is_private_use(char.text) or char.text in "\\{}_^"
    )
    mark_ratio = formula_marks / len(visible)
    centered_or_indented = x0 >= 120 and width <= 360
    compact_math = width <= 260 and (mark_ratio >= 0.08 or FORMULA_WORD_RE.search(text))
    return centered_or_indented or compact_math


def is_formula_seed(group: list[TextChar]) -> bool:
    score, reasons = formula_route_score(group)
    if "cjk_dense" in reasons:
        return False
    if not is_standalone_formula_line(group):
        return False
    return score >= 0.45 or (score >= 0.32 and "math_glyph" in reasons)


def formula_confidence(group: list[TextChar]) -> float:
    score, _ = formula_route_score(group)
    return round(score, 2)


def formula_route_reasons(group: list[TextChar]) -> list[str]:
    _, reasons = formula_route_score(group)
    return reasons


def group_bbox(group: list[TextChar], padding: float = 3.0) -> tuple[float, float, float, float]:
    visible = [char for char in group if char.text.strip()]
    source = visible or group
    return (
        max(min(char.x0 for char in source) - padding, 0.0),
        max(min(char.y0 for char in source) - padding, 0.0),
        max(char.x1 for char in source) + padding,
        max(char.y1 for char in source) + padding,
    )


def same_visual_line(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    a_center = (a[1] + a[3]) / 2
    b_center = (b[1] + b[3]) / 2
    a_height = max(a[3] - a[1], 1.0)
    b_height = max(b[3] - b[1], 1.0)
    center_threshold = max(8.0, min(max(a_height, b_height) * 0.75, 11.0))
    if abs(a_center - b_center) <= center_threshold:
        return True

    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap > min(a_height, b_height) * 0.25


def merge_visual_line_groups(groups: list[list[TextChar]]) -> list[list[TextChar]]:
    line_groups: list[list[TextChar]] = []
    line_boxes: list[tuple[float, float, float, float]] = []
    ordered = sorted(
        groups,
        key=lambda group: (
            (group_bbox(group, padding=0.0)[1] + group_bbox(group, padding=0.0)[3]) / 2,
            group_bbox(group, padding=0.0)[0],
        ),
    )

    for group in ordered:
        bbox = group_bbox(group, padding=0.0)
        target_index: int | None = None
        for index in range(len(line_boxes) - 1, max(len(line_boxes) - 4, -1), -1):
            if same_visual_line(line_boxes[index], bbox):
                target_index = index
                break

        if target_index is None:
            line_groups.append(list(group))
            line_boxes.append(bbox)
            continue

        line_groups[target_index].extend(group)
        lx0, ly0, lx1, ly1 = line_boxes[target_index]
        gx0, gy0, gx1, gy1 = bbox
        line_boxes[target_index] = (min(lx0, gx0), min(ly0, gy0), max(lx1, gx1), max(ly1, gy1))

    return sorted(line_groups, key=lambda group: (group_bbox(group, padding=0.0)[1], group_bbox(group, padding=0.0)[0]))


def merge_formula_groups(groups: list[list[TextChar]]) -> list[list[TextChar]]:
    groups = merge_visual_line_groups(groups)
    seed_regions: list[tuple[float, float, float, float]] = []
    for group in groups:
        if not is_formula_seed(group):
            continue
        x0, y0, x1, y1 = group_bbox(group, padding=0.0)
        if seed_regions and y0 - seed_regions[-1][3] <= 14:
            px0, py0, px1, py1 = seed_regions[-1]
            seed_regions[-1] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
        else:
            seed_regions.append((x0, y0, x1, y1))

    merged: list[list[TextChar]] = []
    for region in seed_regions:
        rx0, ry0, rx1, ry1 = region
        expanded = (max(rx0 - 45, 0), max(ry0 - 10, 0), rx1 + 90, ry1 + 10)
        region_groups: list[TextChar] = []
        for group in groups:
            gx0, gy0, gx1, gy1 = group_bbox(group, padding=0.0)
            intersects = gx1 >= expanded[0] and gx0 <= expanded[2] and gy1 >= expanded[1] and gy0 <= expanded[3]
            if intersects and is_standalone_formula_line(group):
                region_groups.extend(group)
        if region_groups and "cjk_dense" not in formula_route_reasons(region_groups):
            merged.append(region_groups)
    return merged


def extract_line_groups(page: pymupdf.Page) -> list[list[TextChar]]:
    raw = page.get_text("rawdict")
    groups: list[list[TextChar]] = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            group: list[TextChar] = []
            for span in line["spans"]:
                font = span.get("font", "")
                size = float(span.get("size", 0.0))
                for char in span["chars"]:
                    x0, y0, x1, y1 = char["bbox"]
                    if x0 <= 40 or not (40 < y0 < 800):
                        continue
                    group.append(TextChar(char["c"], x0, y0, x1, y1, size, font))
            if group:
                groups.append(group)
    return groups


def detect_formula_candidates(page: pymupdf.Page, page_number: int) -> list[FormulaCandidate]:
    formulas: list[FormulaCandidate] = []
    for index, group in enumerate(merge_formula_groups(extract_line_groups(page)), start=1):
        formulas.append(
            FormulaCandidate(
                id=f"formula_p{page_number:04d}_{index:03d}",
                page=page_number,
                bbox=group_bbox(group),
                preview_text=group_preview_text(group),
                confidence=formula_confidence(group),
                route_reasons=formula_route_reasons(group),
            )
        )
    return formulas


def boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[2] >= b[0] and a[0] <= b[2] and a[3] >= b[1] and a[1] <= b[3]


def render_text_line(group: list[TextChar]) -> str:
    text = group_preview_text(group)
    heading_match = re.match(r"^(\d+(?:\.\d+)*\.)\s*", text)
    if heading_match:
        level = min(heading_match.group(1).rstrip(".").count(".") + 1, 6)
        return f"{'#' * level} {text}"
    return text


def render_page_with_formula_placeholders(
    page: pymupdf.Page, page_number: int, formulas: list[FormulaCandidate]
) -> str:
    groups = sorted(
        extract_line_groups(page),
        key=lambda group: (group_bbox(group, padding=0.0)[1], group_bbox(group, padding=0.0)[0]),
    )
    formulas_by_y = sorted(formulas, key=lambda formula: (formula.bbox[1], formula.bbox[0]))
    emitted: set[str] = set()
    lines: list[str] = []

    for group in groups:
        bbox = group_bbox(group, padding=0.0)
        for formula in formulas_by_y:
            if formula.id not in emitted and formula.bbox[1] <= bbox[1] and boxes_intersect(bbox, formula.bbox):
                lines.append(f"{{{{formula:{formula.id}}}}}")
                emitted.add(formula.id)

        if any(boxes_intersect(bbox, formula.bbox) for formula in formulas_by_y):
            continue

        for formula in formulas_by_y:
            if formula.id not in emitted and formula.bbox[3] < bbox[1]:
                lines.append(f"{{{{formula:{formula.id}}}}}")
                emitted.add(formula.id)

        text = render_text_line(group)
        if text and not text.startswith("数量化专题报告") and not text.startswith("请务必阅读正文之后"):
            lines.append(text)

    for formula in formulas_by_y:
        if formula.id not in emitted:
            lines.append(f"{{{{formula:{formula.id}}}}}")

    compacted: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("{{formula:"):
            if compacted and compacted[-1] != "":
                compacted.append("")
            compacted.append(stripped)
            compacted.append("")
        else:
            compacted.append(stripped)

    markdown = "\n".join(compacted).strip()
    return f"<!-- page {page_number} -->\n\n{markdown}" if markdown else ""


def liteparse_page_markdown(pdf_path: Path, target_pages: str | None) -> tuple[dict[int, str], list[MarkdownImage]]:
    parser = LiteParse(
        output_format="markdown",
        target_pages=target_pages,
        quiet=True,
        ocr_enabled=False,
        image_mode="embed",
    )
    result = parser.parse(pdf_path)
    replacements = detect_private_use_replacements(pdf_path)
    pages = {
        page.page_num: normalize_private_use_text(page.markdown or page.text, replacements) for page in result.pages
    }
    images = [
        MarkdownImage(filename=f"image_{image.id}.{image.format}", content=image.bytes) for image in result.images
    ]
    return pages, images


def markdown_assets_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_assets"


def materialize_markdown_images(markdown: str, output_path: Path, images: list[MarkdownImage]) -> str:
    if not images:
        return markdown

    images_dir = markdown_assets_dir(output_path) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    references: dict[str, str] = {}
    for image in images:
        image_path = images_dir / image.filename
        image_path.write_bytes(image.content)
        relative_path = image_path.relative_to(output_path.parent).as_posix()
        references[image.filename] = quote(relative_path, safe="/-._~")

    def replace(match: re.Match[str]) -> str:
        destination = unquote(match.group(2))
        replacement = references.get(Path(destination).name)
        if replacement is None:
            return match.group(0)
        return f"{match.group(1)}{replacement}{match.group(3)}"

    return MARKDOWN_IMAGE_RE.sub(replace, markdown)


def rebundle_markdown_images(markdown: str, source_path: Path, output_path: Path) -> str:
    """Copy local images referenced by an existing Markdown file into a new bundle."""
    images: list[MarkdownImage] = []
    for match in MARKDOWN_IMAGE_RE.finditer(markdown):
        destination = unquote(match.group(2))
        if destination.startswith(("http://", "https://", "data:")):
            continue
        source_image = source_path.parent / destination
        if not source_image.exists():
            legacy_image = source_path.parent / "images" / Path(destination).name
            if legacy_image.exists():
                source_image = legacy_image
            else:
                continue
        images.append(MarkdownImage(filename=source_image.name, content=source_image.read_bytes()))
    return materialize_markdown_images(markdown, output_path, images)


def default_artifacts_dir(pdf_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.parent / f"{output_path.stem}_formula_artifacts"
    return pdf_path.parent / f"{pdf_path.stem}_formula_artifacts"


def write_page_snapshots(pdf_path: Path, page_numbers: list[int], artifacts_dir: Path) -> dict[int, dict[str, Any]]:
    if not page_numbers:
        return {}

    snapshots_dir = artifacts_dir / "page_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    parser = LiteParse(quiet=True)
    snapshots: dict[int, dict[str, Any]] = {}
    for snapshot in parser.screenshot(pdf_path, page_numbers=sorted(set(page_numbers))):
        snapshot_path = snapshots_dir / f"page_p{snapshot.page_num:04d}.png"
        snapshot_path.write_bytes(snapshot.image_bytes)
        snapshots[snapshot.page_num] = {
            "path": snapshot_path,
            "width": snapshot.width,
            "height": snapshot.height,
        }
    return snapshots


def pixmap_to_image(pixmap: pymupdf.Pixmap) -> Any:
    from PIL import Image

    mode = "RGBA" if pixmap.alpha else "RGB"
    return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)


def normalize_formula_crop_image(image: Any, strategy: FormulaCropStrategy = DEFAULT_FORMULA_CROP_STRATEGY) -> Any:
    from PIL import Image, ImageChops

    if image.mode == "RGBA":
        base = Image.new("RGB", image.size, "white")
        base.paste(image, mask=image.getchannel("A"))
    else:
        base = image.convert("RGB")

    if strategy.trim_to_ink:
        white = Image.new("RGB", base.size, "white")
        diff = ImageChops.difference(base, white).convert("L")
        ink_mask = diff.point(lambda pixel: 255 if pixel > FORMULA_INK_THRESHOLD else 0)
        ink_bbox = ink_mask.getbbox()
        if ink_bbox is not None:
            x0, y0, x1, y1 = ink_bbox
            content = base.crop(
                (
                    max(x0 - strategy.ink_padding, 0),
                    max(y0 - strategy.ink_padding, 0),
                    min(x1 + strategy.ink_padding, base.width),
                    min(y1 + strategy.ink_padding, base.height),
                )
            )
        else:
            content = base
    else:
        content = base

    if strategy.target_content_height is not None and 0 < content.height < strategy.target_content_height:
        scale = strategy.target_content_height / content.height
        width = max(1, round(content.width * scale))
        content = content.resize((width, strategy.target_content_height), Image.Resampling.LANCZOS)

    if not strategy.trim_to_ink:
        return content

    x_margin = max(6, min(12, round(content.height * strategy.margin_ratio_x)))
    y_margin = max(6, min(10, round(content.height * strategy.margin_ratio_y)))
    normalized = Image.new("RGB", (content.width + x_margin * 2, content.height + y_margin * 2), "white")
    normalized.paste(content, (x_margin, y_margin))
    return normalized


def save_formula_crop(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float],
    crop_path: Path,
    strategy: FormulaCropStrategy = DEFAULT_FORMULA_CROP_STRATEGY,
) -> None:
    page_rect = page.rect
    x0, y0, x1, y1 = bbox
    clip = pymupdf.Rect(
        max(x0 - strategy.page_padding_x, page_rect.x0),
        max(y0 - strategy.page_padding_y, page_rect.y0),
        min(x1 + strategy.page_padding_x, page_rect.x1),
        min(y1 + strategy.page_padding_y, page_rect.y1),
    )
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(strategy.render_scale, strategy.render_scale),
        clip=clip,
        alpha=True,
    )
    image = normalize_formula_crop_image(pixmap_to_image(pixmap), strategy=strategy)
    image.save(crop_path)


def write_formula_artifacts(pdf_path: Path, formulas: list[FormulaCandidate], artifacts_dir: Path) -> Path | None:
    if not formulas:
        return None

    crops_dir = artifacts_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    page_snapshots = write_page_snapshots(pdf_path, [formula.page for formula in formulas], artifacts_dir)
    page_sizes: dict[int, tuple[float, float]] = {}
    with pymupdf.open(pdf_path) as doc:
        for formula in formulas:
            page = doc[formula.page - 1]
            page_sizes[formula.page] = (page.rect.width, page.rect.height)
            crop_path = crops_dir / f"{formula.id}.png"
            save_formula_crop(page, formula.bbox, crop_path)
            formula.crop_path = crop_path

    manifest_path = artifacts_dir / "formula_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for formula in formulas:
            crop_path = formula.crop_path
            crop_ref = str(crop_path)
            if crop_path is not None:
                try:
                    crop_ref = str(crop_path.relative_to(manifest_path.parent))
                except ValueError:
                    crop_ref = str(crop_path)
            record = {
                "id": formula.id,
                "page": formula.page,
                "bbox": [round(value, 3) for value in formula.bbox],
                "crop_path": crop_ref,
                "preview_text": formula.preview_text,
                "context_before": formula.context_before,
                "context_after": formula.context_after,
                "kind": formula.kind,
                "status": formula.status,
                "confidence": formula.confidence,
                "route_reasons": formula.route_reasons,
                "crop_strategy": FORMULA_CROP_STRATEGY,
                "processor": DEFAULT_FORMULA_PROCESSOR,
            }
            snapshot = page_snapshots.get(formula.page)
            if snapshot is not None:
                snapshot_path = snapshot["path"]
                try:
                    snapshot_ref = str(snapshot_path.relative_to(manifest_path.parent))
                except ValueError:
                    snapshot_ref = str(snapshot_path)
                record["page_snapshot_path"] = snapshot_ref
                record["page_snapshot_size"] = [snapshot["width"], snapshot["height"]]
                page_size = page_sizes.get(formula.page)
                if page_size is not None:
                    record["page_size"] = [round(page_size[0], 3), round(page_size[1], 3)]
                record["page_snapshot_source"] = "liteparse_screenshot"
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path


def default_formula_manifest_path(markdown_path: Path) -> Path:
    return markdown_path.parent / f"{markdown_path.stem}_formula_artifacts" / "formula_manifest.jsonl"


def read_formula_manifest(manifest_path: Path) -> list[FormulaRecord]:
    records: list[FormulaRecord] = []
    with manifest_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FormulaFillError(f"{manifest_path}:{line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise FormulaFillError(f"{manifest_path}:{line_number} must be a JSON object.")
            record.setdefault("processor", DEFAULT_FORMULA_PROCESSOR)
            records.append(record)
    return records


def write_formula_manifest_records(manifest_path: Path, records: list[FormulaRecord]) -> None:
    with manifest_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_formula_crop_path(manifest_path: Path, record: FormulaRecord) -> Path:
    crop_value = record.get("crop_path")
    formula_id = record.get("id", "<unknown>")
    if not isinstance(crop_value, str) or not crop_value.strip():
        raise FormulaFillError(f"{formula_id} has no crop_path in {manifest_path}.")

    crop_path = Path(crop_value)
    if not crop_path.is_absolute():
        crop_path = manifest_path.parent / crop_path
    return crop_path


def create_pix2tex_recognizer() -> FormulaRecognizer:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings:*")
        warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pix2tex\..*")
        try:
            from PIL import Image
            from pix2tex.cli import LatexOCR
        except ImportError as exc:
            raise FormulaFillError("pix2tex is not installed. Run `uv add pix2tex` first.") from exc

        model = LatexOCR()

    def recognize(image_path: Path) -> str:
        with Image.open(image_path) as image:
            return str(model(image)).strip()

    return recognize


def create_paddleocr_formula_pipeline(direct_formula_input: bool = False) -> Any:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import FormulaRecognitionPipeline
    except ImportError as exc:
        raise FormulaFillError(
            "PaddleOCR formula backend is not installed. Run with "
            "`uv run --with 'paddleocr[doc-parser]' --with paddlepaddle python main.py ...`."
        ) from exc

    if direct_formula_input:
        return FormulaRecognitionPipeline(
            device="cpu",
            engine="paddle_static",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
        )
    return FormulaRecognitionPipeline(device="cpu", engine="paddle_static")


def apply_formula_ocr(
    records: list[FormulaRecord],
    manifest_path: Path,
    recognizer: FormulaRecognizer,
    force: bool = False,
) -> int:
    recognized_count = 0
    for record in records:
        formula_id = record.get("id")
        if not isinstance(formula_id, str) or not formula_id:
            raise FormulaFillError(f"Formula manifest record is missing a string id in {manifest_path}.")

        existing_latex = record.get("latex")
        if not force and isinstance(existing_latex, str) and existing_latex.strip():
            continue

        crop_path = resolve_formula_crop_path(manifest_path, record)
        if not crop_path.exists():
            raise FormulaFillError(f"{formula_id} crop does not exist: {crop_path}")

        latex = recognizer(crop_path)
        if not latex:
            raise FormulaFillError(f"pix2tex returned empty LaTeX for {formula_id}.")

        record["latex"] = latex
        record["status"] = "formula_ocr_complete"
        record["ocr_engine"] = "pix2tex"
        record.pop("ocr_error", None)
        append_ocr_candidate(record, "pix2tex", latex)
        recognized_count += 1
    return recognized_count


def record_has_latex(record: FormulaRecord) -> bool:
    latex = record.get("latex")
    return isinstance(latex, str) and bool(latex.strip())


def append_ocr_candidate(
    record: FormulaRecord, engine: str, latex: str, metadata: dict[str, Any] | None = None
) -> None:
    candidates = record.setdefault("ocr_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
        record["ocr_candidates"] = candidates
    candidate: dict[str, Any] = {"engine": engine, "latex": latex}
    if metadata:
        candidate.update(metadata)
    candidates.append(candidate)


def formula_bbox_to_snapshot_bbox(record: FormulaRecord) -> tuple[float, float, float, float]:
    bbox = record.get("bbox")
    page_size = record.get("page_size")
    snapshot_size = record.get("page_snapshot_size")
    formula_id = record.get("id", "<unknown>")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not isinstance(page_size, list)
        or len(page_size) != 2
        or not isinstance(snapshot_size, list)
        or len(snapshot_size) != 2
    ):
        raise FormulaFillError(f"{formula_id} is missing bbox/page_size/page_snapshot_size for page snapshot matching.")

    page_width, page_height = float(page_size[0]), float(page_size[1])
    snapshot_width, snapshot_height = float(snapshot_size[0]), float(snapshot_size[1])
    if page_width <= 0 or page_height <= 0:
        raise FormulaFillError(f"{formula_id} has invalid page_size: {page_size}")

    sx = snapshot_width / page_width
    sy = snapshot_height / page_height
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    if intersection <= 0:
        return 0.0
    a_area = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    b_area = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = a_area + b_area - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def paddle_result_bbox(result: PageFormulaResult) -> tuple[float, float, float, float] | None:
    dt_polys = result.get("dt_polys")
    if not isinstance(dt_polys, list) or len(dt_polys) < 4:
        return None
    values = [float(value) for value in dt_polys[:4]]
    return (values[0], values[1], values[2], values[3])


def match_paddleocr_page_results(
    records: list[FormulaRecord],
    page_results: list[PageFormulaResult],
) -> dict[str, tuple[PageFormulaResult, float]]:
    available: list[tuple[PageFormulaResult, tuple[float, float, float, float]]] = []
    for result in page_results:
        latex = result.get("rec_formula")
        result_bbox = paddle_result_bbox(result)
        if isinstance(latex, str) and latex.strip() and result_bbox is not None:
            available.append((result, result_bbox))

    matches: dict[str, tuple[PageFormulaResult, float]] = {}
    used_result_ids: set[int] = set()
    for record in sorted(records, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        formula_id = record.get("id")
        if not isinstance(formula_id, str):
            continue
        expected_bbox = formula_bbox_to_snapshot_bbox(record)
        best: tuple[int, PageFormulaResult, float, float] | None = None
        for index, (result, result_bbox) in enumerate(available):
            if index in used_result_ids:
                continue
            overlap = bbox_iou(expected_bbox, result_bbox)
            distance = center_distance(expected_bbox, result_bbox)
            if overlap < 0.08 and distance > 80:
                continue
            if best is None or overlap > best[2] or (overlap == best[2] and distance < best[3]):
                best = (index, result, overlap, distance)
        if best is not None:
            used_result_ids.add(best[0])
            matches[formula_id] = (best[1], best[2])
    return matches


def resolve_page_snapshot_path(manifest_path: Path, record: FormulaRecord) -> Path:
    snapshot_value = record.get("page_snapshot_path")
    formula_id = record.get("id", "<unknown>")
    if not isinstance(snapshot_value, str) or not snapshot_value.strip():
        raise FormulaFillError(f"{formula_id} has no page_snapshot_path. Re-run `parse` to generate page snapshots.")
    snapshot_path = Path(snapshot_value)
    if not snapshot_path.is_absolute():
        snapshot_path = manifest_path.parent / snapshot_path
    return snapshot_path


def extract_paddle_formula_results(result: Any) -> list[PageFormulaResult]:
    formula_results = result.get("formula_res_list") if hasattr(result, "get") else None
    if not isinstance(formula_results, list):
        return []
    return [item for item in formula_results if isinstance(item, dict)]


def apply_paddleocr_page_ocr(records: list[FormulaRecord], manifest_path: Path, force: bool = False) -> int:
    pending = [
        record for record in records if (force or not record_has_latex(record)) and isinstance(record.get("page"), int)
    ]
    if not pending:
        return 0

    pipeline = create_paddleocr_formula_pipeline()
    records_by_page: dict[int, list[FormulaRecord]] = {}
    for record in pending:
        records_by_page.setdefault(int(record["page"]), []).append(record)

    recognized_count = 0
    for page, page_records in sorted(records_by_page.items()):
        snapshot_path = resolve_page_snapshot_path(manifest_path, page_records[0])
        if not snapshot_path.exists():
            raise FormulaFillError(f"Page snapshot for page {page} does not exist: {snapshot_path}")

        results = pipeline.predict(str(snapshot_path))
        page_formula_results: list[PageFormulaResult] = []
        for result in results:
            page_formula_results.extend(extract_paddle_formula_results(result))
        matches = match_paddleocr_page_results(page_records, page_formula_results)

        for record in page_records:
            formula_id = record.get("id")
            if not isinstance(formula_id, str):
                continue
            match = matches.get(formula_id)
            if match is None:
                record["ocr_error"] = "paddleocr_page_no_matching_formula_region"
                continue

            result, overlap = match
            latex = str(result["rec_formula"]).strip()
            record["latex"] = latex
            record["status"] = "formula_ocr_complete"
            record["ocr_engine"] = "paddleocr-page"
            record.pop("ocr_error", None)
            append_ocr_candidate(
                record,
                "paddleocr-page",
                latex,
                {
                    "formula_region_id": result.get("formula_region_id"),
                    "bbox_iou": round(overlap, 3),
                },
            )
            recognized_count += 1

    return recognized_count


def apply_paddleocr_crop_ocr(records: list[FormulaRecord], manifest_path: Path, force: bool = False) -> int:
    pending = [record for record in records if force or not record_has_latex(record)]
    if not pending:
        return 0

    pipeline = create_paddleocr_formula_pipeline(direct_formula_input=True)
    recognized_count = 0
    for record in pending:
        formula_id = record.get("id")
        if not isinstance(formula_id, str) or not formula_id:
            raise FormulaFillError(f"Formula manifest record is missing a string id in {manifest_path}.")

        crop_path = resolve_formula_crop_path(manifest_path, record)
        if not crop_path.exists():
            raise FormulaFillError(f"{formula_id} crop does not exist: {crop_path}")

        page_formula_results: list[PageFormulaResult] = []
        for result in pipeline.predict(
            str(crop_path),
            use_layout_detection=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        ):
            page_formula_results.extend(extract_paddle_formula_results(result))

        formula_results = [
            result
            for result in page_formula_results
            if isinstance(result.get("rec_formula"), str) and result["rec_formula"].strip()
        ]
        if not formula_results:
            record["ocr_error"] = "paddleocr_crop_no_formula_result"
            continue

        result = formula_results[0]
        latex = str(result["rec_formula"]).strip()
        record["latex"] = latex
        record["status"] = "formula_ocr_complete"
        record["ocr_engine"] = "paddleocr-crop"
        record.pop("ocr_error", None)
        append_ocr_candidate(
            record,
            "paddleocr-crop",
            latex,
            {"formula_region_id": result.get("formula_region_id")},
        )
        recognized_count += 1

    return recognized_count


FORMULA_PROCESSORS = FormulaProcessorRegistry()


def process_with_pix2tex(records: list[FormulaRecord], manifest_path: Path, force: bool = False) -> int:
    if not force and all(record_has_latex(record) for record in records):
        return 0
    recognizer = create_pix2tex_recognizer()
    return apply_formula_ocr(records, manifest_path, recognizer, force=force)


FORMULA_PROCESSORS.register("pix2tex", process_with_pix2tex)
FORMULA_PROCESSORS.register("paddleocr-crop", apply_paddleocr_crop_ocr)
FORMULA_PROCESSORS.register("paddleocr-page", apply_paddleocr_page_ocr)


def dispatch_formula_ocr(
    records: list[FormulaRecord],
    manifest_path: Path,
    *,
    processor_override: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    try:
        return FORMULA_PROCESSORS.dispatch(
            records,
            manifest_path,
            processor_override=processor_override,
            force=force,
        )
    except FormulaDispatchError as exc:
        raise FormulaFillError(str(exc)) from exc


def extract_first_paddle_formula_latex(result_items: list[PageFormulaResult]) -> str:
    for result in result_items:
        latex = result.get("rec_formula")
        if isinstance(latex, str) and latex.strip():
            return latex.strip()
    return ""


def create_crop_formula_recognizer(engine: str) -> FormulaRecognizer:
    if engine == "pix2tex":
        return create_pix2tex_recognizer()
    if engine == "paddleocr-crop":
        pipeline = create_paddleocr_formula_pipeline(direct_formula_input=True)

        def recognize(image_path: Path) -> str:
            results: list[PageFormulaResult] = []
            for result in pipeline.predict(
                str(image_path),
                use_layout_detection=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            ):
                results.extend(extract_paddle_formula_results(result))
            return extract_first_paddle_formula_latex(results)

        return recognize
    raise FormulaFillError(f"Unsupported crop OCR engine for benchmark: {engine}")


def make_benchmark_result(
    record: FormulaRecord,
    engine: str,
    strategy_name: str,
    input_path: Path | None,
    output_dir: Path,
    latex: str,
    elapsed_s: float,
    error: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_ref = ""
    if input_path is not None:
        try:
            input_ref = str(input_path.relative_to(output_dir))
        except ValueError:
            input_ref = str(input_path)

    flags = latex_quality_flags(latex)
    result = {
        "id": str(record["id"]),
        "page": record.get("page"),
        "preview_text": record.get("preview_text", ""),
        "engine": engine,
        "crop_strategy": strategy_name,
        "input_path": input_ref,
        "crop_path": input_ref if strategy_name != PAGE_SNAPSHOT_STRATEGY else "",
        "page_snapshot_path": input_ref if strategy_name == PAGE_SNAPSHOT_STRATEGY else "",
        "latex": latex,
        "quality_score": latex_quality_score(latex),
        "quality_flags": flags,
        "elapsed_s": round(elapsed_s, 3),
        "error": error,
    }
    if metadata:
        result.update(metadata)
    return result


def latex_quality_flags(latex: str) -> list[str]:
    flags: list[str] = []
    stripped = latex.strip()
    if not stripped:
        return ["empty"]

    normalized = re.sub(r"\\[,;:! ]", " ", stripped)
    normalized = re.sub(r"\s+", " ", normalized)
    spaced_patterns = {
        "spaced_alpha": r"A\s+l\s+p\s+h\s+a",
        "spaced_model": r"M\s+o\s+d\s+e\s+l",
        "spaced_corr": r"c\s+o\s+r\s+r",
        "spaced_industry": r"i\s+n\s+d\s+u\s+s\s+t\s+r\s+y",
        "spaced_style": r"s\s+t\s+y\s+l\s+e",
        "spaced_volume": r"v\s+o\s+l\s+u\s+m\s+e",
    }
    for flag, pattern in spaced_patterns.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            flags.append(flag)

    typo_patterns = {
        "industry_typo": r"i\s*n\s*d\s*u\s*s\s*t\s*r\s*v|industrv",
        "style_typo": r"s\s*t\s*v\s*l\s*e|stvle",
    }
    for flag, pattern in typo_patterns.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            flags.append(flag)

    if "corrr" in stripped.lower():
        flags.append("corr_typo")
    if "Alphad" in stripped or "dodel" in stripped or "dmodel" in stripped:
        flags.append("alpha_model_typo")
    if stripped.count("{") != stripped.count("}"):
        flags.append("unbalanced_braces")
    if stripped.count("(") != stripped.count(")"):
        flags.append("unbalanced_parentheses")
    if len(re.findall(r"\\left\b", stripped)) != len(re.findall(r"\\right\b", stripped)):
        flags.append("unbalanced_left_right")
    if stripped.count("\\cal") + stripped.count("\\mathcal") > 8:
        flags.append("excessive_cal")
    if "\\nonumber" in stripped:
        flags.append("nonumber_noise")
    if len(stripped) > 260:
        flags.append("very_long")
    return flags


def latex_quality_score(latex: str) -> int:
    penalties = {
        "empty": 100,
        "spaced_alpha": 10,
        "spaced_model": 8,
        "spaced_corr": 12,
        "spaced_industry": 6,
        "spaced_style": 6,
        "spaced_volume": 6,
        "industry_typo": 12,
        "style_typo": 12,
        "corr_typo": 10,
        "alpha_model_typo": 10,
        "unbalanced_braces": 18,
        "unbalanced_parentheses": 14,
        "unbalanced_left_right": 14,
        "excessive_cal": 10,
        "nonumber_noise": 8,
        "very_long": 6,
    }
    score = 100
    for flag in latex_quality_flags(latex):
        score -= penalties.get(flag, 0)
    return max(score, 0)


def formula_record_bbox(record: FormulaRecord) -> tuple[float, float, float, float]:
    bbox = record.get("bbox")
    formula_id = record.get("id", "<unknown>")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise FormulaFillError(f"{formula_id} is missing a valid bbox.")
    return tuple(float(value) for value in bbox)


def save_benchmark_crop_variants(
    pdf_path: Path,
    records: list[FormulaRecord],
    crop_strategies: list[FormulaCropStrategy],
    output_dir: Path,
) -> dict[tuple[str, str], Path]:
    crops_by_key: dict[tuple[str, str], Path] = {}
    crops_dir = output_dir / "benchmark_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    records_by_page: dict[int, list[FormulaRecord]] = {}
    for record in records:
        page = record.get("page")
        if isinstance(page, int):
            records_by_page.setdefault(page, []).append(record)

    with pymupdf.open(pdf_path) as doc:
        for page_number, page_records in records_by_page.items():
            page = doc[page_number - 1]
            for strategy in crop_strategies:
                strategy_dir = crops_dir / strategy.name
                strategy_dir.mkdir(parents=True, exist_ok=True)
                for record in page_records:
                    formula_id = str(record["id"])
                    crop_path = strategy_dir / f"{formula_id}.png"
                    save_formula_crop(page, formula_record_bbox(record), crop_path, strategy=strategy)
                    crops_by_key[(strategy.name, formula_id)] = crop_path
    return crops_by_key


def benchmark_formula_page_snapshots(
    records: list[FormulaRecord],
    manifest_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    pipeline = create_paddleocr_formula_pipeline()
    records_by_page: dict[int, list[FormulaRecord]] = {}
    for record in records:
        page = record.get("page")
        if isinstance(page, int):
            records_by_page.setdefault(page, []).append(record)

    results: list[dict[str, Any]] = []
    for page, page_records in sorted(records_by_page.items()):
        snapshot_path = resolve_page_snapshot_path(manifest_path, page_records[0])
        if not snapshot_path.exists():
            raise FormulaFillError(f"Page snapshot for page {page} does not exist: {snapshot_path}")

        started = time.perf_counter()
        predictions = list(pipeline.predict(str(snapshot_path)))
        elapsed = time.perf_counter() - started

        page_formula_results: list[PageFormulaResult] = []
        for prediction in predictions:
            page_formula_results.extend(extract_paddle_formula_results(prediction))
        matches = match_paddleocr_page_results(page_records, page_formula_results)

        per_formula_elapsed = elapsed / max(len(page_records), 1)
        for record in page_records:
            formula_id = str(record["id"])
            match = matches.get(formula_id)
            if match is None:
                results.append(
                    make_benchmark_result(
                        record,
                        "paddleocr-page",
                        PAGE_SNAPSHOT_STRATEGY,
                        snapshot_path,
                        output_dir,
                        "",
                        per_formula_elapsed,
                        error="paddleocr_page_no_matching_formula_region",
                    )
                )
                continue

            result, overlap = match
            latex = str(result["rec_formula"]).strip()
            results.append(
                make_benchmark_result(
                    record,
                    "paddleocr-page",
                    PAGE_SNAPSHOT_STRATEGY,
                    snapshot_path,
                    output_dir,
                    latex,
                    per_formula_elapsed,
                    metadata={
                        "formula_region_id": result.get("formula_region_id"),
                        "bbox_iou": round(overlap, 3),
                        "page_elapsed_s": round(elapsed, 3),
                        "page_formula_count": len(page_records),
                    },
                )
            )

    return results


def select_best_benchmark_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_by_id: dict[str, tuple[tuple[float, ...], int, dict[str, Any]]] = {}
    for index, result in enumerate(results):
        formula_id = str(result["id"])
        latex = str(result.get("latex", "")).strip()
        quality_score = float(result.get("quality_score", 0))
        flags = result.get("quality_flags")
        flag_count = len(flags) if isinstance(flags, list) else 0
        elapsed_s = float(result.get("elapsed_s", 0.0))
        key = (
            1.0 if latex else 0.0,
            quality_score,
            -float(flag_count),
            -elapsed_s,
            -float(index),
        )
        current = best_by_id.get(formula_id)
        if current is None or key > current[0]:
            best_by_id[formula_id] = (key, index, result)

    selected_by_id: dict[str, dict[str, Any]] = {}
    selected_indexes: set[int] = set()
    for formula_id, (_, index, result) in best_by_id.items():
        selected_by_id[formula_id] = result
        selected_indexes.add(index)

    for index, result in enumerate(results):
        result["selected"] = index in selected_indexes

    return selected_by_id


def attach_best_results_to_records(records: list[FormulaRecord], selected_by_id: dict[str, dict[str, Any]]) -> None:
    results_by_id = selected_by_id
    for record in records:
        formula_id = record.get("id")
        if not isinstance(formula_id, str):
            continue
        best = results_by_id.get(formula_id)
        if best is None:
            continue

        candidates = record.get("ocr_candidates")
        if not isinstance(candidates, list):
            candidates = []
        record["ocr_candidates"] = candidates

        latex = str(best.get("latex", "")).strip()
        if latex:
            record["latex"] = latex
            record["status"] = "formula_ocr_complete"
            record["ocr_engine"] = best.get("engine")
            record["ocr_strategy"] = best.get("crop_strategy")
            record["ocr_quality_score"] = best.get("quality_score")
            record["ocr_quality_flags"] = best.get("quality_flags")
            record.pop("ocr_error", None)
        else:
            record["ocr_error"] = best.get("error") or "no_formula_ocr_result"

        candidates.append(
            {
                "engine": best.get("engine"),
                "crop_strategy": best.get("crop_strategy"),
                "latex": best.get("latex"),
                "quality_score": best.get("quality_score"),
                "quality_flags": best.get("quality_flags"),
                "elapsed_s": best.get("elapsed_s"),
                "selected": True,
                "error": best.get("error", ""),
            }
        )


def summarize_benchmark_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((str(result["engine"]), str(result["crop_strategy"])), []).append(result)

    summary: list[dict[str, Any]] = []
    for (engine, crop_strategy), items in sorted(grouped.items()):
        completed = [item for item in items if item.get("latex")]
        scores = [int(item["quality_score"]) for item in items]
        seconds = sum(float(item["elapsed_s"]) for item in items)
        summary.append(
            {
                "engine": engine,
                "crop_strategy": crop_strategy,
                "formulas": len(items),
                "completed": len(completed),
                "complete_rate": round(len(completed) / len(items), 4) if items else 0.0,
                "total_s": round(seconds, 3),
                "avg_s": round(seconds / len(items), 3) if items else 0.0,
                "avg_quality_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
                "flagged": sum(1 for item in items if item.get("quality_flags")),
                "selected": sum(1 for item in items if item.get("selected")),
            }
        )
    return summary


def write_benchmark_report(
    output_dir: Path,
    pdf_path: Path,
    pages: str | None,
    parse_seconds: float,
    records: list[FormulaRecord],
    results: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_benchmark_results(results)
    jsonl_path = output_dir / "formula_ocr_benchmark.jsonl"
    summary_path = output_dir / "formula_ocr_benchmark_summary.json"
    report_path = output_dir / "formula_ocr_benchmark.md"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary_payload = {
        "pdf": str(pdf_path),
        "pages": pages,
        "parse_seconds": round(parse_seconds, 3),
        "formula_count": len(records),
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Formula OCR Benchmark",
        "",
        f"- PDF: `{pdf_path}`",
        f"- Pages: `{pages or 'all'}`",
        f"- Formula records: {len(records)}",
        f"- Parse seconds: {parse_seconds:.3f}",
        "",
        "## Summary",
        "",
        "| Engine | Crop strategy | Complete | Selected | Total s | Avg s | Avg score | Flagged |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            (
                "| {engine} | {crop_strategy} | {completed}/{formulas} | {selected} | "
                "{total_s:.3f} | {avg_s:.3f} | {avg_quality_score:.1f} | {flagged} |"
            ).format(**item)
        )
    lines.extend(["", "## Per Formula", ""])
    for record in records:
        formula_id = str(record["id"])
        lines.extend(
            [
                f"### {formula_id}",
                "",
                f"- page: {record.get('page')}",
                f"- preview: `{record.get('preview_text', '')}`",
                "",
                "| Engine | Crop strategy | Score | Selected | Flags | LaTeX |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for result in [item for item in results if item["id"] == formula_id]:
            flags = ", ".join(result.get("quality_flags") or [])
            latex = str(result.get("latex", "")).replace("|", "\\|")
            selected = "yes" if result.get("selected") else ""
            lines.append(
                f"| {result['engine']} | {result['crop_strategy']} | {result['quality_score']} | "
                f"{selected} | {flags} | `{latex}` |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return jsonl_path, summary_path, report_path


def benchmark_formula_ocr(
    pdf_path: Path,
    pages: str | None,
    output_dir: Path,
    engine_names: list[str],
    crop_strategy_names: list[str],
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parse_output = output_dir / "source.md"
    artifacts_dir = output_dir / "source_formula_artifacts"
    started = time.perf_counter()
    parse_result = parse_smart(pdf_path, pages, output_path=parse_output, artifacts_dir=artifacts_dir)
    parse_seconds = time.perf_counter() - started
    parse_output.write_text(
        materialize_markdown_images(parse_result.markdown, parse_output, parse_result.images),
        encoding="utf-8",
    )
    if parse_result.manifest_path is None:
        raise FormulaFillError("No formula candidates were detected.")

    records = read_formula_manifest(parse_result.manifest_path)
    if limit is not None:
        records = records[:limit]
    crop_strategies = [CROP_STRATEGIES[name] for name in crop_strategy_names]

    results: list[dict[str, Any]] = []
    crop_engine_names = [engine for engine in engine_names if engine in CROP_OCR_ENGINES]
    page_engine_names = [engine for engine in engine_names if engine in PAGE_OCR_ENGINES]
    if crop_engine_names:
        crops_by_key = save_benchmark_crop_variants(pdf_path, records, crop_strategies, output_dir)
    else:
        crops_by_key = {}

    for engine in crop_engine_names:
        recognizer = create_crop_formula_recognizer(engine)
        for strategy in crop_strategies:
            for record in records:
                formula_id = str(record["id"])
                crop_path = crops_by_key[(strategy.name, formula_id)]
                started = time.perf_counter()
                try:
                    latex = recognizer(crop_path)
                    error = ""
                except Exception as exc:
                    latex = ""
                    error = str(exc)
                elapsed = time.perf_counter() - started
                results.append(
                    make_benchmark_result(
                        record,
                        engine,
                        strategy.name,
                        crop_path,
                        output_dir,
                        latex,
                        elapsed,
                        error=error,
                    )
                )

    for engine in page_engine_names:
        if engine == "paddleocr-page":
            results.extend(benchmark_formula_page_snapshots(records, parse_result.manifest_path, output_dir))
        else:
            raise FormulaFillError(f"Unsupported page OCR engine for benchmark: {engine}")

    selected_by_id = select_best_benchmark_results(results)
    attach_best_results_to_records(records, selected_by_id)
    write_formula_manifest_records(parse_result.manifest_path, records)

    jsonl_path, summary_path, report_path = write_benchmark_report(
        output_dir,
        pdf_path,
        pages,
        parse_seconds,
        records,
        results,
    )
    return {
        "formula_count": len(records),
        "parse_seconds": parse_seconds,
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "summary": summarize_benchmark_results(results),
        "source_markdown_path": parse_output,
        "manifest_path": parse_result.manifest_path,
        "records": records,
        "markdown": parse_result.markdown,
        "images": parse_result.images,
        "results": results,
        "selected_by_id": selected_by_id,
    }


def render_formula_markdown(latex: str, kind: str | None) -> str:
    stripped = latex.strip()
    if not stripped:
        return stripped
    if stripped.startswith("$$") or stripped.startswith("\\[") or (stripped.startswith("$") and stripped.endswith("$")):
        return stripped
    if kind == "inline":
        return f"${stripped}$"
    return f"$$\n{stripped}\n$$"


def compact_markdown_blank_lines(markdown: str) -> str:
    """Keep at most one blank line outside fenced code blocks."""
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
            result.append(line)
            continue

        if fence_char is None and not line.strip():
            if result and result[-1] != "":
                result.append("")
            continue
        result.append(line)

    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


def normalize_table_of_contents(markdown: str) -> str:
    """Normalize dot leaders and page labels inside an identified TOC block."""
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
    """Return the Markdown character span occupied by the first TOC block."""
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


def fill_formula_placeholders(
    markdown: str,
    records: list[FormulaRecord],
    strict: bool = True,
) -> tuple[str, int]:
    latex_by_id: dict[str, str] = {}
    for record in records:
        formula_id = record.get("id")
        latex = record.get("latex")
        if isinstance(formula_id, str) and isinstance(latex, str) and latex.strip():
            kind = record.get("kind")
            latex_by_id[formula_id] = render_formula_markdown(latex, kind if isinstance(kind, str) else None)

    replaced_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replaced_count
        formula_id = match.group(1)
        replacement = latex_by_id.get(formula_id)
        if replacement is None:
            if strict:
                raise FormulaFillError(f"No LaTeX result is available for placeholder {formula_id}.")
            return match.group(0)
        replaced_count += 1
        return replacement

    filled = normalize_table_of_contents(FORMULA_PLACEHOLDER_RE.sub(replace, markdown))
    return compact_markdown_blank_lines(filled), replaced_count


def normalize_text(text: str) -> str:
    """Normalize text by removing spaces and converting to lowercase for matching."""
    return re.sub(r"[\s*_\\]", "", text).lower()


def formula_text_similarity(preview: str, rendered: str) -> float:
    """Measure fuzzy character overlap after PDF/Markdown spacing damage."""
    left = normalize_text(preview)
    right = normalize_text(rendered)
    if not left or not right:
        return 0.0
    width = 2 if min(len(left), len(right)) < 8 else 3
    left_grams = {left[index : index + width] for index in range(len(left) - width + 1)}
    right_grams = {right[index : index + width] for index in range(len(right) - width + 1)}
    if not left_grams or not right_grams:
        return 1.0 if left == right else 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def is_formula_line(line: str) -> bool:
    """Check if a line looks like a poorly-rendered formula from LiteParse."""
    stripped = line.strip()
    if not stripped:
        return False

    if stripped.startswith("#"):
        return False

    # Skip normal prose and emphasized prose. Asterisks alone are Markdown,
    # not evidence that a line is a display formula.
    if len(stripped) > 80:
        return False

    plain = stripped.replace("*", "").strip()
    if cjk_ratio(plain) > 0.15 or INLINE_PROSE_RE.search(plain):
        return False

    normalized = normalize_text(line)

    # Check for formula-like patterns
    math_indicators = ["=", "+", "/", "(", ")", "{", "}", "^"]
    has_math = any(ind in line for ind in math_indicators)
    math_mark_count = sum(line.count(indicator) for indicator in math_indicators)

    formula_words = ["industry", "style", "alpha", "model", "corr", "stas", "return", "factor"]
    has_formula_word = any(word in normalized for word in formula_words)

    unicode_math_count = sum(1 for char in plain if char in MATH_CHARS or is_private_use(char))
    unicode_math = unicode_math_count > 0

    # Lines with repeated formula keywords (like "*t* *in d u s tr y in d u s tr y s ty le s ty le*")
    has_repeated_formula_words = sum(1 for word in formula_words if word in normalized) >= 2

    if (
        (has_formula_word and (has_math or unicode_math))
        or (has_repeated_formula_words and unicode_math)
        or (has_math and ("=" in plain or unicode_math or math_mark_count >= 2))
        or unicode_math_count >= 2
    ):
        return True

    return False


def is_formula_continuation_line(line: str) -> bool:
    """Recognize script/name fragments only after a formula span has started."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or stripped.startswith("#"):
        return False
    plain = stripped.replace("*", "").strip()
    if cjk_ratio(plain) > 0.15 or INLINE_PROSE_RE.search(plain):
        return False
    normalized = normalize_text(plain)
    formula_words = ("industry", "style", "alpha", "model", "corr", "stas", "return", "factor")
    has_math_glyph = any(char in MATH_CHARS or is_private_use(char) for char in plain)
    return has_math_glyph or any(word in normalized for word in formula_words)


def insert_formula_placeholders(markdown: str, formulas: list[FormulaCandidate]) -> str:
    """Replace poorly-rendered formulas in LiteParse markdown with placeholders."""
    if not formulas:
        return markdown

    lines = markdown.split("\n")
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not is_formula_line(lines[index]):
            index += 1
            continue
        start = index
        last_formula = index
        cursor = index + 1
        while cursor < len(lines):
            if is_formula_line(lines[cursor]):
                last_formula = cursor
                cursor += 1
                continue
            if is_formula_continuation_line(lines[cursor]):
                last_formula = cursor
                cursor += 1
                continue
            if not lines[cursor].strip():
                cursor += 1
                continue
            break
        spans.append((start, last_formula + 1))
        index = cursor

    replacements: dict[int, tuple[int, FormulaCandidate]] = {}
    next_span_index = 0
    for formula in formulas:
        scored = [
            (
                formula_text_similarity(formula.preview_text, "\n".join(lines[start:end])),
                span_index,
                start,
                end,
            )
            for span_index, (start, end) in enumerate(spans[next_span_index:], start=next_span_index)
        ]
        if not scored:
            break
        score, span_index, start, end = max(scored, key=lambda item: (item[0], -item[1]))
        if score < 0.15:
            continue
        replacements[start] = (end, formula)
        next_span_index = span_index + 1

    result_lines: list[str] = []
    index = 0
    used_formulas: set[str] = set()
    while index < len(lines):
        replacement = replacements.get(index)
        if replacement is None:
            result_lines.append(lines[index])
            index += 1
            continue
        end, formula = replacement
        if result_lines and result_lines[-1].strip():
            result_lines.append("")
        result_lines.extend([f"{{{{formula:{formula.id}}}}}", ""])
        used_formulas.add(formula.id)
        index = end

    # A detected PDF formula may be absent from LiteParse output entirely. Keep
    # its placeholder rather than deleting unrelated text while guessing.
    for formula in formulas:
        if formula.id not in used_formulas:
            if result_lines and result_lines[-1].strip():
                result_lines.append("")
            result_lines.append(f"{{{{formula:{formula.id}}}}}")

    return compact_markdown_blank_lines("\n".join(result_lines))


def parse_smart(
    pdf_path: Path,
    target_pages: str | None,
    output_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> SmartParseResult:
    page_indexes = resolve_page_indexes(pdf_path, target_pages)
    liteparse_pages, images = liteparse_page_markdown(pdf_path, target_pages)

    with pymupdf.open(pdf_path) as doc:
        rendered_pages: list[str] = []
        formulas: list[FormulaCandidate] = []
        for index in page_indexes:
            page_number = index + 1
            page_formulas = detect_formula_candidates(doc[index], page_number)

            # Always use LiteParse as base
            lite_markdown = liteparse_pages.get(page_number, "").strip()

            if page_formulas:
                formulas.extend(page_formulas)
                # Insert formula placeholders into the LiteParse output
                page_text = insert_formula_placeholders(lite_markdown, page_formulas.copy())
            else:
                page_text = lite_markdown

            if page_text:
                page_text = f"<!-- page {page_number} -->\n\n{page_text}"
                rendered_pages.append(page_text)

    result = SmartParseResult(
        markdown=compact_markdown_blank_lines(
            normalize_table_of_contents("\n\n---\n\n".join(page for page in rendered_pages if page))
        ),
        formulas=formulas,
        images=images,
    )
    if formulas:
        result.artifacts_dir = artifacts_dir or default_artifacts_dir(pdf_path, output_path)
        result.manifest_path = write_formula_artifacts(pdf_path, formulas, result.artifacts_dir)
    return result


def resolve_page_indexes(pdf_path: Path, target_pages: str | None) -> list[int]:
    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
    if not target_pages:
        return list(range(page_count))

    indexes: set[int] = set()
    for part in target_pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise FormulaFillError(f"Invalid page selection: {part!r}") from exc
            if start > end:
                raise FormulaFillError(f"Page range start exceeds end: {part}")
            if start < 1 or end > page_count:
                raise FormulaFillError(f"Page range is outside 1-{page_count}: {part}")
            indexes.update(range(start - 1, end))
        else:
            try:
                page_number = int(part)
            except ValueError as exc:
                raise FormulaFillError(f"Invalid page selection: {part!r}") from exc
            if not 1 <= page_number <= page_count:
                raise FormulaFillError(f"Page number is outside 1-{page_count}: {page_number}")
            indexes.add(page_number - 1)
    if not indexes:
        raise FormulaFillError("Page selection did not contain any pages.")
    return sorted(indexes)


@click.group()
def cli() -> None:
    """PDF parsing utilities with automatic formula routing."""


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "output", type=click.Path(dir_okay=False, path_type=Path), help="Markdown output path.")
@click.option("--pages", help="1-indexed page selection such as '1,4-5'.")
@click.option(
    "--formula-artifacts-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for formula crops and formula_manifest.jsonl. Defaults next to the output Markdown.",
)
def parse(pdf: Path, output: Path | None, pages: str | None, formula_artifacts_dir: Path | None) -> None:
    try:
        result = parse_smart(pdf, pages, output_path=output, artifacts_dir=formula_artifacts_dir)
    except FormulaFillError as exc:
        raise click.ClickException(str(exc)) from exc
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(materialize_markdown_images(result.markdown, output, result.images), encoding="utf-8")
    else:
        click.echo(result.markdown)


@cli.command("fill-formulas")
@click.argument("markdown", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--manifest",
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Formula manifest jsonl. Defaults to <markdown_stem>_formula_artifacts/formula_manifest.jsonl.",
)
@click.option("-o", "output", type=click.Path(dir_okay=False, path_type=Path), help="Filled Markdown output path.")
@click.option("--in-place", is_flag=True, help="Replace the input Markdown file in place.")
@click.option(
    "--engine",
    type=click.Choice(["auto", *FORMULA_PROCESSORS.names]),
    default="auto",
    show_default=True,
    help="Formula OCR processor. Auto dispatches each manifest record by its processor route.",
)
@click.option("--force", is_flag=True, help="Run OCR again even when a record already has LaTeX.")
@click.option("--no-strict", is_flag=True, help="Keep placeholders that have no LaTeX instead of failing.")
def fill_formulas(
    markdown: Path,
    manifest: Path | None,
    output: Path | None,
    in_place: bool,
    engine: str,
    force: bool,
    no_strict: bool,
) -> None:
    if in_place and output is not None:
        raise click.ClickException("Use either --in-place or -o/--output, not both.")

    manifest_path = manifest or default_formula_manifest_path(markdown)
    if not manifest_path.exists():
        raise click.ClickException(f"Formula manifest does not exist: {manifest_path}")

    try:
        records = read_formula_manifest(manifest_path)
        dispatch_counts = dispatch_formula_ocr(
            records,
            manifest_path,
            processor_override=None if engine == "auto" else engine,
            force=force,
        )
        recognized_count = sum(dispatch_counts.values())
        write_formula_manifest_records(manifest_path, records)
        source_markdown = markdown.read_text(encoding="utf-8")
        filled_markdown, replaced_count = fill_formula_placeholders(source_markdown, records, strict=not no_strict)
    except FormulaFillError as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = markdown if in_place else output
    if output_path is None:
        click.echo(filled_markdown)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path != markdown:
            filled_markdown = rebundle_markdown_images(filled_markdown, markdown, output_path)
        output_path.write_text(filled_markdown, encoding="utf-8")

    target = str(output_path) if output_path is not None else "stdout"
    click.echo(
        f"filled {replaced_count} formula placeholders in {target}; ran {engine} OCR for {recognized_count} records",
        err=True,
    )


@cli.command("benchmark-formulas")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pages", help="1-indexed page selection such as '1,4-5'.")
@click.option(
    "--engine",
    "engines",
    multiple=True,
    type=click.Choice(list(FORMULA_OCR_ENGINES)),
    default=("pix2tex", "paddleocr-crop"),
    show_default=True,
    help="Formula OCR engine to benchmark. Can be used more than once.",
)
@click.option(
    "--crop-strategy",
    "crop_strategies",
    multiple=True,
    type=click.Choice(list(CROP_STRATEGIES)),
    default=("raw_3x", "tight_h40", "tight_h48", "tight_h56"),
    show_default=True,
    help="Crop normalization strategy to benchmark. Can be used more than once.",
)
@click.option("--limit", type=int, help="Limit formula records for a quick smoke benchmark.")
@click.option(
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Benchmark output directory. Defaults to <pdf_stem>_formula_benchmark.",
)
@click.option(
    "--filled-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional Markdown output filled with the best OCR candidate per formula.",
)
@click.option("--no-strict", is_flag=True, help="Keep unfilled placeholders in --filled-output instead of failing.")
def benchmark_formulas(
    pdf: Path,
    pages: str | None,
    engines: tuple[str, ...],
    crop_strategies: tuple[str, ...],
    limit: int | None,
    output_dir: Path | None,
    filled_output: Path | None,
    no_strict: bool,
) -> None:
    if not engines:
        raise click.ClickException("At least one --engine is required.")
    if not crop_strategies:
        raise click.ClickException("At least one --crop-strategy is required.")
    if limit is not None and limit <= 0:
        raise click.ClickException("--limit must be positive.")

    target_dir = output_dir or pdf.parent / f"{pdf.stem}_formula_benchmark"
    try:
        result = benchmark_formula_ocr(
            pdf,
            pages,
            target_dir,
            list(engines),
            list(crop_strategies),
            limit=limit,
        )
        if filled_output is not None:
            filled_markdown, replaced_count = fill_formula_placeholders(
                str(result["markdown"]),
                result["records"],
                strict=not no_strict,
            )
            filled_output.parent.mkdir(parents=True, exist_ok=True)
            filled_output.write_text(
                materialize_markdown_images(filled_markdown, filled_output, result["images"]),
                encoding="utf-8",
            )
    except FormulaFillError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"formula records: {result['formula_count']}")
    click.echo(f"parse seconds: {result['parse_seconds']:.3f}")
    for item in result["summary"]:
        click.echo(
            (
                "{engine} {crop_strategy}: {completed}/{formulas}, total={total_s:.3f}s, "
                "avg={avg_s:.3f}s, score={avg_quality_score:.1f}, flagged={flagged}"
            ).format(**item)
        )
    click.echo(f"report: {result['report_path']}")
    click.echo(f"jsonl: {result['jsonl_path']}")
    if filled_output is not None:
        click.echo(f"filled output: {filled_output} ({replaced_count} placeholders)")


@cli.command("parse-final")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o", "output", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Final Markdown output path."
)
@click.option("--pages", help="1-indexed page selection such as '1,4-5'.")
@click.option(
    "--engine",
    "engines",
    multiple=True,
    type=click.Choice(list(FORMULA_OCR_ENGINES)),
    default=("pix2tex",),
    show_default=True,
    help="Formula OCR engine to use. Can be used more than once for comparison and best-candidate selection.",
)
@click.option(
    "--crop-strategy",
    "crop_strategies",
    multiple=True,
    type=click.Choice(list(CROP_STRATEGIES)),
    default=("tight_h48",),
    show_default=True,
    help="Crop normalization strategy for crop OCR engines. Can be used more than once.",
)
@click.option(
    "--report-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for formula OCR comparison artifacts. Defaults to <output_stem>_formula_benchmark.",
)
@click.option("--no-strict", is_flag=True, help="Keep placeholders with no OCR result instead of failing.")
def parse_final(
    pdf: Path,
    output: Path,
    pages: str | None,
    engines: tuple[str, ...],
    crop_strategies: tuple[str, ...],
    report_dir: Path | None,
    no_strict: bool,
) -> None:
    if not engines:
        raise click.ClickException("At least one --engine is required.")
    if not crop_strategies:
        raise click.ClickException("At least one --crop-strategy is required.")

    target_dir = report_dir or output.parent / f"{output.stem}_formula_benchmark"
    try:
        result = benchmark_formula_ocr(
            pdf,
            pages,
            target_dir,
            list(engines),
            list(crop_strategies),
        )
        filled_markdown, replaced_count = fill_formula_placeholders(
            str(result["markdown"]),
            result["records"],
            strict=not no_strict,
        )
    except FormulaFillError as exc:
        raise click.ClickException(str(exc)) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        materialize_markdown_images(filled_markdown, output, result["images"]),
        encoding="utf-8",
    )
    click.echo(f"final markdown: {output}")
    click.echo(f"filled formula placeholders: {replaced_count}/{result['formula_count']}")
    click.echo(f"benchmark report: {result['report_path']}")


@cli.command("parse-cpu")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="CPU 混合解析生成的 Markdown。",
)
@click.option("--pages", help="1-indexed page selection such as '1,4-5'.")
@click.option("--model", default="PP-FormulaNet_plus-M", show_default=True, help="PaddleOCR 公式识别模型。")
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--render-scale", type=click.FloatRange(min=1.0), default=4.0, show_default=True)
@click.option(
    "--formula-device",
    type=click.Choice(["cpu", "dgx"]),
    default="cpu",
    show_default=True,
    help="公式模型运行位置；DGX 使用 SSH 上的 MinerU UniMERNet CUDA。",
)
@click.option("--dgx-host", default="dgx-aliyun", show_default=True)
@click.option(
    "--no-formula-model",
    is_flag=True,
    help="只测试 LiteParse CPU 探针；复杂公式全部使用局部矢量裁剪回退。",
)
def parse_cpu(
    pdf: Path,
    output: Path,
    pages: str | None,
    model: str,
    batch_size: int,
    render_scale: float,
    formula_device: str,
    dgx_host: str,
    no_formula_model: bool,
) -> None:
    """Parse a native-vector PDF and inject validated formulas into LiteParse Grid."""
    from .cpu_hybrid import parse_cpu_hybrid

    try:
        result = parse_cpu_hybrid(
            pdf,
            output,
            pages=pages,
            model_name=model,
            batch_size=batch_size,
            render_scale=render_scale,
            run_model=not no_formula_model,
            formula_device=formula_device,
            dgx_host=dgx_host,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"markdown: {result.markdown_path}")
    click.echo(f"report: {result.report_path}")
    click.echo(f"manifest: {result.manifest_path}")
    click.echo(
        "LiteParse={:.3f}s, final Grid={:.3f}s, model init={:.3f}s, inference={:.3f}s; "
        "native={}, vision={}, latex={}, fallback={}".format(
            result.parse_seconds,
            result.reproject_seconds,
            result.model_init_seconds,
            result.model_inference_seconds,
            result.native_count,
            result.vision_count,
            result.accepted_latex_count,
            result.fallback_image_count,
        )
    )
    if result.skipped_pages:
        click.echo(f"skipped scanned pages: {result.skipped_pages}", err=True)


@cli.command("parse-best")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Final Markdown output path.",
)
@click.option(
    "--dgx-host",
    default="dgx-aliyun",
    envvar="RESEARCH_PDF_PARSER_DGX_HOST",
    show_default=True,
    help="SSH host used for the DGX MinerU worker.",
)
@click.option(
    "--remote-uvx",
    default="~/.local/bin/uvx",
    envvar="RESEARCH_PDF_PARSER_DGX_UVX",
    show_default=True,
    help="uvx executable on the DGX worker.",
)
@click.option(
    "--backend",
    type=click.Choice(["hybrid-engine", "pipeline"]),
    default="hybrid-engine",
    show_default=True,
    help="MinerU backend. hybrid-engine is the high-accuracy default.",
)
@click.option(
    "--effort",
    type=click.Choice(["low", "medium", "high"]),
    default="high",
    show_default=True,
    help="MinerU VLM effort for the hybrid backend.",
)
@click.option("--keep-remote", is_flag=True, help="Keep the remote run directory for debugging.")
def parse_best(
    pdf: Path,
    output: Path,
    dgx_host: str,
    remote_uvx: str,
    backend: str,
    effort: str,
    keep_remote: bool,
) -> None:
    """Run the high-accuracy MinerU path on DGX and package local Markdown."""
    config = RemoteMineruConfig(
        host=dgx_host,
        uvx_path=remote_uvx,
        backend=backend,
        effort=effort,
        keep_remote=keep_remote,
    )
    try:
        with TemporaryDirectory(prefix="research-pdf-parser-mineru-") as directory:
            raw_markdown_path = convert_pdf_on_dgx(pdf, Path(directory), config)
            markdown = clean_extraction_markers(raw_markdown_path.read_text(encoding="utf-8"))

            with pymupdf.open(pdf) as document:
                has_toc_candidate_page = document.page_count >= 2
            if has_toc_candidate_page:
                try:
                    toc_pages, _ = liteparse_page_markdown(pdf, "2")
                    reference_toc = toc_pages.get(2, "")
                    if reference_toc:
                        markdown = prefer_table_of_contents(markdown, reference_toc)
                except Exception as exc:  # TOC repair is an optional secondary parser pass.
                    click.echo(f"warning: vector TOC repair skipped: {exc}", err=True)

            markdown = compact_markdown_blank_lines(normalize_table_of_contents(markdown))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                rebundle_markdown_images(markdown, raw_markdown_path, output),
                encoding="utf-8",
            )
    except RemoteMineruError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"final markdown: {output}")
    click.echo(f"backend: MinerU {backend} ({effort}) on {dgx_host}")
    click.echo(f"assets: {markdown_assets_dir(output)}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
