"""CPU-first LiteParse formula reconstruction for native-vector PDFs.

The LiteParse fork emits formula candidates before grid projection. This
module turns those candidates into final Markdown atoms without running OCR on
the whole page: native code rows stay as text, genuinely two-dimensional
regions go through PP-FormulaNet, and rejected model output falls back to the
crisp vector crop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pymupdf
from liteparse import FormulaAtom as GridFormulaAtom
from liteparse import LiteParse

from .assets import materialize_liteparse_images
from .formula_runtime import PaddleFormulaRuntime
from .formula_service import recognize_formula_files
from .markdown_cleanup import normalize_markdown
from .pdf_utils import (
    normalize_private_use,
    resolve_page_numbers,
    scanned_page_reason,
)

IMPORTANT_IDENTIFIERS = (
    "alphamodel",
    "industry",
    "style",
    "dastd",
    "cmra",
    "stom",
    "stoq",
    "stoa",
    "rstr",
    "lncap",
    "epibs",
    "etop",
    "cetop",
    "btop",
    "dtoa",
    "blev",
    "mlev",
    "hsigma",
    "corr",
)
NOISE_RE = re.compile(r"数量化专题报告|请务必阅读正文之后|\b\d+\s+of\s+\d+\b")
HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)+\.?\s+|\d+\.\s+|附录\s*\d*)")


@dataclass
class FormulaAtom:
    id: str
    page: int
    bbox: tuple[float, float, float, float]
    route: str
    native_text: str
    confidence: float
    reasons: list[str]
    context_label: str = ""
    crop_path: Path | None = None
    latex: str = ""
    model_score: float | None = None
    validation_flags: list[str] = field(default_factory=list)
    accepted: bool = False
    model_source: str = ""


@dataclass
class CPUHybridResult:
    markdown_path: Path
    report_path: Path
    manifest_path: Path
    parse_seconds: float
    reproject_seconds: float
    model_init_seconds: float
    model_inference_seconds: float
    native_count: int
    vision_count: int
    accepted_latex_count: int
    fallback_image_count: int
    skipped_pages: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RuleBand:
    top: float
    bottom: float


def compact_formula_spacing(text: str) -> str:
    text = normalize_private_use(text)
    # PDF glyph runs often spell identifiers as "A l p h a". Only collapse
    # spaces surrounded by ASCII identifier characters; Chinese prose and
    # normal punctuation spacing remain intact.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?<=[A-Za-z0-9_])\s+(?=[A-Za-z0-9_])", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_recognized_latex(latex: str) -> str:
    """Repair harmless PDF-style letter spacing without guessing semantics."""
    separator = r"(?:\s+|\s*\\(?:[,;:!]|quad|qquad)\s*)"
    replacements = {
        "industry": "industry",
        "style": "style",
        "DASTD": "DASTD",
        "CMRA": "CMRA",
        "RSTR": "RSTR",
        "EPIBS": "EPIBS",
        "DTOA": "DTOA",
        "TD": "TD",
        "TA": "TA",
        "ME": "ME",
        "LD": "LD",
        "BE": "BE",
        "HSIGMA": "HSIGMA",
        "BLEV": "BLEV",
        "MLEV": "MLEV",
        "STOM": "STOM",
        "STOQ": "STOQ",
        "STOA": "STOA",
        "ln": r"\ln",
        "max": r"\max",
        "min": r"\min",
        "exp": r"\exp",
        "est": "est",
        "eps": "eps",
        "STD": "STD",
    }
    math_blocks = re.findall(r"(?<!\\)\$([^$]+)\$", latex)
    normalized = math_blocks[0] if math_blocks else latex
    for word, replacement in replacements.items():
        pattern = r"(?<![A-Za-z])" + separator.join(re.escape(char) for char in word) + r"(?![A-Za-z])"
        normalized = re.sub(pattern, lambda _match, value=replacement: value, normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\\mathrm\{cor\}(?:\s+|\s*\\[,;:!]\s*)*r",
        lambda _match: r"\operatorname{corr}",
        normalized,
        flags=re.IGNORECASE,
    )
    math_spacing = r"(?:\s+|\s*\\[,;:!]\s*)*"
    normalized = re.sub(
        rf"\\mathrm\{{m\}}{math_spacing}\\mathrm\{{a\s*x\}}",
        lambda _match: r"\max",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        rf"\\mathrm\{{m\}}{math_spacing}\\mathrm\{{i\s*n\}}",
        lambda _match: r"\min",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.replace(r"\mathrm{\exp}", r"\exp")
    normalized = normalized.replace(r"\operatorname{\ln}", r"\ln")
    normalized = normalized.replace(r"\text{\max}", r"\max")
    normalized = normalized.replace(r"\text{\min}", r"\min")
    normalized = re.sub(
        r"\\textsuperscript\s*\{\s*\\textit\s*\{\s*([^{}]+?)\s*\}\s*\}",
        lambda match: "^{" + match.group(1).strip() + "}",
        normalized,
    )
    normalized = normalized.replace(r"D\mathop{T O A}", "DTOA")
    normalized = normalized.replace(r"\mathop{T D}", "TD")
    normalized = normalized.replace(r"\mathop{TD}", "TD")
    normalized = normalized.replace(r"\mathop{T A}", "TA")
    normalized = normalized.replace(r"\mathop{TA}", "TA")
    normalized = normalized.replace(r"\mathop{/}", "/")
    normalized = normalized.replace(r"B\mathop{{L}{E}{V}}", "BLEV")
    normalized = normalized.replace(r"\mathop{{B}{E}}", "BE")
    normalized = normalized.replace(r"\mathop{{L}{D}}", "LD")
    # PP-FormulaNet occasionally recognizes ``S T D`` in two passes: the
    # generic identifier repair above first turns ``T D`` into ``TD``, leaving
    # ``s TD`` behind. Finish that deterministic repair here so HSIGMA table
    # cells remain readable without adding any semantic guesswork.
    normalized = re.sub(
        r"(?<![A-Za-z])s(?:\s+|\s*\\(?:[,;:!]|quad|qquad)\s*)*TD(?![A-Za-z])",
        "STD",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"HSIGMA(?:\s*\\quad\s*)+HSIGMA", "HSIGMA", normalized)
    normalized = re.sub(r"(?<=MLEV=\()M\s+E", "ME", normalized)
    normalized = re.sub(r"(?<=\+)L\s+D", "LD", normalized)
    normalized = re.sub(r"/M\s+E", "/ME", normalized)
    normalized = re.sub(
        r"\\mathrm\{CETOP\}\s*\\quad\s*\\text\{CETOP\}",
        "CETOP",
        normalized,
    )
    normalized = re.sub(r"est\s*\\quad\s*_\{-\}\s*\\quad\s*eps", r"est_{eps}", normalized)
    normalized = re.sub(r"(?:\\;)+;+", ";", normalized)
    normalized = re.sub(r";{2,}", ";", normalized)
    for _ in range(6):
        previous = normalized
        normalized = re.sub(r"\{\s*\}", "", normalized)
        normalized = re.sub(r"([_^])\{\s*\{([^{}]+)\}\s*\}", r"\1{\2}", normalized)
        normalized = re.sub(r"_\{\s*_\{([^{}]+)\}\s*\}", r"_{\1}", normalized)
        normalized = re.sub(r"\^\{\s*\^\{([^{}]+)\}\s*\}", r"^{\1}", normalized)
        normalized = re.sub(r"_\{\s*_([A-Za-z0-9]+)\s*\}", r"_{\1}", normalized)
        normalized = re.sub(r"\^\{\s*\^([A-Za-z0-9]+)\s*\}", r"^{\1}", normalized)
        if normalized == previous:
            break
    return normalized.strip()


def command_braced_arguments(latex: str, command: str) -> list[str]:
    """Return balanced braced arguments immediately following a LaTeX command."""
    arguments: list[str] = []
    cursor = 0
    while True:
        start = latex.find(command, cursor)
        if start < 0:
            return arguments
        brace = start + len(command)
        while brace < len(latex) and latex[brace].isspace():
            brace += 1
        if brace >= len(latex) or latex[brace] != "{":
            cursor = brace
            continue
        depth = 0
        for end in range(brace, len(latex)):
            if latex[end] == "{":
                depth += 1
            elif latex[end] == "}":
                depth -= 1
                if depth == 0:
                    arguments.append(latex[brace + 1 : end])
                    cursor = end + 1
                    break
        else:
            return arguments


def atom_from_candidate(page_number: int, candidate: Any) -> FormulaAtom:
    return FormulaAtom(
        id=str(candidate.id),
        page=page_number,
        bbox=(
            float(candidate.x),
            float(candidate.y),
            float(candidate.width),
            float(candidate.height),
        ),
        route=str(candidate.route),
        native_text=str(candidate.text),
        confidence=float(candidate.confidence),
        reasons=list(candidate.reasons),
        accepted=str(candidate.route) == "native_text",
    )


def atom_rect(atom: FormulaAtom) -> pymupdf.Rect:
    x, y, width, height = atom.bbox
    return pymupdf.Rect(x, y, x + width, y + height)


def render_formula_crops(
    pdf_path: Path,
    atoms: list[FormulaAtom],
    crops_dir: Path,
    scale: float = 4.0,
) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(scale, scale)
    with pymupdf.open(pdf_path) as document:
        for atom in atoms:
            if atom.route != "vision":
                continue
            page = document[atom.page - 1]
            clip = atom_rect(atom)
            if "ruled_table_cell" not in atom.reasons:
                clip.x0 -= 60.0
                clip.x1 += 16.0
            normalized = normalize_private_use(atom.native_text).strip()
            if normalized.startswith("="):
                clip.x0 -= 90.0
            if normalized.endswith(("/", "(")):
                clip.x1 += 42.0
            clip &= page.rect
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            path = crops_dir / f"{atom.id}.png"
            pixmap.save(path)
            atom.crop_path = path


def run_formula_model(
    atoms: list[FormulaAtom],
    model_name: str = "PP-FormulaNet_plus-S",
    fallback_model_name: str | None = "PP-FormulaNet_plus-M",
    batch_size: int = 4,
    device: str = "auto",
) -> tuple[float, float]:
    pending = sorted(
        [atom for atom in atoms if atom.route == "vision" and atom.crop_path],
        key=lambda atom: (round(atom.bbox[3] / 16), round(atom.bbox[2] / 32), atom.id),
    )
    if not pending:
        return 0.0, 0.0
    runtime = PaddleFormulaRuntime(model_name=model_name, device=device)
    predictions, inference_seconds = runtime.predict(
        [atom.crop_path for atom in pending if atom.crop_path],
        batch_size=batch_size,
    )
    for atom, prediction in zip(pending, predictions, strict=False):
        atom.latex = normalize_recognized_latex(prediction.latex)
        atom.model_score = prediction.score
        atom.model_source = f"{model_name}:{runtime.device}"

    init_seconds = runtime.init_seconds
    retry = [atom for atom in pending if latex_validation_flags(atom.native_text, atom.latex, atom.model_score)]
    if retry and fallback_model_name and fallback_model_name != model_name:
        fallback = PaddleFormulaRuntime(model_name=fallback_model_name, device=device)
        fallback_predictions, fallback_seconds = fallback.predict(
            [atom.crop_path for atom in retry if atom.crop_path],
            batch_size=max(1, min(batch_size, 2)),
        )
        init_seconds += fallback.init_seconds
        inference_seconds += fallback_seconds
        for atom, prediction in zip(retry, fallback_predictions, strict=False):
            candidate = normalize_recognized_latex(prediction.latex)
            current_flags = latex_validation_flags(atom.native_text, atom.latex, atom.model_score)
            candidate_flags = latex_validation_flags(atom.native_text, candidate, prediction.score)
            current_score = atom.model_score if atom.model_score is not None else -1.0
            candidate_score = prediction.score if prediction.score is not None else -1.0
            if len(candidate_flags) < len(current_flags) or (
                len(candidate_flags) == len(current_flags) and candidate_score > current_score
            ):
                atom.latex = candidate
                atom.model_score = prediction.score
                atom.model_source = f"{fallback_model_name}:{fallback.device}-fallback"
    return init_seconds, inference_seconds


def run_formula_model_http(
    atoms: list[FormulaAtom],
    server_url: str,
    batch_size: int = 4,
) -> tuple[float, float]:
    pending = [atom for atom in atoms if atom.route == "vision" and atom.crop_path]
    if not pending:
        return 0.0, 0.0
    response = recognize_formula_files(
        server_url,
        [(atom.id, atom.crop_path) for atom in pending if atom.crop_path],
        batch_size=batch_size,
    )
    by_id = {atom.id: atom for atom in pending}
    seen: set[str] = set()
    for item in response["results"]:
        atom = by_id.get(str(item.get("id", "")))
        if atom is None:
            continue
        atom.latex = normalize_recognized_latex(str(item.get("latex", "")))
        score = item.get("score")
        atom.model_score = float(score) if score is not None else None
        atom.model_source = (
            f"{response.get('model', 'formula-service')}:"
            f"{response.get('device', 'remote')}-http"
        )
        seen.add(atom.id)
    missing = sorted(set(by_id) - seen)
    if missing:
        raise RuntimeError(f"Formula service omitted {len(missing)} result(s): {missing[:5]}")
    # A persistent service paid model startup before this document request.
    # Keep per-document timings honest; /health exposes the service cold start.
    return 0.0, float(response.get("inference_seconds", 0.0))


def latex_validation_flags(native_text: str, latex: str, score: float | None = None) -> list[str]:
    flags: list[str] = []
    stripped = latex.strip()
    if not stripped:
        return ["empty"]
    if stripped.count("{") != stripped.count("}"):
        flags.append("unbalanced_braces")
    if stripped.count("(") != stripped.count(")"):
        flags.append("unbalanced_parentheses")
    if stripped.count("[") != stripped.count("]"):
        flags.append("unbalanced_brackets")
    if re.search(r"\\[+=-]", stripped):
        flags.append("escaped_operator_noise")
    if re.search(r"\\[#%&]", stripped):
        flags.append("escaped_text_symbol_noise")
    if re.search(r"[,;:]\s*=|=\s*[,;:]", stripped):
        flags.append("punctuation_near_equals")
    if "\\\\" in stripped and "\\begin{" not in stripped:
        flags.append("unexpected_latex_linebreak")
    if re.search(r"(?:_\{\s*_\^?|\^\{\s*\^)", stripped):
        flags.append("nested_script_operator")
    if (
        r"\ldots" in stripped
        and re.search(r"f_\{?K\}?", stripped)
        and not re.search(r"\\varepsilon_\{?K\}?(?!\d)", stripped)
    ):
        flags.append("terminal_k_index_mismatch")

    native_symbols = normalize_private_use(native_text)
    greek_commands = {
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\varepsilon": "ε",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\tau": "τ",
        r"\phi": "φ",
        r"\psi": "ψ",
        r"\Psi": "Ψ",
        r"\omega": "ω",
    }
    for command, symbol in greek_commands.items():
        if command in stripped and symbol not in native_symbols:
            command_name = command[1:] if command.startswith("\\") else command
            flags.append(f"unexpected_greek:{command_name}")
    if len(re.findall(r"\\left\b", stripped)) != len(re.findall(r"\\right\b", stripped)):
        flags.append("unbalanced_left_right")
    begins = re.findall(r"\\begin\{([^}]+)\}", stripped)
    ends = re.findall(r"\\end\{([^}]+)\}", stripped)
    if begins != ends:
        flags.append("unbalanced_environment")
    if "\ufffd" in stripped or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", stripped):
        flags.append("invalid_character")
    if len(stripped) > 700:
        flags.append("implausibly_long")
    if re.search(r"[\u3400-\u9fff]", stripped):
        flags.append("mixed_prose")
    if "$" in stripped:
        flags.append("embedded_math_delimiter")
    if stripped.count("_{") + stripped.count("^{") > 18:
        flags.append("excessive_scripts")
    if re.search(r"/(?:\s|\\[,;:!])*[A-Z][,.;]?\s*$", stripped):
        flags.append("trailing_single_letter_denominator")
    if re.search(r"(?<![A-Za-z])(?:[A-Za-z]\s+){3,}[A-Za-z](?![A-Za-z])", stripped):
        flags.append("spaced_ascii_word")
    if "corrr" in stripped.lower():
        flags.append("corr_typo")
    if any(
        command in stripped
        for command in (
            r"\cal",
            r"\boldmath",
            r"\nwarrow",
            r"\stackrel",
            r"\ddag",
            r"\ddagger",
            r"\sharp",
            r"\not \equiv",
        )
    ):
        flags.append("formula_command_noise")
    if stripped.count(r"\textbf") >= 3 or re.search(r"(?:\\#\s*){2,}", stripped):
        flags.append("excessive_style_noise")
    if re.search(r"(?:_|\^|\\frac|\\sqrt)\s*$", stripped):
        flags.append("trailing_operator")
    if "ir" in compact_formula_spacing(native_text).lower():
        for radicand in command_braced_arguments(stripped, r"\sqrt"):
            compact_radicand = re.sub(r"\s+", "", radicand)
            if "252" in compact_radicand and ("*" in compact_radicand or r"\sigma" in radicand):
                flags.append("implausible_ir_sqrt_scope")
                break
    if re.search(r"odeel|indstry|\\arg\s+e|nu\s+e\s+r", stripped, re.IGNORECASE):
        flags.append("ocr_word_corruption")
    if re.match(r"\s*(?:I\s+a\s+x|a\s+x)(?:\s|\\quad)", stripped):
        flags.append("max_token_corruption")
    if score is not None and score < 0.35:
        flags.append("low_model_score")

    native_compact = re.sub(r"[^a-z]", "", compact_formula_spacing(native_text).lower())
    latex_without_commands = re.sub(r"\\[A-Za-z]+", "", stripped)
    latex_compact = re.sub(r"[^a-z]", "", latex_without_commands.lower())
    for identifier in IMPORTANT_IDENTIFIERS:
        if identifier in native_compact and identifier not in latex_compact:
            flags.append(f"missing_identifier:{identifier}")
    native_normalized = normalize_private_use(native_text)
    if re.search(r"(?:^|\s)K(?:$|\s)", native_normalized) and not re.search(
        r"(?<![A-Za-z])K(?![A-Za-z])", latex_without_commands
    ):
        flags.append("missing_identifier:k")
    if re.search(r"[τt]\s*=\s*1", native_normalized, re.IGNORECASE) and not re.search(
        r"(?:\\tau|(?<![A-Za-z])t)\s*=\s*1", stripped, re.IGNORECASE
    ):
        flags.append("missing_sum_lower_bound")
    return flags


def validate_atoms(atoms: list[FormulaAtom]) -> None:
    for atom in atoms:
        if atom.route != "vision":
            atom.accepted = True
            continue
        validation_text = "\n".join(part for part in (atom.context_label, atom.native_text) if part)
        atom.validation_flags = latex_validation_flags(validation_text, atom.latex, atom.model_score)
        atom.accepted = not atom.validation_flags


def _relative_asset(path: Path, output_path: Path) -> str:
    relative = os.path.relpath(path, output_path.parent).replace(os.sep, "/")
    return quote(relative, safe="/-._~")


def render_atom(atom: FormulaAtom, output_path: Path) -> str:
    if atom.route == "native_text":
        text = atom.native_text.split("\n", 1)[-1] if "alpha_code_row" in atom.reasons else atom.native_text
        text = compact_formula_spacing(text)
        if "alpha_code_row" not in atom.reasons:
            text = re.sub(r"^([A-Z][A-Z0-9_]{2,}?)\1(?=\s*[A-Z=])", r"\1", text)
        return f"```text\n{text}\n```"
    if atom.accepted:
        latex = atom.latex.strip().removeprefix("$$").removesuffix("$$").strip()
        return f"$$\n{latex}\n$$"
    if atom.crop_path is None:
        return f"`[公式 {atom.id} 无可用输出]`"
    return f"![公式 {atom.id}（矢量裁剪回退）]({_relative_asset(atom.crop_path, output_path)})"


def grid_atom_markdown(atom: FormulaAtom, output_path: Path) -> str:
    """Build the trusted Markdown payload inserted into LiteParse's final Grid."""
    if atom.route == "native_text":
        if "alpha_code_row" in atom.reasons:
            label, _, body = atom.native_text.partition("\n")
            body = compact_formula_spacing(body)
            return f"**{label.strip()}**\n\n```text\n{body}\n```"
        text = compact_formula_spacing(atom.native_text)
        text = re.sub(r"^([A-Z][A-Z0-9_]{2,}?)\1(?=\s*[A-Z=])", r"\1", text)
        if "ruled_table_cell" in atom.reasons:
            fence = "``" if "`" in text else "`"
            padding = " " if fence == "``" else ""
            return f"{fence}{padding}{text}{padding}{fence}"
        return f"```text\n{text}\n```"
    if atom.accepted:
        latex = atom.latex.strip().removeprefix("$$").removesuffix("$$").strip()
        delimiter = "$" if "ruled_table_cell" in atom.reasons or atom.context_label else "$$"
        return f"{delimiter}{latex}{delimiter}"
    if atom.crop_path is None:
        return f"`[公式 {atom.id} 无可用输出]`"
    return f"![公式 {atom.id}（矢量裁剪回退）]({_relative_asset(atom.crop_path, output_path)})"


def to_grid_atom(atom: FormulaAtom, output_path: Path) -> GridFormulaAtom:
    x, y, width, height = atom.bbox
    if atom.context_label and x < 145.0 and x + width > 149.0:
        right = x + width
        x = 145.0
        width = right - x
    normalized_native = normalize_private_use(atom.native_text).strip()
    if "incomplete_table_formula" in atom.reasons and normalized_native.endswith("("):
        width += 22.0
    if atom.route == "native_text":
        source = "liteparse-native-vector"
        confidence = atom.confidence
    elif atom.accepted:
        source = atom.model_source or "validated-formula-model"
        confidence = atom.model_score if atom.model_score is not None else atom.confidence
    else:
        source = "vector-image-fallback"
        confidence = atom.confidence
    return GridFormulaAtom(
        id=atom.id,
        page_num=atom.page,
        x=x,
        y=y,
        width=width,
        height=height,
        markdown=grid_atom_markdown(atom, output_path),
        confidence=max(0.0, min(1.0, float(confidence))),
        source=source,
    )


def _item_center(item: Any) -> tuple[float, float]:
    return float(item.x) + float(item.width) / 2, float(item.y) + float(item.height) / 2


def _item_in_atom(item: Any, atom: FormulaAtom) -> bool:
    x, y = _item_center(item)
    rect = atom_rect(atom)
    return rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1


def _join_items(items: list[Any]) -> str:
    if not items:
        return ""
    ordered = sorted(items, key=lambda item: float(item.x))
    parts: list[str] = []
    previous_right: float | None = None
    for item in ordered:
        text = normalize_private_use(str(item.text)).strip()
        if not text:
            continue
        left = float(item.x)
        if parts and previous_right is not None and left - previous_right > 2.5:
            parts.append(" ")
        parts.append(text)
        previous_right = left + float(item.width)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _line_groups(items: list[Any]) -> list[tuple[float, list[Any]]]:
    groups: list[tuple[float, list[Any]]] = []
    for item in sorted(items, key=lambda value: (_item_center(value)[1], float(value.x))):
        center = _item_center(item)[1]
        if groups and abs(center - groups[-1][0]) <= 4.6:
            existing_center, existing = groups[-1]
            existing.append(item)
            groups[-1] = ((existing_center * (len(existing) - 1) + center) / len(existing), existing)
        else:
            groups.append((center, [item]))
    return groups


def compose_linear_region(
    page: Any,
    atoms: list[FormulaAtom],
    output_path: Path,
    top: float = 58.0,
    bottom: float | None = None,
    excluded_bands: list[RuleBand] | None = None,
) -> str:
    bottom = float(page.height) - 38.0 if bottom is None else bottom
    excluded_bands = excluded_bands or []
    region_atoms = [
        atom
        for atom in atoms
        if top <= atom.bbox[1] + atom.bbox[3] / 2 < bottom
        and not any(band.top <= atom.bbox[1] + atom.bbox[3] / 2 < band.bottom for band in excluded_bands)
    ]
    items = []
    for item in page.text_items:
        center = _item_center(item)[1]
        if not top <= center < bottom:
            continue
        if any(band.top <= center < band.bottom for band in excluded_bands):
            continue
        if any(_item_in_atom(item, atom) for atom in region_atoms):
            continue
        if NOISE_RE.search(str(item.text)):
            continue
        items.append(item)

    events: list[tuple[float, str, str]] = []
    for center, line_items in _line_groups(items):
        text = _join_items(line_items)
        if text:
            events.append((center, "line", text))
    for atom in region_atoms:
        events.append((atom.bbox[1] + atom.bbox[3] / 2, "atom", render_atom(atom, output_path)))
    events.sort(key=lambda event: (event[0], 0 if event[1] == "atom" else 1))

    output: list[str] = []
    paragraph: list[str] = []
    previous_y: float | None = None

    def flush() -> None:
        if paragraph:
            output.append(" ".join(paragraph))
            paragraph.clear()

    for y, kind, content in events:
        if kind == "atom":
            flush()
            output.append(content)
            previous_y = y
            continue
        if HEADING_RE.match(content):
            flush()
            output.append(f"## {content}")
        elif previous_y is not None and y - previous_y > 17.0:
            flush()
            paragraph.append(content)
        else:
            paragraph.append(content)
        previous_y = y
    flush()
    return "\n\n".join(part for part in output if part.strip())


def horizontal_rule_bands(page: pymupdf.Page) -> list[RuleBand]:
    lines: list[float] = []
    min_width = page.rect.width * 0.3
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.y - end.y) <= 0.8 and abs(start.x - end.x) >= min_width:
                    lines.append((start.y + end.y) / 2)
            elif item[0] == "re":
                rect = item[1]
                if rect.height <= 1.2 and rect.width >= min_width:
                    lines.append((rect.y0 + rect.y1) / 2)
    lines.sort()
    merged: list[float] = []
    for y in lines:
        if merged and abs(y - merged[-1]) <= 1.0:
            merged[-1] = (merged[-1] + y) / 2
        else:
            merged.append(y)
    return [RuleBand(top, bottom) for top, bottom in zip(merged, merged[1:]) if 8.0 <= bottom - top <= 80.0]


def _band_runs(bands: list[RuleBand]) -> list[list[RuleBand]]:
    runs: list[list[RuleBand]] = []
    for band in bands:
        if runs and abs(runs[-1][-1].bottom - band.top) <= 2.0:
            runs[-1].append(band)
        else:
            runs.append([band])
    return runs


def enrich_factor_atom_context(pdf_path: Path, pages: dict[int, Any], atoms: list[FormulaAtom]) -> None:
    """Attach the factor-column label as validation context, not crop text."""
    atoms_by_page: dict[int, list[FormulaAtom]] = {}
    for atom in atoms:
        atoms_by_page.setdefault(atom.page, []).append(atom)
    with pymupdf.open(pdf_path) as document:
        for page_number, page_atoms in atoms_by_page.items():
            page = pages.get(page_number)
            if page is None:
                continue
            for run in _band_runs(horizontal_rule_bands(document[page_number - 1])):
                run_atoms = [
                    atom for atom in page_atoms if run[0].top <= atom.bbox[1] + atom.bbox[3] / 2 < run[-1].bottom
                ]
                if not run_atoms:
                    continue
                run_text = " ".join(_join_items(_band_items(page, band)) for band in run)
                is_function = "变量及函数" in run_text or not any(atom.bbox[0] < 180.0 for atom in run_atoms)
                if is_function:
                    continue
                for band in run:
                    factor = _join_items([item for item in _band_items(page, band) if 105.0 <= float(item.x) < 145.0])
                    if not factor or "小类" in factor:
                        continue
                    for atom in run_atoms:
                        center = atom.bbox[1] + atom.bbox[3] / 2
                        if not band.top < center < band.bottom:
                            continue
                        compact_label = re.sub(r"\W", "", factor).lower()
                        compact_native = re.sub(r"\W", "", compact_formula_spacing(atom.native_text)).lower()
                        formula_text = normalize_private_use(atom.native_text).strip()
                        if formula_text.startswith("=") and compact_label and compact_label not in compact_native:
                            atom.context_label = factor
            # A wide candidate can include the small-factor label itself.
            # Preserve that label so the final replacement bbox can begin at
            # the formula column instead of deleting the label cell.
            for atom in page_atoms:
                if atom.bbox[0] >= 130.0:
                    continue
                atom_center_y = atom.bbox[1] + atom.bbox[3] / 2
                labels = [
                    item
                    for item in page.text_items
                    if 105.0 <= float(item.x) < 145.0
                    and abs(_item_center(item)[1] - atom_center_y) <= max(12.0, atom.bbox[3])
                ]
                if labels:
                    nearest = min(labels, key=lambda item: abs(_item_center(item)[1] - atom_center_y))
                    atom.context_label = _join_items([nearest])


def _band_items(page: Any, band: RuleBand) -> list[Any]:
    return [item for item in page.text_items if band.top < _item_center(item)[1] < band.bottom]


def _render_factor_run(page: Any, run: list[RuleBand], atoms: list[FormulaAtom], output_path: Path) -> str:
    blocks = ["### 大类风格因子定义"]
    current_category = ""
    for band in run:
        row_items = _band_items(page, band)
        row_atoms = [atom for atom in atoms if band.top < atom.bbox[1] + atom.bbox[3] / 2 < band.bottom]
        category = _join_items([item for item in row_items if float(item.x) < 105.0])
        factor = _join_items([item for item in row_items if 105.0 <= float(item.x) < 145.0])
        if "大类" in category or "小类" in factor:
            continue
        if category:
            current_category = category
        if not factor:
            continue
        remaining = [
            item
            for item in row_items
            if float(item.x) >= 145.0 and not any(_item_in_atom(item, atom) for atom in row_atoms)
        ]
        description = _join_items(remaining)
        title = f"**{factor}**"
        if current_category and current_category != factor:
            title += f"（{current_category}）"
        parts = [title]
        parts.extend(render_atom(atom, output_path) for atom in row_atoms)
        if description:
            parts.append(compact_formula_spacing(description))
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def _render_function_run(page: Any, run: list[RuleBand], atoms: list[FormulaAtom], output_path: Path) -> str:
    blocks = ["### 变量与函数"]
    for band in run:
        row_items = _band_items(page, band)
        row_atoms = [atom for atom in atoms if band.top < atom.bbox[1] + atom.bbox[3] / 2 < band.bottom]
        label = _join_items([item for item in row_items if float(item.x) < 180.0])
        if "变量及函数" in label:
            continue
        remaining = [
            item
            for item in row_items
            if float(item.x) >= 180.0 and not any(_item_in_atom(item, atom) for atom in row_atoms)
        ]
        description = _join_items(remaining)
        if not label and not row_atoms:
            continue
        parts = [f"**{compact_formula_spacing(label)}**" if label else "**公式**"]
        if description:
            parts.append(normalize_private_use(description))
        parts.extend(render_atom(atom, output_path) for atom in row_atoms)
        blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks)


def compose_table_page(
    page: Any,
    pdf_page: pymupdf.Page,
    atoms: list[FormulaAtom],
    output_path: Path,
) -> str:
    bands = horizontal_rule_bands(pdf_page)
    if bands:
        last_rule = bands[-1].bottom
        tail_atoms = [atom for atom in atoms if last_rule < atom.bbox[1] + atom.bbox[3] / 2 < last_rule + 80.0]
        if tail_atoms:
            tail_bottom = min(
                float(page.height) - 38.0,
                max(atom.bbox[1] + atom.bbox[3] + 10.0 for atom in tail_atoms),
            )
            if tail_bottom - last_rule >= 8.0:
                bands.append(RuleBand(last_rule, tail_bottom))
    runs = _band_runs(bands)
    blocks: list[tuple[float, str]] = []
    table_ranges: list[tuple[float, float]] = []
    for run in runs:
        run_atoms = [atom for atom in atoms if run[0].top <= atom.bbox[1] + atom.bbox[3] / 2 < run[-1].bottom]
        run_text = " ".join(_join_items(_band_items(page, band)) for band in run)
        if not run_atoms and "变量及函数" not in run_text:
            continue
        is_function = "变量及函数" in run_text or not any(atom.bbox[0] < 180.0 for atom in run_atoms)
        renderer = _render_function_run if is_function else _render_factor_run
        blocks.append((run[0].top, renderer(page, run, atoms, output_path)))
        table_ranges.append((run[0].top, run[-1].bottom))

    cursor = 58.0
    page_bottom = float(page.height) - 38.0
    for start, end in sorted(table_ranges):
        if start > cursor:
            linear = compose_linear_region(page, atoms, output_path, top=cursor, bottom=start)
            if linear:
                blocks.append((cursor, linear))
        cursor = max(cursor, end)
    if cursor < page_bottom:
        linear = compose_linear_region(page, atoms, output_path, top=cursor, bottom=page_bottom)
        if linear:
            blocks.append((cursor, linear))
    blocks.sort(key=lambda block: block[0])
    return "\n\n".join(text for _, text in blocks if text.strip())


def compose_alpha_page(page: Any, atoms: list[FormulaAtom], output_path: Path) -> str:
    alpha_atoms = [atom for atom in atoms if "alpha_code_row" in atom.reasons]
    alpha_atoms.sort(key=lambda atom: atom.bbox[1])
    blocks: list[str] = []
    if alpha_atoms:
        first_label = alpha_atoms[0].native_text.splitlines()[0]
        last_label = alpha_atoms[-1].native_text.splitlines()[0]
        blocks.append(f"## 因子明细：{first_label}–{last_label}")
        for atom in alpha_atoms:
            label, _, body = atom.native_text.partition("\n")
            blocks.append(f"**{label}**\n\n```text\n{compact_formula_spacing(body)}\n```")
    other_atoms = [atom for atom in atoms if atom not in alpha_atoms]
    if alpha_atoms:
        tail_top = max(atom.bbox[1] + atom.bbox[3] for atom in alpha_atoms) + 3.0
        tail = compose_linear_region(page, other_atoms, output_path, top=tail_top)
        if tail:
            blocks.append(tail)
    return "\n\n".join(blocks)


def write_manifest(atoms: list[FormulaAtom], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for atom in atoms:
            payload = asdict(atom)
            payload["crop_path"] = str(atom.crop_path) if atom.crop_path else None
            payload["crop_sha256"] = (
                hashlib.sha256(atom.crop_path.read_bytes()).hexdigest() if atom.crop_path else None
            )
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_report(result: CPUHybridResult, atoms: list[FormulaAtom], output_path: Path) -> None:
    lines = [
        "# LiteParse FormulaAtom 混合解析报告",
        "",
        f"- LiteParse 解析：{result.parse_seconds:.3f} s",
        f"- FormulaAtom 最终 Grid 回注：{result.reproject_seconds:.3f} s",
        f"- 模型初始化：{result.model_init_seconds:.3f} s",
        f"- 模型推理：{result.model_inference_seconds:.3f} s",
        f"- 原生文本公式：{result.native_count}",
        f"- 视觉公式：{result.vision_count}",
        f"- 通过结构校验的 LaTeX：{result.accepted_latex_count}",
        f"- 矢量裁剪回退：{result.fallback_image_count}",
        f"- 扫描页跳过：{result.skipped_pages or '无'}",
        "",
        "## 视觉公式逐项结果",
    ]
    for atom in atoms:
        if atom.route != "vision":
            continue
        status = "LaTeX" if atom.accepted else "矢量图回退"
        lines.extend(["", f"### 第 {atom.page} 页 · {atom.id}", "", f"- 结果：{status}"])
        if atom.validation_flags:
            lines.append(f"- 校验：`{', '.join(atom.validation_flags)}`")
        if atom.latex:
            lines.extend(["", "```latex", atom.latex, "```"])
        if atom.crop_path:
            lines.extend(["", f"![原始矢量裁剪]({_relative_asset(atom.crop_path, result.report_path)})"])
    result.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_cpu_hybrid(
    pdf_path: Path,
    output_path: Path,
    pages: str | None = None,
    model_name: str = "PP-FormulaNet_plus-S",
    fallback_model_name: str | None = "PP-FormulaNet_plus-M",
    batch_size: int = 4,
    render_scale: float = 4.0,
    run_model: bool = True,
    formula_device: str = "auto",
    formula_server_url: str | None = None,
) -> CPUHybridResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output_path.parent / f"{output_path.stem}_assets"
    crops_dir = assets_dir / "formulas"
    assets_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as document:
        selected_pages = resolve_page_numbers(document, pages)
        skipped: dict[int, str] = {}
        for page_number in selected_pages:
            reason = scanned_page_reason(document[page_number - 1])
            if reason:
                skipped[page_number] = reason
    active_pages = [page for page in selected_pages if page not in skipped]
    target_pages = ",".join(str(page) for page in active_pages)

    start = time.perf_counter()
    parser = None
    if active_pages:
        parser = LiteParse(
            output_format="markdown",
            target_pages=target_pages,
            quiet=True,
            ocr_enabled=False,
            emit_word_boxes=True,
            image_mode="embed",
        )
        parsed = parser.parse(pdf_path)
    else:
        parsed = None
    parse_seconds = time.perf_counter() - start

    atoms: list[FormulaAtom] = []
    parsed_pages: dict[int, Any] = {}
    if parsed:
        parsed_pages = {page.page_num: page for page in parsed.pages}
        for page in parsed.pages:
            atoms.extend(atom_from_candidate(page.page_num, candidate) for candidate in page.formula_candidates)
        enrich_factor_atom_context(pdf_path, parsed_pages, atoms)
    render_formula_crops(pdf_path, atoms, crops_dir, scale=render_scale)
    model_init_seconds = model_inference_seconds = 0.0
    if run_model:
        if formula_server_url:
            model_init_seconds, model_inference_seconds = run_formula_model_http(
                atoms,
                formula_server_url,
                batch_size=batch_size,
            )
        elif formula_device in {"auto", "cpu", "gpu"}:
            model_init_seconds, model_inference_seconds = run_formula_model(
                atoms,
                model_name=model_name,
                fallback_model_name=fallback_model_name,
                batch_size=batch_size,
                device=formula_device,
            )
        else:
            raise ValueError(f"未知公式推理设备：{formula_device}")
    validate_atoms(atoms)
    reproject_start = time.perf_counter()
    final_parsed = (
        parser.parse_with_formula_atoms(
            pdf_path,
            [to_grid_atom(atom, output_path) for atom in atoms],
        )
        if parser is not None
        else None
    )
    reproject_seconds = time.perf_counter() - reproject_start

    final_pages = {page.page_num: page for page in final_parsed.pages} if final_parsed else {}
    page_markdown: list[str] = []
    for page_number in selected_pages:
        if page_number in skipped:
            reason = skipped[page_number]
            page_markdown.append(
                f"<!-- page {page_number} skipped: scanned PDF ({reason}) -->\n\n"
                f"> 第 {page_number} 页疑似扫描页；CPU 原生矢量路径已记录并跳过。"
            )
            continue
        page = final_pages[page_number]
        body = normalize_private_use(page.markdown or page.text).strip()
        page_markdown.append(f"<!-- page {page_number} -->\n\n{body}".strip())
    markdown = normalize_markdown("\n\n---\n\n".join(page_markdown).strip())
    markdown = materialize_liteparse_images(final_parsed, markdown, output_path, assets_dir)
    output_path.write_text(markdown + "\n", encoding="utf-8")

    manifest_path = assets_dir / "formula_manifest.jsonl"
    report_path = assets_dir / "cpu_report.md"
    write_manifest(atoms, manifest_path)
    vision = [atom for atom in atoms if atom.route == "vision"]
    result = CPUHybridResult(
        markdown_path=output_path,
        report_path=report_path,
        manifest_path=manifest_path,
        parse_seconds=parse_seconds,
        reproject_seconds=reproject_seconds,
        model_init_seconds=model_init_seconds,
        model_inference_seconds=model_inference_seconds,
        native_count=sum(atom.route == "native_text" for atom in atoms),
        vision_count=len(vision),
        accepted_latex_count=sum(atom.accepted for atom in vision),
        fallback_image_count=sum(not atom.accepted for atom in vision),
        skipped_pages=sorted(skipped),
    )
    write_report(result, atoms, output_path)
    return result
