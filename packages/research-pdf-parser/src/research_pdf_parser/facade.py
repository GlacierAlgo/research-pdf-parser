"""One public parse entry point across fast, CPU-formula, and DGX profiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import cast

from liteparse import __version__ as liteparse_version

from .contracts import (
    ActualProfile,
    ParseResult,
    QualitySummary,
    RequestedProfile,
    project_markdown_blocks,
)
from .cpu_hybrid import parse_cpu_hybrid
from .high_accuracy import parse_best_pdf
from .mineru_remote import RemoteMineruConfig
from .native import parse_native_pdf
from .probe import probe_pdf
from .version import __version__

PROFILES = ("auto", "native-fast", "formula-cpu", "formula-best")


def formula_model_available() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


def parse_pdf(
    pdf_path: Path,
    output_path: Path | None = None,
    *,
    profile: RequestedProfile = "auto",
    pages: str | None = None,
    image_mode: str = "off",
    run_formula_model: bool | None = None,
    formula_model: str = "PP-FormulaNet_plus-S",
    fallback_formula_model: str | None = "PP-FormulaNet_plus-M",
    formula_server_url: str | None = None,
    formula_device: str = "cpu",
    batch_size: int = 4,
    render_scale: float = 4.0,
    dgx_config: RemoteMineruConfig | None = None,
) -> ParseResult:
    """Parse one native-vector PDF and return the neutral v1 result contract.

    ``auto`` never selects DGX. It uses the OCR-free probe to choose either
    ``native-fast``, ``formula-cpu``, or the observable ``scanned-deferred``
    route. Only an explicit ``formula-best`` request can use the remote worker.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {', '.join(PROFILES)}")
    if image_mode not in {"off", "embed"}:
        raise ValueError("image_mode must be 'off' or 'embed'")
    output_path = (output_path or pdf_path.with_suffix(".md")).expanduser().resolve()

    session = probe_pdf(pdf_path, pages=pages, image_mode=image_mode)
    probe = session.probe
    requested = cast(RequestedProfile, profile)
    actual = probe.recommended_profile if requested == "auto" else cast(ActualProfile, requested)
    if probe.recommended_profile == "scanned-deferred":
        actual = "scanned-deferred"

    warnings: list[str] = []
    if probe.scanned_pages:
        warnings.append("scanned-pages-skipped=" + ",".join(map(str, probe.scanned_pages)))
    if requested == "native-fast" and probe.vision_formula_candidates:
        warnings.append(
            f"native-fast-forced-with-vision-formulas={probe.vision_formula_candidates}"
        )

    artifacts: dict[str, str] = {}
    timings: dict[str, float] = {"probe": probe.probe_seconds}
    accepted_latex = 0
    image_fallbacks = 0

    if actual in {"native-fast", "scanned-deferred"}:
        native = parse_native_pdf(
            pdf_path,
            output_path,
            pages=pages,
            image_mode=image_mode,
            parsed=session.parsed,
        )
        timings["native_materialize"] = native.parse_seconds
    elif actual == "formula-cpu":
        available = formula_model_available()
        should_run_model = run_formula_model
        if should_run_model is None:
            should_run_model = bool(formula_server_url or formula_device == "dgx" or available)
        if should_run_model and formula_device == "cpu" and not formula_server_url and not available:
            raise RuntimeError(
                "formula-cpu was required but PP-FormulaNet is not installed; "
                "run `uv sync --extra formula-cpu`, configure --formula-server-url, "
                "or pass --no-formula-model for vector-image fallback."
            )
        if not should_run_model:
            warnings.append("formula-model-unavailable:using-vector-image-fallback")
        cpu = parse_cpu_hybrid(
            pdf_path,
            output_path,
            pages=pages,
            model_name=formula_model,
            fallback_model_name=fallback_formula_model,
            batch_size=batch_size,
            render_scale=render_scale,
            run_model=should_run_model,
            formula_device=formula_device,
            dgx_host=(dgx_config.host if dgx_config else "dgx-aliyun"),
            formula_server_url=formula_server_url,
        )
        timings.update(
            {
                "formula_probe": cpu.parse_seconds,
                "formula_model_init": cpu.model_init_seconds,
                "formula_inference": cpu.model_inference_seconds,
                "final_grid": cpu.reproject_seconds,
            }
        )
        accepted_latex = cpu.accepted_latex_count
        image_fallbacks = cpu.fallback_image_count
        artifacts.update(
            {
                "formula_manifest": str(cpu.manifest_path),
                "formula_report": str(cpu.report_path),
            }
        )
    elif actual == "formula-best":
        if pages:
            raise ValueError("formula-best currently processes the complete PDF; --pages is not supported")
        high = parse_best_pdf(pdf_path, output_path, config=dgx_config)
        timings["formula_best"] = high.elapsed_seconds
        if high.assets_dir.exists():
            artifacts["assets"] = str(high.assets_dir)
    else:  # pragma: no cover - ActualProfile makes this unreachable.
        raise AssertionError(f"Unhandled profile: {actual}")

    markdown = output_path.read_text(encoding="utf-8")
    blocks = project_markdown_blocks(markdown, probe.source_sha256, output_path)
    if actual == "scanned-deferred":
        status = "deferred"
    elif probe.scanned_pages or image_fallbacks:
        status = "partial"
    else:
        status = "ok"
    quality = QualitySummary(
        status=status,
        markdown_chars=len(markdown),
        block_count=len(blocks),
        formula_candidates=probe.formula_candidates,
        accepted_latex=accepted_latex,
        image_fallbacks=image_fallbacks,
        scanned_pages=probe.scanned_pages,
    )
    route_reasons = (
        probe.route_reasons
        if requested == "auto"
        else (f"requested-profile={requested}", *probe.route_reasons)
    )
    return ParseResult(
        source_path=str(pdf_path),
        source_sha256=probe.source_sha256,
        markdown_path=str(output_path),
        requested_profile=requested,
        actual_profile=actual,
        route_reasons=route_reasons,
        probe=probe,
        blocks=blocks,
        quality=quality,
        timings=timings,
        warnings=tuple(warnings),
        artifacts=artifacts,
        provenance={
            "research_pdf_parser": __version__,
            "liteparse": liteparse_version,
        },
    )
