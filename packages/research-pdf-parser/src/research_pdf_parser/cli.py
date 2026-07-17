"""Public Click command tree."""

from __future__ import annotations

import json
from pathlib import Path

import click

from . import __version__, legacy_pipeline
from .contracts import ParseResult
from .doctor import capability_dicts, inspect_capabilities
from .facade import parse_pdf
from .formula_service import serve_formula_runtime
from .mineru_remote import RemoteMineruConfig
from .probe import probe_pdf
from .recursive_help import RecursiveHelpGroup


def _write_optional_result(result: ParseResult, result_json: Path | None) -> None:
    if result_json:
        result.write_json(result_json)
        click.echo(f"result: {result_json}")


def _echo_result(result: ParseResult) -> None:
    click.echo(f"markdown: {result.markdown_path}")
    click.echo(f"route: {result.requested_profile} -> {result.actual_profile}")
    click.echo(f"reason: {', '.join(result.route_reasons)}")
    click.echo(
        "quality: {} · blocks={} · formulas={} · latex={} · image-fallbacks={}".format(
            result.quality.status,
            result.quality.block_count,
            result.quality.formula_candidates,
            result.quality.accepted_latex,
            result.quality.image_fallbacks,
        )
    )
    click.echo("timings: " + ", ".join(f"{name}={seconds:.3f}s" for name, seconds in result.timings.items()))
    for warning in result.warnings:
        click.echo(f"warning: {warning}", err=True)


def _parse_with_errors(**kwargs: object) -> ParseResult:
    try:
        return parse_pdf(**kwargs)  # type: ignore[arg-type]
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group(cls=RecursiveHelpGroup)
@click.version_option(version=__version__)
def cli() -> None:
    """Turn native-vector PDFs into human-readable Markdown."""


@cli.group("parse", cls=RecursiveHelpGroup)
def parse_group() -> None:
    """Use one explicit profile or let the content probe choose locally."""


@parse_group.command("auto")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--pages", help="1-indexed selection such as '1,4-5'.")
@click.option("--images", type=click.Choice(["off", "embed"]), default="off", show_default=True)
@click.option(
    "--formula-model/--no-formula-model",
    default=None,
    help="Auto-detect the local model by default; disabling keeps vector-image fallbacks.",
)
@click.option("--model", default="PP-FormulaNet_plus-S", show_default=True)
@click.option("--fallback-model", default="PP-FormulaNet_plus-M", show_default=True)
@click.option("--formula-server-url", envvar="RESEARCH_PDF_PARSER_FORMULA_URL")
@click.option(
    "--formula-device",
    type=click.Choice(["auto", "cpu", "gpu"]),
    default="auto",
    show_default=True,
    help="Auto uses a local Paddle GPU when available, otherwise CPU.",
)
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--result-json", type=click.Path(dir_okay=False, path_type=Path))
def parse_auto(
    pdf: Path,
    output: Path,
    pages: str | None,
    images: str,
    formula_model: bool | None,
    model: str,
    fallback_model: str,
    formula_server_url: str | None,
    formula_device: str,
    batch_size: int,
    result_json: Path | None,
) -> None:
    """Probe once; choose native-fast or formula-cpu, never a remote worker."""
    result = _parse_with_errors(
        pdf_path=pdf,
        output_path=output,
        profile="auto",
        pages=pages,
        image_mode=images,
        run_formula_model=formula_model,
        formula_model=model,
        fallback_formula_model=fallback_model or None,
        formula_server_url=formula_server_url,
        formula_device=formula_device,
        batch_size=batch_size,
    )
    _echo_result(result)
    _write_optional_result(result, result_json)


@parse_group.command("native-fast")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--pages", help="1-indexed selection such as '1,4-5'.")
@click.option(
    "--images",
    type=click.Choice(["off", "embed"]),
    default="off",
    show_default=True,
    help="The default writes exactly one Markdown file.",
)
@click.option("--result-json", type=click.Path(dir_okay=False, path_type=Path))
def native_fast(pdf: Path, output: Path, pages: str | None, images: str, result_json: Path | None) -> None:
    """One OCR-free LiteParse pass for announcements and simple reports."""
    result = _parse_with_errors(
        pdf_path=pdf,
        output_path=output,
        profile="native-fast",
        pages=pages,
        image_mode=images,
    )
    _echo_result(result)
    _write_optional_result(result, result_json)


@parse_group.command("formula-cpu")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--pages", help="1-indexed selection such as '1,4-5'.")
@click.option("--model", default="PP-FormulaNet_plus-S", show_default=True)
@click.option("--fallback-model", default="PP-FormulaNet_plus-M", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--render-scale", type=click.FloatRange(min=1.0), default=4.0, show_default=True)
@click.option(
    "--formula-device",
    type=click.Choice(["auto", "cpu", "gpu"]),
    default="auto",
    show_default=True,
    help="Auto uses a local Paddle GPU when available, otherwise CPU.",
)
@click.option(
    "--formula-server-url",
    envvar="RESEARCH_PDF_PARSER_FORMULA_URL",
    help="LiteParse-style endpoint such as http://10.0.0.8/formula_ocr.",
)
@click.option("--no-formula-model", is_flag=True, help="Use crisp vector crops for every visual formula.")
@click.option("--result-json", type=click.Path(dir_okay=False, path_type=Path))
def formula_cpu(
    pdf: Path,
    output: Path,
    pages: str | None,
    model: str,
    fallback_model: str,
    batch_size: int,
    render_scale: float,
    formula_device: str,
    formula_server_url: str | None,
    no_formula_model: bool,
    result_json: Path | None,
) -> None:
    """Validate formula candidates, inject FormulaAtoms, then rebuild Grid."""
    result = _parse_with_errors(
        pdf_path=pdf,
        output_path=output,
        profile="formula-cpu",
        pages=pages,
        run_formula_model=not no_formula_model,
        formula_model=model,
        fallback_formula_model=fallback_model or None,
        formula_server_url=formula_server_url,
        formula_device=formula_device,
        batch_size=batch_size,
        render_scale=render_scale,
    )
    _echo_result(result)
    _write_optional_result(result, result_json)


@parse_group.command("formula-best")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--gpu-host",
    required=True,
    envvar="RESEARCH_PDF_PARSER_GPU_HOST",
    help="Explicit SSH host for a remote GPU worker; no machine is assumed.",
)
@click.option(
    "--remote-uvx",
    default="~/.local/bin/uvx",
    envvar="RESEARCH_PDF_PARSER_REMOTE_UVX",
    show_default=True,
)
@click.option("--backend", type=click.Choice(["hybrid-engine", "pipeline"]), default="hybrid-engine")
@click.option("--effort", type=click.Choice(["low", "medium", "high"]), default="high")
@click.option("--keep-remote", is_flag=True)
@click.option("--result-json", type=click.Path(dir_okay=False, path_type=Path))
def formula_best(
    pdf: Path,
    output: Path,
    gpu_host: str,
    remote_uvx: str,
    backend: str,
    effort: str,
    keep_remote: bool,
    result_json: Path | None,
) -> None:
    """Explicit MinerU high-accuracy path on a detected remote GPU."""
    result = _parse_with_errors(
        pdf_path=pdf,
        output_path=output,
        profile="formula-best",
        gpu_config=RemoteMineruConfig(
            host=gpu_host,
            uvx_path=remote_uvx,
            backend=backend,
            effort=effort,
            keep_remote=keep_remote,
        ),
    )
    _echo_result(result)
    _write_optional_result(result, result_json)


@cli.command("probe")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pages", help="1-indexed selection such as '1,4-5'.")
@click.option("--json-output", is_flag=True)
def probe_command(pdf: Path, pages: str | None, json_output: bool) -> None:
    """Explain the cheap content-derived route before expensive inference."""
    try:
        probe = probe_pdf(pdf, pages=pages).probe
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(probe.to_dict(), ensure_ascii=False, indent=2))
        return
    click.echo(f"recommended: {probe.recommended_profile}")
    click.echo(f"reason: {', '.join(probe.route_reasons)}")
    click.echo(
        "page  chars  formula  vision  table  scanned  score  reasons\n"
        "----  -----  -------  ------  -----  -------  -----  -------"
    )
    for page in probe.pages:
        click.echo(
            f"{page.page:>4}  {page.native_chars:>5}  {page.formula_candidates:>7}  "
            f"{page.vision_formula_candidates:>6}  {page.table_formula_candidates:>5}  "
            f"{str(page.scanned):>7}  {page.complexity_score:>5}  {', '.join(page.reasons)}"
        )
    click.echo(f"probe: {probe.probe_seconds:.3f}s")


@cli.command("doctor")
@click.option(
    "--gpu-host",
    envvar="RESEARCH_PDF_PARSER_GPU_HOST",
    help="Optionally verify an explicit SSH GPU worker for formula-best.",
)
@click.option("--remote-uvx", default="~/.local/bin/uvx", show_default=True)
@click.option("--formula-server-url", envvar="RESEARCH_PDF_PARSER_FORMULA_URL")
@click.option("--json-output", is_flag=True)
@click.option("--strict", is_flag=True)
def doctor(
    gpu_host: str | None,
    remote_uvx: str,
    formula_server_url: str | None,
    json_output: bool,
    strict: bool,
) -> None:
    """Detect local formula hardware and optional remote GPU capabilities."""
    capabilities = inspect_capabilities(
        gpu_host=gpu_host,
        remote_uvx=remote_uvx,
        formula_server_url=formula_server_url,
    )
    if json_output:
        click.echo(json.dumps(capability_dicts(capabilities), ensure_ascii=False, indent=2))
    else:
        for capability in capabilities:
            status = "ok" if capability.available else "missing"
            required = " required" if capability.required else " optional"
            click.echo(f"{status:<7} {capability.name:<24} {required}: {capability.detail}")
    if strict and any(item.required and not item.available for item in capabilities):
        raise click.ClickException("one or more required capabilities are missing")


@cli.group("serve", cls=RecursiveHelpGroup)
def serve_group() -> None:
    """Run optional persistent inference services."""


@serve_group.command("formula")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(min=1, max=65535), default=8765, show_default=True)
@click.option("--model", default="PP-FormulaNet_plus-S", show_default=True)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "gpu"]),
    default="auto",
    show_default=True,
)
def serve_formula(host: str, port: int, model: str, device: str) -> None:
    """Serve a LiteParse-style /formula_ocr endpoint."""
    click.echo(f"loading {model} on {device}; endpoint http://{host}:{port}/formula_ocr")
    try:
        serve_formula_runtime(host=host, port=port, model_name=model, device=device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.group("experiment", cls=RecursiveHelpGroup)
def experiment_group() -> None:
    """Legacy placeholder and OCR-comparison surfaces, not product profiles."""


experiment_group.add_command(legacy_pipeline.parse, name="legacy-placeholders")
experiment_group.add_command(legacy_pipeline.fill_formulas, name="fill")
experiment_group.add_command(legacy_pipeline.benchmark_formulas, name="benchmark")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
