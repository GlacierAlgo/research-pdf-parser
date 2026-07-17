"""Public Click command tree."""

from __future__ import annotations

import json
from pathlib import Path

import click

from . import __version__, legacy_pipeline
from .doctor import capability_dicts, inspect_capabilities
from .native import parse_native_pdf
from .recursive_help import RecursiveHelpGroup


@click.group(cls=RecursiveHelpGroup)
@click.version_option(version=__version__)
def cli() -> None:
    """Turn native-vector PDFs into human-readable Markdown."""


@cli.group("parse", cls=RecursiveHelpGroup)
def parse_group() -> None:
    """Choose a parsing profile by document complexity and compute budget."""


@parse_group.command("native-fast")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--pages", help="1-indexed selection such as '1,4-5'.")
@click.option(
    "--images",
    type=click.Choice(["off", "embed"]),
    default="off",
    show_default=True,
    help="Keep one Markdown file by default; embed writes referenced images to _assets.",
)
def native_fast(pdf: Path, output: Path, pages: str | None, images: str) -> None:
    """Single-pass LiteParse for announcements and formula-light PDFs."""
    try:
        result = parse_native_pdf(pdf, output, pages=pages, image_mode=images)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"markdown: {result.markdown_path}")
    click.echo(f"LiteParse: {result.parse_seconds:.3f}s; pages={result.page_count}")
    if result.skipped_pages:
        click.echo(f"skipped scanned pages: {result.skipped_pages}", err=True)


@cli.group("formula", cls=RecursiveHelpGroup)
def formula_group() -> None:
    """Legacy formula experiments retained for comparison and migration."""


@cli.command("doctor")
@click.option("--check-dgx", is_flag=True, help="Also verify SSH connectivity to the DGX worker.")
@click.option("--dgx-host", default="dgx-aliyun", show_default=True)
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--strict", is_flag=True, help="Exit non-zero when a required capability is missing.")
def doctor(check_dgx: bool, dgx_host: str, json_output: bool, strict: bool) -> None:
    """Show installed parser, model, and optional DGX capabilities."""
    capabilities = inspect_capabilities(check_dgx=check_dgx, dgx_host=dgx_host)
    if json_output:
        click.echo(json.dumps(capability_dicts(capabilities), ensure_ascii=False, indent=2))
    else:
        for capability in capabilities:
            status = "ok" if capability.available else "missing"
            required = " required" if capability.required else " optional"
            click.echo(f"{status:<7} {capability.name:<24} {required}: {capability.detail}")
    if strict and any(item.required and not item.available for item in capabilities):
        raise click.ClickException("one or more required capabilities are missing")


parse_group.add_command(legacy_pipeline.parse_cpu, name="formula-cpu")
parse_group.add_command(legacy_pipeline.parse_best, name="formula-best")
parse_group.add_command(legacy_pipeline.parse, name="legacy-placeholders")
formula_group.add_command(legacy_pipeline.fill_formulas, name="fill")
formula_group.add_command(legacy_pipeline.benchmark_formulas, name="benchmark")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
