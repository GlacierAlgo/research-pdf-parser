# LiteParse research PDF fork

This repository is a fixed extension of upstream LiteParse 2.5.0 plus the
high-level `research-pdf-parser` package for native-vector research reports.

## Fork surface

- probes native vector layout for display-formula candidates before Grid
  projection;
- accepts caller-validated `FormulaAtom` values and injects them before the
  final Grid and Markdown classification pass;
- reconstructs formula-bearing Markdown pipe tables, including horizontally
  ruled tables without vertical borders;
- exposes the probe and atom API through the Python binding;
- builds the Python extension without bundled Tesseract by default.

The supported path is intentionally narrow: native-vector PDFs only. Scanned
pages are detected, logged, and skipped by the outer parser. OCR model choice,
formula validation, storage, and accelerator dispatch remain outside LiteParse.

## High-level product surface

`packages/research-pdf-parser` exposes one `parse_pdf(..., profile="auto")`
facade and the matching Click command tree. An OCR-free content probe selects
`native-fast`, `formula-cpu`, or `scanned-deferred`; `auto` never selects a
remote worker. Local formula inference detects a usable GPU and otherwise uses CPU.
`formula-best` is an explicit remote high-accuracy request.

The formula layer can run PP-FormulaNet on an auto-detected local device or call
a machine-neutral `POST /formula_ocr` HTTP service. It tries the small model first, sends structurally
suspect candidates to the medium fallback, validates the result, and only then
creates caller-trusted `FormulaAtom` values. Failed validation uses the vector
crop instead of inserting speculative LaTeX.

Canonical Markdown remains the default artifact. The optional
`research-pdf-parser.result.v1` envelope adds route reasons, warnings, timings,
quality, provenance, and sparse typed Markdown spans for downstream adapters.

## Repository layout

- `crates/liteparse`: Rust/PDFium extraction, formula probing, table recovery,
  FormulaAtom injection and final Grid projection;
- `packages/python`: Python binding for the patched LiteParse core;
- `packages/research-pdf-parser`: CPU/GPU routing, validation, evidence output,
  Click CLI and downstream-facing package.

## Versioning and upstream

Unified releases use one `vX.Y.Z` Git tag for the Rust patch, Python binding and
high-level package. LiteParse Python builds use local versions such as
`2.5.0+research.1`. The `upstream` remote remains
`run-llama/liteparse`; fork changes are kept as a small patch series on top of
an upstream release.

## Verification

```bash
cargo fmt --all -- --check
cargo test -p liteparse --lib --no-default-features
cd packages/research-pdf-parser
uv sync --group dev
uv run pytest
```

The upstream project and all retained source files remain under the Apache-2.0
license.
