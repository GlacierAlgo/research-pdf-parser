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
formula validation, storage, and DGX dispatch remain outside LiteParse.

## Repository layout

- `crates/liteparse`: Rust/PDFium extraction, formula probing, table recovery,
  FormulaAtom injection and final Grid projection;
- `packages/python`: Python binding for the patched LiteParse core;
- `packages/research-pdf-parser`: CPU/DGX routing, validation, evidence output,
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
