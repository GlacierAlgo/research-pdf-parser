# LiteParse research PDF fork

This public fork is a small, fixed extension of upstream LiteParse 2.5.0 for
native-vector research reports. It is consumed by
[`research-pdf-parser`](https://github.com/GlacierAlgo/research-pdf-parser).

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

## Versioning and upstream

Fork releases use `research-formula-vX.Y.Z` Git tags and Python local versions
such as `2.5.0+research.1`. The `upstream` remote remains
`run-llama/liteparse`; fork changes are kept as a small patch series on top of
an upstream release.

## Verification

```bash
cargo fmt --all -- --check
cargo test -p liteparse --lib --no-default-features
```

The upstream project and all retained source files remain under the Apache-2.0
license.
