# Feature status

This matrix is the maintained answer to “what is actually connected?”

| Capability | Public entry | Implementation owner | Status |
| --- | --- | --- | --- |
| Native announcement fast path | `parse auto`, `parse native-fast` | LiteParse + `facade.py` | production |
| Content-derived route explanation | `probe` / `probe_pdf()` | `probe.py` | production |
| Scanned-page detection and observable skip | all local profiles | probe + native/CPU writers | production; OCR intentionally out of scope |
| Display formula detection | `formula-cpu` | LiteParse `formula_probe.rs` | production |
| Markdown-table formula detection | `formula-cpu` | LiteParse probe/table recovery | production |
| Local vector crop rendering | `formula-cpu` | PyMuPDF | production |
| PP-FormulaNet S then M fallback | `formula-cpu` | `formula_runtime.py` | production when optional extra is installed |
| Persistent batch formula service | `serve formula-cpu` | `formula_service.py` | production on trusted network |
| Candidate structural validation | `formula-cpu` | `cpu_hybrid.py` | production |
| Pre-Grid `FormulaAtom` injection | Python API used by `formula-cpu` | patched LiteParse Rust/PyO3 | production |
| Pure Markdown pipe tables | all LiteParse profiles | `markdown_layout/tables.rs` | production; no HTML rowspan |
| Vector-image formula fallback | `formula-cpu --no-formula-model` or failed validation | `cpu_hybrid.py` | production |
| Optional DGX formula candidate source | `formula-cpu --formula-device dgx` | UniMERNet over SSH | optional |
| Full-document high-accuracy DGX path | `formula-best` | MinerU hybrid high | optional, explicit only |
| SSH timeout/keepalive/transient retry | DGX paths and `doctor --check-dgx` | `mineru_remote.py` | production |
| Canonical Markdown cleanup | all production profiles | `markdown_cleanup.py` | production |
| One-Markdown default | `auto`, `native-fast` | facade/CLI | production |
| Sparse typed-block result | `ParseResult`, `--result-json` | `contracts.py` | production |
| Stable formula evidence and crop hash | formula manifest | `cpu_hybrid.py` | production |
| OCR engine comparisons/placeholders | `experiment` group | `legacy_pipeline.py` | benchmark only |
| Tesseract/scanned OCR | none in high-level package | deliberately excluded | deferred |
| AlphaSeeker formula-to-DSL execution | AlphaSeeker adapter | downstream repository | out of parser scope |
| Lighthouse storage/index/writeback | Shadow adapters | downstream repositories | out of parser scope |

“Production” means connected to a public command/API and covered by automated
tests. It does not mean that every optional model is installed in every runtime.
