# Result and storage contract

## Local and personal use

Canonical Markdown is the primary artifact. A simple native document writes one
file and no directory. Embedded images are opt-in. Formula assets are created
only when needed for evidence or a visually faithful fallback.

`ParseResult` is an in-memory, neutral envelope around that Markdown. Its sparse
blocks record type, page, Markdown character range, optional bbox/confidence,
route, versions, warnings, quality, and timings. `--result-json` is an adapter
projection for debugging or integration, not a second canonical document.

## Scale storage

The parser does not prescribe a database. A large consumer should:

1. store raw PDF bytes once under its existing object owner;
2. store canonical Markdown in compact source/period/profile partitions;
3. store sparse blocks/search rows in Parquet or the consumer's read-side store;
4. place images/crops in a global SHA-256 content-addressed store;
5. keep a small run ledger containing source identity, parser/model versions,
   route, quality, warnings, timings, and partition/CAS locations;
6. materialize per-document Markdown plus assets only for personal/export use.

This avoids millions of per-document asset directories while retaining typed
blocks and formula evidence for retrieval or repair.

## Stability

- Schema: `research-pdf-parser.result.v1`
- Probe schema: `research-pdf-parser.probe.v1`
- Block ids are stable for the same source SHA, page, kind, Markdown offset, and text.
- Local filesystem paths are artifacts, never public Lighthouse identifiers.
- OCR LaTeX is evidence, not executable factor truth.
