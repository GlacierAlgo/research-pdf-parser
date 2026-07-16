---
title: CLI Reference
description: Complete reference for all LiteParse CLI commands and options.
sidebar:
  order: 5
---

LiteParse provides the `lit` CLI with these commands: `parse`, `batch-parse`, `screenshot`, `is-complex`, `extract`, and `models`. The CLI is the same whether installed via `npm`, `pip`, or built from Rust source.

## `lit parse`

Parse a single document.

```
lit parse [options] <file>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `file` | Path to the document file, or `-` to read from stdin |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output <file>` | Write output to a file instead of stdout | — |
| `--format <format>` | Output format: `json`, `text`, or `markdown` | `text` |
| `--image-mode <mode>` | Markdown image handling: `off`, `placeholder`, or `embed` | `placeholder` |
| `--image-output-dir <dir>` | Directory to write images to when `--image-mode embed` | — |
| `--no-links` | Emit link anchor text as plain text (no `[text](url)`) in markdown | — |
| `--no-ocr` | Disable OCR entirely | — |
| `--ocr-language <lang>` | OCR language code (Tesseract format) | `eng` |
| `--ocr-server-url <url>` | HTTP OCR server URL | — (uses Tesseract) |
| `--tessdata-path <path>` | Path to tessdata directory | — (uses `TESSDATA_PREFIX` env var) |
| `--num-workers <n>` | Pages to OCR in parallel | CPU cores - 1 |
| `--max-pages <n>` | Maximum pages to parse | `1000` |
| `--target-pages <pages>` | Pages to parse (e.g., `"1-5,10"`) | — (all pages) |
| `--dpi <dpi>` | Rendering DPI | `150` |
| `--preserve-small-text` | Keep very small text | — |
| `--password <password>` | Password for encrypted/protected documents | — |
| `-q, --quiet` | Suppress progress output | — |

### Examples

```bash
# Basic text parsing
lit parse report.pdf

# JSON output with bounding boxes
lit parse report.pdf --format json -o report.json

# Markdown output (headings, tables, lists, images, links)
lit parse report.pdf --format markdown -o report.md

# Markdown with embedded images written to disk
lit parse report.pdf --format markdown --image-mode embed --image-output-dir ./images

# Parse pages 1-5 only, no OCR
lit parse report.pdf --target-pages "1-5" --no-ocr

# High-DPI rendering with French OCR
lit parse report.pdf --dpi 300 --ocr-language fra

# Use an external OCR server
lit parse report.pdf --ocr-server-url http://localhost:8828/ocr

# Pipe output to another tool
lit parse report.pdf -q | wc -l

# Parse a remote file via stdin
curl -sL https://example.com/report.pdf | lit parse --no-ocr -
```

---

## `lit batch-parse`

Parse multiple documents in a directory.

```
lit batch-parse [options] <input-dir> <output-dir>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `input-dir` | Directory containing documents to parse |
| `output-dir` | Directory for output files |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--format <format>` | Output format: `json`, `text`, or `markdown` | `text` |
| `--no-ocr` | Disable OCR entirely | — |
| `--ocr-language <lang>` | OCR language code | `eng` |
| `--ocr-server-url <url>` | HTTP OCR server URL | — (uses Tesseract) |
| `--tessdata-path <path>` | Path to tessdata directory | — |
| `--num-workers <n>` | Pages to OCR in parallel | CPU cores - 1 |
| `--max-pages <n>` | Maximum pages per file | `1000` |
| `--dpi <dpi>` | Rendering DPI | `150` |
| `--recursive` | Search subdirectories | — |
| `--extension <ext>` | Only process this extension (e.g., `".pdf"`) | — (all supported) |
| `--password <password>` | Password for encrypted/protected documents (applied to all files) | — |
| `-q, --quiet` | Suppress progress output | — |

### Examples

```bash
# Parse all supported files in a directory
lit batch-parse ./documents ./output

# Recursively parse only PDFs
lit batch-parse ./documents ./output --recursive --extension ".pdf"

# Batch parse with JSON output and no OCR
lit batch-parse ./documents ./output --format json --no-ocr
```

---

## `lit screenshot`

Generate page images from a document (PDF, DOCX, XLSX, images, etc.).

```
lit screenshot [options] <file>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `file` | Path to the document file |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir <dir>` | Output directory | `./screenshots` |
| `--target-pages <pages>` | Pages to screenshot (e.g., `"1,3,5"` or `"1-5"`) | — (all pages) |
| `--dpi <dpi>` | Rendering DPI | `150` |
| `--password <password>` | Password for encrypted/protected documents | — |
| `-q, --quiet` | Suppress progress output | — |

### Examples

```bash
# Screenshot all pages of a PDF
lit screenshot document.pdf -o ./pages

# Screenshot a Word document (requires LibreOffice)
lit screenshot report.docx -o ./pages

# First 5 pages at high DPI
lit screenshot document.pdf --target-pages "1-5" --dpi 300 -o ./pages

# Specific pages only
lit screenshot document.pdf --target-pages "1,5,10" -o ./pages
```

---

## `lit is-complex`

Check whether a document needs OCR or heavier parsing — a cheap pre-parse pass over the text layer only (no rasterization, no OCR). See the [Document Complexity guide](/liteparse/guides/complexity/) for details.

```
lit is-complex [options] <file>
```

The command prints per-page JSON to **stdout**, a human-readable verdict to **stderr**, and exits **non-zero when any page needs OCR** — so it works as a shell predicate or a `jq` source.

### Arguments

| Argument | Description |
|----------|-------------|
| `file` | Path to the document file |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--compact` | Emit dense, whitespace-free JSON instead of pretty-printed | — |
| `--max-pages <n>` | Maximum pages to check | `1000` |
| `--target-pages <pages>` | Pages to check (e.g., `"1-5,10,15-20"`) | — (all pages) |
| `--password <password>` | Password for encrypted/protected documents | — |
| `-q, --quiet` | Suppress the stderr verdict | — |

### Examples

```bash
# Print the complexity verdict and per-page JSON
lit is-complex document.pdf

# Use as a shell predicate: only parse with --no-ocr when simple
lit is-complex document.pdf --quiet && lit parse document.pdf --no-ocr

# List the page numbers that need OCR
lit is-complex document.pdf --compact | jq '[.[] | select(.needs_ocr) | .page_number]'
```

---

## `lit extract`

Pull specific fields out of a document from a JSON Schema — ranked candidate spans with scores, pages, and bounding boxes. See the [Schema Extraction guide](/liteparse/guides/schema-extraction/) for the output shape.

```
lit extract [options] <file> --schema <schema>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `file` | Path to the document file |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--schema <schema>` | JSON Schema — a file path, or inline JSON starting with `{`. **Required.** | — |
| `-o, --output <file>` | Write the result JSON to a file instead of stdout | — |
| `--compact` | Compact JSON instead of pretty-printed | — |
| `--top-k <n>` | Candidate spans returned per field | `5` |
| `--model <id>` | Static embedding model (Hugging Face id) | `minishlab/potion-retrieval-32M` |
| `--model-path <dir>` | Explicit local model directory (overrides `--model`) | — |
| `--offline` | Never download the model; use a cached one if present, else BM25-only | — |
| `--no-ocr` | Disable OCR | — |
| `--max-pages <n>` | Maximum pages to parse | `1000` |
| `--target-pages <pages>` | Pages to parse (e.g., `"1-5,10"`) | — (all pages) |
| `--password <password>` | Password for encrypted/protected documents | — |
| `-q, --quiet` | Suppress progress logging on stderr | — |

### Examples

```bash
# Inline schema
lit extract invoice.pdf --schema '{"type":"object","properties":{"total":{"type":"number","description":"grand total amount due"}}}'

# Schema from a file, result written to disk
lit extract invoice.pdf --schema schema.json --output result.json

# Run without touching the network (BM25-only if the model isn't cached)
lit extract invoice.pdf --schema schema.json --offline
```

---

## `lit models`

Manage the embedding model used by `lit extract`.

```
lit models pull [model]
```

`lit models pull` downloads an extraction embedding model to the liteparse cache (idempotent — it resolves an existing local copy first) and prints the model directory. The default model is ~250 MB and lands in the platform cache directory; set `LITEPARSE_MODELS_DIR` to override.

### Examples

```bash
# Pre-fetch the default model (e.g. in CI or a Docker build)
lit models pull

# Pull a specific model
lit models pull minishlab/potion-retrieval-32M
```

---

## Global options

These options are available on all commands:

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help for a command |
| `-V, --version` | Show version number |
