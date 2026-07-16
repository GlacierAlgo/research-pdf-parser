---
title: What is LiteParse?
description: Fast, local PDF parsing with spatial text parsing, OCR, and bounding boxes.
sidebar:
  order: 0
---

LiteParse is an open-source document parsing library that parses text with spatial layout information and bounding boxes. Written in Rust for speed and reliability, it runs entirely on your machine with no cloud dependencies, no LLMs, and no API keys.

LiteParse is designed specifically for use cases that require fast, accurate text parsing: real-time applications, coding agents, and local workflows. It provides a simple CLI and library API for parsing PDFs, Office documents, and images, with built-in OCR support.

<img src="https://github.com/user-attachments/assets/07ba6a82-6bb1-4dea-b0ef-cad7df7d1622" alt="out">

## What can LiteParse do?

- **Parse PDFs** with precise spatial layout. Text comes back positioned where it appears on the page
- **Render to Markdown** with headings, tables, lists, images, and links — clean structured output for LLMs and RAG pipelines
- **Extract schema fields** — pull specific fields (invoice number, totals, line items) out of a document from a JSON Schema, with scores, pages, and bounding boxes, and no LLM
- **Extract bounding boxes** for every text line, ready for downstream processing or visualization
- **OCR scanned documents** using built-in Tesseract or plug in your own OCR server
- **Parse Office files and images** with support for DOCX, XLSX, PPTX, PNG, JPG, and more via automatic conversion
- **Screenshot PDF pages** as high-quality images for LLM-based workflows
- **Use from Node.js/TypeScript, Python, Rust, or the browser (WASM)** — whatever fits your stack

## Get started

- [Getting started](/liteparse/getting_started/): Install LiteParse and parse your first document.
- [Markdown output](/liteparse/guides/markdown/): Render documents to clean, structured Markdown.
- [Schema extraction](/liteparse/guides/schema-extraction/): Pull specific fields out of documents from a JSON Schema.
- [Library usage](/liteparse/guides/library-usage/): Use LiteParse from TypeScript or Python code.
- [Browser usage (WASM)](/liteparse/guides/browser-usage/): Run LiteParse in the browser with zero server dependencies.
- [CLI reference](/liteparse/cli-reference/): Complete command and option reference.
- [API reference](/liteparse/api/): Detailed API documentation (rust) for all public types and functions. The same types apply across all language bindings.
