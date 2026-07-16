---
title: Browser Usage (WASM)
description: Run LiteParse entirely in the browser with the WASM package.
sidebar:
  order: 8
---

LiteParse ships a WebAssembly package that runs entirely in the browser — no server, no cloud calls. It supports PDF parsing and custom OCR engines implemented in JavaScript.

## Install

```bash
npm install @llamaindex/liteparse-wasm
```

## Quick start

```typescript
import init, { LiteParse } from "@llamaindex/liteparse-wasm";

// Load the WASM module
await init();

const parser = new LiteParse({
  ocrEnabled: false,
  outputFormat: "json",
});

// data is a Uint8Array (e.g. from <input type="file"> or fetch)
const bytes = new Uint8Array(await file.arrayBuffer());
const result = await parser.parse(bytes);

console.log(result.text);
console.log(result.pages[0]);
```

## What works in the browser

- **PDF parsing** from `Uint8Array` input (use `file.arrayBuffer()` to get bytes from a file picker for example)
- **Custom OCR** via the `ocrEngine` callback interface (see below)
- **Text and JSON output formats**

## What doesn't work

- **File path input** — pass `Uint8Array` instead
- **DOCX/XLSX/PPTX/image conversion** — requires LibreOffice/ImageMagick which aren't available in the browser
- **Built-in Tesseract or HTTP OCR** — use the custom `ocrEngine` interface instead
- **Screenshots** — not available in the WASM build

## OCR in the browser

The native Tesseract and HTTP OCR backends are not available in WASM. To use OCR, pass a custom `ocrEngine` object with a `recognize` method:

```typescript
const parser = new LiteParse({
  ocrEnabled: true,
  ocrLanguage: "eng",
  ocrEngine: {
    /**
     * @param imageData PNG-encoded image bytes
     * @param width  rendered page width in pixels
     * @param height rendered page height in pixels
     * @param language e.g. "eng"
     * @returns array of { text, bbox: [x1, y1, x2, y2], confidence }
     */
    async recognize(imageData, width, height, language) {
      // e.g. call a Web Worker wrapping tesseract.js, or a remote OCR service
      return [
        { text: "Hello", bbox: [10, 20, 80, 40], confidence: 0.98 },
      ];
    },
  },
});
```

This lets you plug in any OCR implementation — a Web Worker running tesseract.js, a cloud OCR API, or anything else that returns text with bounding boxes.

## Config options

All optional, camelCase:

| Option | Type | Default | Description |
|---|---|---|---|
| `ocrLanguage` | `string` | `"eng"` | Language code passed to the OCR engine |
| `ocrEnabled` | `boolean` | `true` | Run OCR on text-sparse pages |
| `maxPages` | `number` | `1000` | Stop after this many pages |
| `targetPages` | `string` | — | e.g. `"1-5,10,15-20"` |
| `dpi` | `number` | `150` | Render DPI for OCR |
| `outputFormat` | `"json" \| "text"` | `"json"` | Format used by `parser.format(...)` |
| `preserveVerySmallText` | `boolean` | `false` | Keep tiny text that's normally filtered |
| `password` | `string` | — | Password for protected PDFs |
| `quiet` | `boolean` | `false` | Suppress progress logging |
| `ocrEngine` | `object` | — | Custom JS-side OCR engine (see above) |
