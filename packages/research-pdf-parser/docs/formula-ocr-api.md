# Formula OCR HTTP API

Remote formula inference is a service capability, not a machine identity. The
parser does not assume a hardware product class, model, hostname, or GPU count. It sends each
formula crop to an explicitly configured endpoint such as:

```text
http://10.0.0.8/formula_ocr
```

## Recognize one formula

```http
POST /formula_ocr
Content-Type: multipart/form-data
```

| Field | Type | Required | Meaning |
| --- | --- | ---: | --- |
| `file` | PNG/JPEG bytes | yes | one tightly cropped formula region |
| `language` | string | no | client sends `math`; servers may ignore it |

The response intentionally follows the LiteParse HTTP OCR shape:

```json
{
  "results": [
    {
      "text": "HSIGMA=STD(e_i)",
      "bbox": [0, 0, 320, 96],
      "confidence": 0.97
    }
  ],
  "model": "PP-FormulaNet_plus-S",
  "device": "gpu:0",
  "inference_seconds": 0.042
}
```

For this endpoint, `text` is LaTeX and `bbox` normally covers the complete crop.
`model`, `device`, and `inference_seconds` are optional diagnostics. The parser
still validates LaTeX against native PDF glyphs and structure before creating a
`FormulaAtom`; a successful HTTP response is not automatically trusted.

The client sends up to `--batch-size` requests concurrently. The server owns
GPU discovery, model residency, queueing, micro-batching, overload control, and
admission. The same endpoint may therefore be backed by PP-FormulaNet,
UniMERNet, another formula model, CPU, one GPU, or a GPU pool.

## Health

```http
GET /health
```

```json
{
  "status": "ok",
  "model": "PP-FormulaNet_plus-S",
  "device": "gpu:0",
  "init_seconds": 8.1
}
```

The included reference server implements both endpoints:

```bash
uv run research-pdf serve formula --device auto
```

It is a trusted-network reference implementation, not an internet-facing auth
gateway. Put authentication, TLS, rate limiting, and multi-tenant quotas at a
reverse proxy or worker platform boundary.
