"""LiteParse-style HTTP formula OCR client and reference service."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from PIL import Image

from .formula_runtime import PaddleFormulaRuntime

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def formula_endpoint(url: str) -> str:
    """Accept either an origin or the explicit /formula_ocr endpoint."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid formula OCR URL: {url!r}")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/formula_ocr"
    return urlunparse(parsed._replace(path=path, params="", fragment=""))


def _health_url(endpoint: str) -> str:
    parsed = urlparse(formula_endpoint(endpoint))
    return urlunparse(parsed._replace(path="/health", params="", query="", fragment=""))


def formula_service_health(endpoint: str, timeout: float = 3.0) -> dict[str, Any]:
    request = Request(_health_url(endpoint), headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Formula service health check failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Formula service health check returned an invalid response")
    return payload


def _multipart_request(path: Path) -> tuple[bytes, str]:
    boundary = f"research-pdf-parser-{uuid.uuid4().hex}"
    content = path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            b"Content-Type: image/png\r\n\r\n",
            content,
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="language"\r\n\r\nmath\r\n',
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _recognize_one(endpoint: str, formula_id: str, path: Path, timeout: float) -> dict[str, Any]:
    body, content_type = _multipart_request(path)
    request = Request(
        formula_endpoint(endpoint),
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Formula service returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Formula service request failed: {exc}") from exc
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise RuntimeError("Formula service returned no formula result")
    item = results[0]
    return {
        "id": formula_id,
        "latex": str(item.get("text", "")),
        "score": item.get("confidence"),
        "model": str(payload.get("model", "formula-service")),
        "device": str(payload.get("device", "unknown")),
        "inference_seconds": float(payload.get("inference_seconds", 0.0)),
    }


def recognize_formula_files(
    endpoint: str,
    files: list[tuple[str, Path]],
    *,
    batch_size: int = 4,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Dispatch formula crops concurrently to a LiteParse-style endpoint."""
    if not files:
        return {"model": "formula-service", "device": "unknown", "inference_seconds": 0.0, "results": []}
    workers = max(1, min(batch_size, len(files), 32))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        responses = list(
            executor.map(
                lambda item: _recognize_one(endpoint, item[0], item[1], timeout),
                files,
            )
        )
    return {
        "model": responses[0]["model"],
        "device": responses[0]["device"],
        "inference_seconds": sum(item["inference_seconds"] for item in responses),
        "results": [
            {"id": item["id"], "latex": item["latex"], "score": item["score"]}
            for item in responses
        ],
    }


class FormulaService:
    """Own one auto-detected local runtime behind the /formula_ocr contract."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.runtime = PaddleFormulaRuntime(model_name=model_name, device=device)
        self.lock = threading.Lock()

    def recognize_image(self, filename: str, content: bytes) -> dict[str, Any]:
        if not content:
            raise ValueError("file must not be empty")
        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
        except OSError as exc:
            raise ValueError("file must be a readable image") from exc
        suffix = Path(filename).suffix or ".png"
        with tempfile.TemporaryDirectory(prefix="research-formula-service-") as directory:
            path = Path(directory) / f"formula{suffix}"
            path.write_bytes(content)
            with self.lock:
                predictions, inference_seconds = self.runtime.predict([path], batch_size=1)
        if not predictions:
            raise RuntimeError("formula model returned no prediction")
        prediction = predictions[0]
        return {
            "results": [
                {
                    "text": prediction.latex,
                    "bbox": [0.0, 0.0, float(width), float(height)],
                    "confidence": prediction.score if prediction.score is not None else 1.0,
                }
            ],
            "model": self.runtime.model_name,
            "device": self.runtime.device,
            "inference_seconds": inference_seconds,
        }


def _multipart_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Content-Type must be multipart/form-data")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    if not message.is_multipart():
        raise ValueError("request body must be multipart/form-data")
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") != "file":
            continue
        filename = part.get_filename() or "formula.png"
        content = part.get_payload(decode=True)
        if isinstance(content, bytes):
            return filename, content
    raise ValueError("multipart request requires a file field")


def make_handler(service: FormulaService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "research-formula-service/1"

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": service.runtime.model_name,
                    "device": service.runtime.device,
                    "init_seconds": service.runtime.init_seconds,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/formula_ocr":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError(f"request body must be 1-{MAX_REQUEST_BYTES} bytes")
                filename, content = _multipart_file(
                    self.headers.get("Content-Type", ""),
                    self.rfile.read(length),
                )
                result = service.recognize_image(filename, content)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve_formula_runtime(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    model_name: str = "PP-FormulaNet_plus-S",
    device: str = "auto",
) -> None:
    """Start the reference /formula_ocr service on a trusted network."""
    service = FormulaService(model_name=model_name, device=device)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
