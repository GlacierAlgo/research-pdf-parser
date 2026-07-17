"""Small persistent HTTP service for amortizing CPU formula-model startup."""

from __future__ import annotations

import base64
import json
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .formula_runtime import PaddleFormulaRuntime

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _service_url(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid formula service URL: {base_url!r}")
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def formula_service_health(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    request = Request(_service_url(base_url, "/health"), headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Formula service health check failed: {exc}") from exc


def recognize_formula_files(
    base_url: str,
    files: list[tuple[str, Path]],
    *,
    batch_size: int = 4,
    timeout: float = 180.0,
) -> dict[str, Any]:
    payload = {
        "batch_size": batch_size,
        "images": [
            {
                "id": formula_id,
                "filename": path.name,
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
            for formula_id, path in files
        ],
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        _service_url(base_url, "/v1/formulas:recognize"),
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Formula service returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Formula service request failed: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        raise RuntimeError("Formula service returned an invalid response")
    return result


class FormulaService:
    """Own one model runtime and serialize access to its predictor."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.runtime = PaddleFormulaRuntime(model_name=model_name, device=device)
        self.lock = threading.Lock()

    def recognize(self, payload: dict[str, Any]) -> dict[str, Any]:
        images = payload.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("images must be a non-empty list")
        batch_size = int(payload.get("batch_size", 4))
        if not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64")

        with tempfile.TemporaryDirectory(prefix="research-formula-service-") as directory:
            root = Path(directory)
            ids: list[str] = []
            paths: list[Path] = []
            for index, item in enumerate(images):
                if not isinstance(item, dict):
                    raise ValueError("each image must be an object")
                formula_id = str(item.get("id", "")).strip()
                encoded = item.get("content_base64")
                if not formula_id or not isinstance(encoded, str):
                    raise ValueError("each image requires id and content_base64")
                try:
                    content = base64.b64decode(encoded, validate=True)
                except ValueError as exc:
                    raise ValueError(f"invalid base64 for {formula_id}") from exc
                path = root / f"{index:06d}.png"
                path.write_bytes(content)
                ids.append(formula_id)
                paths.append(path)
            with self.lock:
                predictions, inference_seconds = self.runtime.predict(paths, batch_size=batch_size)

        return {
            "model": self.runtime.model_name,
            "device": self.runtime.device,
            "init_seconds": self.runtime.init_seconds,
            "inference_seconds": inference_seconds,
            "results": [
                {"id": formula_id, "latex": prediction.latex, "score": prediction.score}
                for formula_id, prediction in zip(ids, predictions, strict=False)
            ],
        }


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
            if self.path != "/v1/formulas:recognize":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError(f"request body must be 1-{MAX_REQUEST_BYTES} bytes")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = service.recognize(payload)
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
    device: str = "cpu",
) -> None:
    """Start a trusted-network service; authentication belongs at the proxy boundary."""
    service = FormulaService(model_name=model_name, device=device)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
