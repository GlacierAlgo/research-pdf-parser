from __future__ import annotations

import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from research_pdf_parser.formula_service import (
    _service_url,
    formula_service_health,
    make_handler,
    recognize_formula_files,
)


class FakeFormulaService:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace(model_name="fake-small", device="cpu", init_seconds=1.25)

    def recognize(self, payload):
        return {
            "model": "fake-small",
            "device": "cpu",
            "init_seconds": 1.25,
            "inference_seconds": 0.5,
            "results": [
                {"id": image["id"], "latex": "x+y", "score": 0.9}
                for image in payload["images"]
            ],
        }


class FormulaServiceTests(unittest.TestCase):
    def test_rejects_non_http_service_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid formula service URL"):
            _service_url("file:///tmp/formula", "/health")

    def test_health_and_batch_recognition_contract(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeFormulaService()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            health = formula_service_health(base_url)
            with tempfile.TemporaryDirectory() as directory:
                crop = Path(directory) / "crop.png"
                crop.write_bytes(b"not-decoded-by-fake-service")
                result = recognize_formula_files(base_url, [("f1", crop)])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(health["model"], "fake-small")
        self.assertEqual(result["results"], [{"id": "f1", "latex": "x+y", "score": 0.9}])


if __name__ == "__main__":
    unittest.main()
