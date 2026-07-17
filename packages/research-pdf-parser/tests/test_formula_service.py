from __future__ import annotations

import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from research_pdf_parser.formula_service import (
    formula_endpoint,
    formula_service_health,
    make_handler,
    recognize_formula_files,
)


class FakeFormulaService:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace(model_name="fake-small", device="cpu", init_seconds=1.25)

    def recognize_image(self, filename: str, content: bytes):
        assert filename == "crop.png"
        assert content == b"formula-image"
        return {
            "model": "fake-small",
            "device": "cpu",
            "init_seconds": 1.25,
            "inference_seconds": 0.5,
            "results": [
                {"text": "x+y", "bbox": [0, 0, 32, 16], "confidence": 0.9}
            ],
        }


class FormulaServiceTests(unittest.TestCase):
    def test_rejects_non_http_service_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid formula OCR URL"):
            formula_endpoint("file:///tmp/formula")

    def test_origin_defaults_to_formula_ocr_endpoint(self) -> None:
        self.assertEqual(
            formula_endpoint("http://10.0.0.8"),
            "http://10.0.0.8/formula_ocr",
        )

    def test_health_and_batch_recognition_contract(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeFormulaService()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            health = formula_service_health(base_url)
            with tempfile.TemporaryDirectory() as directory:
                crop = Path(directory) / "crop.png"
                crop.write_bytes(b"formula-image")
                result = recognize_formula_files(base_url, [("f1", crop)])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(health["model"], "fake-small")
        self.assertEqual(result["results"], [{"id": "f1", "latex": "x+y", "score": 0.9}])
        self.assertEqual(result["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
