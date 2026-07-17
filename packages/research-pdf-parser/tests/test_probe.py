from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from research_pdf_parser.probe import probe_pdf


class ProbeTests(unittest.TestCase):
    def test_routes_native_text_to_fast_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "native.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Native vector announcement with enough text for routing")
            document.save(pdf)
            document.close()

            result = probe_pdf(pdf).probe

            self.assertEqual(result.recommended_profile, "native-fast")
            self.assertEqual(result.scanned_pages, ())
            self.assertGreater(result.pages[0].native_chars, 40)

    def test_routes_full_page_image_to_scanned_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "scan.pdf"
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
            pixmap.clear_with(255)
            document = pymupdf.open()
            page = document.new_page(width=600, height=800)
            page.insert_image(page.rect, stream=pixmap.tobytes("png"))
            document.save(pdf)
            document.close()

            result = probe_pdf(pdf).probe

            self.assertEqual(result.recommended_profile, "scanned-deferred")
            self.assertEqual(result.scanned_pages, (1,))
            self.assertIn("native-vector-only", result.route_reasons)


if __name__ == "__main__":
    unittest.main()
