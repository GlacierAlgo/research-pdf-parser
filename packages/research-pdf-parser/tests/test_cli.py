from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pymupdf
from click.testing import CliRunner

from research_pdf_parser.cli import cli


class CliTests(unittest.TestCase):
    def test_recursive_help_exposes_profile_tree(self) -> None:
        result = CliRunner().invoke(cli, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("● parse", result.output)
        self.assertIn("├─ native-fast", result.output)
        self.assertIn("├─ formula-cpu", result.output)
        self.assertIn("└─ legacy-placeholders", result.output)
        self.assertIn("● formula", result.output)

    def test_doctor_reports_patched_liteparse(self) -> None:
        result = CliRunner().invoke(cli, ["doctor", "--json-output", "--strict"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        liteparse = next(item for item in payload if item["name"] == "liteparse-formula-atom")
        self.assertTrue(liteparse["available"])

    def test_native_fast_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "input.pdf"
            output = root / "output.md"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Native vector research report")
            document.save(pdf)
            document.close()

            result = CliRunner().invoke(cli, ["parse", "native-fast", str(pdf), "-o", str(output)])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(output.exists())
            self.assertIn("Native vector research report", output.read_text(encoding="utf-8"))
            self.assertFalse((root / "output_assets").exists())


if __name__ == "__main__":
    unittest.main()
