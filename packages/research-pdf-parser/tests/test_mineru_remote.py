import tempfile
import unittest
from pathlib import Path

from research_pdf_parser.mineru_remote import RemoteMineruConfig, build_mineru_command, find_mineru_markdown


class RemoteMineruTests(unittest.TestCase):
    def test_builds_high_accuracy_command_with_quoted_paths(self) -> None:
        config = RemoteMineruConfig(backend="hybrid-engine", effort="high")

        command = build_mineru_command(
            "/home/user/run with space",
            "/home/user/bin/uvx",
            config,
        )

        self.assertIn("MINERU_MODEL_SOURCE=modelscope", command)
        self.assertIn("'mineru[all]'", command)
        self.assertIn("--managed-python", command)
        self.assertIn("--effort high", command)
        self.assertIn("'/home/user/run with space/input.pdf'", command)

    def test_finds_hybrid_markdown_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            markdown = output / "document" / "hybrid_txt" / "document.md"
            markdown.parent.mkdir(parents=True)
            markdown.write_text("result", encoding="utf-8")

            self.assertEqual(find_mineru_markdown(output), markdown)


if __name__ == "__main__":
    unittest.main()
