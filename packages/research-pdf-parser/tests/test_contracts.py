from __future__ import annotations

import unittest

from research_pdf_parser.contracts import project_markdown_blocks


class ContractTests(unittest.TestCase):
    def test_single_line_display_formula_does_not_consume_following_page(self) -> None:
        markdown = "<!-- page 1 -->\n\n$$x+y$$\n\n---\n\n<!-- page 2 -->\n\nAfter.\n"

        blocks = project_markdown_blocks(markdown, "b" * 64)

        self.assertEqual([(block.kind, block.page) for block in blocks], [("formula", 1), ("paragraph", 2)])

    def test_projects_sparse_page_table_formula_and_paragraph_blocks(self) -> None:
        markdown = """<!-- page 1 -->

# Title

Body text.

| Name | Formula |
|---|---|
| A | $x+y$ |

$$
x = y
$$
"""

        blocks = project_markdown_blocks(markdown, "a" * 64)

        self.assertEqual([block.kind for block in blocks], ["heading", "paragraph", "table", "formula"])
        self.assertTrue(all(block.page == 1 for block in blocks))
        self.assertEqual(len({block.id for block in blocks}), len(blocks))
        self.assertEqual(markdown[blocks[2].markdown_start : blocks[2].markdown_end].strip(), blocks[2].text)


if __name__ == "__main__":
    unittest.main()
