from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from research_pdf_parser.cpu_hybrid import (
    FormulaAtom,
    command_braced_arguments,
    compact_formula_spacing,
    enrich_factor_atom_context,
    horizontal_rule_bands,
    latex_validation_flags,
    materialize_embedded_images,
    normalize_recognized_latex,
    render_atom,
    to_grid_atom,
)


class CPUHybridTests(unittest.TestCase):
    def test_compacts_spaced_pdf_identifiers(self) -> None:
        self.assertEqual(
            compact_formula_spacing("A l p h a _ M o d e l \uf03d c o r r"),
            "Alpha_Model = corr",
        )

    def test_rejects_unbalanced_latex_and_identifier_loss(self) -> None:
        flags = latex_validation_flags("D A S T D", r"x_{")
        self.assertIn("unbalanced_braces", flags)
        self.assertIn("missing_identifier:dastd", flags)

    def test_accepts_structurally_sound_latex(self) -> None:
        self.assertEqual(latex_validation_flags("D A S T D", r"DASTD=\sqrt{x}"), [])

    def test_normalizes_spaced_model_identifiers_and_operators(self) -> None:
        latex = r"S T O M= l n( e x p(x))+i n d u s t r y"
        self.assertEqual(normalize_recognized_latex(latex), r"STOM= \ln( \exp(x))+industry")

    def test_rejects_balanced_but_exploded_scripts(self) -> None:
        latex = "STOQ=" + "_{x}" * 20
        self.assertIn("excessive_scripts", latex_validation_flags("S T O Q", latex))

    def test_rejects_balanced_dgx_layout_artifacts(self) -> None:
        self.assertIn(
            "formula_command_noise",
            latex_validation_flags("objective", r"\stackrel{\longrightarrow}{x}"),
        )
        self.assertIn(
            "excessive_style_noise",
            latex_validation_flags("Max", r"M\textbf{a}\textbf{x}\textbf{w}"),
        )
        self.assertIn(
            "unbalanced_brackets",
            latex_validation_flags("style", r"f_{s t y[_{e}}X_{style}"),
        )

    def test_requires_standalone_k_identifier_from_native_formula(self) -> None:
        flags = latex_validation_flags("beta X industry epsilon t\nK", r"X^t=\beta X+\epsilon^t")
        self.assertIn("missing_identifier:k", flags)

    def test_requires_visible_sum_lower_bound(self) -> None:
        flags = latex_validation_flags("STOQ=sum τ=1 exp(x)", r"STOQ=\sum^T\exp(x)")
        self.assertIn("missing_sum_lower_bound", flags)

    def test_balanced_command_argument_parser_handles_nested_groups(self) -> None:
        self.assertEqual(
            command_braced_arguments(r"\sqrt{252*(f_{k}/d)/\sigma(f_k)}", r"\sqrt"),
            [r"252*(f_{k}/d)/\sigma(f_k)"],
        )

    def test_rejects_ir_when_ocr_expands_sqrt_past_252(self) -> None:
        self.assertIn(
            "implausible_ir_sqrt_scope",
            latex_validation_flags("IR(f)=252*(f/d)/sigma", r"IR(f)=\sqrt{252*(f/d)/\sigma(f/d)}"),
        )
        self.assertNotIn(
            "implausible_ir_sqrt_scope",
            latex_validation_flags("IR(f)=252*(f/d)/sigma", r"IR(f)=\sqrt{252}*(f/d)/\sigma(f/d)"),
        )

    def test_normalizes_text_superscript_to_math_script(self) -> None:
        self.assertEqual(
            normalize_recognized_latex(r"X\textsuperscript{\textit{t}}"),
            r"X^{t}",
        )

    def test_rejects_formula_output_mixed_with_prose(self) -> None:
        flags = latex_validation_flags("Z(T)", r"Z(T)=\sum x；其中")
        self.assertIn("mixed_prose", flags)

    def test_rejects_single_letter_denominator_at_end(self) -> None:
        flags = latex_validation_flags("B L E V", r"BLEV=(BE+LD)\;/\;B,")
        self.assertIn("trailing_single_letter_denominator", flags)

    def test_extracts_first_model_math_block_from_trailing_prose(self) -> None:
        latex = r"$D\mathop{T O A}=\mathop{TD}/\mathop{TA}$；其中 TD 表示总负债"
        normalized = normalize_recognized_latex(latex)
        self.assertNotIn("其中", normalized)
        self.assertEqual(latex_validation_flags("D T O A", normalized), [])

    def test_rejects_spaced_words_and_known_ocr_typos(self) -> None:
        self.assertIn("spaced_ascii_word", latex_validation_flags("Alpha", "A l p h a=x"))
        self.assertIn("corr_typo", latex_validation_flags("corr", r"\mathrm{corrr}(x,y)"))
        self.assertIn("max_token_corruption", latex_validation_flags("Max", r"a x\quad w_t"))

    def test_canonicalizes_table_formula_formatting(self) -> None:
        dtoa = normalize_recognized_latex(r"D\mathop{T O A}=\mathop{T D}\mathop{/}\mathop{T A}")
        blev = normalize_recognized_latex(r"B\mathop{{L}{E}{V}}=(\mathop{{B}{E}}+\mathop{{L}{D}})/\mathop{{B}{E}}")
        self.assertEqual(dtoa, "DTOA=TD/TA")
        self.assertEqual(blev, "BLEV=(BE+LD)/BE")

    def test_native_table_formula_removes_duplicate_factor_prefix(self) -> None:
        atom = FormulaAtom(
            id="f1",
            page=1,
            bbox=(0, 0, 10, 10),
            route="native_text",
            native_text="L N C A P L N C A P \uf03d x",
            confidence=1.0,
            reasons=["ruled_table_cell"],
            accepted=True,
        )
        rendered = render_atom(atom, Path("out.md"))
        self.assertIn("LNCAP = x", rendered)
        self.assertNotIn("LNCAPLNCAP", rendered)

    def test_grid_atom_preserves_a_covered_table_label_column(self) -> None:
        atom = FormulaAtom(
            id="f1",
            page=1,
            bbox=(114.0, 50.0, 211.0, 16.0),
            route="vision",
            native_text="CETOP=P",
            confidence=0.9,
            reasons=["suspicious_linear_order"],
            context_label="CETOP",
            latex="CETOP=Cash/P",
            accepted=True,
        )
        grid = to_grid_atom(atom, Path("out.md"))
        self.assertEqual(grid.x, 145.0)
        self.assertEqual(grid.width, 180.0)
        self.assertEqual(grid.markdown, "$CETOP=Cash/P$")

    def test_grid_atom_covers_a_trailing_incomplete_formula_fragment(self) -> None:
        atom = FormulaAtom(
            id="f1",
            page=1,
            bbox=(138.0, 50.0, 98.0, 16.0),
            route="vision",
            native_text="MLEV+(",
            confidence=0.9,
            reasons=["incomplete_table_formula", "ruled_table_cell"],
            latex="MLEV=(ME+LD)/ME",
            accepted=True,
        )
        grid = to_grid_atom(atom, Path("out.md"))
        self.assertEqual(grid.width, 120.0)

    def test_enriches_a_table_label_even_when_candidate_lacks_ruled_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.pdf"
            document = pymupdf.open()
            document.new_page(width=600, height=800)
            document.save(path)
            document.close()
            atom = FormulaAtom(
                id="f1",
                page=1,
                bbox=(114.0, 760.0, 211.0, 16.0),
                route="vision",
                native_text="CETOP=P",
                confidence=0.9,
                reasons=["suspicious_linear_order"],
            )
            parsed_page = SimpleNamespace(
                text_items=[SimpleNamespace(text="CETOP", x=112.0, y=762.0, width=23.0, height=9.0)]
            )
            enrich_factor_atom_context(path, {1: parsed_page}, [atom])
            self.assertEqual(atom.context_label, "CETOP")

    def test_horizontal_rules_form_bands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.pdf"
            document = pymupdf.open()
            page = document.new_page(width=600, height=800)
            page.draw_line((50, 100), (550, 100))
            page.draw_line((50, 140), (550, 140))
            page.draw_line((50, 200), (550, 200))
            document.save(path)
            document.close()
            with pymupdf.open(path) as reopened:
                bands = horizontal_rule_bands(reopened[0])
        self.assertEqual([(round(b.top), round(b.bottom)) for b in bands], [(100, 140), (140, 200)])

    def test_materializes_and_restores_page_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out.md"
            parsed = SimpleNamespace(images=[SimpleNamespace(id="p2_0", page=2, format="png", bytes=b"image")])
            markdown = "<!-- page 2 -->\n\n正文"
            rendered = materialize_embedded_images(parsed, markdown, output, root / "out_assets")
            self.assertIn("out_assets/images/image_p2_0.png", rendered)
            self.assertEqual((root / "out_assets/images/image_p2_0.png").read_bytes(), b"image")


if __name__ == "__main__":
    unittest.main()
