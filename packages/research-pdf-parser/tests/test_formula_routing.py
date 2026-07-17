import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research_pdf_parser.formula_dispatch import FormulaDispatchError, FormulaProcessorRegistry
from research_pdf_parser.legacy_pipeline import (
    CROP_STRATEGIES,
    FormulaCandidate,
    MarkdownImage,
    TextChar,
    apply_formula_ocr,
    attach_best_results_to_records,
    fill_formula_placeholders,
    formula_confidence,
    group_preview_text,
    insert_formula_placeholders,
    is_formula_line,
    is_standalone_formula_line,
    latex_quality_flags,
    latex_quality_score,
    match_paddleocr_page_results,
    materialize_markdown_images,
    normalize_formula_crop_image,
    normalize_private_use_text,
    parse_smart,
    read_formula_manifest,
    rebundle_markdown_images,
    resolve_page_indexes,
    select_best_benchmark_results,
    should_externalize_formula,
    summarize_benchmark_results,
    write_formula_manifest_records,
)
from research_pdf_parser.markdown_cleanup import (
    clean_extraction_markers,
    compact_markdown_blank_lines,
    normalize_table_of_contents,
    prefer_table_of_contents,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "基于短周期价量特征的多因子选股体系.pdf"


def make_group(*chars: tuple[str, float, float, float, float, float, str]) -> list[TextChar]:
    return [TextChar(*char) for char in chars]


def main_formula(formula_id: str, preview_text: str) -> FormulaCandidate:
    return FormulaCandidate(
        id=formula_id,
        page=1,
        bbox=(0.0, 0.0, 1.0, 1.0),
        preview_text=preview_text,
        confidence=1.0,
    )


class FormulaRoutingTests(unittest.TestCase):
    def test_preview_text_preserves_visible_order(self) -> None:
        group = make_group(
            ("R", 10, 10, 15, 20, 10, "Times New Roman,Italic"),
            ("=", 18, 10, 22, 20, 10, "Symbol"),
            ("1", 25, 14, 28, 18, 6, "Times New Roman"),
        )
        self.assertEqual(group_preview_text(group), "R=1")

    def test_routes_obvious_formula_group(self) -> None:
        group = make_group(
            ("A", 10, 10, 14, 20, 10, "Times New Roman,Italic"),
            ("l", 15, 10, 17, 20, 10, "Times New Roman,Italic"),
            ("p", 18, 10, 22, 20, 10, "Times New Roman,Italic"),
            ("h", 23, 10, 27, 20, 10, "Times New Roman,Italic"),
            ("a", 28, 10, 32, 20, 10, "Times New Roman,Italic"),
            ("=", 40, 10, 44, 20, 10, "Symbol"),
            ("β", 50, 10, 55, 20, 10, "Symbol"),
            ("1", 56, 14, 59, 18, 6, "Times New Roman"),
        )
        self.assertTrue(should_externalize_formula(group))
        self.assertGreaterEqual(formula_confidence(group), 0.45)

    def test_does_not_route_contact_or_dense_cjk_text(self) -> None:
        contact = make_group(
            ("电", 10, 10, 20, 20, 10, "KaiTi"),
            ("话", 21, 10, 31, 20, 10, "KaiTi"),
            ("：", 32, 10, 36, 20, 10, "KaiTi"),
            ("0", 38, 10, 42, 20, 10, "Times New Roman"),
            ("2", 43, 10, 47, 20, 10, "Times New Roman"),
            ("1", 48, 10, 52, 20, 10, "Times New Roman"),
        )
        prose = make_group(
            ("即", 10, 10, 20, 20, 10, "KaiTi"),
            ("，", 21, 10, 31, 20, 10, "KaiTi"),
            ("股", 32, 10, 42, 20, 10, "KaiTi"),
            ("票", 43, 10, 53, 20, 10, "KaiTi"),
            ("收", 54, 10, 64, 20, 10, "KaiTi"),
            ("益", 65, 10, 75, 20, 10, "KaiTi"),
            ("率", 76, 10, 86, 20, 10, "KaiTi"),
            ("=", 90, 10, 94, 20, 10, "Symbol"),
            ("风", 98, 10, 108, 20, 10, "KaiTi"),
            ("格", 109, 10, 119, 20, 10, "KaiTi"),
        )
        self.assertFalse(should_externalize_formula(contact))
        self.assertFalse(should_externalize_formula(prose))

    def test_does_not_route_inline_formula_inside_prose(self) -> None:
        cjk_inline = make_group(
            ("我", 72, 10, 82, 20, 10, "KaiTi"),
            ("们", 83, 10, 93, 20, 10, "KaiTi"),
            ("使", 94, 10, 104, 20, 10, "KaiTi"),
            ("用", 105, 10, 115, 20, 10, "KaiTi"),
            ("I", 120, 10, 124, 20, 10, "Times New Roman,Italic"),
            ("C", 125, 10, 132, 20, 10, "Times New Roman,Italic"),
            ("=", 136, 10, 140, 20, 10, "Symbol"),
            ("c", 144, 10, 148, 20, 10, "Times New Roman,Italic"),
            ("o", 149, 10, 154, 20, 10, "Times New Roman,Italic"),
            ("r", 155, 10, 158, 20, 10, "Times New Roman,Italic"),
            ("r", 159, 10, 162, 20, 10, "Times New Roman,Italic"),
            ("衡", 168, 10, 178, 20, 10, "KaiTi"),
            ("量", 179, 10, 189, 20, 10, "KaiTi"),
        )
        english_inline = make_group(
            ("w", 72, 10, 80, 20, 10, "Times New Roman"),
            ("h", 81, 10, 88, 20, 10, "Times New Roman"),
            ("e", 89, 10, 95, 20, 10, "Times New Roman"),
            ("r", 96, 10, 100, 20, 10, "Times New Roman"),
            ("e", 101, 10, 107, 20, 10, "Times New Roman"),
            ("I", 116, 10, 120, 20, 10, "Times New Roman,Italic"),
            ("C", 121, 10, 128, 20, 10, "Times New Roman,Italic"),
            ("=", 132, 10, 136, 20, 10, "Symbol"),
            ("c", 140, 10, 144, 20, 10, "Times New Roman,Italic"),
            ("o", 145, 10, 150, 20, 10, "Times New Roman,Italic"),
            ("r", 151, 10, 154, 20, 10, "Times New Roman,Italic"),
            ("r", 155, 10, 158, 20, 10, "Times New Roman,Italic"),
            ("i", 168, 10, 171, 20, 10, "Times New Roman"),
            ("s", 172, 10, 177, 20, 10, "Times New Roman"),
        )

        self.assertFalse(is_standalone_formula_line(cjk_inline))
        self.assertFalse(should_externalize_formula(cjk_inline))
        self.assertFalse(is_standalone_formula_line(english_inline))
        self.assertFalse(should_externalize_formula(english_inline))

    def test_replaces_formula_runs_in_document_order_without_removing_emphasis(self) -> None:
        formulas = [
            main_formula("formula_p0001_001", "AlphaModel=corr(x,y)"),
            main_formula("formula_p0001_002", "T=IC/(sigma/sqrt(n))"),
        ]
        markdown = "\n".join(
            [
                "**Important**",
                "",
                "AlphaModel = corr(x, y)",
                "",
                "middle prose",
                "",
                "T = IC / (sigma / sqrt(n))",
                "",
                "*Summary*",
            ]
        )

        rendered = insert_formula_placeholders(markdown, formulas)

        self.assertIn("**Important**", rendered)
        self.assertIn("*Summary*", rendered)
        self.assertNotIn("AlphaModel = corr", rendered)
        self.assertNotIn("T = IC", rendered)
        first = rendered.index("{{formula:formula_p0001_001}}")
        middle = rendered.index("middle prose")
        second = rendered.index("{{formula:formula_p0001_002}}")
        self.assertLess(first, middle)
        self.assertLess(middle, second)

    def test_short_markdown_emphasis_is_not_a_formula_line(self) -> None:
        self.assertFalse(is_formula_line("**Important**"))
        self.assertFalse(is_formula_line("*Summary*"))

    def test_normalizes_confirmed_symbol_private_use_characters(self) -> None:
        text = "*t* \uf02b1, \uf065, \uf073, \uf03d, \uf02d, \uf0ec\uf0ed\uf0ef\uf0ee"
        replacements = {
            0xF02B: "+",
            0xF065: "ε",
            0xF073: "σ",
            0xF03D: "=",
            0xF02D: "−",
            0xF0EC: "⎧",
            0xF0ED: "⎨",
            0xF0EF: "⎪",
            0xF0EE: "⎩",
        }

        self.assertEqual(normalize_private_use_text(text, replacements), "*t* +1, ε, σ, =, −, ⎧⎨⎪⎩")

    def test_compacts_blank_lines_but_preserves_fenced_code_spacing(self) -> None:
        markdown = "before\n\n\n\nafter\n\n```python\na = 1\n\n\n\nb = 2\n```\n\n\nend\n"

        compacted = compact_markdown_blank_lines(markdown)

        self.assertIn("before\n\nafter", compacted)
        self.assertIn("a = 1\n\n\n\nb = 2", compacted)
        self.assertIn("```\n\nend", compacted)
        self.assertFalse(compacted.endswith("\n"))

    def test_materializes_embedded_images_next_to_markdown_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report final.md"
            markdown = "before\n\n![](image_p1_0.png)\n"
            images = [MarkdownImage(filename="image_p1_0.png", content=b"png-data")]

            bundled = materialize_markdown_images(markdown, output, images)

            image_path = Path(directory) / "report final_assets" / "images" / "image_p1_0.png"
            self.assertEqual(image_path.read_bytes(), b"png-data")
            self.assertIn("![](report%20final_assets/images/image_p1_0.png)", bundled)

    def test_rebundles_legacy_image_directory_for_a_new_markdown_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parsed.md"
            output = root / "final.md"
            legacy_dir = root / "images"
            legacy_dir.mkdir()
            (legacy_dir / "image_p1_0.png").write_bytes(b"legacy-image")

            bundled = rebundle_markdown_images("![](image_p1_0.png)", source, output)

            self.assertEqual(
                (root / "final_assets" / "images" / "image_p1_0.png").read_bytes(),
                b"legacy-image",
            )
            self.assertEqual(bundled, "![](final_assets/images/image_p1_0.png)")

    def test_normalizes_only_identified_table_of_contents_entries(self) -> None:
        markdown = "\n".join(
            [
                "目录",
                "",
                "1. 引言.......... 3",
                "2. 模型........4 2.1. 评价…………5 附录 1 数据······6",
                "",
                "---",
                "",
                "正文中的版本.......... 123",
            ]
        )

        normalized = normalize_table_of_contents(markdown)

        self.assertIn("1. 引言 ... 第3页", normalized)
        self.assertIn("2. 模型 ... 第4页\n2.1. 评价 ... 第5页\n附录 1 数据 ... 第6页", normalized)
        self.assertIn("正文中的版本.......... 123", normalized)

    def test_stops_toc_at_next_heading_and_compacts_leaders_without_page_number(self) -> None:
        markdown = "\n".join(
            [
                "## 目录",
                "",
                "2.2. 一致性问题.... ........",
                "",
                "## 1. 引言",
                "",
                "正文中的版本.......... 123",
            ]
        )

        normalized = normalize_table_of_contents(markdown)

        self.assertIn("2.2. 一致性问题 ...", normalized)
        self.assertIn("正文中的版本.......... 123", normalized)

    def test_prefers_a_better_toc_without_replacing_body_text(self) -> None:
        damaged = "## 目录\n\n1. 引言..........\n\n## 1. 引言\n\n正文"
        reference = "目录\n\n1. 引言.......... 3\n\n---\n\n其他页面"

        merged = prefer_table_of_contents(damaged, reference)

        self.assertIn("目录\n\n1. 引言 ... 第3页", merged)
        self.assertIn("## 1. 引言\n\n正文", merged)
        self.assertNotIn("其他页面", merged)

    def test_cleans_parser_control_markers_only(self) -> None:
        markdown = "\n".join(
            [
                "## <sup>info</sup> [Table\\_Title] 2026.01.01",
                "[Table\\_Summary] 摘要正文",
                "## [Table\\_R相关报告",
                " 第一项",
            ]
        )

        cleaned = clean_extraction_markers(markdown)

        self.assertNotIn("Table\\_", cleaned)
        self.assertIn("摘要正文", cleaned)
        self.assertIn("## 相关报告", cleaned)
        self.assertIn("- 第一项", cleaned)

    @unittest.skipUnless(SAMPLE_PDF.exists(), "sample PDF is not available")
    def test_normalizes_sample_pdf_table_of_contents(self) -> None:
        result = parse_smart(SAMPLE_PDF, "2")

        self.assertIn("1. 引言 ... 第3页", result.markdown)
        self.assertIn("3.1. 短周期交易型阿尔法策略的构建思路 ... 第6页", result.markdown)
        self.assertIn("3.2. 一些显著的价量特征举例 ... 第8页", result.markdown)
        self.assertNotRegex(result.markdown, r"\.{4,}")

    def test_formula_processor_registry_dispatches_records_by_route(self) -> None:
        registry = FormulaProcessorRegistry()
        seen: dict[str, list[str]] = {}

        def processor(name: str):
            def run(records, manifest_path, force):
                seen[name] = [str(record["id"]) for record in records]
                self.assertEqual(manifest_path, Path("manifest.jsonl"))
                self.assertTrue(force)
                return len(records)

            return run

        registry.register("crop", processor("crop"))
        registry.register("page", processor("page"))
        counts = registry.dispatch(
            [
                {"id": "f1", "processor": "crop"},
                {"id": "f2", "processor": "page"},
                {"id": "f3", "processor": "crop"},
            ],
            Path("manifest.jsonl"),
            force=True,
        )

        self.assertEqual(counts, {"crop": 2, "page": 1})
        self.assertEqual(seen, {"crop": ["f1", "f3"], "page": ["f2"]})

    def test_formula_processor_registry_rejects_unknown_route(self) -> None:
        registry = FormulaProcessorRegistry()
        with self.assertRaisesRegex(FormulaDispatchError, "Unknown formula processor"):
            registry.dispatch([{"id": "f1", "processor": "missing"}], Path("manifest.jsonl"))

    @unittest.skipUnless(SAMPLE_PDF.exists(), "sample PDF is not available")
    def test_routes_sample_pdf_formula_lines_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "output.md"
            result = parse_smart(SAMPLE_PDF, "4-5", output_path=output_path)

            self.assertIn("{{formula:", result.markdown)
            self.assertIsNotNone(result.manifest_path)
            self.assertTrue(result.manifest_path.exists())

            manifest = result.manifest_path.read_text(encoding="utf-8")
            records = [json.loads(line) for line in manifest.splitlines() if line.strip()]
            self.assertEqual(len(records), 4)
            self.assertIn('"preview_text"', manifest)
            self.assertIn('"bbox"', manifest)
            self.assertIn('"crop_path"', manifest)
            self.assertIn('"crop_strategy"', manifest)
            self.assertIn('"processor": "pix2tex"', manifest)
            self.assertIn('"page_snapshot_path"', manifest)
            self.assertTrue(any((result.artifacts_dir / "page_snapshots").glob("page_*.png")))
            self.assertIn('"route_reasons"', manifest)
            self.assertTrue(any((result.artifacts_dir / "crops").glob("formula_*.png")))

            page_five = result.markdown.index("<!-- page 5 -->")
            first_formula = result.markdown.index("{{formula:formula_p0005_001}}")
            second_formula = result.markdown.index("{{formula:formula_p0005_002}}")
            third_formula = result.markdown.index("{{formula:formula_p0005_003}}")
            model_explanation = result.markdown.index("换言之", page_five)
            significance_explanation = result.markdown.index("那么，对于足够长", page_five)
            conclusion = result.markdown.index("显著性检验的结果", page_five)
            self.assertLess(page_five, first_formula)
            self.assertLess(first_formula, model_explanation)
            self.assertLess(model_explanation, second_formula)
            self.assertLess(second_formula, significance_explanation)
            self.assertLess(significance_explanation, third_formula)
            self.assertLess(third_formula, conclusion)
            self.assertNotIn("\uf02b", result.markdown)
            self.assertIn("*t* +1", result.markdown)
            self.assertNotRegex(result.markdown, r"\n(?:[ \t]*\n){2,}")

    @unittest.skipUnless(SAMPLE_PDF.exists(), "sample PDF is not available")
    def test_embeds_liteparse_images_for_output_bundling(self) -> None:
        result = parse_smart(SAMPLE_PDF, "1")

        self.assertIn("![](image_p1_0.png)", result.markdown)
        self.assertEqual([image.filename for image in result.images], ["image_p1_0.png"])
        self.assertTrue(result.images[0].content)

    @unittest.skipUnless(SAMPLE_PDF.exists(), "sample PDF is not available")
    def test_parse_does_not_hide_liteparse_failure(self) -> None:
        with mock.patch("main.liteparse_page_markdown", side_effect=RuntimeError("liteparse failed")):
            with self.assertRaisesRegex(RuntimeError, "liteparse failed"):
                parse_smart(SAMPLE_PDF, "4")

    @unittest.skipUnless(SAMPLE_PDF.exists(), "sample PDF is not available")
    def test_rejects_out_of_range_page_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_page_indexes(SAMPLE_PDF, "999")

    def test_normalizes_formula_crop_for_ocr(self) -> None:
        from PIL import Image, ImageChops, ImageDraw

        image = Image.new("RGBA", (240, 80), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 28, 160, 44), fill=(0, 0, 0, 255))

        normalized = normalize_formula_crop_image(image)

        self.assertEqual(normalized.mode, "RGB")
        self.assertGreaterEqual(normalized.height, 60)
        self.assertEqual(normalized.getpixel((0, 0)), (255, 255, 255))

        diff = ImageChops.difference(normalized, Image.new("RGB", normalized.size, "white")).convert("L")
        ink_bbox = diff.point(lambda pixel: 255 if pixel > 12 else 0).getbbox()
        self.assertIsNotNone(ink_bbox)
        left_margin = ink_bbox[0]
        right_margin = normalized.width - ink_bbox[2]
        self.assertLessEqual(left_margin, 18)
        self.assertLessEqual(right_margin, 18)

    def test_crop_strategy_table_includes_raw_and_tight_variants(self) -> None:
        self.assertIn("raw_3x", CROP_STRATEGIES)
        self.assertIn("tight_h48", CROP_STRATEGIES)
        self.assertFalse(CROP_STRATEGIES["raw_3x"].trim_to_ink)
        self.assertTrue(CROP_STRATEGIES["tight_h48"].trim_to_ink)

    def test_latex_quality_flags_spaced_tokens(self) -> None:
        latex = r"A l p h a_{M o d e l}=c o r r(x,y"

        flags = latex_quality_flags(latex)

        self.assertIn("spaced_alpha", flags)
        self.assertIn("spaced_model", flags)
        self.assertIn("spaced_corr", flags)
        self.assertIn("unbalanced_parentheses", flags)
        self.assertLess(latex_quality_score(latex), 100)

    def test_latex_quality_flags_do_not_penalize_normal_words(self) -> None:
        latex = r"IC_{AlphaModel}=corr(E\{\varepsilon_{t+1}\},\varepsilon_{t+1})"

        flags = latex_quality_flags(latex)

        self.assertNotIn("spaced_alpha", flags)
        self.assertNotIn("spaced_model", flags)
        self.assertNotIn("spaced_corr", flags)

    def test_latex_quality_flags_common_formula_word_typos(self) -> None:
        latex = r"f_{i n\,d u s t r v}X_{i n\,d u s t r v}+f_{s t v l e}X_{s t v l e}"

        flags = latex_quality_flags(latex)

        self.assertIn("industry_typo", flags)
        self.assertIn("style_typo", flags)

    def test_formula_ocr_updates_manifest_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crops_dir = root / "crops"
            crops_dir.mkdir()
            crop_path = crops_dir / "formula_p0001_001.png"
            crop_path.write_bytes(b"not-used-by-fake-recognizer")
            manifest_path = root / "formula_manifest.jsonl"
            records = [
                {
                    "id": "formula_p0001_001",
                    "crop_path": "crops/formula_p0001_001.png",
                    "status": "needs_formula_ocr",
                    "kind": "display",
                }
            ]
            write_formula_manifest_records(manifest_path, records)

            loaded = read_formula_manifest(manifest_path)
            seen_paths: list[Path] = []

            def fake_recognizer(path: Path) -> str:
                seen_paths.append(path)
                return r"IC = corr(E\{\epsilon_{t+1}\}, \epsilon_{t+1})"

            recognized_count = apply_formula_ocr(loaded, manifest_path, fake_recognizer)

            self.assertEqual(recognized_count, 1)
            self.assertEqual(seen_paths, [crop_path])
            self.assertEqual(loaded[0]["status"], "formula_ocr_complete")
            self.assertEqual(loaded[0]["ocr_engine"], "pix2tex")
            self.assertIn("latex", loaded[0])
            self.assertEqual(loaded[0]["ocr_candidates"][0]["engine"], "pix2tex")

    def test_fill_formula_placeholders_uses_display_math(self) -> None:
        markdown = "before\n\n{{formula:formula_p0001_001}}\n\nafter"
        records = [
            {
                "id": "formula_p0001_001",
                "kind": "display",
                "latex": r"IC = corr(E\{\epsilon_{t+1}\}, \epsilon_{t+1})",
            }
        ]

        filled, replaced_count = fill_formula_placeholders(markdown, records)

        self.assertEqual(replaced_count, 1)
        self.assertIn("$$\nIC = corr", filled)
        self.assertNotIn("{{formula:", filled)

    def test_matches_paddleocr_page_results_by_manifest_bbox(self) -> None:
        records = [
            {
                "id": "formula_p0005_001",
                "page": 5,
                "bbox": [218.807, 210.047, 344.927, 228.748],
                "page_size": [595.32, 841.92],
                "page_snapshot_size": [1240, 1754],
                "confidence": 0.74,
            },
            {
                "id": "formula_p0005_002",
                "page": 5,
                "bbox": [218.245, 364.774, 367.55, 384.779],
                "page_size": [595.32, 841.92],
                "page_snapshot_size": [1240, 1754],
                "confidence": 0.92,
            },
        ]
        page_results = [
            {
                "rec_formula": r"E\{\varepsilon_{t+1}\}",
                "formula_region_id": 4,
                "dt_polys": [520.54, 569.42, 594.15, 596.59],
            },
            {
                "rec_formula": r"\mathcal{A}lphaModel\rightarrow E\{\varepsilon_{t+1}\}",
                "formula_region_id": 3,
                "dt_polys": [425.03, 433.77, 705.66, 463.70],
            },
            {
                "rec_formula": r"IC_{AlphaModel}=corr(E\{\varepsilon_{t+1}\},\varepsilon_{t+1})",
                "formula_region_id": 2,
                "dt_polys": [416.84, 771.99, 754.21, 803.92],
            },
        ]

        matches = match_paddleocr_page_results(records, page_results)

        self.assertEqual(set(matches), {"formula_p0005_001", "formula_p0005_002"})
        self.assertEqual(matches["formula_p0005_001"][0]["formula_region_id"], 3)
        self.assertEqual(matches["formula_p0005_002"][0]["formula_region_id"], 2)

    def test_summarizes_benchmark_results(self) -> None:
        results = [
            {
                "engine": "pix2tex",
                "crop_strategy": "raw_3x",
                "latex": "x",
                "quality_score": 90,
                "elapsed_s": 0.5,
                "quality_flags": [],
                "selected": True,
            },
            {
                "engine": "pix2tex",
                "crop_strategy": "raw_3x",
                "latex": "",
                "quality_score": 0,
                "elapsed_s": 0.2,
                "quality_flags": ["empty"],
            },
        ]

        summary = summarize_benchmark_results(results)

        self.assertEqual(summary[0]["completed"], 1)
        self.assertEqual(summary[0]["formulas"], 2)
        self.assertEqual(summary[0]["flagged"], 1)
        self.assertEqual(summary[0]["selected"], 1)

    def test_selects_best_benchmark_result_and_updates_record(self) -> None:
        records = [
            {
                "id": "formula_p0001_001",
                "status": "needs_formula_ocr",
                "kind": "display",
            }
        ]
        results = [
            {
                "id": "formula_p0001_001",
                "engine": "pix2tex",
                "crop_strategy": "raw_3x",
                "latex": r"A l p h a=c o r r(x,y",
                "quality_score": 60,
                "quality_flags": ["spaced_alpha", "spaced_corr", "unbalanced_parentheses"],
                "elapsed_s": 0.2,
            },
            {
                "id": "formula_p0001_001",
                "engine": "paddleocr-crop",
                "crop_strategy": "tight_h48",
                "latex": r"Alpha=corr(x,y)",
                "quality_score": 100,
                "quality_flags": [],
                "elapsed_s": 0.8,
            },
        ]

        selected = select_best_benchmark_results(results)
        attach_best_results_to_records(records, selected)

        self.assertTrue(results[1]["selected"])
        self.assertFalse(results[0]["selected"])
        self.assertEqual(records[0]["latex"], r"Alpha=corr(x,y)")
        self.assertEqual(records[0]["ocr_engine"], "paddleocr-crop")
        self.assertEqual(records[0]["ocr_strategy"], "tight_h48")


if __name__ == "__main__":
    unittest.main()
