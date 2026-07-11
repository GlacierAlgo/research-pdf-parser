//! Span → bbox provenance side-channel (Phase 1 of the schema-extraction plan).
//!
//! During grid projection each [`ProjectedLine`] retains `spans: Vec<TextItem>`
//! — the mapping from the line's formatted text back to the source PDF items —
//! but that mapping is dropped before normal output. This module rebuilds it as
//! an [`OffsetMap`]: a side-channel that records, per projected line, the byte
//! range it occupies in an assembled text surface together with the page and
//! union bbox of the items that produced it.
//!
//! It is deliberately *not* serialized into [`crate::types::ParsedPage`] or the
//! public `ParseResult` — ordinary parses are untouched. The extraction engine
//! (Phase 2) builds one of these alongside the parse and uses
//! [`OffsetMap::resolve`] to turn a char/byte span of a retrieved unit into
//! `(page, bbox)` provenance.
//!
//! ## Text surface
//!
//! The map owns the text it indexes. `ParsedPage.text` is produced by
//! `project_to_grid`, a construction separate from `projected_lines`, so the two
//! are not guaranteed to be character-aligned. Rather than depend on that, the
//! map assembles its own canonical surface — every projected line's text, joined
//! by `\n`, pages in order — which is exactly the "natural line" retrieval unit
//! Phase 2 ranks over. Callers that need the surface read it back via
//! [`OffsetMap::text`].
//!
//! ## Offsets
//!
//! Ranges are **byte** offsets into [`OffsetMap::text`], matching what Rust
//! `str` slicing and `regex::Match` produce and consume (Phase 2's value-span
//! regex runs over unit text and yields byte offsets). Conversion to
//! char/codepoint offsets for language bindings is a binding-layer concern.

use crate::types::{ParsedPage, Rect, TextItem};
use std::ops::Range;

impl Rect {
    /// The smallest axis-aligned rectangle covering both `self` and `other`.
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Rect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

/// Union bbox of a slice of items in their (viewport) coordinates, or `None`
/// when the slice is empty.
fn union_of_items(items: &[TextItem]) -> Option<Rect> {
    items
        .iter()
        .map(|it| Rect {
            x: it.x,
            y: it.y,
            width: it.width,
            height: it.height,
        })
        .reduce(|acc, r| acc.union(&r))
}

/// One projected line's footprint in the assembled text surface.
#[derive(Debug, Clone)]
struct LineSpan {
    /// 1-based page number the line came from.
    page: usize,
    /// Byte range `[start, end)` of the line's text within [`OffsetMap::text`].
    /// Excludes the trailing `\n` separator.
    range: Range<usize>,
    /// Union bbox of the line's source items, in viewport coords.
    bbox: Rect,
}

/// Resolved provenance for a span of the text surface.
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    /// 1-based page number.
    pub page: usize,
    /// Union bbox of every line the span touches, in viewport coords.
    pub bbox: Rect,
}

/// Side-channel mapping byte ranges of an assembled projected-line text surface
/// back to `(page, bbox)`. Build with [`OffsetMap::build`]; query with
/// [`OffsetMap::resolve`]. See the module docs for the surface/offset contract.
#[derive(Debug, Clone, Default)]
pub struct OffsetMap {
    text: String,
    lines: Vec<LineSpan>,
}

impl OffsetMap {
    /// Assemble the map from parsed pages. Each projected line contributes its
    /// text (joined by `\n`) and an entry recording its byte range, page, and
    /// union bbox. Pages with no projected lines contribute nothing (their text
    /// has no span→item mapping to offer).
    pub fn build(pages: &[ParsedPage]) -> OffsetMap {
        let mut text = String::new();
        let mut lines = Vec::new();
        for page in pages {
            for line in &page.projected_lines {
                let start = text.len();
                text.push_str(&line.text);
                let end = text.len();
                text.push('\n');
                // Prefer the union of the actual source items; fall back to the
                // line's own bbox when spans are absent (e.g. synthetic lines).
                let bbox = union_of_items(&line.spans).unwrap_or_else(|| line.bbox.clone());
                lines.push(LineSpan {
                    page: page.page_number,
                    range: start..end,
                    bbox,
                });
            }
        }
        OffsetMap { text, lines }
    }

    /// The assembled text surface the byte ranges index into.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// One `(page, byte range)` per indexed line, in surface order. The range
    /// slices [`OffsetMap::text`] to that line's text (excluding its `\n`).
    /// Used to build natural [`crate::extraction_unit::ExtractionUnit`]s.
    pub fn lines(&self) -> impl Iterator<Item = (usize, Range<usize>)> + '_ {
        self.lines.iter().map(|l| (l.page, l.range.clone()))
    }

    /// Resolve a byte range of [`OffsetMap::text`] to `(page, union bbox)`.
    ///
    /// Returns `None` when the range overlaps no line (e.g. it lands entirely on
    /// a `\n` separator, or the map is empty). When a range touches lines on more
    /// than one page — unusual, since retrieval units are single lines — the page
    /// of the first touched line wins and the bbox unions only that page's lines,
    /// since a bbox spanning pages is meaningless.
    pub fn resolve(&self, span: Range<usize>) -> Option<Provenance> {
        let mut page: Option<usize> = None;
        let mut bbox: Option<Rect> = None;
        for line in &self.lines {
            // Overlap test: line and span share at least one byte.
            if line.range.start < span.end && span.start < line.range.end {
                match page {
                    None => {
                        page = Some(line.page);
                        bbox = Some(line.bbox.clone());
                    }
                    Some(p) if p == line.page => {
                        bbox = Some(bbox.unwrap().union(&line.bbox));
                    }
                    // Different page — stop unioning across the page boundary.
                    Some(_) => break,
                }
            }
        }
        Some(Provenance {
            page: page?,
            bbox: bbox?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Anchor, ProjectedLine};

    fn item(x: f32, y: f32, w: f32, h: f32) -> TextItem {
        TextItem {
            x,
            y,
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn line(text: &str, spans: Vec<TextItem>) -> ProjectedLine {
        ProjectedLine {
            text: text.into(),
            bbox: Rect::default(),
            anchor: Anchor::Left,
            indent_x: 0.0,
            dominant_font_size: 10.0,
            font_size_is_estimated: false,
            heading_font_size: None,
            dominant_font_name: None,
            all_bold: false,
            all_italic: false,
            all_mono: false,
            all_strike: false,
            spans,
            region_path: Vec::new(),
            mcid: None,
            in_figure: false,
        }
    }

    fn page(n: usize, lines: Vec<ProjectedLine>) -> ParsedPage {
        ParsedPage {
            page_number: n,
            page_width: 612.0,
            page_height: 792.0,
            text: String::new(),
            markdown: String::new(),
            text_items: vec![],
            projected_lines: lines,
            regions: crate::types::Region::default(),
            graphics: vec![],
            figures: vec![],
            struct_nodes: vec![],
            image_refs: vec![],
        }
    }

    #[test]
    fn rect_union_covers_both() {
        let a = Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = Rect {
            x: 20.0,
            y: 5.0,
            width: 10.0,
            height: 10.0,
        };
        let u = a.union(&b);
        assert_eq!((u.x, u.y, u.width, u.height), (0.0, 0.0, 30.0, 15.0));
    }

    #[test]
    fn surface_joins_lines_with_newline() {
        let map = OffsetMap::build(&[page(1, vec![line("hello", vec![]), line("world", vec![])])]);
        assert_eq!(map.text(), "hello\nworld\n");
    }

    #[test]
    fn resolve_span_returns_line_union_bbox() {
        // Line "Invoice: INV-42" made of two items; a span inside it should
        // resolve to their union bbox on the right page.
        let items = vec![item(10.0, 20.0, 40.0, 8.0), item(55.0, 20.0, 30.0, 8.0)];
        let map = OffsetMap::build(&[page(3, vec![line("Invoice: INV-42", items)])]);
        let prov = map.resolve(0..7).expect("span should resolve");
        assert_eq!(prov.page, 3);
        // Union: x 10..85 (width 75), y 20, height 8.
        assert_eq!((prov.bbox.x, prov.bbox.width), (10.0, 75.0));
        assert_eq!((prov.bbox.y, prov.bbox.height), (20.0, 8.0));
    }

    #[test]
    fn resolve_falls_back_to_line_bbox_without_spans() {
        let mut l = line("no spans here", vec![]);
        l.bbox = Rect {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        };
        let map = OffsetMap::build(&[page(1, vec![l])]);
        let prov = map.resolve(0..2).unwrap();
        assert_eq!((prov.bbox.x, prov.bbox.width), (1.0, 3.0));
    }

    #[test]
    fn resolve_on_separator_only_is_none() {
        let map = OffsetMap::build(&[page(1, vec![line("ab", vec![item(0.0, 0.0, 5.0, 5.0)])])]);
        // Byte 2 is the '\n' separator; an empty-at-2 range touches no line.
        assert_eq!(map.resolve(2..2), None);
    }

    #[test]
    fn resolve_across_lines_unions_same_page() {
        let l1 = line("aaa", vec![item(0.0, 0.0, 30.0, 10.0)]);
        let l2 = line("bbb", vec![item(0.0, 20.0, 30.0, 10.0)]);
        let map = OffsetMap::build(&[page(1, vec![l1, l2])]);
        // Range 0..7 spans "aaa\nbbb", touching both lines.
        let prov = map.resolve(0..7).unwrap();
        assert_eq!(prov.page, 1);
        assert_eq!((prov.bbox.y, prov.bbox.height), (0.0, 30.0));
    }

    #[test]
    fn empty_pages_produce_empty_map() {
        let map = OffsetMap::build(&[]);
        assert_eq!(map.text(), "");
        assert_eq!(map.resolve(0..0), None);
    }
}
