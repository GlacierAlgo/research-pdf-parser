//! `ExtractionUnit` — the retrieval unit the extraction engine ranks over
//! (Phase 2 of the schema-extraction plan).
//!
//! A unit is a short piece of text plus the means to recover its `(page, bbox)`
//! provenance. The plan builds units from three sources, all ranked identically
//! downstream:
//!
//! 1. **Natural lines** — every projected line ([`natural_units`]). Provenance is
//!    a byte range into the [`OffsetMap`] surface, resolved on demand.
//! 2. **Header-joined cell units** — synthetic (not yet built).
//! 3. **Geometry join units** — synthetic (not yet built).
//!
//! Synthetic units have no char range in any emitted text — their text exists
//! only in the index — so provenance is carried [`UnitProvenance::Direct`]ly.
//! This split is the "don't design the unit type char-range-only" requirement
//! from Phase 1.

use crate::offset_map::{OffsetMap, Provenance};
use crate::types::Rect;
use std::ops::Range;

/// Where a unit's text came from. Purely informational — every source ranks
/// identically — but the geometry-join pass leans on it for per-page caps,
/// dedupe-vs-natural-lines, and debug tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitSource {
    /// A projected line, verbatim.
    NaturalLine,
    /// A table cell joined with its column header (and row entity).
    HeaderCell,
    /// A synthetic label→value / header→cell join over raw item geometry.
    GeometryJoin,
}

/// How to recover `(page, bbox)` provenance for a unit.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitProvenance {
    /// Natural line: a byte range into the [`OffsetMap`] surface. Resolved via
    /// [`OffsetMap::resolve`].
    Span(Range<usize>),
    /// Synthetic unit: provenance carried directly, since the text is not in the
    /// surface. `bbox` is the union of the joined source items, in viewport
    /// coords.
    Direct { page: usize, bbox: Rect },
}

/// A retrieval unit: indexable text plus its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionUnit {
    /// The text that gets BM25'd / embedded and that value-span regex runs over.
    pub text: String,
    pub source: UnitSource,
    pub provenance: UnitProvenance,
}

impl ExtractionUnit {
    /// A synthetic unit with directly-carried provenance (header-joined cell or
    /// geometry join).
    pub fn synthetic(text: impl Into<String>, source: UnitSource, page: usize, bbox: Rect) -> Self {
        ExtractionUnit {
            text: text.into(),
            source,
            provenance: UnitProvenance::Direct { page, bbox },
        }
    }

    /// Resolve this unit to `(page, bbox)`. Natural-line units need the
    /// [`OffsetMap`] they were built from; synthetic units ignore it. Returns
    /// `None` only when a natural span resolves to nothing (see
    /// [`OffsetMap::resolve`]).
    pub fn provenance(&self, map: &OffsetMap) -> Option<Provenance> {
        match &self.provenance {
            UnitProvenance::Span(range) => map.resolve(range.clone()),
            UnitProvenance::Direct { page, bbox } => Some(Provenance {
                page: *page,
                bbox: bbox.clone(),
            }),
        }
    }
}

/// Build one natural-line unit per indexed line of the map. Whitespace-only
/// lines are skipped — an empty unit can never be a useful retrieval hit and
/// only dilutes the index.
pub fn natural_units(map: &OffsetMap) -> Vec<ExtractionUnit> {
    map.lines()
        .filter_map(|(_page, range)| {
            let text = map.text().get(range.clone())?;
            if text.trim().is_empty() {
                return None;
            }
            Some(ExtractionUnit {
                text: text.to_string(),
                source: UnitSource::NaturalLine,
                provenance: UnitProvenance::Span(range),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Anchor, ParsedPage, ProjectedLine, TextItem};

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
    fn natural_units_carry_text_and_resolve_to_bbox() {
        let map = OffsetMap::build(&[page(
            2,
            vec![line("Invoice: INV-42", vec![item(10.0, 20.0, 75.0, 8.0)])],
        )]);
        let units = natural_units(&map);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text, "Invoice: INV-42");
        assert_eq!(units[0].source, UnitSource::NaturalLine);
        let prov = units[0].provenance(&map).unwrap();
        assert_eq!(prov.page, 2);
        assert_eq!((prov.bbox.x, prov.bbox.width), (10.0, 75.0));
    }

    #[test]
    fn natural_units_skip_blank_lines() {
        let map = OffsetMap::build(&[page(
            1,
            vec![
                line("real content", vec![item(0.0, 0.0, 50.0, 8.0)]),
                line("   ", vec![item(0.0, 20.0, 5.0, 8.0)]),
                line("", vec![]),
            ],
        )]);
        let units = natural_units(&map);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text, "real content");
    }

    #[test]
    fn synthetic_unit_resolves_to_direct_bbox() {
        let map = OffsetMap::default();
        let bbox = Rect {
            x: 5.0,
            y: 6.0,
            width: 7.0,
            height: 8.0,
        };
        let unit = ExtractionUnit::synthetic(
            "Standard - Price: $146,688",
            UnitSource::GeometryJoin,
            4,
            bbox.clone(),
        );
        let prov = unit.provenance(&map).unwrap();
        assert_eq!(prov.page, 4);
        assert_eq!(prov.bbox, bbox);
    }

    #[test]
    fn natural_unit_ranges_index_the_surface() {
        // A second line's range must slice the correct substring, not the first.
        let map = OffsetMap::build(&[page(
            1,
            vec![
                line("first", vec![item(0.0, 0.0, 30.0, 8.0)]),
                line("second", vec![item(0.0, 20.0, 40.0, 8.0)]),
            ],
        )]);
        let units = natural_units(&map);
        let UnitProvenance::Span(r) = &units[1].provenance else {
            panic!("expected span provenance");
        };
        assert_eq!(&map.text()[r.clone()], "second");
    }
}
