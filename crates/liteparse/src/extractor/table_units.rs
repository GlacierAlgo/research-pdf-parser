//! Table cell-grids with per-row provenance (schema-extraction row-grouping).
//!
//! The object-array extraction path (repeated record groups — `line_items[]`
//! and friends) needs liteparse's *detected* tables as structured rows, plus a
//! `(page, bbox)` for each row so extracted values carry provenance. This module
//! bridges to that: it re-runs the existing detectors through the read-only
//! [`crate::markdown_layout::detect_page_tables`] seam (no parse-path change,
//! same posture as [`super::geometry_units`]) and resolves each detected row
//! back to the projected line it came from.
//!
//! ## Provenance recovery (v1: row-level)
//!
//! Detection flattens its internal `(line_index, &ProjectedLine, cells)` rows to
//! text before the extractor can see them, and threading the line index through
//! every `TableRun` construction + merge pass would churn the whole table hot
//! path. Instead — since v1 provenance is line-level everywhere (word/cell-level
//! x-tightening is the documented follow-up) — each output row is matched back
//! to a source projected line *within the run's line range* by whitespace-
//! collapsed text. A moving cursor keeps identical rows mapping to successive
//! lines. A row that can't be matched (wrapped/merged into multiple lines) gets
//! `None` provenance rather than a wrong box.

use crate::markdown_layout::detect_page_tables;
use crate::offset_map::Provenance;
use crate::types::{ParsedPage, ProjectedLine, Rect};

/// A detected table with per-row provenance, consumed by the row-grouping
/// object-array extraction path. `header`, when present, names the columns;
/// each [`GridRow`] is one record candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct TableGrid {
    /// 1-based page the table was detected on.
    pub page: usize,
    /// Column names, when the detector identified a header row.
    pub header: Option<Vec<String>>,
    pub rows: Vec<GridRow>,
}

/// One detected table row: its cell texts and row-level provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct GridRow {
    pub cells: Vec<String>,
    /// Page + union bbox of the source projected line, or `None` when the row
    /// couldn't be matched back to a single source line.
    pub provenance: Option<Provenance>,
}

/// Build the table cell-grids across all parsed pages. Read-only; nothing here
/// touches parse output.
pub fn table_grids(pages: &[ParsedPage]) -> Vec<TableGrid> {
    let mut grids = Vec::new();
    for page in pages {
        for table in detect_page_tables(page) {
            let lines = &page.projected_lines;
            let end = table.line_end.min(lines.len());
            // Cursor over the run's lines, advanced as rows are matched so
            // duplicate rows resolve to successive source lines rather than all
            // colliding on the first match.
            let mut cursor = table.line_start.min(end);
            let rows = table
                .rows
                .iter()
                .map(|cells| {
                    let provenance = resolve_row(page.page_number, lines, cells, &mut cursor, end);
                    GridRow {
                        cells: cells.clone(),
                        provenance,
                    }
                })
                .collect();
            grids.push(TableGrid {
                page: page.page_number,
                header: table.header.clone(),
                rows,
            });
        }
    }
    grids
}

/// Find the source line for `cells` at or after `*cursor` within `[.., end)`,
/// matching on whitespace-collapsed text. On a hit, advance the cursor past the
/// matched line and return its provenance; on no match, leave the cursor and
/// return `None`.
fn resolve_row(
    page: usize,
    lines: &[ProjectedLine],
    cells: &[String],
    cursor: &mut usize,
    end: usize,
) -> Option<Provenance> {
    let target = collapse_ws(&cells.join(" "));
    if target.is_empty() {
        return None;
    }
    let mut i = *cursor;
    while i < end {
        let line_text = collapse_ws(&lines[i].text);
        if line_text == target || line_text.contains(&target) {
            *cursor = i + 1;
            return Some(Provenance {
                page,
                bbox: row_bbox(&lines[i]),
            });
        }
        i += 1;
    }
    None
}

/// Row-level bbox: union of the line's source item boxes, falling back to the
/// projected line's own bbox when it carries no spans (synthetic lines).
fn row_bbox(line: &ProjectedLine) -> Rect {
    line.spans
        .iter()
        .map(|it| Rect {
            x: it.x,
            y: it.y,
            width: it.width,
            height: it.height,
        })
        .reduce(|acc, r| acc.union(&r))
        .unwrap_or_else(|| line.bbox.clone())
}

/// Lowercase-insensitive whitespace normalization: trim and collapse internal
/// runs of whitespace to a single space. Matches how projection renders a table
/// row (column gaps collapse to single spaces) so detected cell text lines up
/// with `ProjectedLine.text`.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown_layout::test_helpers::line_with_spans;

    fn page(lines: Vec<ProjectedLine>) -> ParsedPage {
        ParsedPage {
            page_number: 1,
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

    /// A borderless table (partial header above a clean 3-column body) should be
    /// detected and each body row resolved to its own source line's bbox. Mirrors
    /// the detector's `absorbs_partial_header_line_above_body` fixture so we test
    /// against real detection, not hand-tuned thresholds.
    #[test]
    fn borderless_table_rows_carry_row_bbox() {
        let lines = vec![
            line_with_spans(&[("Name", 50.0), ("Scores", 150.0)], 100.0, 10.0),
            line_with_spans(&[("A", 50.0), ("1", 150.0), ("2", 250.0)], 115.0, 10.0),
            line_with_spans(&[("B", 50.0), ("3", 150.0), ("4", 250.0)], 130.0, 10.0),
            line_with_spans(&[("C", 50.0), ("5", 150.0), ("6", 250.0)], 145.0, 10.0),
        ];
        let grids = table_grids(&[page(lines)]);
        assert_eq!(grids.len(), 1, "expected one detected table");
        let grid = &grids[0];
        assert_eq!(
            grid.header,
            Some(vec![
                "Name".to_string(),
                "Scores".to_string(),
                String::new()
            ])
        );
        assert_eq!(grid.rows.len(), 3);
        assert_eq!(grid.rows[0].cells, vec!["A", "1", "2"]);
        // Each body row resolves to its own line's bbox (its y), on page 1.
        for (row, y) in grid.rows.iter().zip([115.0, 130.0, 145.0]) {
            let prov = row.provenance.as_ref().expect("row provenance");
            assert_eq!(prov.page, 1);
            assert_eq!(prov.bbox.y, y);
        }
    }

    #[test]
    fn no_table_yields_no_grids() {
        let lines = vec![
            line_with_spans(&[("Just a sentence of prose.", 50.0)], 100.0, 10.0),
            line_with_spans(&[("Another prose line here.", 50.0)], 115.0, 10.0),
        ];
        assert!(table_grids(&[page(lines)]).is_empty());
    }

    #[test]
    fn collapse_ws_normalizes_runs() {
        assert_eq!(collapse_ws("  a   b\tc  "), "a b c");
    }
}
