//! Geometry-join pass — synthetic retrieval units from raw item geometry.
//! The gates and caps here were tuned by measurement; re-measure before
//! loosening them (see the load-bearing note below).
//!
//! Labeled/tabular values often reach the retrieval index as a bare projected
//! line ("$146,688") with no label context, so no query can retrieve them.
//! This pass re-joins values with their labels *in the index only* — parse
//! output is untouched, so it cannot regress any parsing benchmark.
//!
//! Three pairwise joins, cheapest first:
//! 1. **label → value, rightward** — same visual row, WIDE gap (projection
//!    already merges tight pairs into one line; wide survivors need the join).
//! 2. **label → value, below** — boxed form fields: an x-overlapping typed
//!    value 1–2 line-heights below its label.
//! 3. **far-header column projection** — for a typed value, walk *up* to the
//!    nearest x-overlapping column header and *left* to the row's leading
//!    label; emit "«row» - «header»: «value»". Recovers UNDETECTED tables (the
//!    biggest measured win of the pass).
//!
//! The gates are load-bearing, not defensive: loose value-ness floods dense
//! pages with prose joins and makes the whole pass net-NEGATIVE. Hence:
//! `below`/`far-header` require a **typed** value (money / date / percent /
//! strong-ID — bare numbers and years are the flood); a per-page unit cap; and
//! de-dupe on unit text.
//!
//! SCOPE GUARDRAIL (do not cross): pairwise adjacency joins over item geometry
//! only. The moment a join wants reading-order / column-model / table reasoning
//! of its own, that's a shadow projection — stop, and either expose the signal
//! from `projection.rs` or drop the case.

use crate::extraction_unit::{ExtractionUnit, UnitSource};
use crate::types::{ParsedPage, Rect, TextItem};
use std::collections::HashSet;

/// Per-page cap on synthetic units. Measured to matter: without it a dense
/// page floods the index and outranks good natural lines.
pub(crate) const MAX_UNITS_PER_PAGE: usize = 40;

/// Row-clustering tolerance as a fraction of item height.
const Y_TOL_FRAC: f32 = 0.5;
/// Rightward join fires only across a gap wider than this × item height
/// (tighter pairs are already merged into one projected line).
const GAP_FRAC: f32 = 1.5;
/// Below join reach, in label line-heights.
const MAX_GAP_LINES: f32 = 2.0;
/// Hard cap per join kind before the page-level cap (runaway backstop).
const MAX_UNITS_PER_JOIN: usize = 200;

/// Which join produced a unit. Informational — for debug tooling and eval
/// triage; every unit ranks identically downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Rightward,
    Below,
    FarHeader,
}

impl JoinKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            JoinKind::Rightward => "rightward",
            JoinKind::Below => "below",
            JoinKind::FarHeader => "far-header",
        }
    }
}

/// Build all synthetic geometry-join units for a parsed document.
pub fn geometry_units(pages: &[ParsedPage]) -> Vec<ExtractionUnit> {
    geometry_units_by_kind(pages)
        .into_iter()
        .map(|(_, u)| u)
        .collect()
}

/// [`geometry_units`], with each unit tagged by the join that produced it.
/// The entry point for debug tooling (`examples/dump_geometry_units.rs`).
pub fn geometry_units_by_kind(pages: &[ParsedPage]) -> Vec<(JoinKind, ExtractionUnit)> {
    let mut out = Vec::new();
    for page in pages {
        let items: Vec<&TextItem> = page
            .text_items
            .iter()
            .filter(|it| !it.text.trim().is_empty())
            .collect();
        let mut units = page_joins(page.page_number, &items);
        // De-dupe on normalized text, keeping the earliest (cheapest) join,
        // then cap. Join order (rightward, below, far-header) decides what
        // survives the cap.
        let mut seen = HashSet::new();
        units.retain(|(_, u)| seen.insert(u.text.trim().to_lowercase()));
        units.truncate(MAX_UNITS_PER_PAGE);
        out.extend(units);
    }
    out
}

/// Per-item join gates, precomputed once per page.
struct Gates {
    typed: bool,
    value: bool,
    label: bool,
}

fn page_joins(page_no: usize, items: &[&TextItem]) -> Vec<(JoinKind, ExtractionUnit)> {
    let gates: Vec<Gates> = items
        .iter()
        .map(|it| {
            let typed = is_typed_value(&it.text);
            let value = is_value_ish(&it.text, typed);
            Gates {
                typed,
                value,
                label: is_label_ish(&it.text, value),
            }
        })
        .collect();
    let rows = group_rows(items);

    let mut units = Vec::new();
    join_rightward(page_no, items, &gates, &rows, &mut units);
    join_below(page_no, items, &gates, &mut units);
    join_far_header(page_no, items, &gates, &rows, &mut units);
    units
}

// ── row grouping ──────────────────────────────────────────────────────────────

fn x2(it: &TextItem) -> f32 {
    it.x + it.width
}

fn ycenter(it: &TextItem) -> f32 {
    it.y + it.height / 2.0
}

fn x_overlap(a: &TextItem, b: &TextItem) -> bool {
    x2(a).min(x2(b)) - a.x.max(b.x) > 0.0
}

/// Cluster items into visual rows by y-center. The tolerance scales with item
/// height so big and small text both cluster sanely. Each row's items are
/// returned sorted left-to-right (as indices into `items`).
fn group_rows(items: &[&TextItem]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&a, &b| ycenter(items[a]).total_cmp(&ycenter(items[b])));

    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let it = items[i];
        let row = rows.iter_mut().find(|row| {
            let anchor = items[row[0]];
            let tol = Y_TOL_FRAC * anchor.height.max(it.height);
            (ycenter(it) - ycenter(anchor)).abs() <= tol
        });
        match row {
            Some(row) => row.push(i),
            None => rows.push(vec![i]),
        }
    }
    for row in &mut rows {
        row.sort_by(|&a, &b| items[a].x.total_cmp(&items[b].x));
    }
    rows
}

// ── label-ness / value-ness gates ─────────────────────────────────────────────

/// Strong value signal: money / date / percent / strong-ID only. Bare numbers
/// and years are deliberately NOT typed — on dense docs they fire all over
/// prose, and the flood of bad joins is what sinks the pass (measured).
fn is_typed_value(text: &str) -> bool {
    has_money(text)
        || has_percent(text)
        || has_strong_id(text)
        || super::value_span::find_date(text).is_some()
}

/// Loose value signal (rightward join only): typed, or numeric-heavy + short.
fn is_value_ish(text: &str, typed: bool) -> bool {
    if !typed && !text.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    let n_tokens = super::tokenize(text).len();
    let digits = text.chars().filter(char::is_ascii_digit).count();
    n_tokens > 0 && n_tokens <= 4 && digits >= (text.chars().count() / 4).max(1)
}

/// Short-ish and carrying no value of its own. A trailing colon or full
/// title-case/all-caps qualifies; neither alone is required (many form labels
/// lack a colon).
fn is_label_ish(text: &str, value_ish: bool) -> bool {
    let t = text.trim();
    if t.is_empty() || value_ish {
        return false;
    }
    let n_tokens = super::tokenize(t).len();
    if !(1..=6).contains(&n_tokens) {
        return false;
    }
    if t.ends_with(':') {
        return true;
    }
    // Title-case / all-caps label ("Invoice Number", "TOTAL DUE").
    let mut saw_word = false;
    for w in t.split_whitespace() {
        if !w.chars().any(char::is_alphabetic) {
            continue;
        }
        saw_word = true;
        if !w.chars().next().is_some_and(char::is_uppercase) {
            return false;
        }
    }
    saw_word
}

/// A currency symbol followed (allowing one space) by a digit: `$146,688`.
fn has_money(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    chars.iter().enumerate().any(|(i, &c)| {
        matches!(c, '$' | '€' | '£')
            && (chars.get(i + 1).is_some_and(char::is_ascii_digit)
                || (chars.get(i + 1) == Some(&' ')
                    && chars.get(i + 2).is_some_and(char::is_ascii_digit)))
    })
}

/// A digit followed (allowing one space) by `%`: `18 %`, `3.5%`.
fn has_percent(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    chars.iter().enumerate().any(|(i, &c)| {
        c == '%'
            && (i >= 1 && chars[i - 1].is_ascii_digit()
                || (i >= 2 && chars[i - 1] == ' ' && chars[i - 2].is_ascii_digit()))
    })
}

/// An ID-ish token: ≥2 uppercase letters at a word boundary, immediately
/// followed by a digit or hyphen (`INV-2024-0042`, `PO4211`). Plain all-caps
/// words ("TOTAL DUE") don't qualify.
fn has_strong_id(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Word boundary: start of text or previous char non-alphanumeric.
        if chars[i].is_ascii_uppercase() && (i == 0 || !chars[i - 1].is_alphanumeric()) {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_uppercase() {
                j += 1;
            }
            if j - i >= 2
                && chars
                    .get(j)
                    .is_some_and(|&c| c == '-' || c.is_ascii_digit())
            {
                return true;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    false
}

// ── the three joins ───────────────────────────────────────────────────────────

/// `"Invoice Number:  " + "INV-42"` → `"Invoice Number: INV-42"`.
fn join_text(label: &str, value: &str) -> String {
    format!("{}: {}", strip_colon(label), value.trim())
}

fn strip_colon(label: &str) -> &str {
    label.trim().trim_end_matches(':').trim_end()
}

fn union_bbox(items: &[&TextItem]) -> Rect {
    let x1 = items.iter().map(|i| i.x).fold(f32::INFINITY, f32::min);
    let y1 = items.iter().map(|i| i.y).fold(f32::INFINITY, f32::min);
    let x2_ = items
        .iter()
        .map(|i| x2(i))
        .fold(f32::NEG_INFINITY, f32::max);
    let y2 = items
        .iter()
        .map(|i| i.y + i.height)
        .fold(f32::NEG_INFINITY, f32::max);
    Rect {
        x: x1,
        y: y1,
        width: x2_ - x1,
        height: y2 - y1,
    }
}

fn push_unit(
    units: &mut Vec<(JoinKind, ExtractionUnit)>,
    kind: JoinKind,
    page: usize,
    text: String,
    joined: &[&TextItem],
) {
    units.push((
        kind,
        ExtractionUnit::synthetic(text, UnitSource::GeometryJoin, page, union_bbox(joined)),
    ));
}

/// Join 1: same visual row, a label-ish item followed by a value-ish item
/// across a WIDE gap (tight pairs are projection's job, not ours).
fn join_rightward(
    page: usize,
    items: &[&TextItem],
    gates: &[Gates],
    rows: &[Vec<usize>],
    units: &mut Vec<(JoinKind, ExtractionUnit)>,
) {
    let mut emitted = 0;
    for row in rows {
        for w in row.windows(2) {
            let (a, b) = (items[w[0]], items[w[1]]);
            let gap = b.x - x2(a);
            if gap <= GAP_FRAC * a.height.max(b.height) {
                continue;
            }
            if gates[w[0]].label && gates[w[1]].value {
                push_unit(
                    units,
                    JoinKind::Rightward,
                    page,
                    join_text(&a.text, &b.text),
                    &[a, b],
                );
                emitted += 1;
                if emitted >= MAX_UNITS_PER_JOIN {
                    return;
                }
            }
        }
    }
}

/// Join 2: boxed form fields — a label-ish item with an x-overlapping TYPED
/// value 1–2 line-heights directly below. Reading order often does not make
/// these adjacent lines; geometry does.
fn join_below(
    page: usize,
    items: &[&TextItem],
    gates: &[Gates],
    units: &mut Vec<(JoinKind, ExtractionUnit)>,
) {
    for (ai, &a) in items.iter().enumerate() {
        if !gates[ai].label {
            continue;
        }
        let below = a.y + a.height;
        let mut best: Option<&TextItem> = None;
        for (bi, &b) in items.iter().enumerate() {
            if bi == ai || !gates[bi].typed || !x_overlap(a, b) {
                continue;
            }
            let dy = b.y - below;
            if (0.0..=MAX_GAP_LINES * a.height).contains(&dy) && best.is_none_or(|cur| b.y < cur.y)
            {
                best = Some(b);
            }
        }
        if let Some(b) = best {
            push_unit(
                units,
                JoinKind::Below,
                page,
                join_text(&a.text, &b.text),
                &[a, b],
            );
        }
    }
}

/// Join 3: far-header column projection. For a TYPED value, walk UP to the
/// nearest x-overlapping label-ish item (column header) and LEFT within its
/// row to the leading label-ish item (row entity); emit
/// `"«row» - «header»: «value»"`. Recovers undetected tables. Requiring BOTH
/// a header and a row entity is what keeps dense prose quiet.
fn join_far_header(
    page: usize,
    items: &[&TextItem],
    gates: &[Gates],
    rows: &[Vec<usize>],
    units: &mut Vec<(JoinKind, ExtractionUnit)>,
) {
    let mut row_of = vec![usize::MAX; items.len()];
    for (ri, row) in rows.iter().enumerate() {
        for &i in row {
            row_of[i] = ri;
        }
    }

    let mut emitted = 0;
    for (vi, &v) in items.iter().enumerate() {
        if !gates[vi].typed {
            continue;
        }
        // Walk UP: nearest x-overlapping label-ish item strictly above.
        let mut header: Option<&TextItem> = None;
        for (ci, &c) in items.iter().enumerate() {
            if ci == vi || c.y >= v.y || !x_overlap(c, v) {
                continue;
            }
            if !gates[ci].label || gates[ci].typed {
                continue;
            }
            if header.is_none_or(|cur| c.y > cur.y) {
                header = Some(c);
            }
        }
        let Some(header) = header else { continue };

        // Walk LEFT: leading label-ish item in v's row, left of v.
        let Some(entity) = rows[row_of[vi]]
            .iter()
            .map(|&i| (i, items[i]))
            .find(|(i, it)| it.x < v.x && gates[*i].label)
            .map(|(_, it)| it)
        else {
            continue;
        };

        push_unit(
            units,
            JoinKind::FarHeader,
            page,
            format!(
                "{} - {}: {}",
                entity.text.trim(),
                strip_colon(&header.text),
                v.text.trim()
            ),
            &[entity, header, v],
        );
        emitted += 1;
        if emitted >= MAX_UNITS_PER_JOIN {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str, x: f32, y: f32, w: f32, h: f32) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn page(n: usize, text_items: Vec<TextItem>) -> ParsedPage {
        ParsedPage {
            page_number: n,
            page_width: 612.0,
            page_height: 792.0,
            text: String::new(),
            markdown: String::new(),
            text_items,
            projected_lines: vec![],
            regions: crate::types::Region::default(),
            graphics: vec![],
            figures: vec![],
            struct_nodes: vec![],
            image_refs: vec![],
        }
    }

    fn texts(units: &[(JoinKind, ExtractionUnit)]) -> Vec<&str> {
        units.iter().map(|(_, u)| u.text.as_str()).collect()
    }

    #[test]
    fn typed_value_gate() {
        for v in [
            "$146,688",
            "€ 18",
            "3.5%",
            "18 %",
            "INV-2024-0042",
            "PO4211",
            "01/15/2024",
            "January 15, 2024",
        ] {
            assert!(is_typed_value(v), "{v:?} should be typed");
        }
        // Bare numbers, years, and plain all-caps are the measured flood — NOT typed.
        for v in ["42", "1,024", "2024", "TOTAL DUE", "plain prose"] {
            assert!(!is_typed_value(v), "{v:?} should NOT be typed");
        }
    }

    #[test]
    fn label_gate() {
        for l in ["Invoice Number:", "Invoice Number", "TOTAL DUE", "Standard"] {
            assert!(is_label_ish(l, false), "{l:?} should be label-ish");
        }
        // Lowercase prose, over-long, and value-ish are not labels.
        assert!(!is_label_ish("the following cases apply", false));
        assert!(!is_label_ish(
            "A very long heading with far too many words in it",
            false
        ));
        assert!(!is_label_ish("$146,688", true));
    }

    #[test]
    fn rightward_joins_wide_gap_only() {
        let pages = [page(
            1,
            vec![
                item("Invoice Number:", 0.0, 0.0, 80.0, 10.0),
                item("INV-2024-0042", 200.0, 0.0, 80.0, 10.0), // gap 120 > 15
                item("Due Date:", 400.0, 0.0, 50.0, 10.0),
                item("01/15/2024", 458.0, 0.0, 60.0, 10.0), // gap 8 <= 15 → projection's job
            ],
        )];
        let units = geometry_units_by_kind(&pages);
        assert_eq!(texts(&units), ["Invoice Number: INV-2024-0042"]);
        assert_eq!(units[0].0, JoinKind::Rightward);
        // Union bbox spans label through value.
        let crate::extraction_unit::UnitProvenance::Direct { page, bbox } = &units[0].1.provenance
        else {
            panic!("synthetic units carry direct provenance");
        };
        assert_eq!(*page, 1);
        assert_eq!((bbox.x, bbox.width), (0.0, 280.0));
    }

    #[test]
    fn below_joins_typed_value_within_reach() {
        let pages = [page(
            1,
            vec![
                item("Ship Date", 0.0, 0.0, 60.0, 10.0),
                item("01/15/2024", 5.0, 15.0, 60.0, 10.0), // dy 5 <= 20
                item("Order Total", 200.0, 0.0, 60.0, 10.0),
                item("$1,250.00", 205.0, 60.0, 55.0, 10.0), // dy 50 > 20 → out of reach
            ],
        )];
        let units = geometry_units_by_kind(&pages);
        assert_eq!(texts(&units), ["Ship Date: 01/15/2024"]);
        assert_eq!(units[0].0, JoinKind::Below);
    }

    #[test]
    fn below_requires_typed_value() {
        // A bare number under a label is the measured flood case — no join.
        let pages = [page(
            1,
            vec![
                item("Statement(s)", 0.0, 0.0, 60.0, 10.0),
                item("1", 10.0, 15.0, 8.0, 10.0),
            ],
        )];
        assert!(geometry_units_by_kind(&pages).is_empty());
    }

    #[test]
    fn far_header_joins_row_entity_and_column_header() {
        // The rate-card shape: a column header high above, row entities at the
        // left, typed values in the column — no detected table anywhere.
        let pages = [page(
            1,
            vec![
                item("Price", 200.0, 0.0, 40.0, 10.0),
                item("Standard", 0.0, 50.0, 60.0, 10.0),
                item("$146,688", 200.0, 50.0, 55.0, 10.0),
                item("Premium", 0.0, 70.0, 60.0, 10.0),
                item("$213,024", 200.0, 70.0, 55.0, 10.0),
            ],
        )];
        let units = geometry_units_by_kind(&pages);
        let all = texts(&units);
        assert!(
            all.contains(&"Standard - Price: $146,688"),
            "missing far-header unit; got {all:?}"
        );
        assert!(all.contains(&"Premium - Price: $213,024"), "got {all:?}");
        let (kind, unit) = units
            .iter()
            .find(|(_, u)| u.text == "Standard - Price: $146,688")
            .unwrap();
        assert_eq!(*kind, JoinKind::FarHeader);
        // Union bbox spans row entity → header → value.
        let crate::extraction_unit::UnitProvenance::Direct { bbox, .. } = &unit.provenance else {
            panic!("direct provenance");
        };
        assert_eq!((bbox.x, bbox.y), (0.0, 0.0));
        assert_eq!((bbox.width, bbox.height), (255.0, 60.0));
    }

    #[test]
    fn far_header_requires_both_header_and_row_entity() {
        // Typed value with a header above but nothing label-ish to its left:
        // dense-prose quietness depends on requiring BOTH.
        let pages = [page(
            1,
            vec![
                item("Price", 200.0, 0.0, 40.0, 10.0),
                item("$146,688", 200.0, 50.0, 55.0, 10.0),
            ],
        )];
        let units = geometry_units_by_kind(&pages);
        assert!(
            !texts(&units).iter().any(|t| t.contains(" - ")),
            "far-header must not fire without a row entity: {units:?}"
        );
    }

    #[test]
    fn dense_prose_stays_quiet() {
        let pages = [page(
            1,
            vec![
                item(
                    "the parties agree that all disputes arising",
                    0.0,
                    0.0,
                    300.0,
                    10.0,
                ),
                item(
                    "shall be resolved through binding arbitration",
                    0.0,
                    12.0,
                    300.0,
                    10.0,
                ),
                item(
                    "pursuant to the rules then in effect in 2024",
                    0.0,
                    24.0,
                    300.0,
                    10.0,
                ),
            ],
        )];
        assert!(geometry_units_by_kind(&pages).is_empty());
    }

    #[test]
    fn per_page_cap_and_dedupe() {
        // 50 identical label/value rows: dedupe collapses the repeats first,
        // then distinct rows are capped at MAX_UNITS_PER_PAGE.
        let mut items = Vec::new();
        for i in 0..50 {
            let y = i as f32 * 20.0;
            items.push(item(&format!("Fee Code {i}:"), 0.0, y, 60.0, 10.0));
            items.push(item(&format!("FEE-{i:04}"), 200.0, y, 60.0, 10.0));
        }
        let one_page = page(1, items.clone());
        assert_eq!(
            geometry_units_by_kind(&[one_page]).len(),
            MAX_UNITS_PER_PAGE
        );

        let dup = page(
            1,
            vec![
                item("Total:", 0.0, 0.0, 40.0, 10.0),
                item("$100", 200.0, 0.0, 40.0, 10.0),
                item("Total:", 0.0, 40.0, 40.0, 10.0),
                item("$100", 200.0, 40.0, 40.0, 10.0),
            ],
        );
        assert_eq!(geometry_units_by_kind(&[dup]).len(), 1);
    }

    #[test]
    fn pages_are_independent_and_carry_page_number() {
        let pages = [
            page(
                3,
                vec![
                    item("Invoice Number:", 0.0, 0.0, 80.0, 10.0),
                    item("INV-2024-0042", 200.0, 0.0, 80.0, 10.0),
                ],
            ),
            page(
                4,
                vec![
                    item("PO Number:", 0.0, 0.0, 80.0, 10.0),
                    item("PO-7788", 200.0, 0.0, 80.0, 10.0),
                ],
            ),
        ];
        let units = geometry_units_by_kind(&pages);
        assert_eq!(units.len(), 2);
        let pages_of: Vec<usize> = units
            .iter()
            .map(|(_, u)| match &u.provenance {
                crate::extraction_unit::UnitProvenance::Direct { page, .. } => *page,
                _ => panic!("direct provenance"),
            })
            .collect();
        assert_eq!(pages_of, [3, 4]);
    }
}
