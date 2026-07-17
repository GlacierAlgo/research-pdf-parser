//! Read-only formula-region probe run immediately before grid projection.
//!
//! The probe intentionally stops at routing.  It preserves native tokens for
//! code-like rows and marks genuinely two-dimensional regions for a caller-
//! supplied image-to-LaTeX model; it never guesses LaTeX in the Rust core.

use crate::types::{FormulaCandidate, FormulaRoute, GraphicPrimitive, Page, Rect, TextItem};

const MIN_RULE_WIDTH_RATIO: f32 = 0.35;
const MIN_CANDIDATE_WIDTH: f32 = 44.0;

#[derive(Debug, Clone, Copy)]
struct RowBand {
    top: f32,
    bottom: f32,
}

fn item_rect(item: &TextItem) -> Rect {
    Rect {
        x: item.x,
        y: item.y,
        width: item.width,
        height: item.height,
    }
}

fn rect_right(rect: &Rect) -> f32 {
    rect.x + rect.width
}

fn rect_bottom(rect: &Rect) -> f32 {
    rect.y + rect.height
}

fn union_rects<'a>(rects: impl Iterator<Item = &'a Rect>) -> Option<Rect> {
    let mut x0 = f32::INFINITY;
    let mut y0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    let mut seen = false;
    for rect in rects {
        seen = true;
        x0 = x0.min(rect.x);
        y0 = y0.min(rect.y);
        x1 = x1.max(rect_right(rect));
        y1 = y1.max(rect_bottom(rect));
    }
    seen.then_some(Rect {
        x: x0,
        y: y0,
        width: (x1 - x0).max(0.0),
        height: (y1 - y0).max(0.0),
    })
}

fn padded_rect(rect: &Rect, x_pad: f32, y_pad: f32, page: &Page) -> Rect {
    let x0 = (rect.x - x_pad).max(0.0);
    let y0 = (rect.y - y_pad).max(0.0);
    let x1 = (rect_right(rect) + x_pad).min(page.page_width);
    let y1 = (rect_bottom(rect) + y_pad).min(page.page_height);
    Rect {
        x: x0,
        y: y0,
        width: (x1 - x0).max(0.0),
        height: (y1 - y0).max(0.0),
    }
}

fn is_private_use(ch: char) -> bool {
    ('\u{e000}'..='\u{f8ff}').contains(&ch)
}

fn has_private_use(text: &str) -> bool {
    text.chars().any(is_private_use)
}

fn is_cjk(ch: char) -> bool {
    matches!(ch, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}')
}

fn cjk_ratio(text: &str) -> f32 {
    let visible: Vec<char> = text.chars().filter(|ch| !ch.is_whitespace()).collect();
    if visible.is_empty() {
        return 0.0;
    }
    visible.iter().filter(|ch| is_cjk(**ch)).count() as f32 / visible.len() as f32
}

fn has_math_font(item: &TextItem) -> bool {
    item.font_name.as_deref().is_some_and(|name| {
        let lower = name.to_ascii_lowercase();
        lower.contains("symbol") || lower.contains("math") || lower.contains("mt extra")
    })
}

fn is_alpha_label(text: &str) -> bool {
    let compact: String = text.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact
        .strip_prefix("Alpha")
        .is_some_and(|tail| !tail.is_empty() && tail.chars().all(|ch| ch.is_ascii_digit()))
}

fn has_formula_word(text: &str) -> bool {
    let upper = text.to_ascii_uppercase();
    [
        "RANK(", "SUM(", "SUMAC(", "MEAN(", "STD(", "CORR(", "DELAY(", "DELTA(", "MAX(", "MIN(",
        "LOG(", "LN(", "EXP(", "SIGN(", "SMA(", "REGRES",
    ]
    .iter()
    .any(|needle| upper.contains(needle))
}

fn has_math_operator(text: &str) -> bool {
    text.chars().any(|ch| {
        is_private_use(ch)
            || matches!(
                ch,
                '=' | '+' | '-' | '*' | '/' | '^' | '<' | '>' | '−' | '∑' | '√' | '±' | '×'
            )
    })
}

fn is_formula_seed(item: &TextItem) -> bool {
    let text = item.text.trim();
    !text.is_empty()
        && cjk_ratio(text) < 0.25
        && (has_math_font(item)
            || has_private_use(text)
            || has_formula_word(text)
            || is_alpha_label(text)
            || (has_math_operator(text) && text.chars().count() >= 2))
}

fn is_component_token(item: &TextItem) -> bool {
    let text = item.text.trim();
    if text.is_empty() || cjk_ratio(text) >= 0.25 {
        return false;
    }
    text.chars().all(|ch| {
        ch.is_ascii()
            || is_private_use(ch)
            || ch.is_whitespace()
            || matches!(ch, '（' | '）' | '，' | '：' | '；' | '？')
            || matches!(ch, '±' | '×' | 'α'..='ω' | '∑' | '√' | '−')
    })
}

fn horizontal_gap(a: &Rect, b: &Rect) -> f32 {
    if rect_right(a) < b.x {
        b.x - rect_right(a)
    } else if rect_right(b) < a.x {
        a.x - rect_right(b)
    } else {
        0.0
    }
}

fn vertical_gap(a: &Rect, b: &Rect) -> f32 {
    if rect_bottom(a) < b.y {
        b.y - rect_bottom(a)
    } else if rect_bottom(b) < a.y {
        a.y - rect_bottom(b)
    } else {
        0.0
    }
}

fn overlap_1d(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

fn items_connected(a: &TextItem, b: &TextItem) -> bool {
    let ar = item_rect(a);
    let br = item_rect(b);
    let a_center_y = ar.y + ar.height / 2.0;
    let b_center_y = br.y + br.height / 2.0;
    let y_overlap = overlap_1d(ar.y, rect_bottom(&ar), br.y, rect_bottom(&br));
    let x_overlap = overlap_1d(ar.x, rect_right(&ar), br.x, rect_right(&br));
    let same_line = (y_overlap >= ar.height.min(br.height) * 0.2
        || (a_center_y - b_center_y).abs() <= ar.height.max(br.height) * 0.65)
        && horizontal_gap(&ar, &br) <= 16.0;
    let stacked =
        x_overlap >= ar.width.min(br.width).min(8.0) * 0.2 && vertical_gap(&ar, &br) <= 4.5;
    let script_neighbor =
        horizontal_gap(&ar, &br) <= 5.0 && (a_center_y - b_center_y).abs() <= 12.0;
    same_line || stacked || script_neighbor
}

fn horizontal_rule_ys(page: &Page) -> Vec<f32> {
    let min_width = page.page_width * MIN_RULE_WIDTH_RATIO;
    let mut ys = Vec::new();
    for graphic in &page.graphics {
        match graphic {
            GraphicPrimitive::Stroke { x1, y1, x2, y2, .. }
                if (y1 - y2).abs() <= 0.8 && (x1 - x2).abs() >= min_width =>
            {
                ys.push((y1 + y2) / 2.0);
            }
            GraphicPrimitive::Rect { bbox, .. }
                if bbox.height <= 1.2 && bbox.width >= min_width =>
            {
                ys.push(bbox.y + bbox.height / 2.0);
            }
            _ => {}
        }
    }
    ys.sort_by(f32::total_cmp);
    let mut merged: Vec<f32> = Vec::new();
    for y in ys {
        if let Some(last) = merged.last_mut()
            && (y - *last).abs() <= 1.0
        {
            *last = (*last + y) / 2.0;
            continue;
        }
        merged.push(y);
    }
    merged
}

fn row_bands(page: &Page) -> Vec<RowBand> {
    horizontal_rule_ys(page)
        .windows(2)
        .filter_map(|pair| {
            let height = pair[1] - pair[0];
            // Very long Alpha expressions can occupy seven wrapped lines.
            (8.0..=120.0).contains(&height).then_some(RowBand {
                top: pair[0],
                bottom: pair[1],
            })
        })
        .collect()
}

fn item_center_y(item: &TextItem) -> f32 {
    item.y + item.height / 2.0
}

fn item_in_band(item: &TextItem, band: RowBand) -> bool {
    let center = item_center_y(item);
    center > band.top && center < band.bottom
}

fn text_in_reading_order(page: &Page, indices: &[usize], join_lines: bool) -> String {
    let mut ordered = indices.to_vec();
    ordered.sort_by(|a, b| {
        let left = &page.text_items[*a];
        let right = &page.text_items[*b];
        left.y
            .total_cmp(&right.y)
            .then_with(|| left.x.total_cmp(&right.x))
    });

    let mut lines: Vec<Vec<&TextItem>> = Vec::new();
    for index in ordered {
        let item = &page.text_items[index];
        let center = item_center_y(item);
        let belongs = lines.last().is_some_and(|line| {
            let previous =
                line.iter().map(|span| item_center_y(span)).sum::<f32>() / line.len() as f32;
            (center - previous).abs() <= 5.5
        });
        if belongs {
            lines.last_mut().unwrap().push(item);
        } else {
            lines.push(vec![item]);
        }
    }

    let rendered: Vec<String> = lines
        .into_iter()
        .map(|mut line| {
            line.sort_by(|a, b| a.x.total_cmp(&b.x));
            line.into_iter().map(|item| item.text.trim()).collect()
        })
        .filter(|line: &String| !line.is_empty())
        .collect();
    if join_lines {
        rendered.concat()
    } else {
        rendered.join("\n")
    }
}

fn build_alpha_candidate(
    page: &Page,
    label_index: usize,
    body: Vec<usize>,
    used: &mut [bool],
) -> Option<FormulaCandidate> {
    if body.is_empty() {
        return None;
    }
    let label = &page.text_items[label_index];
    let mut indices = vec![label_index];
    indices.extend(body.iter().copied());
    let rects: Vec<Rect> = indices
        .iter()
        .map(|index| item_rect(&page.text_items[*index]))
        .collect();
    let raw_bbox = union_rects(rects.iter())?;
    let label_text = label.text.trim();
    let body_text = text_in_reading_order(page, &body, true);
    for index in indices {
        used[index] = true;
    }
    Some(FormulaCandidate {
        id: String::new(),
        bbox: padded_rect(&raw_bbox, 3.0, 2.0, page),
        route: FormulaRoute::NativeText,
        text: format!("{label_text}\n{body_text}"),
        confidence: 0.99,
        reasons: vec!["alpha_code_row".into(), "ruled_table_band".into()],
    })
}

fn alpha_row_candidates(
    page: &Page,
    bands: &[RowBand],
    used: &mut [bool],
) -> Vec<FormulaCandidate> {
    let mut candidates = Vec::new();
    for band in bands {
        let labels: Vec<usize> = page
            .text_items
            .iter()
            .enumerate()
            .filter(|(_, item)| item_in_band(item, *band) && is_alpha_label(item.text.trim()))
            .map(|(index, _)| index)
            .collect();
        for label_index in labels {
            let label = &page.text_items[label_index];
            let body: Vec<usize> = page
                .text_items
                .iter()
                .enumerate()
                .filter(|(index, item)| {
                    !used[*index]
                        && item_in_band(item, *band)
                        && item.x >= label.x + label.width + 4.0
                        && is_component_token(item)
                })
                .map(|(index, _)| index)
                .collect();
            if let Some(candidate) = build_alpha_candidate(page, label_index, body, used) {
                candidates.push(candidate);
            }
        }
    }

    // A final table row may omit its closing rule. Infer its vertical extent
    // from neighboring Alpha labels instead of dropping it or sending it to
    // vision OCR.
    let mut all_labels: Vec<usize> = page
        .text_items
        .iter()
        .enumerate()
        .filter(|(_, item)| is_alpha_label(item.text.trim()))
        .map(|(index, _)| index)
        .collect();
    all_labels.sort_by(|a, b| {
        item_center_y(&page.text_items[*a]).total_cmp(&item_center_y(&page.text_items[*b]))
    });
    for (position, label_index) in all_labels.iter().copied().enumerate() {
        if used[label_index] {
            continue;
        }
        let label = &page.text_items[label_index];
        let center = item_center_y(label);
        let previous = position
            .checked_sub(1)
            .map(|index| item_center_y(&page.text_items[all_labels[index]]));
        let next = all_labels
            .get(position + 1)
            .map(|index| item_center_y(&page.text_items[*index]));
        let typical_gap = previous
            .map(|value| center - value)
            .or_else(|| next.map(|value| value - center))
            .unwrap_or(24.0)
            .clamp(12.0, 64.0);
        let top = previous.map_or(center - typical_gap, |value| (value + center) / 2.0);
        let bottom = next.map_or(center + typical_gap, |value| (value + center) / 2.0);
        let body: Vec<usize> = page
            .text_items
            .iter()
            .enumerate()
            .filter(|(index, item)| {
                !used[*index]
                    && item_center_y(item) > top
                    && item_center_y(item) < bottom
                    && item.x >= label.x + label.width + 4.0
                    && is_component_token(item)
            })
            .map(|(index, _)| index)
            .collect();
        if let Some(mut candidate) = build_alpha_candidate(page, label_index, body, used) {
            candidate.reasons = vec!["alpha_code_row".into(), "label_midpoint_band".into()];
            candidates.push(candidate);
        }
    }
    candidates
}

fn component_indices(page: &Page, used: &[bool]) -> Vec<Vec<usize>> {
    let eligible: Vec<usize> = page
        .text_items
        .iter()
        .enumerate()
        .filter(|(index, item)| !used[*index] && is_component_token(item))
        .map(|(index, _)| index)
        .collect();
    let mut visited = vec![false; eligible.len()];
    let mut components = Vec::new();
    for start in 0..eligible.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(position) = stack.pop() {
            let item_index = eligible[position];
            component.push(item_index);
            for other in 0..eligible.len() {
                if visited[other] {
                    continue;
                }
                if items_connected(
                    &page.text_items[item_index],
                    &page.text_items[eligible[other]],
                ) {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }
        components.push(component);
    }
    components
}

fn component_is_complex(page: &Page, indices: &[usize], bbox: &Rect) -> bool {
    let mut heights: Vec<f32> = indices
        .iter()
        .map(|index| page.text_items[*index].height)
        .filter(|height| *height > 0.0)
        .collect();
    if heights.is_empty() {
        return false;
    }
    heights.sort_by(f32::total_cmp);
    let median = heights[heights.len() / 2].max(1.0);
    let has_large_math_glyph = indices.iter().any(|index| {
        let item = &page.text_items[*index];
        has_math_font(item) && item.height >= median * 1.35
    });
    let has_script = indices.iter().any(|a| {
        indices.iter().any(|b| {
            if a == b {
                return false;
            }
            let left = &page.text_items[*a];
            let right = &page.text_items[*b];
            horizontal_gap(&item_rect(left), &item_rect(right)) <= 6.0
                && (item_center_y(left) - item_center_y(right)).abs() >= median * 0.35
        })
    });
    bbox.height >= median * 1.6 || has_large_math_glyph || has_script
}

fn compact_ascii_identifier(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .flat_map(char::to_uppercase)
        .collect()
}

fn drop_duplicate_table_label(page: &Page, indices: &mut Vec<usize>) {
    let mut remove = Vec::new();
    for left in indices.iter().copied() {
        let left_item = &page.text_items[left];
        let left_text = compact_ascii_identifier(&left_item.text);
        if left_text.len() < 3 {
            continue;
        }
        for right in indices.iter().copied() {
            let right_item = &page.text_items[right];
            if left_item.x >= right_item.x
                || horizontal_gap(&item_rect(left_item), &item_rect(right_item)) > 22.0
                || left_item.height < right_item.height
            {
                continue;
            }
            let right_text = compact_ascii_identifier(&right_item.text);
            let near_duplicate = left_text == right_text
                || (right_text.starts_with(&left_text) && right_text.len() - left_text.len() <= 1);
            if near_duplicate {
                remove.push(left);
                break;
            }
        }
    }
    indices.retain(|index| !remove.contains(index));
}

fn component_candidates(page: &Page, bands: &[RowBand], used: &[bool]) -> Vec<FormulaCandidate> {
    let mut candidates = Vec::new();
    for mut indices in component_indices(page, used) {
        drop_duplicate_table_label(page, &mut indices);
        if !indices
            .iter()
            .any(|index| is_formula_seed(&page.text_items[*index]))
        {
            continue;
        }
        let rects: Vec<Rect> = indices
            .iter()
            .map(|index| item_rect(&page.text_items[*index]))
            .collect();
        let Some(raw_bbox) = union_rects(rects.iter()) else {
            continue;
        };
        let complex = component_is_complex(page, &indices, &raw_bbox);
        let has_assignment = indices.iter().any(|index| {
            let text = &page.text_items[*index].text;
            text.contains('=') || text.contains('\u{f03d}')
        });
        let has_glyph_math_signal = indices.iter().any(|index| {
            let item = &page.text_items[*index];
            has_math_font(item) || has_private_use(&item.text)
        });
        let has_strong_math_signal = has_glyph_math_signal
            || indices.iter().any(|index| {
                let item = &page.text_items[*index];
                has_formula_word(&item.text) || is_alpha_label(item.text.trim())
            });
        let in_ruled_table = bands.iter().any(|band| {
            let center = raw_bbox.y + raw_bbox.height / 2.0;
            center > band.top && center < band.bottom
        });
        let incomplete_table_formula = in_ruled_table && !has_assignment && has_glyph_math_signal;
        let minimum_width = if incomplete_table_formula {
            // Compact table definitions such as A_{i-n} can collapse to a
            // very narrow, scrambled PDF text run even though their vector
            // glyphs are perfectly usable by the local formula recognizer.
            10.0
        } else if in_ruled_table {
            24.0
        } else {
            MIN_CANDIDATE_WIDTH
        };
        if raw_bbox.width < minimum_width
            || !has_strong_math_signal
            || (!complex && !has_assignment && !incomplete_table_formula)
        {
            continue;
        }
        let x_pad = if in_ruled_table {
            // Formula glyphs at the left edge of a table cell are commonly
            // absent from PDF text extraction. Keep enough vector context for
            // a vision recognizer without swallowing the factor-name column.
            12.0
        } else {
            (raw_bbox.width * 0.25).clamp(8.0, 50.0)
        };
        let candidate_text = text_in_reading_order(page, &indices, false);
        let trimmed_text = candidate_text.trim_end();
        let suspicious_linear_order = has_assignment
            && (trimmed_text.ends_with('/')
                || trimmed_text.trim_start().starts_with(['=', '\u{f03d}']));
        let route = if complex || incomplete_table_formula || suspicious_linear_order {
            FormulaRoute::Vision
        } else {
            FormulaRoute::NativeText
        };
        let mut reasons = Vec::new();
        if indices
            .iter()
            .any(|index| has_math_font(&page.text_items[*index]))
        {
            reasons.push("math_font".into());
        }
        if indices
            .iter()
            .any(|index| has_private_use(&page.text_items[*index].text))
        {
            reasons.push("private_use_glyph".into());
        }
        if complex {
            reasons.push("multi_baseline_or_large_glyph".into());
        }
        if incomplete_table_formula {
            reasons.push("incomplete_table_formula".into());
        }
        if suspicious_linear_order {
            reasons.push("suspicious_linear_order".into());
        }
        if in_ruled_table {
            reasons.push("ruled_table_cell".into());
        }
        candidates.push(FormulaCandidate {
            id: String::new(),
            bbox: padded_rect(&raw_bbox, x_pad, 4.0, page),
            route,
            text: candidate_text,
            confidence: if complex {
                0.9
            } else if incomplete_table_formula {
                0.72
            } else {
                0.78
            },
            reasons,
        });
    }
    candidates
}

/// Detect formula-shaped regions without altering the page or interpreting
/// mathematical semantics.  Candidates are sorted spatially and receive stable
/// page-scoped identifiers.
pub fn probe_formula_candidates(page: &Page) -> Vec<FormulaCandidate> {
    if page.text_items.is_empty() {
        return Vec::new();
    }
    let bands = row_bands(page);
    let mut used = vec![false; page.text_items.len()];
    let mut candidates = alpha_row_candidates(page, &bands, &mut used);
    let mut components = component_candidates(page, &bands, &used);

    // A long Alpha row can start a fraction of a point above the detected
    // horizontal rule while its vertically-centred label remains below it.
    // Reattach that isolated leading line to the following Alpha candidate.
    components.retain(|component| {
        if component.route != FormulaRoute::NativeText {
            return true;
        }
        let component_bottom = rect_bottom(&component.bbox);
        let Some(next) = candidates
            .iter_mut()
            .filter(|candidate| {
                candidate
                    .reasons
                    .iter()
                    .any(|reason| reason == "alpha_code_row")
                    && candidate.bbox.y >= component_bottom
                    && candidate.bbox.y - component_bottom <= 4.0
            })
            .min_by(|left, right| left.bbox.y.total_cmp(&right.bbox.y))
        else {
            return true;
        };
        let Some((label, body)) = next.text.split_once('\n') else {
            return true;
        };
        next.text = format!("{label}\n{}{body}", component.text.replace('\n', ""));
        let merged = [component.bbox.clone(), next.bbox.clone()];
        if let Some(bbox) = union_rects(merged.iter()) {
            next.bbox = bbox;
        }
        false
    });
    candidates.extend(components);
    candidates.sort_by(|a, b| {
        a.bbox
            .y
            .total_cmp(&b.bbox.y)
            .then_with(|| a.bbox.x.total_cmp(&b.bbox.x))
    });
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.id = format!("formula_p{:04}_{:03}", page.page_number, index + 1);
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str, x: f32, y: f32, width: f32, height: f32, font: &str) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width,
            height,
            font_name: Some(font.into()),
            font_size: Some(height),
            ..Default::default()
        }
    }

    fn page(items: Vec<TextItem>, graphics: Vec<GraphicPrimitive>) -> Page {
        Page {
            page_number: 5,
            page_width: 600.0,
            page_height: 800.0,
            text_items: items,
            graphics,
            struct_nodes: vec![],
            image_refs: vec![],
        }
    }

    fn rule(y: f32) -> GraphicPrimitive {
        GraphicPrimitive::Rect {
            bbox: Rect {
                x: 45.0,
                y,
                width: 500.0,
                height: 0.25,
            },
            fill: Some("ff000000".into()),
            stroke: None,
        }
    }

    #[test]
    fn alpha_row_uses_native_tokens_and_joins_pdf_line_wraps() {
        let page = page(
            vec![
                item("Alpha159", 51.0, 125.0, 37.0, 8.0, "Times"),
                item("CLO", 100.0, 110.0, 20.0, 8.0, "Times"),
                item("SE+", 100.0, 126.0, 24.0, 8.0, "Times"),
                item("DELAY(CLOSE,1)", 125.0, 126.0, 90.0, 8.0, "Times"),
            ],
            vec![rule(100.0), rule(150.0)],
        );
        let candidates = probe_formula_candidates(&page);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].route, FormulaRoute::NativeText);
        assert_eq!(candidates[0].text, "Alpha159\nCLOSE+DELAY(CLOSE,1)");
        assert!(candidates[0].reasons.contains(&"alpha_code_row".into()));
    }

    #[test]
    fn multi_baseline_math_routes_to_vision() {
        let page = page(
            vec![
                item("STOM", 150.0, 200.0, 30.0, 8.0, "Times"),
                item("\u{f03d} ln ( \u{f0e5}", 185.0, 199.0, 45.0, 13.0, "Symbol"),
                item("21", 220.0, 193.0, 8.0, 5.0, "Times"),
                item("t=1", 220.0, 211.0, 12.0, 5.0, "Times"),
                item("(Vt/St)", 234.0, 200.0, 42.0, 8.0, "Times"),
                item("；其中成交量", 280.0, 200.0, 70.0, 9.0, "CJK"),
            ],
            vec![rule(180.0), rule(225.0)],
        );
        let candidates = probe_formula_candidates(&page);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].route, FormulaRoute::Vision);
        assert!(candidates[0].bbox.x < 150.0);
        assert!(rect_right(&candidates[0].bbox) < 300.0);
        assert!(!candidates[0].text.contains("其中"));
    }

    #[test]
    fn ordinary_prose_is_ignored() {
        let page = page(
            vec![item(
                "This is ordinary prose without a formula.",
                50.0,
                100.0,
                220.0,
                10.0,
                "Arial",
            )],
            vec![],
        );
        assert!(probe_formula_candidates(&page).is_empty());
    }

    #[test]
    fn hyphenated_contact_footer_is_ignored() {
        let page = page(
            vec![
                item("200120", 20.0, 740.0, 45.0, 9.0, "Arial"),
                item("(021) 38676666", 70.0, 740.0, 90.0, 9.0, "Arial"),
                item(
                    "E-mail: research@example.com",
                    165.0,
                    740.0,
                    170.0,
                    9.0,
                    "Arial",
                ),
            ],
            vec![],
        );
        assert!(probe_formula_candidates(&page).is_empty());
    }

    #[test]
    fn incomplete_ruled_table_formula_routes_to_vision() {
        let page = page(
            vec![
                item("CMRA", 150.0, 120.0, 30.0, 8.0, "Times"),
                item(
                    "ln \u{f02b} (1 max{Z(T)})",
                    190.0,
                    120.0,
                    100.0,
                    8.0,
                    "Symbol",
                ),
            ],
            vec![rule(110.0), rule(140.0)],
        );
        let candidates = probe_formula_candidates(&page);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].route, FormulaRoute::Vision);
        assert!(
            candidates[0]
                .reasons
                .contains(&"incomplete_table_formula".into())
        );
    }

    #[test]
    fn compact_script_formula_in_ruled_table_routes_to_vision() {
        let page = page(
            vec![item(
                "i n A \u{f02d}",
                199.0,
                420.0,
                13.0,
                8.0,
                "Times New Roman,Italic",
            )],
            vec![rule(410.0), rule(430.0)],
        );
        let candidates = probe_formula_candidates(&page);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].route, FormulaRoute::Vision);
        assert!(
            candidates[0]
                .reasons
                .contains(&"incomplete_table_formula".into())
        );
    }
}
