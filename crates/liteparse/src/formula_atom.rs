//! Validated formula-region replacement before final grid projection.

use crate::error::LiteParseError;
use crate::types::{FormulaAtom, Page, Rect, TextItem, TextItemKind};

fn right(rect: &Rect) -> f32 {
    rect.x + rect.width
}

fn bottom(rect: &Rect) -> f32 {
    rect.y + rect.height
}

fn materially_overlaps(a: &Rect, b: &Rect) -> bool {
    let width = (right(a).min(right(b)) - a.x.max(b.x)).max(0.0);
    let height = (bottom(a).min(bottom(b)) - a.y.max(b.y)).max(0.0);
    let smaller_area = (a.width * a.height).min(b.width * b.height).max(1.0);
    width * height / smaller_area >= 0.25
}

fn item_is_replaced(item: &TextItem, bbox: &Rect) -> bool {
    let center_x = item.x + item.width * 0.5;
    let center_y = item.y + item.height * 0.5;
    let center_inside = center_x >= bbox.x
        && center_x <= right(bbox)
        && center_y >= bbox.y
        && center_y <= bottom(bbox);
    if center_inside {
        return true;
    }

    let ix = right(bbox).min(item.x + item.width) - bbox.x.max(item.x);
    let iy = bottom(bbox).min(item.y + item.height) - bbox.y.max(item.y);
    let item_area = item.width.max(0.0) * item.height.max(0.0);
    item_area > 0.0 && ix.max(0.0) * iy.max(0.0) / item_area >= 0.5
}

fn validate_atom(atom: &FormulaAtom, page_number: usize) -> Result<(), LiteParseError> {
    if atom.page_number != page_number {
        return Err(format!(
            "formula atom {} targets page {}, not page {}",
            atom.id, atom.page_number, page_number
        )
        .into());
    }
    if atom.id.trim().is_empty() || atom.markdown.trim().is_empty() {
        return Err("formula atom id and markdown must be non-empty".into());
    }
    let values = [
        atom.bbox.x,
        atom.bbox.y,
        atom.bbox.width,
        atom.bbox.height,
        atom.confidence,
    ];
    if values.iter().any(|value| !value.is_finite())
        || atom.bbox.width <= 0.0
        || atom.bbox.height <= 0.0
        || !(0.0..=1.0).contains(&atom.confidence)
    {
        return Err(format!("formula atom {} has invalid geometry/confidence", atom.id).into());
    }
    Ok(())
}

/// Replace every native text item covered by `atoms` and insert one semantic
/// formula item per region. Overlapping formula atoms are rejected because
/// their replacement order would otherwise change document meaning.
pub(crate) fn apply_formula_atoms(
    page: &mut Page,
    atoms: &[FormulaAtom],
) -> Result<(), LiteParseError> {
    for (index, atom) in atoms.iter().enumerate() {
        validate_atom(atom, page.page_number)?;
        if atoms[..index]
            .iter()
            .any(|previous| materially_overlaps(&previous.bbox, &atom.bbox))
        {
            return Err(format!("formula atom {} overlaps another atom", atom.id).into());
        }
    }

    for atom in atoms {
        let mut removed = Vec::new();
        page.text_items.retain(|item| {
            if item_is_replaced(item, &atom.bbox) {
                removed.push(item.clone());
                false
            } else {
                true
            }
        });

        let mut sizes: Vec<f32> = removed.iter().filter_map(|item| item.font_size).collect();
        sizes.sort_by(f32::total_cmp);
        let font_size = sizes
            .get(sizes.len() / 2)
            .copied()
            .unwrap_or_else(|| atom.bbox.height.clamp(8.0, 14.0));
        page.text_items.push(TextItem {
            text: atom.markdown.clone(),
            x: atom.bbox.x,
            y: atom.bbox.y,
            width: atom.bbox.width,
            height: atom.bbox.height,
            font_size: Some(font_size),
            font_height: Some(font_size),
            text_width: Some(atom.bbox.width),
            confidence: Some(atom.confidence),
            kind: TextItemKind::FormulaAtom {
                id: atom.id.clone(),
                source: atom.source.clone(),
            },
            ..Default::default()
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str, x: f32, y: f32, width: f32, height: f32) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width,
            height,
            font_size: Some(10.0),
            ..Default::default()
        }
    }

    fn atom(id: &str, x: f32) -> FormulaAtom {
        FormulaAtom {
            id: id.into(),
            page_number: 1,
            bbox: Rect {
                x,
                y: 10.0,
                width: 30.0,
                height: 12.0,
            },
            markdown: r"$x^2$".into(),
            confidence: 0.9,
            source: "test".into(),
        }
    }

    #[test]
    fn replaces_covered_items_with_one_formula_item() {
        let mut page = Page {
            page_number: 1,
            page_width: 100.0,
            page_height: 100.0,
            text_items: vec![
                item("x", 10.0, 10.0, 5.0, 10.0),
                item("2", 17.0, 8.0, 4.0, 6.0),
                item("keep", 60.0, 10.0, 20.0, 10.0),
            ],
            graphics: vec![],
            struct_nodes: vec![],
            image_refs: vec![],
        };
        apply_formula_atoms(&mut page, &[atom("f1", 8.0)]).unwrap();
        assert_eq!(page.text_items.len(), 2);
        let formula = page
            .text_items
            .iter()
            .find(|item| item.is_formula_atom())
            .unwrap();
        assert_eq!(formula.text, r"$x^2$");
        assert!(page.text_items.iter().any(|item| item.text == "keep"));
    }

    #[test]
    fn rejects_overlapping_atoms() {
        let mut page = Page {
            page_number: 1,
            page_width: 100.0,
            page_height: 100.0,
            text_items: vec![],
            graphics: vec![],
            struct_nodes: vec![],
            image_refs: vec![],
        };
        let error =
            apply_formula_atoms(&mut page, &[atom("f1", 10.0), atom("f2", 20.0)]).unwrap_err();
        assert!(error.to_string().contains("overlaps"));
    }
}
