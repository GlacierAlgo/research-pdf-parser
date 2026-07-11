//! Label-anchored cell extraction for plain-string fields — the value path's
//! biggest single win (Phase 0 `Result 3d`: plain-string exact-match 8.8% →
//! 19.0%, 38 won / 0 lost). Ported 1:1 from the frozen Python prototype
//! (`dataset_eval_utils/extract_poc/extract_cli.py::span_search`).
//!
//! The idea: a retrieved projected line often carries several fields' values as
//! whitespace-separated *cells* of one visually-multi-column line
//! (`"Settlement Agent    Zeta Title    Lender    Ficus Bank"`). Values are
//! opaque (IDs, names, amounts carry no embedding gradient toward the query),
//! but **labels are query paraphrases** — the regime static embeddings are good
//! at. So we score each cell as a candidate *label* against the field's label
//! vocabulary (lexical echo + optional embedding cosine, equal weight), then
//! return the value that *follows* the winning label. This recovers the
//! geometry rightward-join from intra-line column structure that projection had
//! merged into a single line.
//!
//! The embedding half is optional: pass a closure that returns per-cell cosines
//! when a model is attached, or `None` for a pure-lexical score (the Phase 0
//! ablation put lexical-only at 20.1% vs 21.1% with the embedding — most of the
//! win survives with no model, so a BM25-only build still benefits).

use std::collections::HashSet;

/// Minimum label-match score to trust a cell as *the* label (`SPAN_LABEL_FLOOR`).
const LABEL_FLOOR: f32 = 0.35;
/// Embedding share of the label score; the rest is lexical echo (`SPAN_EMBED_W`).
const EMBED_W: f32 = 0.5;

/// Query stopwords — dropped from label vocabulary and query terms so the echo
/// match keys on content words. Verbatim from the Python `_STOP` set.
const STOP: &[&str] = &[
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "be", "was",
    "were", "this", "that", "with", "by", "from", "as", "it", "its", "their", "his", "her", "our",
    "your", "any", "all", "per", "each", "such", "via", "if", "no", "not", "do", "does", "did",
    "done", "has", "have", "had", "what", "which", "who", "whom", "whose", "how", "when", "where",
    "why", "will", "would", "can", "could", "should", "may", "might", "must",
];

/// Content terms of a string: lowercased ASCII-alphanumeric tokens, longer than
/// one char and not a stopword. (`_content_terms`.)
fn content_terms(text: &str) -> HashSet<String> {
    ascii_tokens(text)
        .into_iter()
        .filter(|t| t.len() > 1 && !STOP.contains(&t.as_str()))
        .collect()
}

/// Lowercased ASCII-alphanumeric tokens of `text` (runs of anything else split).
fn ascii_tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Token matches a query content term, allowing morphological variants
/// (lender/lending, dated/date) via a shared ≥4-char prefix. (`_echoes`.)
fn echoes(tok: &str, qterms: &HashSet<String>) -> bool {
    if qterms.contains(tok) {
        return true;
    }
    qterms
        .iter()
        .any(|q| matches!((tok.get(..4), q.get(..4)), (Some(a), Some(b)) if a == b))
}

/// Split a projected line into visual cells on runs of 2+ whitespace; single
/// spaces stay inside a cell. (`_cells` / `re.split(r"\s{2,}", text)`.)
pub(super) fn cells(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut ws_run = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            ws_run += 1;
            continue;
        }
        if ws_run >= 2 {
            let trimmed = cur.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            cur.clear();
        } else if ws_run == 1 && !cur.is_empty() {
            cur.push(' ');
        }
        ws_run = 0;
        cur.push(ch);
    }
    let trimmed = cur.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

/// Terms that can appear in the document's LABEL for this field: the field
/// name's own tokens plus the description's first clause. Everything from
/// `e.g.`/`Do NOT` on is cut — bench/LLM descriptions embed the example value
/// and negative instructions there, which would let the VALUE cell echo the
/// query and win as its own label. (`label_terms_for`.)
pub(super) fn label_terms_for(name: &str, desc: Option<&str>) -> HashSet<String> {
    let clause = desc.map(first_clause).unwrap_or("");
    content_terms(&format!("{} {}", name.replace('_', " "), clause))
}

/// The description up to the first `e.g.` / `Do NOT` / `do not` marker.
fn first_clause(desc: &str) -> &str {
    let cut = ["e.g.", "Do NOT", "do not"]
        .iter()
        .filter_map(|m| desc.find(m))
        .min();
    match cut {
        Some(i) => &desc[..i],
        None => desc,
    }
}

/// Text after the longest run of query-echoing words (ties → last run):
/// `"PAGE 1 OF 5 . LOAN ID # 123456789"` → `"123456789"`. `None` when the label
/// is a prefix-free miss or sits at the end. (`_echo_run_tail`.)
fn echo_run_tail(words: &[&str], qterms: &HashSet<String>) -> Option<String> {
    let norm = |w: &str| -> String {
        w.to_lowercase()
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect()
    };
    // Maximal runs of echoing words (bare-punctuation words extend a run).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let tok = norm(words[i]);
        if !tok.is_empty() && echoes(&tok, qterms) {
            let mut j = i;
            while j < words.len() {
                let t = norm(words[j]);
                if !t.is_empty() && !echoes(&t, qterms) {
                    break;
                }
                j += 1; // echoing word, or bare punctuation like '#'
            }
            runs.push((i, j));
            i = j;
        } else {
            i += 1;
        }
    }
    // Longest run wins; ties → later start (matches Python's max key `(len, start)`).
    let (_, end) = runs.into_iter().max_by_key(|&(s, e)| (e - s, s))?;
    let mut tail = &words[end..];
    while let [first, rest @ ..] = tail {
        if norm(first).is_empty() {
            tail = rest; // drop leading bare punctuation
        } else {
            break;
        }
    }
    if tail.is_empty() {
        None
    } else {
        Some(tail.join(" "))
    }
}

/// Reject "values" that are label debris: trailing-colon fragments (`"To:"`,
/// `"NO.:"`) or text with no non-echoing alphanumeric content. (`_plausible_value`.)
fn plausible_value(v: &str, qterms: &HashSet<String>) -> bool {
    if v.trim_end().ends_with(':') {
        return false;
    }
    ascii_tokens(v).iter().any(|t| !echoes(t, qterms))
}

/// Label-anchored cell extraction over a single unit's text (`SPAN_TOPK = 1` —
/// the Phase 0 evidence was that cross-unit search only destroys). Returns the
/// extracted value, or `None` so the caller falls back to the baseline
/// label-stripped span.
///
/// `embed_cells` yields a per-cell cosine to the query when a model is attached
/// (aligned to the returned cells), or `None` for a pure-lexical score.
pub(super) fn span_search(
    unit_text: &str,
    qterms: &HashSet<String>,
    embed_cells: impl Fn(&[String]) -> Option<Vec<f32>>,
) -> Option<String> {
    let cells = cells(unit_text);
    if cells.len() < 2 {
        return None; // prose (one cell) → no harm possible
    }
    let sims = embed_cells(&cells);

    // Score each cell as a candidate LABEL.
    let mut best: Option<usize> = None;
    let mut best_s = LABEL_FLOOR;
    for (i, cell) in cells.iter().enumerate() {
        let label = match cell.rfind(':') {
            Some(x) => &cell[..x],
            None => cell.as_str(),
        };
        let toks = ascii_tokens(label);
        if toks.is_empty() {
            continue;
        }
        let echo = toks.iter().filter(|t| echoes(t, qterms)).count() as f32 / toks.len() as f32;
        if echo == 0.0 {
            continue; // embedding alone may not nominate a label
        }
        let s = match &sims {
            Some(sv) => (1.0 - EMBED_W) * echo + EMBED_W * sv[i],
            None => echo,
        };
        if s > best_s {
            best_s = s;
            best = Some(i);
        }
    }
    let bi = best?;
    let cell = &cells[bi];

    // Value = first plausible of: colon-residual in the label cell → the cell's
    // text after the label's echo run → the next cell.
    let mut candidates: Vec<String> = Vec::new();
    if let Some(x) = cell.rfind(':') {
        candidates.push(cell[x + 1..].trim().to_string());
    }
    let words: Vec<&str> = cell.split_whitespace().collect();
    if let Some(tail) = echo_run_tail(&words, qterms) {
        candidates.push(tail);
    }
    if let Some(next) = cells.get(bi + 1) {
        candidates.push(next.clone());
    }
    candidates
        .into_iter()
        .find(|v| !v.is_empty() && plausible_value(v, qterms))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(words: &[&str]) -> HashSet<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn cells_split_on_two_plus_spaces() {
        assert_eq!(cells("a  b"), ["a", "b"]);
        assert_eq!(cells("a b"), ["a b"]);
        assert_eq!(
            cells("Settlement Agent    Zeta Title    Lender    Ficus Bank"),
            ["Settlement Agent", "Zeta Title", "Lender", "Ficus Bank"]
        );
    }

    #[test]
    fn echoes_allows_shared_prefix() {
        let q = terms(&["lender"]);
        assert!(echoes("lender", &q));
        assert!(echoes("lending", &q)); // shared 4-char prefix
        assert!(!echoes("led", &q)); // too short to prefix-match
        assert!(!echoes("borrower", &q));
    }

    #[test]
    fn label_terms_cut_at_eg() {
        let t = label_terms_for("lender", Some("the lending bank e.g. Ficus Bank"));
        assert!(t.contains("lender") && t.contains("lending") && t.contains("bank"));
        assert!(!t.contains("ficus")); // cut at e.g.
    }

    #[test]
    fn echo_run_tail_returns_text_after_longest_run() {
        let q = terms(&["loan", "id"]);
        let words: Vec<&str> = "PAGE 1 OF 5 # LOAN ID # 123456789"
            .split_whitespace()
            .collect();
        assert_eq!(echo_run_tail(&words, &q).as_deref(), Some("123456789"));
    }

    #[test]
    fn picks_value_after_matching_label_cell() {
        let q = label_terms_for("lender", Some("the lending bank"));
        let got = span_search(
            "Settlement Agent    Zeta Title    Lender    Ficus Bank",
            &q,
            |_| None,
        );
        assert_eq!(got.as_deref(), Some("Ficus Bank"));
    }

    #[test]
    fn colon_residual_is_preferred() {
        let q = label_terms_for("total", Some("the total amount"));
        let got = span_search("Subtotal $10.00    Total: $18.00", &q, |_| None);
        assert_eq!(got.as_deref(), Some("$18.00"));
    }

    #[test]
    fn prose_line_yields_nothing() {
        let q = label_terms_for("lender", Some("the lending bank"));
        assert_eq!(
            span_search("the quick brown fox jumped", &q, |_| None),
            None
        );
    }

    #[test]
    fn no_echoing_label_yields_nothing() {
        let q = label_terms_for("lender", Some("the lending bank"));
        // Two cells but neither cell's label echoes the query.
        assert_eq!(span_search("Buyer    John Smith", &q, |_| None), None);
    }
}
