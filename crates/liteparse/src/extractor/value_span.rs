//! Value-span extraction: pull a field's *value* out of a retrieved unit's
//! text. Given the span `"Invoice Number: INV-2024-0042"` and a field, return
//! `"INV-2024-0042"`; given `"Total VAT $18.00"` and a `number` field, return
//! `"$18.00"`.
//!
//! Values are returned **verbatim as they appear** — normalization (dates → ISO,
//! amounts → floats) is a documented v1 non-goal. These scanners only *locate
//! and slice* the value substring; they don't validate or reformat it.
//!
//! No `regex` dependency: the codebase does its text work with char scanning
//! (see `pdf_read.rs`), and these shapes are simple enough to match by hand. If
//! date coverage proves too weak this is the place to swap in `regex`.

use super::{FieldType, SchemaField};

/// Try to extract a typed/format/enum value from `text` for `field`. Returns
/// `None` when the expected shape isn't present ([`narrowed_value`] then decides
/// the fallback). Precedence: `format` → `choices` (enum) → `type`.
pub(super) fn typed_value(field: &SchemaField, text: &str) -> Option<String> {
    if let Some(fmt) = field.format.as_deref()
        && let Some(v) = match_format(fmt, text)
    {
        return Some(v);
    }
    if !field.choices.is_empty() {
        return classify_enum(&field.choices, text);
    }
    match field.field_type {
        FieldType::Date => find_date(text),
        FieldType::Int => find_integer(text),
        FieldType::Number => {
            if money_hint(&field.name) {
                find_money(text)
            } else {
                find_number(text)
            }
        }
        FieldType::Bool => find_bool(text),
        FieldType::Str | FieldType::List | FieldType::Other => None,
    }
}

/// Does the field's name say the number is an amount of *money* (price, fee,
/// total, pay, …)? Word-boundaried on the retrieval tokenizer, like
/// `schema_json::name_hints`. Count-ish numeric hints (qty, count, rate) are
/// deliberately absent — their values are legitimately bare digit runs, which
/// the money shape gate would reject.
fn money_hint(name: &str) -> bool {
    const MONEY: &[&str] = &[
        "amount", "total", "subtotal", "price", "cost", "fee", "balance", "tax", "discount",
        "shipping", "paid", "pay", "salary", "wage", "charge", "charged", "due",
    ];
    super::tokenize(name)
        .iter()
        .any(|t| MONEY.contains(&t.as_str()))
}

/// The narrowed sub-value for a retrieved span, or `None` when the engine can't
/// isolate one narrower than the full span. Present when a typed/format/enum
/// scanner fired, or (for plain fields) when a clean `Label:` prefix strips off
/// and leaves a shorter value. A **factual** narrowing, not a trust claim: the
/// caller always keeps the candidate's full `text` for provenance.
///
/// Never-guess fields (a closed `enum`, or an implemented `format` scanner) that
/// find no match return `None` — they do not fall back to the raw span.
pub(super) fn narrowed_value(field: &SchemaField, text: &str) -> Option<String> {
    if let Some(v) = typed_value(field, text) {
        return Some(v);
    }
    // Enum / implemented-format fields never guess: no scanner match → no value.
    if !field.choices.is_empty() || field.format.as_deref().is_some_and(has_format_scanner) {
        return None;
    }
    // Plain fields: keep the label-stripped span only when it actually narrows.
    let trimmed = text.trim();
    let stripped = strip_label_prefix(trimmed);
    (stripped != trimmed).then(|| stripped.to_string())
}

/// A `Label:` prefix is only stripped when the label side stays under both
/// limits — a longer, many-word prefix is prose with an incidental colon
/// (`"Note: the following cases apply, ..."`), not a real label.
const LABEL_MAX_CHARS: usize = 40;
const LABEL_MAX_WORDS: usize = 5;

/// `"Invoice Number: INV-42"` → `"INV-42"`. Only strips when the label side is
/// short and the value side is non-empty, so prose with a mid-sentence colon
/// (`"Note: the following cases apply, ..."`) isn't truncated to nonsense.
fn strip_label_prefix(text: &str) -> &str {
    let text = text.trim();
    if let Some((label, rest)) = text.split_once(':') {
        let rest = rest.trim();
        if !rest.is_empty()
            && label.chars().count() <= LABEL_MAX_CHARS
            && label.split_whitespace().count() <= LABEL_MAX_WORDS
            && !label.contains(". ")
        {
            return rest;
        }
    }
    text
}

// ── enum classification ───────────────────────────────────────────────────────

/// Match `text` against a closed choice set. The longest choice whose
/// normalized form is a substring of the normalized text wins (most specific);
/// `None` when no choice is supported — an enum never guesses. Returns the
/// choice in its original (canonical) spelling.
fn classify_enum(choices: &[String], text: &str) -> Option<String> {
    let hay = normalize(text);
    choices
        .iter()
        .filter(|c| {
            let needle = normalize(c);
            !needle.is_empty() && hay.contains(&needle)
        })
        .max_by_key(|c| normalize(c).len())
        .cloned()
}

/// Lowercase, keep only alphanumerics and single spaces. So `"Net-30"` and
/// `"net 30"` both normalize to `"net 30"` and match.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
            prev_space = false;
        } else if !prev_space && !out.is_empty() {
            out.push(' ');
            prev_space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

// ── numeric values ────────────────────────────────────────────────────────────

/// First standalone integer-looking run, thousands-separator commas included:
/// `"Line 1,024 items"` → `"1,024"`. Digit runs embedded in a larger token
/// (`"MSTRL-API-6819"`, `"01/15/2024"`) are skipped — see [`standalone`].
fn find_integer(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut first = 0;
    while let Some(start) = next_digit(&chars, first) {
        let mut end = start;
        while end < n {
            let c = chars[end];
            if c.is_ascii_digit() || (c == ',' && next_is_digit(&chars, end)) {
                end += 1;
            } else {
                break;
            }
        }
        if standalone(&chars, start, end) {
            return Some(chars[start..end].iter().collect());
        }
        first = end;
    }
    None
}

/// First standalone number-looking run: optional leading currency symbol,
/// digits with thousands commas and at most one decimal point.
/// `"Total $146,688.00 due"` → `"$146,688.00"`. Digit runs embedded in a larger
/// token (invoice ids, date fragments) are skipped — see [`standalone`].
fn find_number(text: &str) -> Option<String> {
    scan_number(text, false)
}

/// [`find_number`] restricted to *money-shaped* runs, for money-hinted fields:
/// a run qualifies only with a currency symbol, a decimal point, or a
/// thousands comma, and percent runs (`"11.4 %"`) are skipped. Bare digit runs
/// — street numbers, phone fragments, OCR noise — never promote; the scan
/// walks on to the next run, so `"Price: 11.4 % $ 12.8"` yields `"$ 12.8"`.
fn find_money(text: &str) -> Option<String> {
    scan_number(text, true)
}

fn scan_number(text: &str, money_only: bool) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut from = 0;
    while let Some(first) = next_digit(&chars, from) {
        // Forward extent from the first digit.
        let mut end = first;
        let mut seen_dot = false;
        let mut seen_comma = false;
        while end < n {
            let c = chars[end];
            if c.is_ascii_digit() {
                end += 1;
            } else if c == ',' && next_is_digit(&chars, end) {
                seen_comma = true;
                end += 1;
            } else if c == '.' && !seen_dot && next_is_digit(&chars, end) {
                seen_dot = true;
                end += 1;
            } else {
                break;
            }
        }
        if standalone(&chars, first, end) {
            // Back up to include an immediately-preceding currency symbol
            // (allowing one space, e.g. "$ 18").
            let mut start = first;
            if start > 0 && chars[start - 1] == ' ' && start >= 2 && is_currency(chars[start - 2]) {
                start -= 2;
            } else if start > 0 && is_currency(chars[start - 1]) {
                start -= 1;
            }
            let money_shaped = start < first || seen_dot || seen_comma;
            let percent = {
                let mut k = end;
                if chars.get(k) == Some(&' ') {
                    k += 1;
                }
                chars.get(k) == Some(&'%')
            };
            if !money_only || (money_shaped && !percent) {
                return Some(chars[start..end].iter().collect::<String>());
            }
        }
        from = end;
    }
    None
}

/// Is the digit run `[start, end)` its own token, rather than a fragment of an
/// id or date? Rejects runs directly adjoining a letter/digit, and runs whose
/// neighboring `-` / `/` / `.` glues them into a larger token (`API-6819-014`,
/// `01/15/2024`, `v2.5`) — while keeping standalone negatives (`"-42"`) and
/// ranges' leading edge alone ends the scan for that run only.
fn standalone(chars: &[char], start: usize, end: usize) -> bool {
    // Left boundary: a letter/digit glues; `-`/`/`/`.` glue only when they in
    // turn follow a letter/digit (mid-token), not at a token start (`" -42"`).
    if start > 0 {
        let prev = chars[start - 1];
        if prev.is_alphanumeric() {
            return false;
        }
        if matches!(prev, '-' | '/' | '.') && start >= 2 && chars[start - 2].is_alphanumeric() {
            return false;
        }
    }
    // Right boundary: a letter glues (`42nd`); `-`/`/` glue when followed by a
    // letter/digit (`6819-014`, `01/15`). A bare trailing `.`/`,` is sentence
    // punctuation and fine.
    if let Some(&next) = chars.get(end) {
        if next.is_alphabetic() {
            return false;
        }
        if matches!(next, '-' | '/') && chars.get(end + 1).is_some_and(|c| c.is_alphanumeric()) {
            return false;
        }
    }
    true
}

fn next_digit(chars: &[char], from: usize) -> Option<usize> {
    (from..chars.len()).find(|&i| chars[i].is_ascii_digit())
}

fn is_currency(c: char) -> bool {
    matches!(c, '$' | '€' | '£' | '¥')
}

fn next_is_digit(chars: &[char], i: usize) -> bool {
    chars.get(i + 1).is_some_and(char::is_ascii_digit)
}

// ── bool ──────────────────────────────────────────────────────────────────────

/// First standalone yes/no/true/false token (word-boundaried, so `"notice"`
/// doesn't read as `"no"`).
fn find_bool(text: &str) -> Option<String> {
    let tokens = super::tokenize(text);
    ["yes", "no", "true", "false"]
        .into_iter()
        .find(|kw| tokens.iter().any(|t| t == kw))
        .map(str::to_string)
}

// ── format (email / uri / date) ───────────────────────────────────────────────

fn match_format(fmt: &str, text: &str) -> Option<String> {
    match fmt.to_ascii_lowercase().as_str() {
        "email" | "idn-email" => find_email(text),
        "uri" | "url" | "iri" => find_uri(text),
        "date" | "date-time" => find_date(text),
        _ => None,
    }
}

/// Does `fmt` select a scanner this engine actually implements? Distinguishes
/// "the scanner looked and found nothing" (never guess — the field goes null,
/// like an enum with no matching choice) from "we don't understand this format
/// keyword" (no scanner ever ran, so the raw-span fallback stands).
pub(super) fn has_format_scanner(fmt: &str) -> bool {
    matches!(
        fmt.to_ascii_lowercase().as_str(),
        "email" | "idn-email" | "uri" | "url" | "iri" | "date" | "date-time"
    )
}

/// First whitespace-delimited token that looks like an email address. Trims
/// surrounding punctuation first (`"(a@b.com)"` → `"a@b.com"`).
fn find_email(text: &str) -> Option<String> {
    text.split_whitespace().find_map(|tok| {
        let t = tok.trim_matches(|c: char| !c.is_alphanumeric());
        let at = t.find('@')?;
        let (local, domain) = (&t[..at], &t[at + 1..]);
        let local_ok = !local.is_empty()
            && local
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-'));
        let domain_ok = domain.contains('.')
            && !domain.starts_with('.')
            && domain
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '.' | '-'));
        (local_ok && domain_ok).then(|| t.to_string())
    })
}

/// First token beginning `http://`, `https://`, or `www.`. Trailing sentence
/// punctuation is trimmed.
fn find_uri(text: &str) -> Option<String> {
    text.split_whitespace().find_map(|tok| {
        let t = tok.trim_end_matches(['.', ',', ';', ')', ']', '}', '>']);
        (t.starts_with("http://") || t.starts_with("https://") || t.starts_with("www."))
            .then(|| t.to_string())
    })
}

// ── date ──────────────────────────────────────────────────────────────────────

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// First date-looking substring: numeric (`2024-01-15`, `01/15/2024`) or
/// month-name (`January 15, 2024`, `15 Jan 2024`, `May 2024`). Returned
/// verbatim. Numeric shapes are tried first. Also serves as the geometry-join
/// pass's typed-date gate (`pub(super)`).
pub(super) fn find_date(text: &str) -> Option<String> {
    find_numeric_date(text).or_else(|| find_monthname_date(text))
}

/// `d{1,4}[-/]d{1,2}[-/]d{1,4}` — the year can lead or trail. The 1–2 digit
/// middle group keeps invoice ids like `INV-2024-0042` (4-digit tail) from
/// matching.
fn find_numeric_date(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_digit() {
            if let Some(end) = numeric_date_at(&chars, i) {
                return Some(chars[i..end].iter().collect());
            }
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    None
}

fn numeric_date_at(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut i = start;
    let run = |chars: &[char], mut i: usize| {
        let s = i;
        while i < n && chars[i].is_ascii_digit() {
            i += 1;
        }
        (i - s, i)
    };
    let (g1, after1) = run(chars, i);
    if !(1..=4).contains(&g1) {
        return None;
    }
    i = after1;
    let sep = *chars.get(i)?;
    if sep != '-' && sep != '/' {
        return None;
    }
    i += 1;
    let (g2, after2) = run(chars, i);
    if !(1..=2).contains(&g2) {
        return None;
    }
    i = after2;
    if chars.get(i) != Some(&sep) {
        return None;
    }
    i += 1;
    let (g3, after3) = run(chars, i);
    if !(1..=4).contains(&g3) {
        return None;
    }
    Some(after3)
}

/// A month name (full or 3-letter abbrev) with a nearby 4-digit year, plus an
/// optional day before or after the month.
fn find_monthname_date(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let lower: Vec<char> = chars.iter().map(char::to_ascii_lowercase).collect();
    let n = chars.len();

    for i in 0..n {
        let Some(month_len) = month_at(&lower, i) else {
            continue;
        };
        // Find the year: scan forward over spaces / digits / commas / periods,
        // stopping once a 4-digit run completes.
        let mut k = i + month_len;
        let mut year_end = None;
        let mut digit_run = 0;
        while k < n {
            let c = lower[k];
            if c.is_ascii_digit() {
                digit_run += 1;
                k += 1;
                if digit_run == 4 {
                    year_end = Some(k);
                    break;
                }
            } else if matches!(c, ' ' | ',' | '.') {
                digit_run = 0;
                k += 1;
            } else {
                break;
            }
        }
        let end = year_end?;

        // Include a leading day ("15 January 2024"): 1–2 digits + optional space
        // directly before the month.
        let mut start = i;
        let mut b = i;
        while b > 0 && lower[b - 1] == ' ' {
            b -= 1;
        }
        let day_end = b;
        while b > 0 && lower[b - 1].is_ascii_digit() {
            b -= 1;
        }
        if (1..=2).contains(&(day_end - b)) {
            start = b;
        }
        return Some(
            chars[start..end]
                .iter()
                .collect::<String>()
                .trim()
                .to_string(),
        );
    }
    None
}

/// If a month name (full, or 3-letter abbreviation optionally followed by `.`)
/// starts at `i` on a word boundary, return its length in chars.
fn month_at(lower: &[char], i: usize) -> Option<usize> {
    if i > 0 && lower[i - 1].is_ascii_alphabetic() {
        return None;
    }
    for full in MONTHS {
        let fc: Vec<char> = full.chars().collect();
        if starts_with_at(lower, i, &fc) && !alpha_at(lower, i + fc.len()) {
            return Some(fc.len());
        }
        let abbr = &fc[..3];
        if starts_with_at(lower, i, abbr) {
            let after = i + abbr.len();
            let has_dot = lower.get(after) == Some(&'.');
            let boundary_at = if has_dot { after + 1 } else { after };
            if !alpha_at(lower, after) {
                return Some(boundary_at - i);
            }
        }
    }
    None
}

fn starts_with_at(chars: &[char], i: usize, pat: &[char]) -> bool {
    pat.iter()
        .enumerate()
        .all(|(k, p)| chars.get(i + k) == Some(p))
}

fn alpha_at(chars: &[char], i: usize) -> bool {
    chars.get(i).is_some_and(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(field_type: FieldType) -> SchemaField {
        SchemaField {
            name: "f".into(),
            description: None,
            field_type,
            choices: Vec::new(),
            format: None,
        }
    }

    #[test]
    fn strip_label_prefix_cases() {
        assert_eq!(strip_label_prefix("Invoice Number: INV-42"), "INV-42");
        assert_eq!(strip_label_prefix("no label here"), "no label here");
        // Empty value after colon → keep whole.
        assert_eq!(strip_label_prefix("Section 4:"), "Section 4:");
        // Long label side → not a label, keep whole.
        let long = "In a landmark ruling the court decided: appeal denied";
        assert_eq!(strip_label_prefix(long), long);
    }

    #[test]
    fn integer_and_number() {
        assert_eq!(find_integer("Line 1,024 items"), Some("1,024".into()));
        assert_eq!(find_integer("no digits"), None);
        assert_eq!(
            find_number("Total $146,688.00 due"),
            Some("$146,688.00".into())
        );
        assert_eq!(find_number("about 3.5 percent"), Some("3.5".into()));
        assert_eq!(find_number("$ 18"), Some("$ 18".into()));
        // A trailing decimal point is not consumed.
        assert_eq!(find_number("ends at 42."), Some("42".into()));
    }

    #[test]
    fn embedded_digit_runs_are_not_numbers() {
        // Digit fragments of ids must not be mined out as numeric values.
        assert_eq!(find_number("Invoice Number: MSTRL-API-6819-014"), None);
        assert_eq!(find_number("Tax ID: FR95952418325"), None);
        assert_eq!(find_integer("INV-2024-0042"), None);
        // Date fragments aren't numbers either.
        assert_eq!(find_number("on 01/15/2024"), None);
        // But a real amount later in the span is still found.
        assert_eq!(
            find_number("Invoice INV-2024-0042 total $118.50"),
            Some("$118.50".into())
        );
        assert_eq!(find_integer("ref A-1, qty 12"), Some("12".into()));
        // Standalone negatives keep working (sign not captured, as before).
        assert_eq!(find_number("delta -42.5 today"), Some("42.5".into()));
    }

    #[test]
    fn money_hinted_fields_require_money_shape() {
        let money = |name: &str| SchemaField {
            name: name.into(),
            ..field(FieldType::Number)
        };
        // Bare digit runs never promote for a money-hinted field (e.g. a street
        // number or a garbled OCR phone fragment).
        assert_eq!(typed_value(&money("total_charged"), "123 Lane, eld"), None);
        assert_eq!(
            typed_value(&money("price_or_fee"), "Panes 55 173-567"),
            None
        );
        // Currency / decimal / thousands-comma shapes all qualify.
        assert_eq!(
            typed_value(&money("standard_build_price"), "Standard - Price: $146,688"),
            Some("$146,688".into())
        );
        assert_eq!(
            typed_value(&money("total_gross"), "Total gross payroll 30,600 due"),
            Some("30,600".into())
        );
        assert_eq!(
            typed_value(&money("late_fee"), "late fee of 2.50"),
            Some("2.50".into())
        );
        // A percent run is skipped in favor of the money-shaped run after it.
        assert_eq!(
            typed_value(&money("share_price"), "Price: 11.4 % $ 12.8"),
            Some("$ 12.8".into())
        );
        // Percent-only span → no money value at all.
        assert_eq!(typed_value(&money("total_cost"), "up 11.4 % overall"), None);
        // Non-money number fields keep the permissive scanner (name "f").
        assert_eq!(
            typed_value(&field(FieldType::Number), "population 338"),
            Some("338".into())
        );
    }

    #[test]
    fn enum_longest_match_wins() {
        let choices = vec!["basic".into(), "basic plus".into(), "premium".into()];
        assert_eq!(
            classify_enum(&choices, "the Basic Plus tier applies"),
            Some("basic plus".into())
        );
        assert_eq!(classify_enum(&choices, "no tier stated"), None);
        // Normalization bridges punctuation/case.
        let net = vec!["net 30".into()];
        assert_eq!(
            classify_enum(&net, "terms: Net-30 days"),
            Some("net 30".into())
        );
    }

    #[test]
    fn numeric_dates() {
        assert_eq!(find_date("due 2024-01-15 sharp"), Some("2024-01-15".into()));
        assert_eq!(find_date("on 01/15/2024"), Some("01/15/2024".into()));
        // Invoice id with a 4-digit tail group is NOT a date.
        assert_eq!(find_date("INV-2024-0042"), None);
    }

    #[test]
    fn monthname_dates() {
        assert_eq!(
            find_date("dated January 15, 2024 hereby"),
            Some("January 15, 2024".into())
        );
        assert_eq!(find_date("15 Jan 2024"), Some("15 Jan 2024".into()));
        assert_eq!(find_date("as of May 2024"), Some("May 2024".into()));
        // "mayor" must not match "may".
        assert_eq!(find_date("the mayor spoke in 2024"), None);
        // Month with no year nearby → no match.
        assert_eq!(find_date("in September we met"), None);
    }

    #[test]
    fn email_and_uri() {
        let f = SchemaField {
            format: Some("email".into()),
            ..field(FieldType::Str)
        };
        assert_eq!(
            typed_value(&f, "write to (jo@acme.co)."),
            Some("jo@acme.co".into())
        );
        assert_eq!(typed_value(&f, "no address"), None);
        let u = SchemaField {
            format: Some("uri".into()),
            ..field(FieldType::Str)
        };
        assert_eq!(
            typed_value(&u, "see https://example.com/x."),
            Some("https://example.com/x".into())
        );
    }

    #[test]
    fn bool_is_word_boundaried() {
        assert_eq!(find_bool("Answer: Yes it applies"), Some("yes".into()));
        assert_eq!(find_bool("notice of termination"), None);
    }

    #[test]
    fn typed_value_precedence_and_fallback() {
        // Number type pulls the amount.
        assert_eq!(
            typed_value(&field(FieldType::Number), "Grand total 118.50"),
            Some("118.50".into())
        );
        // Str type has no scanner → None (caller falls back).
        assert_eq!(typed_value(&field(FieldType::Str), "some prose"), None);
    }
}
