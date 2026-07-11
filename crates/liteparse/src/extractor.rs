//! Schema-extraction engine (Phase 2 of the schema-extraction plan).
//!
//! Given a flat [`ExtractionSchema`] (named fields with descriptions/types) and
//! a parsed document, rank the document's [`ExtractionUnit`]s per field and
//! return the best span(s) with a relevance `score` and `(page, bbox)`
//! provenance. This is the *narrowing / easy-field* extractor described in
//! `EXTRACT_PLAN.md` — top-k candidate spans per field, not a single reasoned
//! LLM-grade answer.
//!
//! ## Retrieval: BM25 ∪ static embedding, always-fuse (RRF)
//!
//! Two signals, combined per the frozen Phase 0 decision (see the plan's
//! "Validation results", Result 1):
//!
//! - **BM25** — a pure-Rust, zero-download keyword ranker over the unit
//!   index. Always available; the whole engine degrades to this when no
//!   embedding model is attached (or the `static-embed` feature is off).
//! - **Static embedding** ([`static_embed`]) — model2vec cosine ranking,
//!   attached via [`LocalExtractor::with_embedder`]. Unit vectors are
//!   embedded once at attach time.
//! - **Fusion** — [`FusionMode::Auto`] is *always-fuse*: unconditional
//!   reciprocal-rank fusion of the two rankings (`1/(60+rank)` summed; no
//!   adaptive gate, no IDF, no threshold — Phase 0 measured the gate
//!   Pareto-dominated). `Bm25`/`Embed` stay selectable; pure `Embed` is the
//!   right mode for known-paraphrastic corpus routing.
//!
//! ## What is deliberately NOT here yet
//!
//! - **Header-joined cell units** (from *detected* tables). Geometry-join
//!   synthetic units are in ([`geometry_units`], via
//!   [`LocalExtractor::from_pages`]); the detected-table variant is not.
//! - **Model download.** [`static_embed::resolve_model_dir`] only finds
//!   already-local model files; [`model_fetch::ensure_model`] downloads on
//!   first use (native builds; `extract_fusion = bm25` never downloads).

use crate::extraction_unit::{ExtractionUnit, UnitSource, natural_units};
use crate::offset_map::OffsetMap;
use crate::types::{ParsedPage, Rect};
use serde::Serialize;
use std::collections::HashMap;

pub mod geometry_units;
#[cfg(all(feature = "static-embed", not(target_arch = "wasm32")))]
pub mod model_fetch;
pub mod schema_json;
mod span_search;
#[cfg(feature = "static-embed")]
pub mod static_embed;
pub mod table_units;
mod value_span;

/// Default number of candidate spans returned per field.
pub const DEFAULT_TOP_K: usize = 5;

/// A field's declared value type. A *hint* that will drive value-span
/// extraction later; today it only tags the output. Unknown strings deserialize
/// to [`FieldType::Str`] so an over-rich schema (written for an LLM engine)
/// still loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    #[default]
    Str,
    Int,
    Number,
    Date,
    Bool,
    List,
    /// Any type keyword we don't (yet) special-case. Kept distinct from `Str`
    /// so a future value path can see "the schema said something typed".
    #[serde(other)]
    Other,
}

/// One field to extract.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct SchemaField {
    /// Machine name — the output key and the query fallback.
    pub name: String,
    /// Natural-language description. Used as the retrieval query when present;
    /// this is where paraphrase / label vocabulary lives.
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "type")]
    pub field_type: FieldType,
    /// Closed choice set (`enum` in JSON Schema terms). When non-empty the value
    /// path classifies the span against these and returns the best-matching
    /// choice (or `None` — it never guesses).
    #[serde(default)]
    pub choices: Vec<String>,
    /// JSON-Schema `format` hint (`email` / `uri` / `date` …). Selects a
    /// value-span scanner and overrides `field_type` for value extraction.
    #[serde(default)]
    pub format: Option<String>,
}

impl SchemaField {
    /// The retrieval query for this field: the description, else the name with
    /// underscores/hyphens softened to spaces so `invoice_number` tokenizes like
    /// "invoice number".
    fn query(&self) -> String {
        match &self.description {
            Some(d) if !d.trim().is_empty() => d.clone(),
            _ => self.name.replace(['_', '-'], " "),
        }
    }
}

/// A flat schema: an ordered list of fields.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ExtractionSchema {
    pub fields: Vec<SchemaField>,
}

/// A single ranked candidate span for a field.
#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    /// The extracted value. In the BM25 milestone this is the trimmed unit text
    /// (value-span extraction will later narrow it to just the value).
    pub value: String,
    /// The full source unit text — always carried so callers can verify the
    /// span even once `value` is narrowed.
    pub text: String,
    /// Raw, unitless fusion/BM25 score. Comparable *within a run* for ranking
    /// and filtering; explicitly **not** a probability. See the plan's
    /// "Scoring & ranking".
    pub score: f32,
    pub page: usize,
    pub bbox: Rect,
    /// Which unit source produced this candidate (informational / debug).
    pub source: UnitSource,
    /// Did a type/format/enum scanner actually pull a value out (vs. `value`
    /// falling back to the raw span)? Lets callers see *why* a field resolved.
    pub typed_match: bool,
}

/// Coarse trust tier for a resolved field — the escalation signal callers
/// (reviewers, agents) branch on: act on `strong`, verify `weak`, escalate or
/// skip `none`. This is a **heuristic over observable value-path facts**, not a
/// calibrated confidence (scores stay unitless, see the plan's "Scoring &
/// ranking"): `strong` means a value-shaped scanner actually isolated the value
/// (typed/format/enum match or a label-anchored cell), `weak` means the headline
/// is a raw retrieved span the engine could not narrow, `none` means no answer
/// (nothing retrieved, or an enum with no matching choice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Signal {
    Strong,
    Weak,
    None,
}

/// Per-field extraction result. The headline `value`/`score`/`page`/`bbox`
/// mirror the top candidate; `None` throughout when retrieval found nothing.
#[derive(Debug, Clone, Serialize)]
pub struct FieldResult {
    pub name: String,
    pub value: Option<String>,
    /// How much to trust `value` — see [`Signal`].
    pub signal: Signal,
    pub score: Option<f32>,
    pub page: Option<usize>,
    pub bbox: Option<Rect>,
    /// Top-k deduped candidates in rank order. Empty on a miss.
    pub candidates: Vec<Candidate>,
}

impl FieldResult {
    fn miss(name: &str) -> Self {
        FieldResult {
            name: name.to_string(),
            value: None,
            signal: Signal::None,
            score: None,
            page: None,
            bbox: None,
            candidates: Vec::new(),
        }
    }
}

/// Extraction output: one [`FieldResult`] per schema field, in schema order.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractResult {
    pub fields: Vec<FieldResult>,
}

/// An array-of-objects group (repeated records — `line_items[]` and friends):
/// extract one record per detected table row, each sub-field pulled from the
/// column whose header best matches it. The scalar-schema counterpart is
/// [`ExtractionSchema`]; this is the row-grouping path over detected tables.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ObjectArraySchema {
    /// Sub-fields of each record. Flat — nested objects are flattened to dotted
    /// names upstream, the same convention scalar leaves use.
    pub fields: Vec<SchemaField>,
    /// Optional description of the group as a whole ("line items on the
    /// invoice"). Reserved for disambiguating which table to group over; not
    /// yet consulted (v1 picks the best-matching detected table by header).
    #[serde(default)]
    pub description: Option<String>,
}

/// One extracted record: a [`FieldResult`] per sub-field, in schema order.
#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub fields: Vec<FieldResult>,
}

/// Object-array extraction output: one [`Record`] per grouped table row.
#[derive(Debug, Clone, Serialize)]
pub struct ObjectArrayResult {
    pub records: Vec<Record>,
}

/// One object-array group in a document-level extraction: the group's dotted
/// schema path, a group-level [`Signal`], and the grouped records.
#[derive(Debug, Clone, Serialize)]
pub struct ArrayFieldResult {
    pub name: String,
    /// `none` = no table matched (no records); `strong` = at least one record
    /// resolved half or more of its sub-fields; `weak` = records exist but are
    /// sparse (partial table detection or poor header matching).
    pub signal: Signal,
    pub records: Vec<Record>,
}

impl ArrayFieldResult {
    pub(crate) fn new(name: String, records: Vec<Record>) -> Self {
        let signal = if records.is_empty() {
            Signal::None
        } else if records.iter().any(|r| {
            !r.fields.is_empty()
                && r.fields.iter().filter(|f| f.value.is_some()).count() * 2 >= r.fields.len()
        }) {
            Signal::Strong
        } else {
            Signal::Weak
        };
        ArrayFieldResult {
            name,
            signal,
            records,
        }
    }
}

/// Document-level extraction result: the deliverable of `LiteParse::extract`.
/// Scalar leaves (nested objects flattened to dotted names) plus object-array
/// groups. The contract is **narrowing signal with provenance** — top-k
/// candidate spans, a score, a page and a bbox per field — not LLM-grade
/// single-answer extraction; callers branch on [`Signal`] and verify through
/// `candidates`.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentExtraction {
    /// One result per scalar leaf, in schema order, named by dotted path
    /// (`vendor.address.city`).
    pub fields: Vec<FieldResult>,
    /// One result per `array<object>` group (`line_items`), in schema order.
    pub arrays: Vec<ArrayFieldResult>,
}

// ── BM25 ────────────────────────────────────────────────────────────────────

/// BM25 free parameters (Robertson/Sparck-Jones defaults).
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Split text into lowercased alphanumeric tokens. Unicode-aware
/// (`char::is_alphanumeric`), so accented letters and non-Latin scripts survive;
/// runs of punctuation/whitespace are separators. Numbers stay as tokens
/// (they carry real signal — invoice numbers, amounts).
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// One occurrence of a term in a document: `(doc index, term frequency)`.
#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: usize,
    tf: u32,
}

/// Inverted BM25 index over a fixed unit set. `postings[term]` lists the docs
/// containing `term` with per-doc term frequency, so scoring a query touches
/// only docs that share a query token — work is proportional to postings-list
/// length, not corpus size.
#[derive(Debug, Default)]
struct Bm25Index {
    postings: HashMap<String, Vec<Posting>>,
    /// Token count per document, parallel to the unit vec.
    doc_len: Vec<u32>,
    /// Average document length; 0 when there are no documents.
    avgdl: f32,
    n_docs: usize,
}

impl Bm25Index {
    fn build(units: &[ExtractionUnit]) -> Bm25Index {
        let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut doc_len = Vec::with_capacity(units.len());
        let mut total_len: u64 = 0;

        for (doc, unit) in units.iter().enumerate() {
            let tokens = tokenize(&unit.text);
            doc_len.push(tokens.len() as u32);
            total_len += tokens.len() as u64;

            // Per-document term frequencies, then one posting per distinct term.
            let mut tf: HashMap<&str, u32> = HashMap::new();
            for t in &tokens {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            for (term, count) in tf {
                postings
                    .entry(term.to_string())
                    .or_default()
                    .push(Posting { doc, tf: count });
            }
        }

        let n_docs = units.len();
        let avgdl = if n_docs == 0 {
            0.0
        } else {
            total_len as f32 / n_docs as f32
        };
        Bm25Index {
            postings,
            doc_len,
            avgdl,
            n_docs,
        }
    }

    /// IDF for a term appearing in `df` documents (BM25's `+1`-inside-`ln`
    /// variant, which stays non-negative even for terms in >half the docs).
    fn idf(&self, df: usize) -> f32 {
        let n = self.n_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score every document that shares at least one *distinct* query term.
    /// Returns `(doc, score)` sorted by score descending (ties broken by lower
    /// doc index for determinism).
    fn score(&self, query: &str) -> Vec<(usize, f32)> {
        if self.n_docs == 0 || self.avgdl <= 0.0 {
            return Vec::new();
        }
        // Dedupe query terms: BM25 sums over query *terms*; for short field
        // descriptions a repeated word shouldn't double-count.
        let mut seen = std::collections::HashSet::new();
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in tokenize(query) {
            if !seen.insert(term.clone()) {
                continue;
            }
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let idf = self.idf(postings.len());
            for p in postings {
                let dl = self.doc_len[p.doc] as f32;
                let tf = p.tf as f32;
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / self.avgdl));
                let contrib = idf * (tf * (BM25_K1 + 1.0)) / denom;
                *scores.entry(p.doc).or_insert(0.0) += contrib;
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked
    }
}

// ── Fusion ──────────────────────────────────────────────────────────────────

/// How the BM25 and static-embedding rankings combine. Mirrors the plan's
/// `extract_fusion` config (`auto` | `bm25` | `embed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionMode {
    /// Always-fuse: reciprocal-rank fusion of BM25 ∪ embedding,
    /// unconditionally (the Phase 0 ship decision). Degrades to BM25 when no
    /// embedder is attached.
    #[default]
    Auto,
    /// BM25 only — the zero-download mode.
    Bm25,
    /// Embedding only — the mode for known-paraphrastic corpus routing.
    /// Degrades to BM25 when no embedder is attached.
    Embed,
}

/// RRF constant, matching the Python reference (`crux.py` `RRF_K`).
const RRF_K: f32 = 60.0;

/// Reciprocal-rank-fuse two rankings over `n` units: each unit contributes
/// `1/(RRF_K + position)` per ranking, summed. Units missing from a (sparse
/// BM25) ranking take the tail positions in unit-index order — the Python
/// reference argsorts a dense score vector, so every unit always has a
/// position there; index order is our deterministic stand-in for its
/// arbitrary zero-score tie order.
fn rrf_fuse(rankings: [&[(usize, f32)]; 2], n: usize) -> Vec<(usize, f32)> {
    let mut fused = vec![0.0f32; n];
    for ranking in rankings {
        let mut pos = vec![usize::MAX; n];
        for (p, (doc, _)) in ranking.iter().enumerate() {
            pos[*doc] = p;
        }
        let mut tail = ranking.len();
        for doc in 0..n {
            let p = if pos[doc] == usize::MAX {
                let p = tail;
                tail += 1;
                p
            } else {
                pos[doc]
            };
            fused[doc] += 1.0 / (RRF_K + p as f32);
        }
    }
    let mut ranked: Vec<(usize, f32)> = fused.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked
}

/// The per-unit embedding matrix plus the model that produced it (needed at
/// query time to embed the field description).
#[cfg(feature = "static-embed")]
struct EmbedIndex {
    embedder: std::sync::Arc<static_embed::StaticEmbedder>,
    /// Row-major `[n_units × dim]` unit vectors (L2-normalized by the model,
    /// so dot product = cosine).
    vecs: Vec<f32>,
    dim: usize,
}

#[cfg(feature = "static-embed")]
impl EmbedIndex {
    /// Dense cosine ranking of every unit, best first (ties → lower index).
    fn rank(&self, query: &str) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(query);
        let n = self.vecs.len() / self.dim.max(1);
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = &self.vecs[i * self.dim..(i + 1) * self.dim];
                let dot = row.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>();
                (i, dot)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored
    }
}

// ── LocalExtractor ────────────────────────────────────────────────────────────

/// The default, always-compiled extraction engine. Owns the unit index and the
/// [`OffsetMap`] needed to resolve natural-line provenance.
///
/// Build with [`LocalExtractor::from_pages`] (natural + geometry units from a
/// parse), attach an embedding model with
/// [`with_embedder`](LocalExtractor::with_embedder) for the full always-fuse
/// engine, and query with [`extract`](LocalExtractor::extract).
pub struct LocalExtractor {
    map: OffsetMap,
    units: Vec<ExtractionUnit>,
    index: Bm25Index,
    top_k: usize,
    fusion: FusionMode,
    /// Grid `page.text` rows (column-padded surface), kept for span_search's
    /// label-anchored cell extraction. `projected_lines` collapse column gaps to
    /// single spaces, so the plain-string value path looks the matching grid row
    /// up by whitespace-collapsed text to recover the 2+ space cell structure the
    /// mechanic keys on. Empty unless built via [`from_pages`](LocalExtractor::from_pages).
    grid_lines: Vec<GridLine>,
    #[cfg(feature = "static-embed")]
    embed: Option<EmbedIndex>,
}

/// One row of the grid `page.text` surface, with its whitespace-collapsed form
/// precomputed for matching against a retrieved unit's (single-spaced) text.
struct GridLine {
    page: usize,
    text: String,
    collapsed: String,
}

/// Collapse every run of whitespace to a single space (and trim). Used to match
/// a column-padded grid row against a single-spaced projected-line unit.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Index every non-blank grid `page.text` row, keyed by page, with its
/// whitespace-collapsed form precomputed for lookup.
fn build_grid_lines(pages: &[ParsedPage]) -> Vec<GridLine> {
    let mut out = Vec::new();
    for page in pages {
        for line in page.text.lines() {
            let collapsed = collapse_ws(line);
            if collapsed.is_empty() {
                continue;
            }
            out.push(GridLine {
                page: page.page_number,
                text: line.to_string(),
                collapsed,
            });
        }
    }
    out
}

impl LocalExtractor {
    /// Build from a parse's [`OffsetMap`], indexing its natural-line units.
    pub fn new(map: OffsetMap) -> LocalExtractor {
        let units = natural_units(&map);
        LocalExtractor::from_units(map, units)
    }

    /// Build the full v1 index from a parse: natural-line units plus the
    /// synthetic geometry-join units ([`geometry_units`]). This is the default
    /// entry point; [`LocalExtractor::new`] indexes natural lines only.
    pub fn from_pages(pages: &[ParsedPage]) -> LocalExtractor {
        let map = OffsetMap::build(pages);
        let mut units = natural_units(&map);
        units.extend(geometry_units::geometry_units(pages));
        let mut extractor = LocalExtractor::from_units(map, units);
        extractor.grid_lines = build_grid_lines(pages);
        extractor
    }

    /// Build from an explicit unit set (natural + future synthetic units). The
    /// `map` is still needed to resolve any [`UnitProvenance::Span`] units;
    /// synthetic units carry their own provenance and ignore it.
    ///
    /// [`UnitProvenance::Span`]: crate::extraction_unit::UnitProvenance::Span
    pub fn from_units(map: OffsetMap, units: Vec<ExtractionUnit>) -> LocalExtractor {
        let index = Bm25Index::build(&units);
        LocalExtractor {
            map,
            units,
            index,
            top_k: DEFAULT_TOP_K,
            fusion: FusionMode::default(),
            grid_lines: Vec::new(),
            #[cfg(feature = "static-embed")]
            embed: None,
        }
    }

    /// Override how many candidates each field returns (default
    /// [`DEFAULT_TOP_K`]).
    pub fn with_top_k(mut self, top_k: usize) -> LocalExtractor {
        self.top_k = top_k.max(1);
        self
    }

    /// Attach a static embedding model, enabling the embedding signal and
    /// (under [`FusionMode::Auto`]) always-fuse ranking. Every indexed unit is
    /// embedded here, once (~0.05 ms/line).
    #[cfg(feature = "static-embed")]
    pub fn with_embedder(
        mut self,
        embedder: std::sync::Arc<static_embed::StaticEmbedder>,
    ) -> LocalExtractor {
        let dim = embedder.dim();
        let mut vecs = Vec::with_capacity(self.units.len() * dim);
        for unit in &self.units {
            vecs.extend(embedder.embed(&unit.text));
        }
        self.embed = Some(EmbedIndex {
            embedder,
            vecs,
            dim,
        });
        self
    }

    /// Override the fusion mode (default [`FusionMode::Auto`] = always-fuse).
    pub fn with_fusion(mut self, fusion: FusionMode) -> LocalExtractor {
        self.fusion = fusion;
        self
    }

    /// Rank every indexed unit against `query` under the configured
    /// [`FusionMode`], best first. This is the raw retrieval surface behind
    /// [`extract`](LocalExtractor::extract) — no top-k cap, no text dedupe,
    /// no value path. In pure-BM25 mode, units sharing no query token are
    /// omitted entirely (BM25 gives them no score, not a zero rank); embed
    /// and fused rankings are dense. Indices refer into
    /// [`units`](LocalExtractor::units). Used by the eval harness
    /// (`examples/eval_extract.rs`) and useful for corpus routing.
    pub fn rank(&self, query: &str) -> Vec<(usize, f32)> {
        self.rank_with(query, self.fusion)
    }

    /// [`rank`](LocalExtractor::rank) under an explicit mode, ignoring the
    /// configured default. Modes needing the embedder degrade to BM25 when
    /// none is attached (or the `static-embed` feature is off).
    pub fn rank_with(&self, query: &str, fusion: FusionMode) -> Vec<(usize, f32)> {
        #[cfg(feature = "static-embed")]
        if let Some(embed) = &self.embed {
            match fusion {
                FusionMode::Bm25 => {}
                FusionMode::Embed => return embed.rank(query),
                FusionMode::Auto => {
                    let bm25 = self.index.score(query);
                    let cos = embed.rank(query);
                    return rrf_fuse([&bm25, &cos], self.units.len());
                }
            }
        }
        let _ = fusion;
        self.index.score(query)
    }

    /// The indexed units, in the order [`rank`](LocalExtractor::rank) indices
    /// refer to.
    pub fn units(&self) -> &[ExtractionUnit] {
        &self.units
    }

    /// The column-padded grid row on `page` corresponding to `unit_text` (a
    /// single-spaced projected line), matched by whitespace-collapsed text —
    /// exact first, then containment. `None` when no grid rows were indexed
    /// (built via [`new`](LocalExtractor::new)/[`from_units`](LocalExtractor::from_units))
    /// or nothing matches; the caller then uses `unit_text` unchanged.
    fn grid_line_for(&self, page: usize, unit_text: &str) -> Option<&str> {
        let target = collapse_ws(unit_text);
        if target.is_empty() {
            return None;
        }
        let on_page = || self.grid_lines.iter().filter(|g| g.page == page);
        on_page()
            .find(|g| g.collapsed == target)
            .or_else(|| on_page().find(|g| g.collapsed.contains(&target)))
            .map(|g| g.text.as_str())
    }

    /// Extract every field in `schema`, in order.
    pub fn extract(&self, schema: &ExtractionSchema) -> ExtractResult {
        let fields = schema
            .fields
            .iter()
            .map(|f| self.extract_field(f))
            .collect();
        ExtractResult { fields }
    }

    fn extract_field(&self, field: &SchemaField) -> FieldResult {
        let ranked = self.rank(&field.query());

        // Build candidates, resolving provenance and deduping on unit text.
        // A field can retrieve the same text via distinct units (e.g. a natural
        // line and a synthetic join over the same items) — collapse those.
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut seen_text = std::collections::HashSet::new();
        for (doc, score) in ranked {
            if candidates.len() >= self.top_k {
                break;
            }
            let unit = &self.units[doc];
            let text = unit.text.trim();
            if text.is_empty() || !seen_text.insert(text.to_string()) {
                continue;
            }
            // A natural-line unit that fails to resolve carries no usable
            // provenance — skip it rather than emit a boxless candidate.
            let Some(prov) = unit.provenance(&self.map) else {
                continue;
            };
            // Value path: try to pull a typed/enum/format value out of the span;
            // fall back to the raw span (label-trimmed) when nothing fires.
            let typed = value_span::typed_value(field, &unit.text);
            let typed_match = typed.is_some();
            let value = typed.unwrap_or_else(|| value_span::fallback_value(text));
            candidates.push(Candidate {
                value,
                text: unit.text.clone(),
                score,
                page: prov.page,
                bbox: prov.bbox,
                source: unit.source,
                typed_match,
            });
        }

        // Plain-string value path: label-anchored cell extraction. For untyped
        // string fields (no enum/format) the baseline value is the whole
        // label-stripped span; span_search narrows it to the value cell that
        // follows the matching label. Top unit only (SPAN_TOPK = 1); degrades to
        // a pure-lexical score when no embedder is attached.
        if !candidates.is_empty() && is_plain_string(field) && span_search_enabled() {
            let qterms = span_search::label_terms_for(&field.name, field.description.as_deref());
            // span_search keys on 2+ space column gaps; the retrieved unit came
            // from projected_lines (gaps collapsed), so recover the column-padded
            // grid row for this line. Falls back to the unit text when no grid row
            // matches (e.g. built without pages, or projection/grid diverged).
            let top = &candidates[0];
            let top_text = self
                .grid_line_for(top.page, &top.text)
                .unwrap_or(&top.text)
                .to_string();
            #[cfg(feature = "static-embed")]
            let q_emb: Option<Vec<f32>> = self
                .embed
                .as_ref()
                .map(|e| e.embedder.embed(&field.query()));
            let embed_cells = |cells: &[String]| -> Option<Vec<f32>> {
                #[cfg(feature = "static-embed")]
                {
                    if let (Some(e), Some(q)) = (self.embed.as_ref(), q_emb.as_ref()) {
                        return Some(cells.iter().map(|c| dot(&e.embedder.embed(c), q)).collect());
                    }
                }
                let _ = cells;
                None
            };
            if let Some(v) = span_search::span_search(&top_text, &qterms, embed_cells) {
                candidates[0].value = v;
                // A label-anchored cell hit IS a value isolation — count it for
                // the signal tier. Safe for headline selection: span_search only
                // fires on plain strings, where typed_value never fires, so the
                // headline is index 0 either way.
                candidates[0].typed_match = true;
            }
        }

        if candidates.is_empty() {
            return FieldResult::miss(&field.name);
        }

        // Headline = the highest-ranked candidate that actually yielded a typed
        // value ("pick a value-bearing candidate from top-k, not blind top-1"),
        // else fall back to rank 0. For enum fields with no match anywhere the
        // field value is null — enum never guesses — but the top retrieved span
        // is still surfaced for provenance.
        let headline = candidates.iter().position(|c| c.typed_match).unwrap_or(0);
        let enum_no_match = !field.choices.is_empty() && !candidates.iter().any(|c| c.typed_match);
        let h = &candidates[headline];
        let signal = if enum_no_match {
            Signal::None
        } else if h.typed_match {
            Signal::Strong
        } else {
            Signal::Weak
        };
        FieldResult {
            name: field.name.clone(),
            value: (!enum_no_match).then(|| h.value.clone()),
            signal,
            score: Some(h.score),
            page: Some(h.page),
            bbox: Some(h.bbox.clone()),
            candidates,
        }
    }

    // ── Object-array (row grouping) ──────────────────────────────────────────
    //
    // Repeated record groups (`line_items[]`) are recovered from liteparse's
    // *detected* tables (via `table_units::table_grids`): match each column
    // header to a sub-field, then emit one record per body row with the cell
    // value + row-level provenance. Retrieval, value coercion, and provenance
    // are all reused — the only new logic is the header→sub-field assignment.

    /// Extract one [`Record`] per row of the detected table that best matches
    /// `group`, pulling each sub-field from its matched column. `grids` come from
    /// [`table_units::table_grids`]. Returns no records when no table's columns
    /// match the schema (empty doc, no detected table, or no lexical overlap).
    pub fn extract_object_array(
        &self,
        group: &ObjectArraySchema,
        grids: &[table_units::TableGrid],
    ) -> ObjectArrayResult {
        let Some(matched) = best_grid_for(group, grids) else {
            return ObjectArrayResult {
                records: Vec::new(),
            };
        };
        let records = matched.grid.rows[matched.body_start..]
            .iter()
            .map(|row| row_to_record(&group.fields, row, &matched.mapping, matched.grid.page))
            .collect();
        ObjectArrayResult { records }
    }
}

/// The table chosen to group over, with its column→sub-field mapping and the
/// index of the first body row (1 when the header was synthesized from row 0).
struct MatchedGrid<'g> {
    grid: &'g table_units::TableGrid,
    mapping: Vec<Option<(usize, f32)>>,
    body_start: usize,
}

/// The header cells to match against, and where the body rows start: the
/// detector's header when it identified one, else the first row promoted to
/// header (the common case where detection mis-binned the header as a body row).
fn grid_header(grid: &table_units::TableGrid) -> Option<(Vec<String>, usize)> {
    match &grid.header {
        Some(h) => Some((h.clone(), 0)),
        None => grid.rows.first().map(|r| (r.cells.clone(), 1)),
    }
}

/// Choose the detected table to group over: the grid that maps the most
/// sub-fields (ties broken by higher total match score). A grid that maps zero
/// sub-fields (no lexical overlap with any column) is skipped — this is what
/// keeps an unrelated table (an address block detected as a table) from
/// producing garbage records.
fn best_grid_for<'g>(
    group: &ObjectArraySchema,
    grids: &'g [table_units::TableGrid],
) -> Option<MatchedGrid<'g>> {
    let mut best: Option<(MatchedGrid<'g>, usize, f32)> = None;
    for grid in grids {
        let Some((header, body_start)) = grid_header(grid) else {
            continue;
        };
        if grid.rows.len() <= body_start {
            continue; // no body rows to group
        }
        let mapping = match_columns(&header, &group.fields);
        let mapped = mapping.iter().filter(|m| m.is_some()).count();
        if mapped == 0 {
            continue;
        }
        let total: f32 = mapping.iter().filter_map(|m| m.map(|(_, s)| s)).sum();
        let better = best.as_ref().is_none_or(|(_, bmapped, btotal)| {
            mapped > *bmapped || (mapped == *bmapped && total > *btotal)
        });
        if better {
            best = Some((
                MatchedGrid {
                    grid,
                    mapping,
                    body_start,
                },
                mapped,
                total,
            ));
        }
    }
    best.map(|(m, _, _)| m)
}

/// Assign each sub-field to at most one header column. Columns are scored by
/// **BM25** over the header cells — a lexical gate: a (field, column) pair only
/// exists when they share a token, so a field can't be mapped to an unrelated
/// column by embedding noise (headers are terse keywords, BM25's strength; the
/// paraphrased-header case, e.g. "QTT" for quantity, is an accepted v1 residual).
/// Pairs are then taken greedily highest-score-first, one-to-one, so two fields
/// can't claim the same column. Blank headers are never assigned.
fn match_columns(header: &[String], fields: &[SchemaField]) -> Vec<Option<(usize, f32)>> {
    let units: Vec<ExtractionUnit> = header
        .iter()
        .map(|h| ExtractionUnit::synthetic(h.clone(), UnitSource::HeaderCell, 0, Rect::default()))
        .collect();
    let index = Bm25Index::build(&units);

    // (field, column, score) triples, best score first, for greedy 1:1 assign.
    let mut triples: Vec<(usize, usize, f32)> = Vec::new();
    for (fi, f) in fields.iter().enumerate() {
        for (col, score) in index.score(&f.query()) {
            if header[col].trim().is_empty() {
                continue;
            }
            triples.push((fi, col, score));
        }
    }
    triples.sort_by(|a, b| b.2.total_cmp(&a.2));

    let mut mapping = vec![None; fields.len()];
    let mut used_col = vec![false; header.len()];
    for (fi, col, score) in triples {
        if mapping[fi].is_none() && !used_col[col] {
            mapping[fi] = Some((col, score));
            used_col[col] = true;
        }
    }
    mapping
}

/// Build a record from one table row: each sub-field takes its matched column's
/// cell (empty/out-of-range → a miss), coerced through the same value path as
/// scalar extraction and stamped with the row's provenance.
fn row_to_record(
    fields: &[SchemaField],
    row: &table_units::GridRow,
    mapping: &[Option<(usize, f32)>],
    grid_page: usize,
) -> Record {
    let fields_out = fields
        .iter()
        .zip(mapping)
        .map(|(field, m)| match m {
            Some((col, score)) if *col < row.cells.len() => cell_field_result(
                field,
                &row.cells[*col],
                *score,
                row.provenance.as_ref(),
                grid_page,
            ),
            _ => FieldResult::miss(&field.name),
        })
        .collect();
    Record { fields: fields_out }
}

/// Build a [`FieldResult`] for one table cell: coerce the cell through the
/// scalar value path (typed/enum/format, else the label-stripped text) and
/// attach the row's `(page, bbox)`. An empty cell is a miss; an enum with no
/// matching choice yields a null value but still surfaces the cell candidate.
fn cell_field_result(
    field: &SchemaField,
    cell: &str,
    score: f32,
    provenance: Option<&crate::offset_map::Provenance>,
    grid_page: usize,
) -> FieldResult {
    let cell = cell.trim();
    if cell.is_empty() {
        return FieldResult::miss(&field.name);
    }
    let typed = value_span::typed_value(field, cell);
    let typed_match = typed.is_some();
    let value = typed.unwrap_or_else(|| value_span::fallback_value(cell));
    let enum_no_match = !field.choices.is_empty() && !typed_match;

    let page = provenance.map(|p| p.page).unwrap_or(grid_page);
    let bbox = provenance.map(|p| p.bbox.clone());
    let candidate = Candidate {
        value: value.clone(),
        text: cell.to_string(),
        score,
        page,
        bbox: bbox.clone().unwrap_or_default(),
        source: UnitSource::HeaderCell,
        typed_match,
    };
    let signal = if enum_no_match {
        Signal::None
    } else if typed_match {
        Signal::Strong
    } else {
        Signal::Weak
    };
    FieldResult {
        name: field.name.clone(),
        value: (!enum_no_match).then_some(value),
        signal,
        score: Some(score),
        page: Some(page),
        // Only claim a bbox when the row actually resolved to a source line.
        bbox,
        candidates: vec![candidate],
    }
}

/// A field whose value path is the plain-string branch: no enum, no format, and
/// an untyped/string declared type. These are the fields [`span_search`] targets
/// — typed/enum/format fields already resolve through [`value_span::typed_value`].
fn is_plain_string(field: &SchemaField) -> bool {
    matches!(field.field_type, FieldType::Str | FieldType::Other)
        && field.choices.is_empty()
        && field.format.is_none()
}

/// Label-anchored cell extraction is on by default — it *is* the plain-string
/// value path (Phase 0 `Result 3d`). Set `LITEPARSE_EXTRACT_SPAN_SEARCH=0` to
/// disable it for an A/B byte-diff against the label-stripped baseline.
fn span_search_enabled() -> bool {
    !matches!(
        std::env::var("LITEPARSE_EXTRACT_SPAN_SEARCH").as_deref(),
        Ok("0")
    )
}

/// Dot product of two equal-length vectors. For L2-normalized model2vec
/// embeddings this is the cosine similarity.
#[cfg(feature = "static-embed")]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction_unit::ExtractionUnit;

    fn synth(text: &str) -> ExtractionUnit {
        // Synthetic units carry direct provenance, so tests need no OffsetMap.
        ExtractionUnit::synthetic(
            text,
            UnitSource::NaturalLine,
            1,
            Rect {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 5.0,
            },
        )
    }

    fn field(name: &str, description: &str) -> SchemaField {
        SchemaField {
            name: name.to_string(),
            description: Some(description.to_string()),
            field_type: FieldType::Str,
            choices: Vec::new(),
            format: None,
        }
    }

    fn extractor(texts: &[&str]) -> LocalExtractor {
        let units = texts.iter().map(|t| synth(t)).collect();
        LocalExtractor::from_units(OffsetMap::default(), units)
    }

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(
            tokenize("Invoice #INV-2024_0042!"),
            ["invoice", "inv", "2024", "0042"]
        );
        assert_eq!(tokenize("   "), Vec::<String>::new());
    }

    #[test]
    fn ranks_the_matching_line_first() {
        let ex = extractor(&[
            "Payment terms are net 30 days",
            "Invoice Number: INV-2024-0042",
            "Thank you for your business",
        ]);
        let schema = ExtractionSchema {
            fields: vec![field("invoice_number", "the invoice or document number")],
        };
        let out = ex.extract(&schema);
        assert_eq!(out.fields.len(), 1);
        let r = &out.fields[0];
        // Ranking picks the invoice line; the raw span is preserved on the
        // candidate, while the headline value is label-trimmed.
        assert_eq!(r.candidates[0].text, "Invoice Number: INV-2024-0042");
        assert_eq!(r.value.as_deref(), Some("INV-2024-0042"));
        assert_eq!(r.page, Some(1));
        assert!(r.score.unwrap() > 0.0);
    }

    #[test]
    fn miss_when_no_query_term_appears() {
        let ex = extractor(&["Payment terms are net 30 days", "Thank you"]);
        let schema = ExtractionSchema {
            fields: vec![field("shipment_weight", "gross shipment weight kilograms")],
        };
        let r = &ex.extract(&schema).fields[0];
        assert_eq!(r.value, None);
        assert!(r.candidates.is_empty());
    }

    #[test]
    fn rarer_term_outweighs_common_one() {
        // "total" appears in every doc (idf→low); "vat" is rare (idf→high), so
        // the line carrying it should win for a query mentioning both.
        let ex = extractor(&["Subtotal total 100", "Grand total 118", "Total VAT 18"]);
        let schema = ExtractionSchema {
            fields: vec![field("vat", "total vat amount")],
        };
        let r = &ex.extract(&schema).fields[0];
        assert_eq!(r.value.as_deref(), Some("Total VAT 18"));
    }

    #[test]
    fn top_k_caps_candidate_count() {
        let ex = extractor(&["total amount one", "total amount two", "total amount three"])
            .with_top_k(2);
        let schema = ExtractionSchema {
            fields: vec![field("amount", "total amount")],
        };
        let r = &ex.extract(&schema).fields[0];
        assert_eq!(r.candidates.len(), 2);
    }

    #[test]
    fn name_is_query_fallback_when_no_description() {
        let ex = extractor(&["Delivery date 2024-01-15", "unrelated"]);
        let schema = ExtractionSchema {
            fields: vec![SchemaField {
                name: "delivery_date".into(),
                description: None,
                field_type: FieldType::Date,
                choices: Vec::new(),
                format: None,
            }],
        };
        let r = &ex.extract(&schema).fields[0];
        // Date type now pulls the value substring out of the span.
        assert_eq!(r.value.as_deref(), Some("2024-01-15"));
        assert!(r.candidates[0].typed_match);
    }

    #[test]
    fn number_field_extracts_value_and_selects_value_bearing_candidate() {
        // Top BM25 hit is a labelled header with no number; the value lives one
        // rank down. Value-bearing selection should surface the amount.
        let ex = extractor(&["Total amount payable", "Amount 1,250.00"]);
        let schema = ExtractionSchema {
            fields: vec![SchemaField {
                name: "amount".into(),
                description: Some("total amount payable".into()),
                field_type: FieldType::Number,
                choices: Vec::new(),
                format: None,
            }],
        };
        let r = &ex.extract(&schema).fields[0];
        assert_eq!(r.value.as_deref(), Some("1,250.00"));
    }

    #[test]
    fn enum_returns_choice_or_null_never_guesses() {
        let schema = || ExtractionSchema {
            fields: vec![SchemaField {
                name: "plan_tier".into(),
                description: Some("the subscription plan tier".into()),
                field_type: FieldType::Str,
                choices: vec!["basic".into(), "premium".into(), "enterprise".into()],
                format: None,
            }],
        };
        // Match present → canonical choice.
        let ex = extractor(&["Your plan tier is Premium", "unrelated line"]);
        let r = &ex.extract(&schema()).fields[0];
        assert_eq!(r.value.as_deref(), Some("premium"));
        // No choice supported → null value, but the top span is still surfaced.
        let ex2 = extractor(&["Your plan tier is Ultimate", "unrelated line"]);
        let r2 = &ex2.extract(&schema()).fields[0];
        assert_eq!(r2.value, None);
        assert!(!r2.candidates.is_empty());
    }

    #[test]
    fn str_field_strips_label_prefix() {
        let ex = extractor(&["Invoice Number: INV-2024-0042", "other"]);
        let schema = ExtractionSchema {
            fields: vec![field("invoice_number", "the invoice number")],
        };
        let r = &ex.extract(&schema).fields[0];
        assert_eq!(r.value.as_deref(), Some("INV-2024-0042"));
        // But the raw span is preserved on the candidate for verification.
        assert_eq!(r.candidates[0].text, "Invoice Number: INV-2024-0042");
    }

    #[test]
    fn duplicate_unit_text_is_deduped() {
        let ex = extractor(&["Invoice total 100", "Invoice total 100"]);
        let schema = ExtractionSchema {
            fields: vec![field("total", "invoice total")],
        };
        let r = &ex.extract(&schema).fields[0];
        assert_eq!(r.candidates.len(), 1);
    }

    #[test]
    fn rrf_fuses_sparse_and_dense_rankings() {
        // bm25 is sparse (only doc 2 ranked); unranked docs 0,1 take the tail
        // positions 1,2 in index order. embed is dense: doc 1, 0, 2.
        let bm25 = vec![(2usize, 5.0f32)];
        let embed = vec![(1usize, 0.9f32), (0, 0.5), (2, 0.1)];
        let fused = rrf_fuse([&bm25, &embed], 3);
        // doc1 = 1/62 + 1/60, doc2 = 1/60 + 1/62 — an exact tie, broken by
        // lower index; doc0 = 1/61 + 1/61 loses to both.
        let order: Vec<usize> = fused.iter().map(|(d, _)| *d).collect();
        assert_eq!(order, vec![1, 2, 0]);
        assert!((fused[0].1 - fused[1].1).abs() < 1e-9, "docs 1 and 2 tie");
    }

    #[test]
    fn auto_mode_degrades_to_bm25_without_embedder() {
        let ex = extractor(&["Invoice Number: INV-1", "Payment terms net 30"]);
        let q = "invoice number";
        assert_eq!(
            ex.rank_with(q, FusionMode::Auto),
            ex.rank_with(q, FusionMode::Bm25)
        );
        assert_eq!(
            ex.rank_with(q, FusionMode::Embed),
            ex.rank_with(q, FusionMode::Bm25)
        );
    }

    #[test]
    fn field_type_deserializes_unknown_as_other() {
        let f: SchemaField = serde_json::from_str(r#"{"name":"x","type":"currency"}"#).unwrap();
        assert_eq!(f.field_type, FieldType::Other);
        let d: SchemaField = serde_json::from_str(r#"{"name":"y"}"#).unwrap();
        assert_eq!(d.field_type, FieldType::Str);
    }

    // ── Object-array (row grouping) ──────────────────────────────────────────

    use crate::extractor::table_units::{GridRow, TableGrid};
    use crate::offset_map::Provenance;

    fn prov(y: f32) -> Provenance {
        Provenance {
            page: 1,
            bbox: Rect {
                x: 10.0,
                y,
                width: 100.0,
                height: 10.0,
            },
        }
    }

    fn row(cells: &[&str], y: f32) -> GridRow {
        GridRow {
            cells: cells.iter().map(|s| s.to_string()).collect(),
            provenance: Some(prov(y)),
        }
    }

    /// Headers match sub-fields by vocabulary; each body row becomes one record
    /// with the cell value + the row's bbox.
    #[test]
    fn object_array_groups_rows_into_records() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![field("item", "the item name"), field("price", "unit price")],
            description: None,
        };
        let grid = TableGrid {
            page: 1,
            header: Some(vec!["Item".to_string(), "Price".to_string()]),
            rows: vec![
                row(&["Widget", "$9.00"], 680.0),
                row(&["Gadget", "$5.00"], 660.0),
            ],
        };
        let out = ex.extract_object_array(&group, &[grid]);
        assert_eq!(out.records.len(), 2);

        let r0 = &out.records[0];
        assert_eq!(r0.fields[0].value.as_deref(), Some("Widget"));
        assert_eq!(r0.fields[1].value.as_deref(), Some("$9.00"));
        // Row-level provenance flows onto every cell of the record.
        assert_eq!(r0.fields[0].bbox.as_ref().map(|b| b.y), Some(680.0));
        assert_eq!(r0.fields[1].page, Some(1));

        assert_eq!(out.records[1].fields[0].value.as_deref(), Some("Gadget"));
    }

    /// The grid that maps more sub-fields wins over an unrelated table (e.g. an
    /// address block detected as a table).
    #[test]
    fn object_array_picks_best_matching_table() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![
                field("description", "line item description"),
                field("quantity", "quantity ordered"),
            ],
            description: None,
        };
        let address = TableGrid {
            page: 1,
            header: Some(vec!["From".to_string(), "To".to_string()]),
            rows: vec![row(&["Acme", "Globex"], 700.0)],
        };
        let items = TableGrid {
            page: 2,
            header: Some(vec!["Description".to_string(), "Quantity".to_string()]),
            rows: vec![row(&["Bolt", "12"], 500.0)],
        };
        let out = ex.extract_object_array(&group, &[address, items]);
        assert_eq!(out.records.len(), 1);
        assert_eq!(out.records[0].fields[0].value.as_deref(), Some("Bolt"));
        assert_eq!(out.records[0].fields[1].value.as_deref(), Some("12"));
    }

    /// Headerless grids can't map columns → no records rather than a wrong guess.
    #[test]
    fn object_array_skips_headerless_grid() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![field("item", "the item")],
            description: None,
        };
        let grid = TableGrid {
            page: 1,
            header: None,
            rows: vec![row(&["Widget", "$9"], 680.0)],
        };
        assert!(ex.extract_object_array(&group, &[grid]).records.is_empty());
    }

    /// A row missing a mapped column's cell (empty or short) is a per-field miss,
    /// not a crash or a wrong value.
    #[test]
    fn object_array_missing_cell_is_a_field_miss() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![field("item", "the item name"), field("price", "unit price")],
            description: None,
        };
        let grid = TableGrid {
            page: 1,
            header: Some(vec!["Item".to_string(), "Price".to_string()]),
            rows: vec![row(&["Widget", ""], 680.0)],
        };
        let out = ex.extract_object_array(&group, &[grid]);
        assert_eq!(out.records[0].fields[0].value.as_deref(), Some("Widget"));
        assert_eq!(out.records[0].fields[1].value, None);
    }
}
