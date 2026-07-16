//! Schema-extraction engine.
//!
//! Given a flat [`ExtractionSchema`] (named fields with descriptions/types) and
//! a parsed document, rank the document's [`ExtractionUnit`]s per field and
//! return the best span(s) with a relevance `score` and `(page, bbox)`
//! provenance. This is a *narrowing* extractor — top-k candidate spans per
//! field, not a single reasoned LLM-grade answer.
//!
//! ## Retrieval: BM25 ∪ static embedding, always-fuse (RRF)
//!
//! Two signals:
//!
//! - **BM25** — a pure-Rust, zero-download keyword ranker over the unit
//!   index. Always available; the whole engine degrades to this when no
//!   embedding model is attached (or the `static-embed` feature is off).
//! - **Static embedding** ([`static_embed`]) — model2vec cosine ranking,
//!   attached via [`LocalExtractor::with_embedder`]. Unit vectors are
//!   embedded once at attach time.
//! - **Fusion** — [`FusionMode::Auto`] is *always-fuse*: unconditional
//!   reciprocal-rank fusion of the two rankings (`1/(60+rank)` summed; no
//!   adaptive gate, no IDF, no threshold — the gate was measured
//!   Pareto-dominated). `Bm25`/`Embed` stay selectable on [`LocalExtractor`]
//!   for evaluation; pure `Embed` suits known-paraphrastic corpus routing.
//!
//! ## What is deliberately not here
//!
//! - **Header-joined cell units** (from *detected* tables). Geometry-join
//!   synthetic units are in ([`geometry_units`], via
//!   [`LocalExtractor::from_pages`]); the detected-table variant is not.
//! - **Model download.** [`static_embed::resolve_model_dir`] only finds
//!   already-local model files; [`model_fetch::ensure_model`] downloads on
//!   first use (native builds), which the `extract_offline` config skips.

use crate::extraction_unit::{ExtractionUnit, UnitSource, natural_units};
use crate::offset_map::OffsetMap;
use crate::types::{ParsedPage, Rect};
use serde::Serialize;
use std::collections::HashMap;

pub mod geometry_units;
#[cfg(all(feature = "static-embed", not(target_arch = "wasm32")))]
pub mod model_fetch;
pub mod schema_json;
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
    /// The narrowed sub-value isolated out of `text` — by a type/format/enum
    /// scanner (span `"Total: $146,688"` → `"$146,688"`) or, for plain fields, a
    /// clean `Label:` strip. Present **only** when the engine found something
    /// narrower than the full span; absent otherwise, where `text` is the value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// The full source unit text — always carried so callers can verify the
    /// span (it's what you'd highlight for provenance).
    pub text: String,
    /// Relevance of this span to the field, in `[0,1]`, higher = closer. With
    /// the embedding model it is cosine similarity to the field description
    /// (absolute, comparable across documents — usable to gate a corpus); in
    /// bm25-only mode it is a saturating squash of the lexical score (coarser,
    /// comparable only within one run). Explicitly **not** a probability — it
    /// orders candidates; it is not a calibrated confidence.
    pub score: f32,
    pub page: usize,
    /// `None` when the candidate's source row couldn't be resolved back to a
    /// source line (some table rows) — never a made-up `{0,0,0,0}` box.
    pub bbox: Option<Rect>,
    /// Which unit source produced this candidate (informational / debug).
    pub source: UnitSource,
}

/// Per-field extraction result: the field name and its ranked candidate spans.
///
/// Deliberately **not** a single headline answer — extraction is a narrowing
/// tool, so it reports the ranked [`Candidate`]s (by descending [`score`], with
/// page/bbox provenance) and lets the caller decide. `candidates` is empty when
/// retrieval found nothing.
///
/// [`score`]: Candidate::score
#[derive(Debug, Clone, Serialize)]
pub struct FieldResult {
    pub name: String,
    /// Top-k deduped candidates, sorted by [`Candidate::score`] descending.
    /// Empty on a miss.
    pub candidates: Vec<Candidate>,
}

impl FieldResult {
    fn miss(name: &str) -> Self {
        FieldResult {
            name: name.to_string(),
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
    /// yet consulted — the best-matching detected table is picked by header.
    #[serde(default)]
    pub description: Option<String>,
    /// The record schema contains its own `array<object>` sub-group. Such a
    /// record is a *container* describing a table or section (census-style
    /// `tables[]` each carrying `rows[]`), not a row of one — row-grouping
    /// refuses it rather than force-mapping meta fields (`table_title`…) onto
    /// data columns. Set by `schema_json::flatten_json_schema`.
    #[serde(default)]
    pub has_nested_groups: bool,
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
/// schema path and the grouped records (one per detected table row).
#[derive(Debug, Clone, Serialize)]
pub struct ArrayFieldResult {
    pub name: String,
    pub records: Vec<Record>,
}

impl ArrayFieldResult {
    pub(crate) fn new(name: String, records: Vec<Record>) -> Self {
        ArrayFieldResult { name, records }
    }
}

/// Document-level extraction result: the deliverable of `LiteParse::extract`.
/// Scalar leaves (nested objects flattened to dotted names) plus object-array
/// groups. The contract is **narrowing with provenance** — top-k ranked
/// candidate spans (each with a score, page, and bbox) per field, not
/// LLM-grade single-answer extraction; callers rank and verify through
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

/// How the BM25 and static-embedding rankings combine. A builder-only knob on
/// [`LocalExtractor`] (the document API always uses [`Auto`](FusionMode::Auto));
/// the other modes exist for evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionMode {
    /// Always-fuse: unconditional reciprocal-rank fusion of BM25 ∪ embedding.
    /// Degrades to BM25 when no embedder is attached.
    #[default]
    Auto,
    /// BM25 only — the zero-download mode.
    Bm25,
    /// Embedding only — the mode for known-paraphrastic corpus routing.
    /// Degrades to BM25 when no embedder is attached.
    Embed,
}

/// RRF constant. `1/(60+rank)` is the long-standing reciprocal-rank-fusion
/// default (Cormack et al.); only the embedding side fuses, so this and
/// [`rrf_fuse`] are compiled only with the `static-embed` feature.
#[cfg(feature = "static-embed")]
const RRF_K: f32 = 60.0;

/// Saturating constant for [`squash_bm25`]. BM25 sums are unbounded and
/// query-length dependent, so there is no principled max to divide by; `s/(s+K)`
/// maps a "decent" top-1 match (~K) to ~0.5 and saturates toward 1. Chosen by
/// eyeballing top-1 scores on the demo corpus — a coarse ruler for ranking, not
/// a calibrated probability.
const BM25_SQUASH_K: f32 = 3.0;

/// Squash an unbounded BM25 score into `(0, 1)`, monotonically. Used for the
/// bm25-only value path and for detected-table cell scores (whose column match
/// is always lexical). Comparable only *within* a bm25 run.
fn squash_bm25(score: f32) -> f32 {
    let s = score.max(0.0);
    s / (s + BM25_SQUASH_K)
}

/// Reciprocal-rank-fuse two rankings over `n` units: each unit contributes
/// `1/(RRF_K + position)` per ranking, summed. Units missing from a (sparse
/// BM25) ranking take the tail positions in unit-index order — a dense embed
/// ranking always has a position for every unit, so index order is our
/// deterministic stand-in for the sparse side's arbitrary zero-score tie order.
#[cfg(feature = "static-embed")]
fn rrf_fuse(rankings: &[&[(usize, f32)]], n: usize) -> Vec<(usize, f32)> {
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
    #[cfg(feature = "static-embed")]
    embed: Option<EmbedIndex>,
}

impl LocalExtractor {
    /// Build from a parse's [`OffsetMap`], indexing its natural-line units.
    pub fn new(map: OffsetMap) -> LocalExtractor {
        let units = natural_units(&map);
        LocalExtractor::from_units(map, units)
    }

    /// Build the full index from a parse: natural-line units plus the
    /// synthetic geometry-join units ([`geometry_units`]). This is the default
    /// entry point; [`LocalExtractor::new`] indexes natural lines only.
    pub fn from_pages(pages: &[ParsedPage]) -> LocalExtractor {
        let map = OffsetMap::build(pages);
        let mut units = natural_units(&map);
        units.extend(geometry_units::geometry_units(pages));
        LocalExtractor::from_units(map, units)
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
                    return rrf_fuse(&[&bm25, &cos], self.units.len());
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
        let query = field.query();
        let ranked = self.rank(&query);

        // Build candidates, resolving provenance and deduping on unit text.
        // A field can retrieve the same text via distinct units (e.g. a natural
        // line and a synthetic join over the same items) — collapse those.
        // `cand_docs` remembers each candidate's unit index (parallel vec) so
        // `rescore_relevance` can look the unit vector up afterwards.
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut cand_docs: Vec<usize> = Vec::new();
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
            // Value path: narrow the span to its sub-value when we can (typed /
            // enum / format scanner, else a label strip). `None` leaves the
            // candidate carrying only its full `text`.
            let value = value_span::narrowed_value(field, &unit.text);
            candidates.push(Candidate {
                value,
                text: unit.text.clone(),
                score,
                page: prov.page,
                bbox: Some(prov.bbox),
                source: unit.source,
            });
            cand_docs.push(doc);
        }

        if candidates.is_empty() {
            return FieldResult::miss(&field.name);
        }

        // Replace the raw ranking score with a [0,1] relevance for output. The
        // fusion score above chose the ordering; callers get a comparable number.
        self.rescore_relevance(&query, &mut candidates, &cand_docs);

        // Output contract: candidates sorted by displayed score descending, so
        // `candidates[0]` is never shown beaten by a lower entry. Stable, so
        // score ties keep retrieval order.
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        FieldResult {
            name: field.name.clone(),
            candidates,
        }
    }

    /// Rewrite each candidate's `score` into a documented `[0,1]` relevance,
    /// replacing the raw (unitless, mode-dependent) fusion/BM25 score that drove
    /// ranking. With an embedder in play (auto/embed fusion) the score becomes
    /// the **cosine similarity** of the field query to the candidate's unit
    /// vector — an absolute match strength that is comparable across documents
    /// (so it can gate a corpus: "keep docs scoring > 0.5 for this field").
    /// Otherwise it is a saturating [`squash_bm25`] of the raw BM25 score —
    /// monotonic but coarser, comparable only within one bm25 run. Neither is a
    /// probability — a ranking aid, not a trust score.
    ///
    /// `docs[i]` is `candidates[i]`'s unit index (the parallel vec the caller
    /// already tracks). Cosine is decoupled from the RRF *ranking* on purpose:
    /// fusion decides which span wins; this reports how close that span is.
    fn rescore_relevance(&self, query: &str, candidates: &mut [Candidate], docs: &[usize]) {
        #[cfg(feature = "static-embed")]
        if self.fusion != FusionMode::Bm25
            && let Some(embed) = &self.embed
        {
            let q = embed.embedder.embed(query);
            for (c, &doc) in candidates.iter_mut().zip(docs) {
                let row = &embed.vecs[doc * embed.dim..(doc + 1) * embed.dim];
                let cos: f32 = row.iter().zip(&q).map(|(a, b)| a * b).sum();
                c.score = cos.clamp(0.0, 1.0);
            }
            return;
        }
        let _ = (query, docs);
        for c in candidates.iter_mut() {
            c.score = squash_bm25(c.score);
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
    /// match the schema (empty doc, no detected table, or no lexical overlap),
    /// or when the group is refused by the container-shape gate below.
    ///
    /// Two emission gates keep the output honest:
    ///
    /// - **Group gate:** a record schema that contains its own `array<object>`
    ///   (`has_nested_groups`) describes tables, it isn't a row of one — header
    ///   tokens leak into BM25 for *any* table ("number", a description's
    ///   example phrases), so without this gate meta fields get force-mapped
    ///   onto data columns and every body row becomes a garbage record.
    /// - **Row gates:** a body row is dropped when (a) no mapped column has a
    ///   non-empty cell (an all-null record carries nothing), or (b) every
    ///   resolved cell merely echoes its own field name / column header — the
    ///   signature of a wrapped header line binned as a body row ("At Closing"
    ///   under `borrower_paid_at_closing`), not of data.
    pub fn extract_object_array(
        &self,
        group: &ObjectArraySchema,
        grids: &[table_units::TableGrid],
    ) -> ObjectArrayResult {
        if group.has_nested_groups {
            return ObjectArrayResult {
                records: Vec::new(),
            };
        }
        let Some(matched) = best_grid_for(group, grids) else {
            return ObjectArrayResult {
                records: Vec::new(),
            };
        };
        debug_dump_matched_grid(group, &matched);
        let header = grid_header(matched.grid)
            .map(|(h, _)| h)
            .unwrap_or_default();
        let records = matched.grid.rows[matched.body_start..]
            .iter()
            .filter(|row| !is_label_echo_row(&group.fields, row, &matched.mapping, &header))
            .map(|row| row_to_record(&group.fields, row, &matched.mapping, matched.grid.page))
            // All-miss records carry nothing — not even a candidate span (an
            // enum-no-match cell still surfaces its candidate, so it survives).
            .filter(|rec| rec.fields.iter().any(|f| !f.candidates.is_empty()))
            .collect();
        ObjectArrayResult { records }
    }
}

/// `LITEPARSE_EXTRACT_DEBUG=1` stderr dump of an object-array grouping decision:
/// which grid won, its header, and the column each sub-field mapped to. The
/// row-grouping counterpart of `dump_geometry_units` / `dump_table_grids`.
fn debug_dump_matched_grid(group: &ObjectArraySchema, matched: &MatchedGrid) {
    if !std::env::var("LITEPARSE_EXTRACT_DEBUG").is_ok_and(|v| v != "0") {
        return;
    }
    let header = grid_header(matched.grid)
        .map(|(h, _)| h)
        .unwrap_or_default();
    eprintln!(
        "[extract-debug] group{} → grid p{} ({} rows, body from {}), header: {:?}",
        group
            .description
            .as_deref()
            .map(|d| format!(" '{}'", &d[..d.len().min(60)]))
            .unwrap_or_default(),
        matched.grid.page,
        matched.grid.rows.len(),
        matched.body_start,
        header
    );
    for (field, m) in group.fields.iter().zip(&matched.mapping) {
        match m {
            Some((col, score)) => eprintln!(
                "[extract-debug]   {} → col {} {:?} (score {:.2})",
                field.name,
                col,
                header.get(*col).map(String::as_str).unwrap_or("?"),
                score
            ),
            None => eprintln!("[extract-debug]   {} → (unmapped)", field.name),
        }
    }
}

/// A body row whose every resolved cell just echoes its own field's name tokens
/// or its column's header tokens is a mis-binned (wrapped/continuation) header
/// line, not data: "At Closing | Others" under `borrower_paid_at_closing` /
/// `paid_by_others`. Real values ("405.00", "YES", "Application Fee") carry at
/// least one token outside that label vocabulary. Field *descriptions* are
/// deliberately not part of the vocabulary — bench-style descriptions embed
/// example values verbatim, which would make real data look like an echo.
fn is_label_echo_row(
    fields: &[SchemaField],
    row: &table_units::GridRow,
    mapping: &[Option<(usize, f32)>],
    header: &[String],
) -> bool {
    let mut resolved = 0usize;
    for (field, m) in fields.iter().zip(mapping) {
        let Some((col, _)) = m else { continue };
        let Some(cell) = row.cells.get(*col).map(|c| c.trim()) else {
            continue;
        };
        if cell.is_empty() {
            continue;
        }
        resolved += 1;
        let mut label_vocab: Vec<String> = tokenize(&field.name);
        label_vocab.extend(tokenize(header.get(*col).map(String::as_str).unwrap_or("")));
        if !tokenize(cell).iter().all(|t| label_vocab.contains(t)) {
            return false; // a real (non-echo) value — keep the row
        }
    }
    resolved > 0
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
/// paraphrased-header case, e.g. "QTT" for quantity, is an accepted residual).
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
    // Column matching is lexical (BM25 over header cells), so the cell score is
    // squashed into the same [0,1] relevance the bm25 scalar path reports.
    let score = squash_bm25(score);
    // Narrow the cell to its sub-value when a scanner fires (a bare data cell
    // usually is the value already, so this is often `None` and `text` carries
    // it); never-guess fields with no match yield `None`.
    let value = value_span::narrowed_value(field, cell);
    let page = provenance.map(|p| p.page).unwrap_or(grid_page);
    let bbox = provenance.map(|p| p.bbox.clone());
    let candidate = Candidate {
        value,
        text: cell.to_string(),
        score,
        page,
        // Only claim a bbox when the row actually resolved to a source line.
        bbox,
        source: UnitSource::HeaderCell,
    };
    FieldResult {
        name: field.name.clone(),
        candidates: vec![candidate],
    }
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
        // Ranking picks the invoice line; the candidate keeps the full span as
        // `text` and the label-stripped value in `value`.
        assert_eq!(r.candidates[0].text, "Invoice Number: INV-2024-0042");
        assert_eq!(r.candidates[0].value.as_deref(), Some("INV-2024-0042"));
        assert_eq!(r.candidates[0].page, 1);
        assert!(r.candidates[0].score > 0.0);
    }

    #[test]
    fn miss_when_no_query_term_appears() {
        let ex = extractor(&["Payment terms are net 30 days", "Thank you"]);
        let schema = ExtractionSchema {
            fields: vec![field("shipment_weight", "gross shipment weight kilograms")],
        };
        let r = &ex.extract(&schema).fields[0];
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
        assert_eq!(r.candidates[0].text, "Total VAT 18");
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
        // Date type pulls the value substring out onto the candidate.
        assert_eq!(r.candidates[0].value.as_deref(), Some("2024-01-15"));
    }

    #[test]
    fn number_field_extracts_value_substring() {
        // The amount lives on the second line; its candidate carries the
        // isolated number as `value` while the header line does not.
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
        assert!(
            r.candidates
                .iter()
                .any(|c| c.value.as_deref() == Some("1,250.00"))
        );
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
        assert_eq!(r.candidates[0].value.as_deref(), Some("premium"));
        // No choice supported → no isolated value, but the span is still surfaced.
        let ex2 = extractor(&["Your plan tier is Ultimate", "unrelated line"]);
        let r2 = &ex2.extract(&schema()).fields[0];
        assert!(!r2.candidates.is_empty());
        assert!(r2.candidates.iter().all(|c| c.value.is_none()));
    }

    #[test]
    fn str_field_strips_label_prefix() {
        let ex = extractor(&["Invoice Number: INV-2024-0042", "other"]);
        let schema = ExtractionSchema {
            fields: vec![field("invoice_number", "the invoice number")],
        };
        let r = &ex.extract(&schema).fields[0];
        // The candidate isolates the value via the label strip, but keeps the
        // full span as `text` for verification.
        assert_eq!(r.candidates[0].value.as_deref(), Some("INV-2024-0042"));
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

    #[cfg(feature = "static-embed")]
    #[test]
    fn rrf_fuses_sparse_and_dense_rankings() {
        // bm25 is sparse (only doc 2 ranked); unranked docs 0,1 take the tail
        // positions 1,2 in index order. embed is dense: doc 1, 0, 2.
        let bm25 = vec![(2usize, 5.0f32)];
        let embed = vec![(1usize, 0.9f32), (0, 0.5), (2, 0.1)];
        let fused = rrf_fuse(&[&bm25, &embed], 3);
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
            has_nested_groups: false,
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
        assert_eq!(r0.fields[0].candidates[0].text, "Widget");
        assert_eq!(r0.fields[1].candidates[0].text, "$9.00");
        // Row-level provenance flows onto every cell of the record.
        assert_eq!(
            r0.fields[0].candidates[0].bbox.as_ref().map(|b| b.y),
            Some(680.0)
        );
        assert_eq!(r0.fields[1].candidates[0].page, 1);

        assert_eq!(out.records[1].fields[0].candidates[0].text, "Gadget");
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
            has_nested_groups: false,
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
        assert_eq!(out.records[0].fields[0].candidates[0].text, "Bolt");
        assert_eq!(out.records[0].fields[1].candidates[0].text, "12");
    }

    /// Headerless grids can't map columns → no records rather than a wrong guess.
    #[test]
    fn object_array_skips_headerless_grid() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![field("item", "the item")],
            description: None,
            has_nested_groups: false,
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
            has_nested_groups: false,
        };
        let grid = TableGrid {
            page: 1,
            header: Some(vec!["Item".to_string(), "Price".to_string()]),
            rows: vec![row(&["Widget", ""], 680.0)],
        };
        let out = ex.extract_object_array(&group, &[grid]);
        assert_eq!(out.records[0].fields[0].candidates[0].text, "Widget");
        // The empty price cell is a miss — no candidate at all.
        assert!(out.records[0].fields[1].candidates.is_empty());
    }

    /// A record schema that contains its own repeated group (census-style
    /// `tables[]` carrying `rows[]`) is a container, not a table row — the
    /// group gate refuses it instead of force-mapping meta fields onto data
    /// columns.
    #[test]
    fn object_array_refuses_container_shaped_record() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![
                field("table_number", "the printed table number"),
                field("table_title", "the table title"),
            ],
            description: None,
            has_nested_groups: true,
        };
        let grid = TableGrid {
            page: 1,
            header: Some(vec!["Number".to_string(), "Title of measure".to_string()]),
            rows: vec![row(&["39,507", "Persons 65+"], 680.0)],
        };
        assert!(ex.extract_object_array(&group, &[grid]).records.is_empty());
    }

    /// Body rows where no mapped column has a non-empty cell would emit an
    /// all-null record — dropped, they carry nothing at all.
    #[test]
    fn object_array_drops_all_null_rows() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![field("item", "the item name"), field("price", "unit price")],
            description: None,
            has_nested_groups: false,
        };
        let grid = TableGrid {
            page: 1,
            header: Some(vec![
                "Item".to_string(),
                "Price".to_string(),
                "Ref".to_string(),
            ]),
            rows: vec![
                row(&["Widget", "$9.00", "A1"], 680.0),
                // Only the unmapped "Ref" column is filled → all-null record.
                row(&["", "", "B2"], 660.0),
            ],
        };
        let out = ex.extract_object_array(&group, &[grid]);
        assert_eq!(out.records.len(), 1);
        assert_eq!(out.records[0].fields[0].candidates[0].text, "Widget");
    }

    /// A body row whose resolved cells only echo their field names / column
    /// headers is a mis-binned wrapped-header line ("At Closing" under
    /// `borrower_paid_at_closing`), not data — dropped.
    #[test]
    fn object_array_drops_label_echo_rows() {
        let ex = LocalExtractor::from_units(OffsetMap::default(), vec![]);
        let group = ObjectArraySchema {
            fields: vec![
                field("borrower_paid_at_closing", "amount the borrower paid"),
                field("paid_by_others", "amount paid by third parties"),
            ],
            description: None,
            has_nested_groups: false,
        };
        let grid = TableGrid {
            page: 1,
            header: Some(vec!["Borrower-Paid".to_string(), "Paid by".to_string()]),
            rows: vec![
                // The split sub-header line detection binned as a body row.
                row(&["At Closing", "Others"], 700.0),
                row(&["$405.00", ""], 680.0),
            ],
        };
        let out = ex.extract_object_array(&group, &[grid]);
        assert_eq!(out.records.len(), 1);
        assert_eq!(out.records[0].fields[0].candidates[0].text, "$405.00");
    }

    /// An implemented `format` scanner that matches nowhere never guesses — no
    /// value is isolated on the candidate — while the retrieved span stays for
    /// provenance. When the shape IS present, the candidate carries it.
    #[test]
    fn format_scanner_isolates_value_or_nothing() {
        let mut f = field("contact_email", "contact email support team");
        f.format = Some("email".into());

        // No email present → the candidate has a span but no isolated value.
        let ex = extractor(&["Contact our sales team for email support today"]);
        let out = ex.extract_field(&f);
        assert!(!out.candidates.is_empty(), "spans stay for provenance");
        assert!(out.candidates.iter().all(|c| c.value.is_none()));

        // Email present → its address is isolated onto the candidate.
        let ex2 = extractor(&["Email our support team at help@acme.com"]);
        let out2 = ex2.extract_field(&f);
        assert!(
            out2.candidates
                .iter()
                .any(|c| c.value.as_deref() == Some("help@acme.com"))
        );
    }

    /// A plain (untyped) field with lexical support surfaces its span as a
    /// candidate; nothing was narrowed, so `value` stays absent.
    #[test]
    fn plain_field_surfaces_span_without_narrowing() {
        let ex = extractor(&["Payment terms net 30"]);
        let out = ex.extract_field(&field("terms", "payment terms"));
        assert_eq!(out.candidates[0].text, "Payment terms net 30");
        assert!(out.candidates[0].value.is_none());
    }

    /// A typed (money) hit isolates the amount onto the candidate's `value`.
    #[test]
    fn number_field_isolates_money_value() {
        let ex = extractor(&["Total due $118.50 today"]);
        let mut f = field("total_amount", "total amount due");
        f.field_type = FieldType::Number;
        let out = ex.extract_field(&f);
        assert_eq!(out.candidates[0].value.as_deref(), Some("$118.50"));
    }
}
