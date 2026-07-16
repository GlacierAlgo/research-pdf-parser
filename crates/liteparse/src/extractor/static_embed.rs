//! model2vec static-embedding inference.
//!
//! A hand-port of the Python `model2vec.StaticModel.encode` path for the
//! default model (`potion-retrieval-32M`): tokenize
//! with the model's `tokenizer.json`, look up one embedding row per token id,
//! mean-pool, L2-normalize. There is no ML runtime here — inference is a
//! matrix row lookup, which is what makes the static engine ~0.05 ms/line.
//!
//! The tokenizer stage is parity-by-construction: the Python `model2vec`
//! package's tokenizer *is* the HF `tokenizers` Rust crate (via its Python
//! binding), and this module uses the same crate on the same `tokenizer.json`.
//! What this file actually ports is the encode recipe around it, replicated
//! 1:1 from `model2vec/model.py` (`StaticModel.tokenize` + `_encode_batch`):
//!
//! 1. Char-truncate the input to `MAX_TOKENS × median vocab-token char
//!    length` BEFORE tokenizing (model2vec's cheap guard against pathological
//!    inputs), then tokenize with `add_special_tokens=false`, strip `[UNK]`
//!    ids, and cap at [`MAX_TOKENS`] ids.
//! 2. Mean-pool the embedding rows; an empty id list embeds to the zero
//!    vector.
//! 3. When the model config says `normalize: true` (potion models do),
//!    L2-normalize with model2vec's `+1e-32` denominator guard.
//!
//! Parity is enforced against a reference fixture
//! (`dataset_eval_utils/extract_poc/parity_fixture.json` — 28 strings with
//! reference token ids plus fp32 and int8 vectors; see
//! `export_parity_fixture.py`). The fixture's `token_ids` were exported with
//! special tokens *included*, so the tokenizer test encodes with
//! `add_special_tokens=true`; the embedding path uses `false`, matching the
//! reference encode (the empty string maps to ids `[CLS][SEP]` but a zero
//! vector).
//!
//! ## int8
//!
//! model2vec's `quantize_to="int8"` rescales the whole matrix by one GLOBAL
//! `max|w| / 127` factor at load (not per-row) and then encodes with the raw
//! int8 rows — the scale is discarded upstream, which only round-trips
//! because L2-normalization cancels a global factor. We keep the scale and
//! apply it before normalization so the math is also right for a
//! hypothetical `normalize: false` model. Two knowingly-accepted
//! micro-divergences from numpy, both far inside the fixture's 0.02 int8
//! tolerance: no float16 intermediate during quantization, and summation
//! order (we accumulate in f64; numpy pairwise-sums in f32).
//!
//! Model files are read from a local directory (`tokenizer.json` +
//! `model.safetensors` + `config.json`); [`resolve_model_dir`] finds an
//! already-downloaded copy (explicit path, env var, HF-hub snapshot, or the
//! liteparse model cache). Download-on-first-use lives in
//! [`super::model_fetch`]. On native targets the weights are
//! mmapped, not read: fp32 rows are decoded from the mapping per lookup, so
//! load cost is ~the tokenizer parse and only touched rows ever page in.

use crate::error::LiteParseError;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// model2vec's `encode(max_length=512)` default: token-id cap per input.
const MAX_TOKENS: usize = 512;

/// L2-norm denominator guard, exactly model2vec's.
const NORM_EPS: f64 = 1e-32;

/// Embedding-matrix precision. `F32` is the shipped default
/// (`potion-retrieval-32M` weights are fp32); `Int8` quantizes at load with
/// model2vec's global max-abs/127 scale — 4× smaller resident matrix, drift
/// ≤ ~0.015 on the parity fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedPrecision {
    F32,
    Int8,
}

enum Matrix {
    /// Owned fp32 matrix — the wasm path (no mmap in the browser).
    #[cfg(target_arch = "wasm32")]
    F32(Vec<f32>),
    /// The fp32 matrix served straight from the mmapped `model.safetensors`
    /// (`offset` = the embeddings tensor's byte offset in the file). Rows are
    /// decoded per lookup with `f32::from_le_bytes` — a plain load on
    /// little-endian targets — so load time is ~the header parse and untouched
    /// rows are never paged in.
    #[cfg(not(target_arch = "wasm32"))]
    F32Mapped {
        map: memmap2::Mmap,
        offset: usize,
    },
    Int8 {
        data: Vec<i8>,
        scale: f32,
    },
}

impl Matrix {
    /// Add row `id` into `acc` (length = dim). int8 rows are dequantized by
    /// the caller-visible contract that `acc` accumulates *scaled* values.
    fn accumulate_row(&self, id: usize, dim: usize, acc: &mut [f64]) {
        match self {
            #[cfg(target_arch = "wasm32")]
            Matrix::F32(m) => {
                for (a, &v) in acc.iter_mut().zip(&m[id * dim..(id + 1) * dim]) {
                    *a += v as f64;
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Matrix::F32Mapped { map, offset } => {
                let row = &map[offset + id * dim * 4..offset + (id + 1) * dim * 4];
                for (a, b) in acc.iter_mut().zip(row.chunks_exact(4)) {
                    *a += f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
                }
            }
            Matrix::Int8 { data, scale } => {
                let s = *scale as f64;
                for (a, &v) in acc.iter_mut().zip(&data[id * dim..(id + 1) * dim]) {
                    *a += v as f64 * s;
                }
            }
        }
    }
}

/// A loaded model2vec static embedding model. Build with
/// [`StaticEmbedder::load`], embed with [`StaticEmbedder::embed`].
pub struct StaticEmbedder {
    tokenizer: Tokenizer,
    matrix: Matrix,
    dim: usize,
    unk_id: Option<u32>,
    /// Pre-tokenize char cap: `MAX_TOKENS × median vocab-token char length`.
    char_cap: usize,
    normalize: bool,
}

fn extract_err(stage: &str, e: impl std::fmt::Display) -> LiteParseError {
    LiteParseError::Other(format!("static-embed {stage}: {e}"))
}

impl StaticEmbedder {
    /// Load a model2vec model directory (`tokenizer.json`, `model.safetensors`,
    /// `config.json`) at fp32.
    pub fn load(dir: &Path) -> Result<StaticEmbedder, LiteParseError> {
        StaticEmbedder::load_with(dir, EmbedPrecision::F32)
    }

    /// Load with an explicit matrix precision.
    pub fn load_with(
        dir: &Path,
        precision: EmbedPrecision,
    ) -> Result<StaticEmbedder, LiteParseError> {
        let tok_raw = std::fs::read(dir.join("tokenizer.json"))?;
        let tokenizer = Tokenizer::from_bytes(&tok_raw).map_err(|e| extract_err("tokenizer", e))?;

        // The unk token NAME lives in the tokenizer.json model section
        // (`[UNK]` for WordPiece here, but read it rather than assume) —
        // model2vec strips its id so unknown words don't drag every embedding
        // toward one shared vector.
        let tok_json: serde_json::Value = serde_json::from_slice(&tok_raw)?;
        let unk_id = tok_json
            .get("model")
            .and_then(|m| m.get("unk_token"))
            .and_then(|u| u.as_str())
            .and_then(|u| tokenizer.token_to_id(u));

        // `normalize` comes from config.json (model2vec defaults it to false;
        // all potion models set true).
        #[derive(serde::Deserialize, Default)]
        struct EmbedConfig {
            #[serde(default)]
            normalize: bool,
        }
        let config: EmbedConfig = match std::fs::read(dir.join("config.json")) {
            Ok(raw) => serde_json::from_slice(&raw)?,
            Err(_) => EmbedConfig::default(),
        };

        // The weights are mmapped rather than read: fp32 serves rows straight
        // from the mapping (load = header parse; untouched rows never page in)
        // and int8 quantizes in one streaming pass with no f32 copy. wasm has
        // no mmap and takes the owned-buffer path.
        let st_path = dir.join("model.safetensors");
        #[cfg(not(target_arch = "wasm32"))]
        let st_raw: memmap2::Mmap = {
            let file = std::fs::File::open(&st_path)?;
            // SAFETY: standard mmap-a-file caveat — undefined behavior only if
            // the file is truncated/modified while mapped. Model files are
            // written once (atomic rename in model_fetch) and then immutable.
            unsafe { memmap2::Mmap::map(&file)? }
        };
        #[cfg(target_arch = "wasm32")]
        let st_raw: Vec<u8> = std::fs::read(&st_path)?;

        // Borrow scope: `SafeTensors` borrows `st_raw`; extract the tensor's
        // byte offset + shape, then drop it so `st_raw` can move into `Matrix`.
        let (offset, vocab_rows, dim) = {
            let st = safetensors::SafeTensors::deserialize(&st_raw)
                .map_err(|e| extract_err("safetensors", e))?;
            let view = st
                .tensor("embeddings")
                .map_err(|e| extract_err("safetensors[embeddings]", e))?;
            if view.dtype() != safetensors::Dtype::F32 {
                return Err(extract_err(
                    "safetensors[embeddings]",
                    format!("expected F32 weights, got {:?}", view.dtype()),
                ));
            }
            let [vocab_rows, dim]: [usize; 2] = view
                .shape()
                .try_into()
                .map_err(|_| extract_err("safetensors[embeddings]", "expected a 2-D matrix"))?;
            let data = view.data();
            let offset = data.as_ptr() as usize - st_raw.as_ptr() as usize;
            (offset, vocab_rows, dim)
        };

        // Mirror model2vec's constructor check: every token id must have a row
        // (potion models have no vocabulary-quantization token_mapping).
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_rows != vocab_size {
            return Err(extract_err(
                "load",
                format!("{vocab_rows} embedding rows for {vocab_size} vocab tokens"),
            ));
        }

        let char_cap = MAX_TOKENS * median_token_char_length(&tokenizer);

        let tensor_bytes = &st_raw[offset..offset + vocab_rows * dim * 4];
        let matrix = match precision {
            #[cfg(not(target_arch = "wasm32"))]
            EmbedPrecision::F32 => Matrix::F32Mapped {
                offset,
                map: st_raw,
            },
            #[cfg(target_arch = "wasm32")]
            EmbedPrecision::F32 => Matrix::F32(
                tensor_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
            ),
            EmbedPrecision::Int8 => quantize_int8_bytes(tensor_bytes),
        };

        Ok(StaticEmbedder {
            tokenizer,
            matrix,
            dim,
            unk_id,
            char_cap,
            normalize: config.normalize,
        })
    }

    /// Embedding dimensionality (512 for potion-retrieval-32M).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Tokenize `text` the model2vec way for [`embed`](Self::embed):
    /// char-truncate first (cheap), encode with `add_special_tokens=false`,
    /// strip `[UNK]` ids, cap at [`MAX_TOKENS`].
    /// Tokenizer errors are unreachable for plain strings on a validated
    /// tokenizer; treat them as "no tokens" rather than plumbing a `Result` into
    /// every ranking loop.
    fn token_ids(&self, text: &str) -> Vec<u32> {
        let capped: &str = match text.char_indices().nth(self.char_cap) {
            Some((byte_idx, _)) => &text[..byte_idx],
            None => text,
        };
        self.tokenizer
            .encode(capped, false)
            .map(|enc| enc.get_ids().to_vec())
            .unwrap_or_default()
            .into_iter()
            .filter(|id| Some(*id) != self.unk_id)
            .take(MAX_TOKENS)
            .collect()
    }

    /// Embed one string. Inputs with no usable tokens (empty, whitespace,
    /// all-unknown words) embed to the zero vector — cosine 0 against
    /// everything, i.e. "no signal", never an error.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let ids = self.token_ids(text);
        if ids.is_empty() {
            return vec![0.0; self.dim];
        }

        let mut acc = vec![0.0f64; self.dim];
        for id in &ids {
            self.matrix.accumulate_row(*id as usize, self.dim, &mut acc);
        }
        let n = ids.len() as f64;
        for a in &mut acc {
            *a /= n;
        }
        if self.normalize {
            let norm = acc.iter().map(|v| v * v).sum::<f64>().sqrt() + NORM_EPS;
            for a in &mut acc {
                *a /= norm;
            }
        }
        acc.into_iter().map(|v| v as f32).collect()
    }
}

/// model2vec's `median_token_length`: `int(np.median([len(tok) for tok in
/// vocab]))`, token length in chars (codepoints), vocab in id order (order is
/// irrelevant to a median; added tokens included).
fn median_token_char_length(tokenizer: &Tokenizer) -> usize {
    let mut lens: Vec<usize> = tokenizer
        .get_vocab(true)
        .keys()
        .map(|t| t.chars().count())
        .collect();
    if lens.is_empty() {
        return 1;
    }
    lens.sort_unstable();
    let n = lens.len();
    if n % 2 == 1 {
        lens[n / 2]
    } else {
        // int(np.median(..)) on an even count: mean of the two middles,
        // truncated toward zero.
        ((lens[n / 2 - 1] + lens[n / 2]) as f64 / 2.0) as usize
    }
}

/// model2vec's `quantize_embeddings(int8)`: one global symmetric scale
/// `max|w| / 127`, round-half-to-even (numpy `rint`), clamp to [-127, 127].
/// Streams the raw little-endian f32 bytes twice (max-abs, then quantize) so
/// the fp32 matrix is never materialized.
fn quantize_int8_bytes(bytes: &[u8]) -> Matrix {
    let vals = || {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let max_abs = vals().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        return Matrix::Int8 {
            data: vec![0; bytes.len() / 4],
            scale: 1.0,
        };
    }
    let scale = max_abs / 127.0;
    let data = vals()
        .map(|v| (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();
    Matrix::Int8 { data, scale }
}

/// A directory holds a usable model2vec model when it contains both a
/// `tokenizer.json` and a `model.safetensors`. Used to validate an explicit
/// `--model-path` directly (independent of the resolution fallbacks) and by
/// [`resolve_model_dir`] for each candidate directory.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join("tokenizer.json").is_file() && dir.join("model.safetensors").is_file()
}

/// Find a local directory holding the model files, checked in order:
///
/// 1. `explicit` (the `extract_model_path` config),
/// 2. the `LITEPARSE_EXTRACT_MODEL_PATH` env var,
/// 3. an existing HF-hub cache snapshot for `model_id`
///    (`~/.cache/huggingface/hub/models--org--name/snapshots/*`),
/// 4. liteparse's own model cache (a prior
///    [`ensure_model`](super::model_fetch::ensure_model) download).
///
/// Returns the first directory that contains both `tokenizer.json` and
/// `model.safetensors`. On a full miss, native callers download via
/// [`ensure_model`](super::model_fetch::ensure_model) unless offline.
pub fn resolve_model_dir(model_id: &str, explicit: Option<&Path>) -> Option<PathBuf> {
    let has_model = is_model_dir;

    if let Some(dir) = explicit.filter(|d| has_model(d)) {
        return Some(dir.to_path_buf());
    }
    if let Ok(dir) = std::env::var("LITEPARSE_EXTRACT_MODEL_PATH") {
        let dir = PathBuf::from(dir);
        if has_model(&dir) {
            return Some(dir);
        }
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let repo = home
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", model_id.replace('/', "--")))
            .join("snapshots");
        if let Ok(snapshots) = std::fs::read_dir(repo) {
            for entry in snapshots.flatten() {
                let dir = entry.path();
                if has_model(&dir) {
                    return Some(dir);
                }
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = super::model_fetch::model_download_dir(model_id);
        if has_model(&dir) {
            return Some(dir);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_matches_numpy_int_median_semantics() {
        // Odd count → middle element; even count → truncated mean of middles.
        assert_eq!(median_of(&[1, 2, 9]), 2);
        assert_eq!(median_of(&[1, 2, 3, 9]), 2); // int(2.5) = 2
        assert_eq!(median_of(&[2, 2, 3, 3]), 2); // int(2.5) = 2
    }

    fn median_of(lens: &[usize]) -> usize {
        let mut lens = lens.to_vec();
        lens.sort_unstable();
        let n = lens.len();
        if n % 2 == 1 {
            lens[n / 2]
        } else {
            ((lens[n / 2 - 1] + lens[n / 2]) as f64 / 2.0) as usize
        }
    }

    #[test]
    fn int8_quantization_is_global_scale_rint() {
        let bytes: Vec<u8> = [0.5f32, -1.0, 0.25, 0.003_936, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let m = quantize_int8_bytes(&bytes);
        let Matrix::Int8 { data, scale } = m else {
            panic!("expected int8 matrix");
        };
        assert!((scale - 1.0 / 127.0).abs() < 1e-9);
        // 0.5/scale = 63.5 → round-half-even → 64? No: 63.5 ties to 64 (even).
        // numpy rint(63.5) = 64. rint(0.003936/scale = 0.4999) = 0.
        assert_eq!(data, vec![64, -127, 32, 0, 0]);
    }

    // ── Fixture parity (needs the real model on disk; skips otherwise) ──────

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../dataset_eval_utils/extract_poc/parity_fixture.json"
    );

    #[derive(serde::Deserialize)]
    struct Fixture {
        model: String,
        normalize: bool,
        dim: usize,
        cases: Vec<Case>,
    }

    #[derive(serde::Deserialize)]
    struct Case {
        text: String,
        token_ids: Vec<u32>,
        embedding_fp32: Vec<f32>,
        embedding_int8: Vec<f32>,
    }

    fn load_fixture_and_model() -> Option<(Fixture, PathBuf)> {
        // The fixture is checked into the repo — failing to read it is a real
        // failure, not a skip. Only a locally-absent MODEL downgrades to skip.
        let fixture: Fixture = serde_json::from_str(
            &std::fs::read_to_string(FIXTURE).expect("read parity_fixture.json"),
        )
        .expect("parse parity_fixture.json");
        assert!(!fixture.cases.is_empty(), "fixture has no cases");
        let Some(dir) = resolve_model_dir(&fixture.model, None) else {
            eprintln!(
                "SKIP static_embed parity: model {} not found locally \
                 (set LITEPARSE_EXTRACT_MODEL_PATH)",
                fixture.model
            );
            return None;
        };
        Some((fixture, dir))
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn parity_token_ids_and_fp32_embeddings() {
        let Some((fixture, dir)) = load_fixture_and_model() else {
            return;
        };
        let emb = StaticEmbedder::load(&dir).expect("load fp32 model");
        assert_eq!(emb.dim(), fixture.dim);
        assert_eq!(emb.normalize, fixture.normalize);

        let mut worst = 0.0f32;
        for case in &fixture.cases {
            // The fixture exported ids WITH special tokens; assert the raw
            // tokenizer stage matches so a vector mismatch can be localized.
            let ids = emb
                .tokenizer
                .encode(case.text.as_str(), true)
                .expect("encode")
                .get_ids()
                .to_vec();
            assert_eq!(ids, case.token_ids, "token ids diverge on {:?}", case.text);

            let v = emb.embed(&case.text);
            let diff = max_abs_diff(&v, &case.embedding_fp32);
            worst = worst.max(diff);
            assert!(
                diff < 1e-4,
                "fp32 embedding diverges on {:?}: max abs diff {diff}",
                case.text
            );
        }
        eprintln!(
            "fp32 parity: {} cases, worst max-abs diff {worst:.2e}",
            fixture.cases.len()
        );
    }

    #[test]
    fn parity_int8_embeddings() {
        let Some((fixture, dir)) = load_fixture_and_model() else {
            return;
        };
        let emb = StaticEmbedder::load_with(&dir, EmbedPrecision::Int8).expect("load int8 model");
        let mut worst = 0.0f32;
        for case in &fixture.cases {
            let v = emb.embed(&case.text);
            let diff = max_abs_diff(&v, &case.embedding_int8);
            worst = worst.max(diff);
            // ~0.02 is the int8 CI tolerance (fixture drift median 0.006 / max
            // 0.015 vs fp32; our quantization skips numpy's float16
            // intermediate, worth at most 1 int8 step).
            assert!(
                diff < 0.02,
                "int8 embedding diverges on {:?}: max abs diff {diff}",
                case.text
            );
        }
        eprintln!(
            "int8 parity: {} cases, worst max-abs diff {worst:.2e}",
            fixture.cases.len()
        );
    }
}
