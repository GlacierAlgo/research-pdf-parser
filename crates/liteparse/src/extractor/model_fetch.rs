//! Download-on-first-use for the static embedding model — the tessdata pattern
//! from `ocr/tesseract.rs` applied to model files: platform cache dir, one-time
//! size notice, temp-file + atomic rename so concurrent first use is safe and a
//! partial download never resolves.
//!
//! Nothing here runs unless extraction actually needs the model: the
//! `extract_offline` config skips it entirely, and [`ensure_model`] returns
//! immediately when [`resolve_model_dir`] already finds the files (explicit
//! path, env var, HF cache, or a previous download).
//!
//! Files are fetched from the Hugging Face hub (`resolve/main`). The
//! `model.safetensors` weights land last: [`resolve_model_dir`] requires
//! `tokenizer.json` + `model.safetensors`, so an interrupted download leaves a
//! directory that does not resolve and is retried next time.

use super::static_embed::resolve_model_dir;
use crate::error::LiteParseError;
use std::path::{Path, PathBuf};

/// The files a model2vec model directory needs. Weights last — see module doc.
const MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// Platform cache directory for downloaded models, mirroring the tessdata
/// location policy. Override with `LITEPARSE_MODELS_DIR`.
pub fn models_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LITEPARSE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join("Library/Application Support/liteparse/models");
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".liteparse/models");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(base) = std::env::var("APPDATA").ok().or_else(|| {
            std::env::var("USERPROFILE")
                .ok()
                .map(|p| format!("{p}\\AppData\\Roaming"))
        }) {
            return PathBuf::from(base).join("liteparse").join("models");
        }
    }
    PathBuf::from("liteparse-models")
}

/// The directory a downloaded `model_id` lands in (HF-cache-style naming, so
/// the same id never collides across orgs).
pub fn model_download_dir(model_id: &str) -> PathBuf {
    models_cache_dir().join(format!("models--{}", model_id.replace('/', "--")))
}

/// Ensure `model_id` is available locally, downloading it on first use.
/// Returns the model directory. Resolution order is [`resolve_model_dir`]'s
/// (explicit/env/HF cache/prior download); only a full miss downloads.
pub async fn ensure_model(
    model_id: &str,
    explicit: Option<&Path>,
    quiet: bool,
) -> Result<PathBuf, LiteParseError> {
    if let Some(dir) = resolve_model_dir(model_id, explicit) {
        return Ok(dir);
    }
    let dir = model_download_dir(model_id);
    tokio::fs::create_dir_all(&dir).await?;

    for file in MODEL_FILES {
        let final_path = dir.join(file);
        if final_path.exists() {
            continue;
        }
        let url = format!("https://huggingface.co/{model_id}/resolve/main/{file}");
        let response = reqwest::get(&url)
            .await
            .map_err(|e| fetch_err(model_id, &url, e))?;
        if !response.status().is_success() {
            return Err(fetch_err(model_id, &url, response.status()));
        }
        if !quiet && file == "model.safetensors" {
            let size = response
                .content_length()
                .map(|n| format!("{:.0} MB", n as f64 / 1e6))
                .unwrap_or_else(|| "size unknown".into());
            eprintln!(
                "extract: downloading embedding model {model_id} ({size}, one-time) \
                 to {}",
                dir.display()
            );
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| fetch_err(model_id, &url, e))?;

        // Temp file + atomic rename (the tessdata move): concurrent first use
        // is safe, and a partial file never appears at the final path.
        let tmp_path = dir.join(format!("{file}.tmp.{}", std::process::id()));
        tokio::fs::write(&tmp_path, &bytes).await?;
        if let Err(e) = tokio::fs::rename(&tmp_path, &final_path).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            if !final_path.exists() {
                return Err(e.into());
            }
        }
    }
    Ok(dir)
}

fn fetch_err(model_id: &str, url: &str, e: impl std::fmt::Display) -> LiteParseError {
    LiteParseError::Other(format!(
        "downloading extract model \"{model_id}\" from {url}: {e}"
    ))
}
