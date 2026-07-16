use crate::config::{LiteParseConfig, parse_target_pages};
#[cfg(not(target_arch = "wasm32"))]
use crate::conversion;
use crate::error::LiteParseError;
use crate::ocr::OcrEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::ocr::http_simple::HttpOcrEngine;
#[cfg(feature = "tesseract")]
use crate::ocr::tesseract::TesseractOcrEngine;
use crate::ocr_merge;
use crate::output::markdown;
use crate::pdf_read;
use crate::projection;
#[cfg(not(target_arch = "wasm32"))]
use crate::render;
use crate::types::{ExtractedImage, OutlineTarget, Page, ParsedPage, PdfInput};
use pdfium::Library;

/// Result of parsing a document.
pub struct ParseResult {
    /// Parsed pages with projected text layout.
    pub pages: Vec<ParsedPage>,
    /// Full document text, concatenated from all pages.
    pub text: String,
    /// Document outline (bookmarks) when present. Used by the markdown
    /// emitter as a high-priority heading source on untagged PDFs.
    pub outline: Vec<OutlineTarget>,
    /// Raster images extracted from the document. Empty unless the parser
    /// was configured with `ImageMode::Embed`. Each entry carries the same
    /// `id` the markdown emitter referenced in `![](image_{id}.png)`, so the
    /// caller can match them up without parsing markdown.
    pub images: Vec<ExtractedImage>,
}

/// Result of rendering a single page screenshot.
#[derive(Debug, Clone)]
pub struct ScreenshotResult {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub image_bytes: Vec<u8>,
}

/// Env var pointing at a fragmented glyph-outline → unicode font database
/// directory (`%02x%02x.msgpack` shards). When set, [`LiteParse::new`]
/// auto-wires a [`crate::FontDbResolver`] so buggy/obfuscated-font glyphs are
/// recovered without any extra wiring. Unset (default) leaves the hook dormant.
#[cfg(not(target_arch = "wasm32"))]
const FONT_DB_DIR_ENV: &str = "LITEPARSE_FONT_DB_DIR";

/// Build the default glyph resolver from the environment, if configured.
#[cfg(not(target_arch = "wasm32"))]
fn default_glyph_resolver() -> Option<std::sync::Arc<dyn crate::GlyphResolver>> {
    let dir = std::env::var_os(FONT_DB_DIR_ENV)?;
    if dir.is_empty() {
        return None;
    }
    Some(std::sync::Arc::new(crate::FontDbResolver::new(dir)))
}

#[cfg(target_arch = "wasm32")]
fn default_glyph_resolver() -> Option<std::sync::Arc<dyn crate::GlyphResolver>> {
    None
}

/// Main LiteParse orchestrator.
///
/// ### Thread safety
///
/// `LiteParse` is `Send + Sync` and safe to share across threads (e.g.
/// behind an `Arc`, or used concurrently from a multi-threaded `tokio`
/// runtime).
///
/// PDFium itself is **not** thread-safe, so all PDFium FFI work — document
/// loading, page rendering, text extraction — is serialized through a
/// process-global lock held by [`pdfium::Library`]. From a caller's
/// perspective, this means concurrent `parse_*` / `screenshot*` calls are
/// safe but their PDFium portions run sequentially. The OCR pass and grid
/// projection (which dominate runtime for OCR-heavy documents) run outside
/// the lock and remain fully concurrent.
pub struct LiteParse {
    config: LiteParseConfig,
    /// Optional caller-provided OCR engine. When set, this overrides the
    /// built-in selection logic (HTTP OCR / Tesseract). This is the primary
    /// mechanism for plugging an OCR engine in environments without the
    /// built-ins (e.g. WASM, where the JS side supplies a callback engine).
    ocr_engine_override: Option<std::sync::Arc<dyn OcrEngine>>,
    /// Optional caller-provided glyph recovery hook. When set, it is consulted
    /// as a last resort for buggy/obfuscated-font glyphs that liteparse's
    /// built-in cmap/AGL recovery could not decode. The published package ships
    /// none; the platform build injects an outline → unicode font-DB resolver.
    glyph_resolver: Option<std::sync::Arc<dyn crate::GlyphResolver>>,
}

impl LiteParse {
    pub fn new(config: LiteParseConfig) -> Self {
        Self {
            config,
            ocr_engine_override: None,
            glyph_resolver: default_glyph_resolver(),
        }
    }

    /// Override the OCR engine. When set, the engine is used regardless of
    /// `ocr_server_url` / built-in Tesseract availability.
    pub fn with_ocr_engine(mut self, engine: std::sync::Arc<dyn OcrEngine>) -> Self {
        self.ocr_engine_override = Some(engine);
        self
    }

    /// Inject a glyph recovery hook. When set, glyphs that liteparse considers
    /// untrusted and cannot decode with its built-in cmap/AGL recovery are
    /// passed to the resolver as vector-outline segments for a final attempt.
    pub fn with_glyph_resolver(
        mut self,
        resolver: std::sync::Arc<dyn crate::GlyphResolver>,
    ) -> Self {
        self.glyph_resolver = Some(resolver);
        self
    }

    /// Parse the configured `target_pages` string (e.g. `"1-5,10"`) into an
    /// explicit page list, or `None` when no selection was configured.
    fn resolve_target_pages(&self) -> Result<Option<Vec<u32>>, LiteParseError> {
        self.config
            .target_pages
            .as_ref()
            .map(|s| parse_target_pages(s))
            .transpose()
            .map_err(|e| format!("invalid --target-pages: {}", e).into())
    }

    /// Determine the complexity of each page in a document, returning a vector
    /// of `PageComplexityStats` for each page. This is useful for deciding
    /// whether to enable OCR on a per-page basis, or for other heuristics.
    pub async fn is_complex(
        &self,
        input: PdfInput,
    ) -> Result<Vec<ocr_merge::PageComplexityStats>, LiteParseError> {
        let log = |msg: &str| {
            if !self.config.quiet {
                eprintln!("{}", msg);
            }
        };

        let t0 = web_time::Instant::now();

        #[cfg(not(target_arch = "wasm32"))]
        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), false).await?;

        #[cfg(target_arch = "wasm32")]
        let validated_input = input;

        // Determine which pages to extract
        let target_pages = self.resolve_target_pages()?;

        // Load the document and extract text items. Complexity signals derive
        // from the text layer and page objects only — embedded image rasters
        // and hyperlinks are irrelevant here, so both are skipped to keep this
        // pass fast (its whole purpose is a cheap pre-OCR check).
        let password = self.config.password.as_deref();

        let lib = Library::init();
        let document = pdf_read::load_document_from_input(&lib, &validated_input, password)?;

        let (pages, _) = pdf_read::extract_pages_and_images(
            &document,
            target_pages.as_deref(),
            self.config.max_pages,
            false, // render_images: image rasters not needed for complexity
            false, // extract_links: irrelevant for complexity stats
            self.glyph_resolver.as_deref(),
            false, // emit_word_boxes: word boxes not needed for complexity stats
        )?;
        let t_extract = web_time::Instant::now();
        log(&format!(
            "[liteparse] extract: {:.1}ms ({} pages)",
            t_extract.duration_since(t0).as_secs_f64() * 1000.0,
            pages.len()
        ));

        let t_complexity = web_time::Instant::now();
        let page_complexities = pages
            .iter()
            .map(|page| {
                let page_obj = document.page((page.page_number - 1) as i32)?;
                ocr_merge::calculate_page_complexity(page, &page_obj)
            })
            .collect::<Result<Vec<_>, _>>()?;
        log(&format!(
            "[liteparse] complexity: {:.1}ms",
            t_complexity.duration_since(t_extract).as_secs_f64() * 1000.0
        ));

        Ok(page_complexities)
    }

    /// Parse a document from a file path, returning structured results.
    ///
    /// Non-PDF files are automatically converted to PDF first (requires
    /// LibreOffice/ImageMagick on the system).
    ///
    /// Not available on `wasm32` — the browser has no filesystem. Use
    /// [`LiteParse::parse_input`] with [`PdfInput::Bytes`] instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn parse(&self, input: &str) -> Result<ParseResult, LiteParseError> {
        self.parse_input(PdfInput::Path(input.to_string())).await
    }

    /// Schema extraction: parse `input`, then extract every field of a
    /// **standard JSON Schema** from it — nested objects flatten to dotted
    /// names, `array<object>` groups row-group over detected tables.
    ///
    /// The contract is **narrowing with provenance**, not LLM-grade extraction:
    /// each field returns its top-k ranked candidate spans (a score, a page, a
    /// bbox, and — when isolable — a narrowed value per candidate), not a single
    /// asserted answer. Values are verbatim spans — normalization is
    /// deliberately out of scope.
    ///
    /// Retrieval always fuses BM25 with a static embedding model (RRF), degrading
    /// to BM25-only when no model is available. Engine knobs live on
    /// [`LiteParseConfig`]: `extract_model` / `extract_model_path` (local files
    /// preferred; downloaded on first use on native builds — any
    /// resolution/download failure logs and degrades to BM25), `extract_offline`
    /// (skip the download; use a cached model if present), `extract_top_k`.
    ///
    /// A schema that flattens to zero extractable fields is an error, not an
    /// empty result.
    pub async fn extract(
        &self,
        input: PdfInput,
        schema: &serde_json::Value,
    ) -> Result<crate::extractor::DocumentExtraction, LiteParseError> {
        use crate::extractor::{
            ArrayFieldResult, DocumentExtraction, ExtractionSchema, LocalExtractor,
            ObjectArraySchema, schema_json, table_units,
        };

        let flat = schema_json::flatten_json_schema(schema);
        if flat.fields.is_empty() && flat.object_arrays.is_empty() {
            return Err("schema flattened to 0 extractable fields — \
                 check for $ref shapes the flattener doesn't resolve"
                .into());
        }

        let parsed = self.parse_input(input).await?;

        #[cfg_attr(not(feature = "static-embed"), allow(unused_mut))]
        let mut extractor =
            LocalExtractor::from_pages(&parsed.pages).with_top_k(self.config.extract_top_k);
        // Always-fuse: attach the static embedder whenever it resolves.
        // Resolution prefers local files (explicit path, env var, HF cache,
        // prior download); a full miss downloads on first use (native builds,
        // tessdata pattern) unless `extract_offline` is set. Any failure
        // degrades to BM25-only.
        #[cfg(feature = "static-embed")]
        {
            use crate::extractor::static_embed::{StaticEmbedder, is_model_dir, resolve_model_dir};
            let explicit = self
                .config
                .extract_model_path
                .as_deref()
                .map(std::path::Path::new);
            // An explicit `--model-path` the caller asserted must resolve — a
            // typo silently falling through to the env var / cached default
            // model is a trust bug, so validate the path itself and refuse it up
            // front. (`resolve_model_dir` would mask it via its fallbacks.)
            if let Some(path) = explicit
                && !is_model_dir(path)
            {
                return Err(format!(
                    "extract model path does not contain a model \
                     (need tokenizer.json + model.safetensors): {}",
                    path.display()
                )
                .into());
            }
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let mut dir = resolve_model_dir(&self.config.extract_model, explicit);
            #[cfg(not(target_arch = "wasm32"))]
            if dir.is_none() && !self.config.extract_offline {
                match crate::extractor::model_fetch::ensure_model(
                    &self.config.extract_model,
                    explicit,
                    self.config.quiet,
                )
                .await
                {
                    Ok(d) => dir = Some(d),
                    Err(e) if !self.config.quiet => {
                        eprintln!("extract: {e} — running BM25-only");
                    }
                    _ => {}
                }
            }
            match dir.as_deref().map(StaticEmbedder::load) {
                Some(Ok(e)) => extractor = extractor.with_embedder(std::sync::Arc::new(e)),
                Some(Err(e)) if !self.config.quiet => {
                    eprintln!("extract: embedding model failed to load ({e}) — running BM25-only");
                }
                None if !self.config.quiet && cfg!(target_arch = "wasm32") => {
                    eprintln!(
                        "extract: model '{}' not found locally — running BM25-only \
                         (set LITEPARSE_EXTRACT_MODEL_PATH)",
                        self.config.extract_model
                    );
                }
                _ => {}
            }
        }
        // Scalar leaves: extract, then rename to dotted output paths (the
        // engine query used the leaf name; the caller sees the full path).
        let engine_schema = ExtractionSchema {
            fields: flat.fields.iter().map(|f| f.to_schema_field()).collect(),
        };
        let mut fields = extractor.extract(&engine_schema).fields;
        for (out, leaf) in fields.iter_mut().zip(&flat.fields) {
            out.name = leaf.dotted();
        }

        // Object-array groups: row-group over detected tables (grids built once,
        // shared). Record sub-fields are renamed to record-relative dotted paths.
        let grids = table_units::table_grids(&parsed.pages);
        let arrays = flat
            .object_arrays
            .iter()
            .map(|oa| {
                let group = ObjectArraySchema {
                    fields: oa.fields.iter().map(|f| f.to_schema_field()).collect(),
                    description: None,
                    has_nested_groups: oa.has_nested_groups,
                };
                let mut records = extractor.extract_object_array(&group, &grids).records;
                for rec in &mut records {
                    for (out, leaf) in rec.fields.iter_mut().zip(&oa.fields) {
                        out.name = leaf.dotted();
                    }
                }
                ArrayFieldResult::new(oa.dotted(), records)
            })
            .collect();

        Ok(DocumentExtraction { fields, arrays })
    }

    /// Parse a document from either a file path or raw bytes.
    ///
    /// Use `PdfInput::Path` for files on disk or `PdfInput::Bytes` for
    /// in-memory PDF data (e.g. from a network response or Node.js Buffer).
    pub async fn parse_input(&self, input: PdfInput) -> Result<ParseResult, LiteParseError> {
        let log = |msg: &str| {
            if !self.config.quiet {
                eprintln!("{}", msg);
            }
        };

        let t0 = web_time::Instant::now();

        #[cfg(not(target_arch = "wasm32"))]
        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), false).await?;

        #[cfg(target_arch = "wasm32")]
        let validated_input = input;

        // Determine which pages to extract
        let target_pages = self.resolve_target_pages()?;

        // Extract text (and pre-render OCR pages in one PDF load when OCR is on).
        // The PDFium lock is acquired for this entire critical section and
        // released before any `.await` below — OCR (network / CPU) and grid
        // projection (pure Rust) do not touch PDFium, so they can run
        // concurrently with other `LiteParse` calls.
        let password = self.config.password.as_deref();
        let render_images = matches!(self.config.image_mode, crate::config::ImageMode::Embed);

        // Build the OCR engine up front so the renderer knows whether to emit a
        // grayscale buffer (cheaper, for engines that binarize internally) or RGB.
        let ocr_engine: Option<std::sync::Arc<dyn OcrEngine>> = if self.config.ocr_enabled {
            Some(if let Some(e) = self.ocr_engine_override.clone() {
                e
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(ref url) = self.config.ocr_server_url {
                        std::sync::Arc::new(
                            HttpOcrEngine::with_headers(
                                url.clone(),
                                self.config.ocr_server_headers.clone(),
                            )
                            .with_retry(
                                crate::ocr::http_simple::OcrRetryConfig {
                                    hedge_delays_ms: self.config.ocr_hedge_delays_ms.clone(),
                                    ..Default::default()
                                },
                            ),
                        )
                    } else {
                        #[cfg(feature = "tesseract")]
                        {
                            std::sync::Arc::new(TesseractOcrEngine::new(
                                self.config.tessdata_path.clone(),
                            ))
                        }
                        #[cfg(not(feature = "tesseract"))]
                        {
                            return Err("OCR enabled but no --ocr-server-url provided and tesseract feature is disabled".into());
                        }
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err(
                        "OCR enabled but no `ocrEngine` callback was provided (WASM builds have no built-in OCR engine)".into(),
                    );
                }
            })
        } else {
            None
        };
        let ocr_grayscale = ocr_engine.as_ref().is_some_and(|e| e.prefers_grayscale());

        let (pages, ocr_rendered, outline, images) = {
            let lib = Library::init();
            let document = pdf_read::load_document_from_input(&lib, &validated_input, password)?;
            let outline = pdf_read::extract_outline(&document);
            let (pages, images) = pdf_read::extract_pages_and_images(
                &document,
                target_pages.as_deref(),
                self.config.max_pages,
                render_images,
                self.config.extract_links
                    && self.config.output_format == crate::config::OutputFormat::Markdown,
                self.glyph_resolver.as_deref(),
                self.config.emit_word_boxes,
            )?;
            let t_extract = web_time::Instant::now();
            log(&format!(
                "[liteparse] extract: {:.1}ms ({} pages)",
                t_extract.duration_since(t0).as_secs_f64() * 1000.0,
                pages.len()
            ));
            let rendered = if self.config.ocr_enabled {
                let r = ocr_merge::render_pages_for_ocr(
                    &document,
                    &pages,
                    self.config.dpi,
                    ocr_grayscale,
                )?;
                log(&format!(
                    "[liteparse] ocr render: {:.1}ms ({} pages)",
                    web_time::Instant::now()
                        .duration_since(t_extract)
                        .as_secs_f64()
                        * 1000.0,
                    r.len()
                ));
                r
            } else {
                Vec::new()
            };
            // `lib` is dropped here, releasing the PDFium lock.
            (pages, rendered, outline, images)
        };
        let mut pages = pages;
        let t1 = web_time::Instant::now();

        // OCR pass (engine resolved before the render block above).
        if let Some(engine) = ocr_engine {
            ocr_merge::ocr_and_merge_rendered(
                &mut pages,
                ocr_rendered,
                self.config.dpi,
                engine,
                &self.config.ocr_language,
                self.config.num_workers,
                self.config.ocr_failure_fatal,
            )
            .await?;
        }
        let t_ocr = web_time::Instant::now();
        log(&format!(
            "[liteparse] ocr: {:.1}ms",
            t_ocr.duration_since(t1).as_secs_f64() * 1000.0
        ));

        // Caller-requested content filters (page-region crop, diagonal-text
        // removal). Runs after OCR merge so it also drops OCR text outside the
        // crop region, and before projection so filtered items never surface.
        pdf_read::apply_content_filters(
            &mut pages,
            self.config.crop_box.as_ref(),
            self.config.skip_diagonal_text,
        );

        // Grid projection
        let mut parsed_pages = projection::project_pages_to_grid(pages);
        let t2 = web_time::Instant::now();
        log(&format!(
            "[liteparse] project: {:.1}ms",
            t2.duration_since(t_ocr).as_secs_f64() * 1000.0
        ));

        let full_text = if self.config.output_format == crate::config::OutputFormat::Markdown {
            let page_md =
                markdown::format_markdown_pages(&parsed_pages, &outline, self.config.image_mode);
            let md = page_md.join("\n\n-----\n\n");
            for (page, md) in parsed_pages.iter_mut().zip(page_md) {
                page.markdown = md;
            }
            let t3 = web_time::Instant::now();
            log(&format!(
                "[liteparse] markdown: {:.1}ms",
                t3.duration_since(t2).as_secs_f64() * 1000.0
            ));
            md
        } else {
            parsed_pages
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        let total = web_time::Instant::now().duration_since(t0).as_secs_f64() * 1000.0;
        log(&format!("[liteparse] total: {:.1}ms", total));

        Ok(ParseResult {
            pages: parsed_pages,
            text: full_text,
            outline,
            images,
        })
    }

    /// Parse from pre-extracted pages, skipping PDFium text extraction.
    ///
    /// The caller supplies `Page`s already populated with text items (and,
    /// optionally, graphics / struct nodes / image refs) in viewport space
    /// (top-left origin, 72 DPI). This runs only grid projection and the
    /// configured output formatter, so it touches neither PDFium nor OCR and
    /// is fully synchronous. Used when an external extractor (e.g. with its
    /// own font-recovery pipeline) owns text extraction.
    pub fn parse_from_pages(&self, pages: Vec<Page>, outline: Vec<OutlineTarget>) -> ParseResult {
        let mut parsed_pages = projection::project_pages_to_grid(pages);

        let full_text = if self.config.output_format == crate::config::OutputFormat::Markdown {
            let page_md =
                markdown::format_markdown_pages(&parsed_pages, &outline, self.config.image_mode);
            let md = page_md.join("\n\n-----\n\n");
            for (page, md) in parsed_pages.iter_mut().zip(page_md) {
                page.markdown = md;
            }
            md
        } else {
            parsed_pages
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        ParseResult {
            pages: parsed_pages,
            text: full_text,
            outline,
            images: Vec::new(),
        }
    }

    /// Generate screenshots of document pages as PNG bytes.
    ///
    /// Non-PDF files are automatically converted to PDF first (requires
    /// LibreOffice/ImageMagick on the system). Plain-text formats cannot be
    /// rendered and return a clear error.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn screenshot(
        &self,
        input: &str,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        self.screenshot_input(PdfInput::Path(input.to_string()), page_numbers)
            .await
    }

    /// Generate screenshots from a file path or raw bytes.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn screenshot_input(
        &self,
        input: PdfInput,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        let log = |msg: &str| {
            if !self.config.quiet {
                eprintln!("{}", msg);
            }
        };

        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), true).await?;

        if let PdfInput::Path(ref path) = validated_input
            && !conversion::is_pdf(path)
        {
            log("[liteparse] converted input to PDF for screenshot rendering");
        }

        let rendered = render::render_pages_to_png(
            &validated_input,
            page_numbers.as_deref(),
            self.config.dpi,
            self.config.password.as_deref(),
        )?;

        Ok(rendered
            .into_iter()
            .map(|page| ScreenshotResult {
                page_num: page.page_num,
                width: page.width,
                height: page.height,
                image_bytes: page.png_bytes,
            })
            .collect())
    }

    pub fn config(&self) -> &LiteParseConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_new_stores_config() {
        let mut cfg = LiteParseConfig::default();
        cfg.ocr_enabled = false;
        cfg.max_pages = 7;
        let lp = LiteParse::new(cfg);
        assert!(!lp.config().ocr_enabled);
        assert_eq!(lp.config().max_pages, 7);
    }
}
