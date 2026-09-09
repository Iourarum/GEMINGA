//! Resolve a Hugging Face dataset into a list of remote Parquet files.
//!
//! Every dataset on the Hub is exposed in Parquet form on the `refs/convert/parquet`
//! branch (the parquet-converter bot either converts it, or links the original files
//! if they were already Parquet with sane row-group sizes). The datasets-server
//! `/parquet` endpoint lists those files together with their sizes, which lets us
//! skip a HEAD request per file and go straight to range requests on the footer.

use crate::error::{GemingaError, Result};
use crate::reader::{Source, Telemetry};
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{ClientOptions, ObjectStore};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

pub const DATASETS_SERVER: &str = "https://datasets-server.huggingface.co";
pub const HUB: &str = "https://huggingface.co";
const UA: &str = concat!("geminga/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ParquetFile {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub url: String,
    pub filename: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Deserialize)]
struct ParquetResponse {
    #[serde(default)]
    parquet_files: Vec<ParquetFile>,
    #[serde(default)]
    partial: bool,
}

fn default_headers(token: Option<&str>) -> Result<HeaderMap> {
    let mut h = HeaderMap::new();
    h.insert(USER_AGENT, HeaderValue::from_static(UA));
    if let Some(t) = token {
        let v = HeaderValue::from_str(&format!("Bearer {t}"))
            .map_err(|e| GemingaError::Invalid(format!("bad token header: {e}")))?;
        h.insert(AUTHORIZATION, v);
    }
    Ok(h)
}

/// Ask the datasets-server which Parquet files make up a dataset (optionally one config/split).
pub async fn list_parquet(
    dataset: &str,
    config: Option<&str>,
    split: Option<&str>,
    token: Option<&str>,
) -> Result<(Vec<ParquetFile>, bool)> {
    let client = reqwest::Client::builder()
        .default_headers(default_headers(token)?)
        .build()?;
    let url = format!("{DATASETS_SERVER}/parquet?dataset={}", urlencode(dataset));
    let resp = client.get(&url).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(GemingaError::Msg(format!(
            "datasets-server returned {status} for {dataset}: {body}"
        )));
    }
    let parsed: ParquetResponse = resp.json().await?;
    let files: Vec<ParquetFile> = parsed
        .parquet_files
        .into_iter()
        .filter(|f| config.map_or(true, |c| f.config == c))
        .filter(|f| split.map_or(true, |s| f.split == s))
        .collect();
    if files.is_empty() {
        return Err(GemingaError::Invalid(format!(
            "no parquet files for dataset={dataset} config={config:?} split={split:?} \
             (the dataset viewer may be disabled, or the config/split name is wrong)"
        )));
    }
    Ok((files, parsed.partial))
}

/// Build one `HttpStore` per URL origin so many files on the same host share a client.
pub struct StoreCache {
    token: Option<String>,
    stores: HashMap<String, Arc<dyn ObjectStore>>,
    pub tele: Arc<Telemetry>,
}

impl StoreCache {
    pub fn new(token: Option<String>) -> Self {
        Self { token, stores: HashMap::new(), tele: Arc::new(Telemetry::default()) }
    }

    /// Turn an `https://host/some/path.parquet` URL into (store rooted at origin, object path).
    pub fn source_for_url(&mut self, url: &str, size: Option<u64>, label: String) -> Result<Source> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|e| GemingaError::Invalid(format!("bad url {url}: {e}")))?;
        let origin = format!(
            "{}://{}{}",
            parsed.scheme(),
            parsed.host_str().unwrap_or_default(),
            parsed.port().map(|p| format!(":{p}")).unwrap_or_default()
        );
        let store = match self.stores.get(&origin) {
            Some(s) => s.clone(),
            None => {
                // Only send the token to the Hub itself; presigned CDN redirects must not get it
                // (reqwest already drops Authorization on cross-host redirects, this is belt and braces).
                let token = if origin.starts_with(HUB) { self.token.as_deref() } else { None };
                let opts = ClientOptions::new()
                    .with_default_headers(default_headers(token)?)
                    .with_allow_http(parsed.scheme() == "http");
                let store: Arc<dyn ObjectStore> = Arc::new(
                    HttpBuilder::new().with_url(&origin).with_client_options(opts).build()?,
                );
                self.stores.insert(origin.clone(), store.clone());
                store
            }
        };
        // from_url_path percent-decodes, so `refs%2Fconvert%2Fparquet` becomes the segments
        // refs/convert/parquet, which the Hub accepts as a plain path.
        let path = Path::from_url_path(parsed.path().trim_start_matches('/'))
            .map_err(|e| GemingaError::Invalid(format!("bad url path {url}: {e}")))?;
        Ok(Source { store, path, size, label, tele: self.tele.clone(), discovered_size: Arc::new(AtomicU64::new(0)) })
    }

    pub fn source_for_local(&self, path: &str) -> Result<Source> {
        let abs = std::fs::canonicalize(path)?;
        let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new());
        let size = std::fs::metadata(&abs)?.len();
        let opath = Path::from_filesystem_path(&abs)
            .map_err(|e| GemingaError::Invalid(format!("bad local path {}: {e}", abs.display())))?;
        Ok(Source { store, path: opath, size: Some(size), label: abs.display().to_string(), tele: self.tele.clone(), discovered_size: Arc::new(AtomicU64::new(0)) })
    }
}

/// Direct `resolve` URL on the Parquet branch, for when you already know the layout.
pub fn resolve_url(repo: &str, config: &str, split: &str, filename: &str) -> String {
    format!("{HUB}/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{config}/{split}/{filename}")
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
