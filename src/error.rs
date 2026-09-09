use pyo3::exceptions::{PyIOError, PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GemingaError {
    #[error("object store: {0}")]
    ObjectStore(#[from] object_store::Error),
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Budget(String),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, GemingaError>;

impl From<GemingaError> for PyErr {
    fn from(e: GemingaError) -> PyErr {
        match e {
            GemingaError::Invalid(m) => PyValueError::new_err(m),
            GemingaError::Budget(m) => PyMemoryError::new_err(m),
            GemingaError::Io(io) => PyIOError::new_err(io.to_string()),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}
