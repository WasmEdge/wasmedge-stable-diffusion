use thiserror::Error;

/// Error types for the Llama Core library.
#[derive(Error, Debug)]
pub enum SDError {
    /// Errors in General operation.
    #[error("{0}")]
    Operation(String),
}
