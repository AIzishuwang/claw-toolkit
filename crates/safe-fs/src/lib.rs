//! # safe-fs
//!
//! Defensive file system operations for AI agent workspaces.
//!
//! Provides a sandboxed `SafeFs` that ensures all file operations stay within
//! a designated workspace root. Includes protections against:
//!
//! - **Path traversal** (`../` escapes)
//! - **Symlink escapes** (following symlinks outside the workspace)
//! - **Binary file writes** (NUL-byte detection)
//! - **Oversized reads/writes** (configurable limits)
//!
//! # Example
//!
//! ```no_run
//! use safe_fs::SafeFs;
//!
//! let fs = SafeFs::new("/workspace").unwrap();
//! // fs.read("../../etc/passwd") → Err(PathTraversal)
//! // fs.write("data.bin", b"\x00binary") → Err(BinaryContent)
//! ```

mod operations;
mod validators;

pub use operations::{EditResult, GlobMatch, GrepMatch};
pub use validators::SafeFs;

use thiserror::Error;

/// Errors returned by safe file operations.
#[derive(Debug, Error)]
pub enum SafeFsError {
    #[error("path traversal detected: {path:?} resolves outside workspace {workspace:?}")]
    PathTraversal { path: String, workspace: String },

    #[error("symlink escape: {path:?} points to {target:?} outside workspace")]
    SymlinkEscape { path: String, target: String },

    #[error("binary content detected in {path:?}")]
    BinaryContent { path: String },

    #[error("file too large: {size} bytes exceeds limit of {limit} bytes")]
    FileTooLarge { size: u64, limit: u64 },

    #[error("content too large: {size} bytes exceeds write limit of {limit} bytes")]
    ContentTooLarge { size: usize, limit: usize },

    #[error("file not found: {0}")]
    NotFound(String),

    #[error("target string not found in file for edit")]
    EditTargetNotFound,

    #[error("multiple occurrences of target string found (use replace_all=true)")]
    EditMultipleMatches,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("glob pattern error: {0}")]
    GlobPattern(String),
}

/// Configuration for `SafeFs` limits.
#[derive(Debug, Clone)]
pub struct SafeFsConfig {
    /// Maximum file size in bytes for read operations. Default: 10 MB.
    pub max_read_size: u64,
    /// Maximum content size in bytes for write operations. Default: 5 MB.
    pub max_write_size: usize,
    /// Whether to reject writes containing NUL bytes. Default: true.
    pub deny_binary: bool,
    /// Whether to check symlinks for workspace escapes. Default: true.
    pub deny_symlink_escape: bool,
}

impl Default for SafeFsConfig {
    fn default() -> Self {
        Self {
            max_read_size: 10 * 1024 * 1024,
            max_write_size: 5 * 1024 * 1024,
            deny_binary: true,
            deny_symlink_escape: true,
        }
    }
}
