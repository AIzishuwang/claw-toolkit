use std::path::{Path, PathBuf};

use crate::{SafeFsConfig, SafeFsError};

/// A sandboxed file system rooted at a specific workspace directory.
///
/// All paths passed to operations are resolved relative to the workspace root.
/// Any attempt to escape the root (via `../`, absolute paths, or symlinks) is
/// rejected with a [`SafeFsError`].
#[derive(Debug, Clone)]
pub struct SafeFs {
    root: PathBuf,
    config: SafeFsConfig,
}

impl SafeFs {
    /// Create a new `SafeFs` rooted at the given workspace path.
    ///
    /// The path must exist and must be a directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, SafeFsError> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(SafeFsError::NotFound(root.display().to_string()));
        }
        Ok(Self {
            root,
            config: SafeFsConfig::default(),
        })
    }

    /// Create a `SafeFs` with custom configuration.
    pub fn with_config(root: impl AsRef<Path>, config: SafeFsConfig) -> Result<Self, SafeFsError> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(SafeFsError::NotFound(root.display().to_string()));
        }
        Ok(Self { root, config })
    }

    /// Returns the workspace root path.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &SafeFsConfig {
        &self.config
    }

    /// Resolve and validate a user-provided path, ensuring it stays within the
    /// workspace.
    ///
    /// - Relative paths are resolved against the workspace root.
    /// - Absolute paths are checked for workspace containment.
    /// - Symlinks are followed and checked if `deny_symlink_escape` is enabled.
    pub fn resolve_path(&self, user_path: &str) -> Result<PathBuf, SafeFsError> {
        let candidate = if Path::new(user_path).is_absolute() {
            PathBuf::from(user_path)
        } else {
            self.root.join(user_path)
        };

        // For existing paths, canonicalize to resolve symlinks and ..
        let resolved = if candidate.exists() {
            candidate.canonicalize()?
        } else {
            // For non-existing paths (write targets), normalize manually
            normalize_path(&candidate)
        };

        // Check workspace containment
        if !resolved.starts_with(&self.root) {
            return Err(SafeFsError::PathTraversal {
                path: user_path.to_string(),
                workspace: self.root.display().to_string(),
            });
        }

        // Additional symlink escape check for existing paths
        if self.config.deny_symlink_escape && candidate.exists() {
            let metadata = std::fs::symlink_metadata(&candidate)?;
            if metadata.is_symlink() {
                let target = std::fs::read_link(&candidate)?;
                let abs_target = if target.is_absolute() {
                    target.clone()
                } else {
                    candidate
                        .parent()
                        .unwrap_or(Path::new("/"))
                        .join(&target)
                        .canonicalize()?
                };
                if !abs_target.starts_with(&self.root) {
                    return Err(SafeFsError::SymlinkEscape {
                        path: user_path.to_string(),
                        target: abs_target.display().to_string(),
                    });
                }
            }
        }

        Ok(resolved)
    }

    /// Check if content contains NUL bytes (binary indicator).
    pub fn check_binary(&self, path: &str, content: &[u8]) -> Result<(), SafeFsError> {
        if self.config.deny_binary && content.contains(&0) {
            return Err(SafeFsError::BinaryContent {
                path: path.to_string(),
            });
        }
        Ok(())
    }

    /// Check file size against read limit.
    pub fn check_read_size(&self, size: u64) -> Result<(), SafeFsError> {
        if size > self.config.max_read_size {
            return Err(SafeFsError::FileTooLarge {
                size,
                limit: self.config.max_read_size,
            });
        }
        Ok(())
    }

    /// Check content size against write limit.
    pub fn check_write_size(&self, size: usize) -> Result<(), SafeFsError> {
        if size > self.config.max_write_size {
            return Err(SafeFsError::ContentTooLarge {
                size,
                limit: self.config.max_write_size,
            });
        }
        Ok(())
    }
}

/// Normalize a path by resolving `.` and `..` components without hitting the
/// filesystem (for paths that don't exist yet).
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path();
        fs::write(workspace.join("test.txt"), "hello").unwrap();

        let sfs = SafeFs::new(workspace).unwrap();
        let resolved = sfs.resolve_path("test.txt").unwrap();
        assert!(resolved.starts_with(workspace.canonicalize().unwrap()));
    }

    #[test]
    fn reject_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let sfs = SafeFs::new(dir.path()).unwrap();
        let result = sfs.resolve_path("../../etc/passwd");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SafeFsError::PathTraversal { .. }));
    }

    #[test]
    fn reject_absolute_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sfs = SafeFs::new(dir.path()).unwrap();
        let result = sfs.resolve_path("/etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn detect_binary_content() {
        let dir = tempfile::tempdir().unwrap();
        let sfs = SafeFs::new(dir.path()).unwrap();
        let result = sfs.check_binary("test.bin", b"hello\x00world");
        assert!(matches!(result, Err(SafeFsError::BinaryContent { .. })));
    }

    #[test]
    fn enforce_read_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let sfs = SafeFs::new(dir.path()).unwrap();
        let result = sfs.check_read_size(20 * 1024 * 1024);
        assert!(matches!(result, Err(SafeFsError::FileTooLarge { .. })));
    }
}
