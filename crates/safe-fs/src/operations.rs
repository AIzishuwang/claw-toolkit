use std::fs;
use std::path::Path;

use crate::validators::SafeFs;
use crate::SafeFsError;

/// Result of an edit operation.
#[derive(Debug, Clone)]
pub struct EditResult {
    /// Number of replacements made.
    pub replacements: usize,
    /// New file content after edits.
    pub new_content: String,
}

/// A file matching a glob pattern.
#[derive(Debug, Clone)]
pub struct GlobMatch {
    /// Path relative to the workspace root.
    pub relative_path: String,
    /// Whether this entry is a directory.
    pub is_dir: bool,
}

/// A line matching a grep pattern.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    /// Path relative to the workspace root.
    pub relative_path: String,
    /// 1-indexed line number.
    pub line_number: usize,
    /// Content of the matching line.
    pub line_content: String,
}

impl SafeFs {
    /// Read a text file within the workspace.
    ///
    /// Validates path containment, file size, and binary content.
    pub fn read(&self, path: &str) -> Result<String, SafeFsError> {
        let resolved = self.resolve_path(path)?;

        if !resolved.exists() {
            return Err(SafeFsError::NotFound(path.to_string()));
        }

        let metadata = fs::metadata(&resolved)?;
        self.check_read_size(metadata.len())?;

        let content = fs::read(&resolved)?;
        self.check_binary(path, &content)?;

        String::from_utf8(content).map_err(|_| SafeFsError::BinaryContent {
            path: path.to_string(),
        })
    }

    /// Read a portion of a text file (offset and limit are 0-indexed line
    /// numbers).
    pub fn read_lines(
        &self,
        path: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, SafeFsError> {
        let content = self.read(path)?;
        let lines: Vec<&str> = content.lines().skip(offset).take(limit).collect();
        Ok(lines.join("\n"))
    }

    /// Write content to a file within the workspace.
    ///
    /// Creates parent directories if needed. Validates path containment,
    /// content size, and binary content.
    pub fn write(&self, path: &str, content: &str) -> Result<(), SafeFsError> {
        let resolved = self.resolve_path(path)?;
        self.check_write_size(content.len())?;
        self.check_binary(path, content.as_bytes())?;

        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&resolved, content)?;
        Ok(())
    }

    /// Replace occurrences of `old` with `new` in a file.
    ///
    /// With `replace_all = false`, exactly one occurrence must exist.
    pub fn edit(
        &self,
        path: &str,
        old: &str,
        new: &str,
        replace_all: bool,
    ) -> Result<EditResult, SafeFsError> {
        let content = self.read(path)?;

        let count = content.matches(old).count();
        if count == 0 {
            return Err(SafeFsError::EditTargetNotFound);
        }
        if !replace_all && count > 1 {
            return Err(SafeFsError::EditMultipleMatches);
        }

        let new_content = content.replace(old, new);
        self.write(path, &new_content)?;

        Ok(EditResult {
            replacements: count,
            new_content,
        })
    }

    /// Find files matching a glob pattern within the workspace.
    pub fn glob(&self, pattern: &str) -> Result<Vec<GlobMatch>, SafeFsError> {
        let full_pattern = self.root().join(pattern);
        let _pattern_str = full_pattern
            .to_str()
            .ok_or_else(|| SafeFsError::GlobPattern("invalid UTF-8 in pattern".to_string()))?;

        let mut matches = Vec::new();

        // Simple recursive directory walk with pattern matching
        self.walk_glob(&self.root().to_path_buf(), pattern, &mut matches)?;

        Ok(matches)
    }

    /// Search file contents for lines matching a pattern (simple substring
    /// match).
    pub fn grep(
        &self,
        pattern: &str,
        search_path: Option<&str>,
        case_insensitive: bool,
    ) -> Result<Vec<GrepMatch>, SafeFsError> {
        let start_path = match search_path {
            Some(p) => self.resolve_path(p)?,
            None => self.root().to_path_buf(),
        };

        let mut matches = Vec::new();
        let search_pattern = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };

        self.walk_grep(&start_path, &search_pattern, case_insensitive, &mut matches)?;

        Ok(matches)
    }

    fn walk_glob(
        &self,
        dir: &Path,
        pattern: &str,
        matches: &mut Vec<GlobMatch>,
    ) -> Result<(), SafeFsError> {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return Ok(()),
        };

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let relative = path
                .strip_prefix(self.root())
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            let is_dir = path.is_dir();

            // Simple pattern matching: support * and **
            if simple_glob_match(pattern, &relative) {
                matches.push(GlobMatch {
                    relative_path: relative,
                    is_dir,
                });
            }

            if is_dir {
                self.walk_glob(&path, pattern, matches)?;
            }
        }

        Ok(())
    }

    fn walk_grep(
        &self,
        path: &Path,
        pattern: &str,
        case_insensitive: bool,
        matches: &mut Vec<GrepMatch>,
    ) -> Result<(), SafeFsError> {
        if path.is_file() {
            if let Ok(content) = fs::read_to_string(path) {
                let relative = path
                    .strip_prefix(self.root())
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();

                for (i, line) in content.lines().enumerate() {
                    let haystack = if case_insensitive {
                        line.to_lowercase()
                    } else {
                        line.to_string()
                    };

                    if haystack.contains(pattern) {
                        matches.push(GrepMatch {
                            relative_path: relative.clone(),
                            line_number: i + 1,
                            line_content: line.to_string(),
                        });
                    }
                }
            }
        } else if path.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    self.walk_grep(&entry.path(), pattern, case_insensitive, matches)?;
                }
            }
        }

        Ok(())
    }
}

/// Simple glob matching supporting `*` (any within a segment) and `**`
/// (recursive).
fn simple_glob_match(pattern: &str, path: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();
    glob_match_recursive(&pattern_parts, &path_parts)
}

fn glob_match_recursive(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    if pattern[0] == "**" {
        // ** matches zero or more path segments
        for i in 0..=path.len() {
            if glob_match_recursive(&pattern[1..], &path[i..]) {
                return true;
            }
        }
        return false;
    }
    if path.is_empty() {
        return false;
    }
    if segment_matches(pattern[0], path[0]) {
        return glob_match_recursive(&pattern[1..], &path[1..]);
    }
    false
}

fn segment_matches(pattern: &str, segment: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    // Simple wildcard: *.ext
    if let Some(ext) = pattern.strip_prefix("*.") {
        return segment.ends_with(&format!(".{ext}"));
    }
    pattern == segment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("hello.txt"), "world").unwrap();

        let sfs = SafeFs::new(dir.path()).unwrap();
        let content = sfs.read("hello.txt").unwrap();
        assert_eq!(content, "world");
    }

    #[test]
    fn write_creates_nested_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let sfs = SafeFs::new(dir.path()).unwrap();

        sfs.write("a/b/c/deep.txt", "hello").unwrap();
        assert_eq!(fs::read_to_string(dir.path().join("a/b/c/deep.txt")).unwrap(), "hello");
    }

    #[test]
    fn edit_replaces_text() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("code.rs"), "fn old() {}").unwrap();

        let sfs = SafeFs::new(dir.path()).unwrap();
        let result = sfs.edit("code.rs", "old", "new", false).unwrap();
        assert_eq!(result.replacements, 1);
        assert_eq!(result.new_content, "fn new() {}");
    }

    #[test]
    fn edit_rejects_ambiguous_match() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("code.rs"), "old old").unwrap();

        let sfs = SafeFs::new(dir.path()).unwrap();
        let result = sfs.edit("code.rs", "old", "new", false);
        assert!(matches!(result, Err(SafeFsError::EditMultipleMatches)));
    }

    #[test]
    fn grep_finds_matches() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "hello world\nfoo bar\nhello again").unwrap();

        let sfs = SafeFs::new(dir.path()).unwrap();
        let matches = sfs.grep("hello", None, false).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].line_number, 1);
        assert_eq!(matches[1].line_number, 3);
    }

    #[test]
    fn grep_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), "Hello World\nHELLO").unwrap();

        let sfs = SafeFs::new(dir.path()).unwrap();
        let matches = sfs.grep("hello", None, true).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn glob_pattern_matching() {
        assert!(simple_glob_match("*.rs", "main.rs"));
        assert!(!simple_glob_match("*.rs", "main.txt"));
        assert!(simple_glob_match("**/*.rs", "src/main.rs"));
        assert!(simple_glob_match("**/*.rs", "src/deep/nested/file.rs"));
    }
}
