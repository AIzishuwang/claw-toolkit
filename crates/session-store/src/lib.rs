//! # session-store
//!
//! AI conversation session management with JSONL persistence and compaction.
//!
//! Provides a `Session` data structure for tracking multi-turn conversations
//! between users and AI assistants, including tool use. Sessions can be:
//!
//! - Persisted to disk as JSONL files
//! - Loaded and resumed
//! - Compacted to reduce token usage
//! - Forked into branches
//!
//! # Example
//!
//! ```
//! use session_store::{Session, ContentBlock, MessageRole};
//!
//! let mut session = Session::new();
//! session.push_user("Fix the bug in main.rs");
//! session.push_assistant(vec![ContentBlock::text("I'll look at the file...")]);
//!
//! assert_eq!(session.messages().len(), 2);
//! assert_eq!(session.estimate_tokens(), 20); // rough estimate
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

/// Errors from session operations.
#[derive(Debug, Error)]
pub enum SessionError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("session not found: {0}")]
    NotFound(String),
    #[error("invalid message sequence: {0}")]
    InvalidSequence(String),
}

/// The role of a conversation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

/// A content block within a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        tool_name: String,
        content: String,
        is_error: bool,
    },
}

impl ContentBlock {
    /// Create a text content block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a tool use content block.
    #[must_use]
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: impl Into<String>) -> Self {
        Self::ToolUse {
            id: id.into(),
            name: name.into(),
            input: input.into(),
        }
    }

    /// Create a tool result content block.
    #[must_use]
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            tool_name: tool_name.into(),
            content: content.into(),
            is_error,
        }
    }

    /// Estimate token count for this block (rough: 1 token ≈ 4 chars).
    #[must_use]
    pub fn estimate_tokens(&self) -> usize {
        let chars = match self {
            Self::Text { text } => text.len(),
            Self::ToolUse { name, input, .. } => name.len() + input.len(),
            Self::ToolResult { content, .. } => content.len(),
        };
        (chars + 3) / 4 // round up
    }
}

/// A single message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub blocks: Vec<ContentBlock>,
}

impl Message {
    /// Estimate token count for this message.
    #[must_use]
    pub fn estimate_tokens(&self) -> usize {
        self.blocks.iter().map(ContentBlock::estimate_tokens).sum::<usize>() + 4 // role overhead
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl TokenUsage {
    /// Total tokens used.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Accumulate another usage record.
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

/// A conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    id: String,
    messages: Vec<Message>,
    usage: TokenUsage,
    #[serde(default)]
    compaction_count: usize,
}

impl Session {
    /// Create a new session with a generated ID.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: generate_session_id(),
            messages: Vec::new(),
            usage: TokenUsage::default(),
            compaction_count: 0,
        }
    }

    /// Create a session with a specific ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
            usage: TokenUsage::default(),
            compaction_count: 0,
        }
    }

    /// Session ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// All messages in the session.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Cumulative token usage.
    #[must_use]
    pub fn usage(&self) -> TokenUsage {
        self.usage
    }

    /// Number of times this session has been compacted.
    #[must_use]
    pub fn compaction_count(&self) -> usize {
        self.compaction_count
    }

    /// Push a user text message.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(Message {
            role: MessageRole::User,
            blocks: vec![ContentBlock::text(text)],
        });
    }

    /// Push an assistant message with content blocks.
    pub fn push_assistant(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message {
            role: MessageRole::Assistant,
            blocks,
        });
    }

    /// Push a tool result message (user role with tool_result blocks).
    pub fn push_tool_result(
        &mut self,
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) {
        self.messages.push(Message {
            role: MessageRole::User,
            blocks: vec![ContentBlock::tool_result(tool_use_id, tool_name, content, is_error)],
        });
    }

    /// Record token usage from an API response.
    pub fn record_usage(&mut self, usage: TokenUsage) {
        self.usage.add(usage);
    }

    /// Estimate total tokens across all messages.
    #[must_use]
    pub fn estimate_tokens(&self) -> usize {
        self.messages.iter().map(Message::estimate_tokens).sum()
    }

    /// Compact the session by removing old messages, keeping system context
    /// and recent messages.
    ///
    /// Returns the number of messages removed.
    pub fn compact(&mut self, keep_recent: usize) -> CompactionResult {
        let total = self.messages.len();
        if total <= keep_recent {
            return CompactionResult {
                removed_count: 0,
                estimated_tokens_saved: 0,
            };
        }

        let remove_count = total - keep_recent;
        let removed: Vec<Message> = self.messages.drain(..remove_count).collect();
        let tokens_saved: usize = removed.iter().map(Message::estimate_tokens).sum();

        // Insert a compaction marker as the first message
        let marker = format!(
            "[Session compacted: {remove_count} older messages removed to reduce context size]"
        );
        self.messages.insert(
            0,
            Message {
                role: MessageRole::User,
                blocks: vec![ContentBlock::text(marker)],
            },
        );

        self.compaction_count += 1;

        CompactionResult {
            removed_count: remove_count,
            estimated_tokens_saved: tokens_saved,
        }
    }

    /// Fork this session into a new session with an independent message history.
    #[must_use]
    pub fn fork(&self, new_id: Option<String>) -> Session {
        Session {
            id: new_id.unwrap_or_else(generate_session_id),
            messages: self.messages.clone(),
            usage: TokenUsage::default(),
            compaction_count: 0,
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub removed_count: usize,
    pub estimated_tokens_saved: usize,
}

/// JSONL-based session persistence.
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    /// Create a session store at the given directory.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self, SessionError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Save a session to disk as JSONL.
    pub fn save(&self, session: &Session) -> Result<PathBuf, SessionError> {
        let path = self.session_path(&session.id);
        let file = fs::File::create(&path)?;
        let mut writer = std::io::BufWriter::new(file);

        for message in &session.messages {
            let line = serde_json::to_string(message)?;
            writeln!(writer, "{line}")?;
        }

        // Write metadata as last line
        let meta = serde_json::json!({
            "_meta": true,
            "id": session.id,
            "usage": session.usage,
            "compaction_count": session.compaction_count,
        });
        writeln!(writer, "{}", serde_json::to_string(&meta)?)?;
        writer.flush()?;

        Ok(path)
    }

    /// Load a session from disk.
    pub fn load(&self, session_id: &str) -> Result<Session, SessionError> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }

        let file = fs::File::open(&path)?;
        let reader = std::io::BufReader::new(file);

        let mut messages = Vec::new();
        let mut usage = TokenUsage::default();
        let mut compaction_count = 0;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(&line)?;
            if value.get("_meta").is_some() {
                if let Some(u) = value.get("usage") {
                    usage = serde_json::from_value(u.clone())?;
                }
                if let Some(c) = value.get("compaction_count") {
                    compaction_count = c.as_u64().unwrap_or(0) as usize;
                }
            } else {
                let message: Message = serde_json::from_value(value)?;
                messages.push(message);
            }
        }

        Ok(Session {
            id: session_id.to_string(),
            messages,
            usage,
            compaction_count,
        })
    }

    /// List all session IDs in the store.
    pub fn list(&self) -> Result<Vec<String>, SessionError> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(id) = name.strip_suffix(".jsonl") {
                ids.push(id.to_string());
            }
        }
        ids.sort();
        Ok(ids)
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.jsonl"))
    }
}

fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("session_{ts:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_push_and_estimate() {
        let mut session = Session::new();
        session.push_user("hello world");
        session.push_assistant(vec![ContentBlock::text("hi there")]);

        assert_eq!(session.messages().len(), 2);
        assert!(session.estimate_tokens() > 0);
    }

    #[test]
    fn session_compaction() {
        let mut session = Session::new();
        for i in 0..20 {
            session.push_user(format!("message {i}"));
            session.push_assistant(vec![ContentBlock::text(format!("reply {i}"))]);
        }

        assert_eq!(session.messages().len(), 40);

        let result = session.compact(10);
        assert_eq!(result.removed_count, 30);
        // 10 kept + 1 compaction marker
        assert_eq!(session.messages().len(), 11);
        assert_eq!(session.compaction_count(), 1);
    }

    #[test]
    fn session_fork() {
        let mut session = Session::new();
        session.push_user("test");

        let fork = session.fork(Some("fork-1".to_string()));
        assert_eq!(fork.id(), "fork-1");
        assert_eq!(fork.messages().len(), 1);
    }

    #[test]
    fn store_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path()).unwrap();

        let mut session = Session::with_id("test-session");
        session.push_user("save me");
        session.push_assistant(vec![ContentBlock::text("saved")]);
        session.record_usage(TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
        });

        store.save(&session).unwrap();

        let loaded = store.load("test-session").unwrap();
        assert_eq!(loaded.id(), "test-session");
        assert_eq!(loaded.messages().len(), 2);
        assert_eq!(loaded.usage().input_tokens, 100);
    }

    #[test]
    fn store_list() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path()).unwrap();

        for id in ["alpha", "beta", "gamma"] {
            let session = Session::with_id(id);
            store.save(&session).unwrap();
        }

        let mut ids = store.list().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn token_usage_accumulation() {
        let mut usage = TokenUsage::default();
        usage.add(TokenUsage { input_tokens: 10, output_tokens: 5 });
        usage.add(TokenUsage { input_tokens: 20, output_tokens: 15 });
        assert_eq!(usage.total(), 50);
    }
}
