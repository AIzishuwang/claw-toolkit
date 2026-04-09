//! # session-store
//!
//! AI conversation session management with JSONL persistence, structured
//! compaction, and workspace-isolated storage.
//!
//! This crate provides the **Session Memory** layer for AI agent systems:
//!
//! - **Session** — in-memory conversation state with messages and token tracking
//! - **Structured Compaction** — intelligent summarization of old messages
//!   (scope stats, tool mentions, recent requests, key files, timeline)
//! - **Summary Compression** — secondary budget-controlled compression of summaries
//! - **SessionStore** — JSONL persistence with workspace fingerprint isolation
//! - **Session Fork** — branch conversations for parallel exploration
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
//! assert!(session.estimate_tokens() > 0);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Message model
// ---------------------------------------------------------------------------

/// The role of a conversation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
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
    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
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
            Self::ToolResult {
                content,
                tool_name,
                ..
            } => tool_name.len() + content.len(),
        };
        chars / 4 + 1
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
        self.blocks
            .iter()
            .map(ContentBlock::estimate_tokens)
            .sum::<usize>()
            + 4 // role overhead
    }
}

// ---------------------------------------------------------------------------
// Token usage
// ---------------------------------------------------------------------------

/// Token usage statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Metadata recorded after a compaction operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCompaction {
    pub count: usize,
    pub removed_message_count: usize,
    pub summary: String,
}

/// Provenance recorded when a session is forked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionFork {
    pub parent_session_id: String,
    pub branch_name: Option<String>,
}

/// A conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    id: String,
    messages: Vec<Message>,
    usage: TokenUsage,
    #[serde(default)]
    compaction: Option<SessionCompaction>,
    #[serde(default)]
    fork: Option<SessionFork>,
    #[serde(default)]
    workspace_root: Option<PathBuf>,
}

impl Session {
    /// Create a new session with a generated ID.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: generate_session_id(),
            messages: Vec::new(),
            usage: TokenUsage::default(),
            compaction: None,
            fork: None,
            workspace_root: None,
        }
    }

    /// Create a session with a specific ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
            usage: TokenUsage::default(),
            compaction: None,
            fork: None,
            workspace_root: None,
        }
    }

    /// Bind this session to a workspace root.
    #[must_use]
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
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

    /// Mutable access to messages (for compaction replacement).
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    /// Cumulative token usage.
    #[must_use]
    pub fn usage(&self) -> TokenUsage {
        self.usage
    }

    /// Compaction metadata.
    #[must_use]
    pub fn compaction(&self) -> Option<&SessionCompaction> {
        self.compaction.as_ref()
    }

    /// Number of times this session has been compacted.
    #[must_use]
    pub fn compaction_count(&self) -> usize {
        self.compaction.as_ref().map_or(0, |c| c.count)
    }

    /// Fork metadata.
    #[must_use]
    pub fn fork_info(&self) -> Option<&SessionFork> {
        self.fork.as_ref()
    }

    /// Workspace root this session is bound to.
    #[must_use]
    pub fn workspace_root(&self) -> Option<&Path> {
        self.workspace_root.as_deref()
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
            blocks: vec![ContentBlock::tool_result(
                tool_use_id,
                tool_name,
                content,
                is_error,
            )],
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

    /// Compact the session using **structured summarization**.
    ///
    /// Unlike simple truncation, this generates a rich summary containing:
    /// - Scope statistics (message counts by role)
    /// - Tool names mentioned
    /// - Recent user requests
    /// - Key files referenced
    /// - Current work inference
    /// - Key timeline of compacted messages
    ///
    /// Returns a [`CompactionResult`] with details about what was removed.
    pub fn compact(&mut self, keep_recent: usize) -> CompactionResult {
        let total = self.messages.len();

        // Skip any existing compaction system message at index 0
        let compacted_prefix = if self.has_compaction_prefix() { 1 } else { 0 };
        let compactable = &self.messages[compacted_prefix..];

        if compactable.len() <= keep_recent {
            return CompactionResult {
                removed_count: 0,
                estimated_tokens_saved: 0,
                summary: String::new(),
            };
        }

        let keep_from = total.saturating_sub(keep_recent);
        let removed = &self.messages[compacted_prefix..keep_from];
        let preserved = self.messages[keep_from..].to_vec();

        // Extract existing summary if present
        let existing_summary = if compacted_prefix > 0 {
            self.extract_existing_summary()
        } else {
            None
        };

        // Build structured summary
        let new_summary = summarize_messages(removed);
        let merged_summary = merge_summaries(existing_summary.as_deref(), &new_summary);
        let compressed = compress_summary_text(&merged_summary);

        let tokens_saved: usize = removed.iter().map(Message::estimate_tokens).sum();
        let removed_count = removed.len();

        // Build continuation message
        let continuation = build_continuation_message(&compressed, !preserved.is_empty());

        // Replace messages
        let mut new_messages = vec![Message {
            role: MessageRole::System,
            blocks: vec![ContentBlock::text(continuation)],
        }];
        new_messages.extend(preserved);
        self.messages = new_messages;

        // Record compaction metadata
        let count = self.compaction.as_ref().map_or(1, |c| c.count + 1);
        self.compaction = Some(SessionCompaction {
            count,
            removed_message_count: removed_count,
            summary: compressed.clone(),
        });

        CompactionResult {
            removed_count,
            estimated_tokens_saved: tokens_saved,
            summary: compressed,
        }
    }

    /// Fork this session into a new session with an independent message history.
    #[must_use]
    pub fn fork(&self, branch_name: Option<String>) -> Session {
        Session {
            id: generate_session_id(),
            messages: self.messages.clone(),
            usage: TokenUsage::default(),
            compaction: self.compaction.clone(),
            fork: Some(SessionFork {
                parent_session_id: self.id.clone(),
                branch_name,
            }),
            workspace_root: self.workspace_root.clone(),
        }
    }

    /// Check whether the first message is a compaction system message.
    fn has_compaction_prefix(&self) -> bool {
        self.messages.first().is_some_and(|m| {
            m.role == MessageRole::System
                && m.blocks.iter().any(|b| match b {
                    ContentBlock::Text { text } => text.contains(CONTINUATION_PREAMBLE),
                    _ => false,
                })
        })
    }

    /// Extract the summary text from an existing compaction system message.
    fn extract_existing_summary(&self) -> Option<String> {
        let first = self.messages.first()?;
        if first.role != MessageRole::System {
            return None;
        }
        let text = first.blocks.iter().find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })?;
        let rest = text.strip_prefix(CONTINUATION_PREAMBLE)?;
        let rest = rest
            .split_once(&format!("\n\n{RECENT_MESSAGES_NOTE}"))
            .map_or(rest, |(before, _)| before);
        Some(rest.trim().to_string())
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    pub removed_count: usize,
    pub estimated_tokens_saved: usize,
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Structured compaction
// ---------------------------------------------------------------------------

const CONTINUATION_PREAMBLE: &str = "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n";
const RECENT_MESSAGES_NOTE: &str = "Recent messages are preserved verbatim.";
const DIRECT_RESUME: &str = "Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, and do not preface with continuation text.";

/// Build a structured summary from a slice of messages.
fn summarize_messages(messages: &[Message]) -> String {
    let user_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .count();
    let assistant_count = messages
        .iter()
        .filter(|m| m.role == MessageRole::Assistant)
        .count();
    let tool_count = messages.len() - user_count - assistant_count;

    // Collect unique tool names
    let mut tool_names: Vec<String> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse { name, .. } => Some(name.clone()),
            ContentBlock::ToolResult { tool_name, .. } => Some(tool_name.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    tool_names.sort();

    let mut lines = vec![
        "Summary:".to_string(),
        format!(
            "- Scope: {} earlier messages compacted (user={}, assistant={}, tool={}).",
            messages.len(),
            user_count,
            assistant_count,
            tool_count
        ),
    ];

    if !tool_names.is_empty() {
        lines.push(format!("- Tools mentioned: {}.", tool_names.join(", ")));
    }

    // Recent user requests (last 3)
    let recent_requests: Vec<String> = messages
        .iter()
        .rev()
        .filter(|m| m.role == MessageRole::User)
        .filter_map(|m| first_text(m))
        .take(3)
        .map(|t| truncate(t, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !recent_requests.is_empty() {
        lines.push("- Recent user requests:".to_string());
        for req in &recent_requests {
            lines.push(format!("  - {req}"));
        }
    }

    // Pending work
    let pending: Vec<String> = messages
        .iter()
        .rev()
        .filter_map(|m| first_text(m))
        .filter(|t| {
            let lower = t.to_ascii_lowercase();
            lower.contains("todo")
                || lower.contains("next")
                || lower.contains("pending")
                || lower.contains("remaining")
        })
        .take(3)
        .map(|t| truncate(t, 160))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if !pending.is_empty() {
        lines.push("- Pending work:".to_string());
        for item in &pending {
            lines.push(format!("  - {item}"));
        }
    }

    // Key files
    let files = collect_key_files(messages);
    if !files.is_empty() {
        lines.push(format!("- Key files referenced: {}.", files.join(", ")));
    }

    // Current work
    if let Some(current) = messages
        .iter()
        .rev()
        .filter_map(|m| first_text(m))
        .find(|t| !t.trim().is_empty())
    {
        lines.push(format!("- Current work: {}", truncate(current, 200)));
    }

    // Key timeline
    lines.push("- Key timeline:".to_string());
    for msg in messages {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
        };
        let content = msg
            .blocks
            .iter()
            .map(summarize_block)
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("  - {role}: {content}"));
    }

    lines.join("\n")
}

/// Merge an existing compaction summary with a newly generated one.
fn merge_summaries(existing: Option<&str>, new_summary: &str) -> String {
    let Some(existing) = existing else {
        return new_summary.to_string();
    };

    let mut lines = vec!["Summary:".to_string()];
    lines.push("- Previously compacted context:".to_string());
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "Summary:" {
            continue;
        }
        lines.push(format!("  {trimmed}"));
    }
    lines.push("- Newly compacted context:".to_string());
    for line in new_summary.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "Summary:" {
            continue;
        }
        lines.push(format!("  {trimmed}"));
    }
    lines.join("\n")
}

fn build_continuation_message(summary: &str, has_recent: bool) -> String {
    let mut text = format!("{CONTINUATION_PREAMBLE}{summary}");
    if has_recent {
        text.push_str("\n\n");
        text.push_str(RECENT_MESSAGES_NOTE);
    }
    text.push('\n');
    text.push_str(DIRECT_RESUME);
    text
}

fn summarize_block(block: &ContentBlock) -> String {
    let raw = match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::ToolUse { name, input, .. } => format!("tool_use {name}({input})"),
        ContentBlock::ToolResult {
            tool_name,
            content,
            is_error,
            ..
        } => format!(
            "tool_result {tool_name}: {}{}",
            if *is_error { "error " } else { "" },
            content
        ),
    };
    truncate(&raw, 160)
}

fn first_text(msg: &Message) -> Option<&str> {
    msg.blocks.iter().find_map(|b| match b {
        ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.as_str()),
        _ => None,
    })
}

fn collect_key_files(messages: &[Message]) -> Vec<String> {
    let mut files: Vec<String> = messages
        .iter()
        .flat_map(|m| m.blocks.iter())
        .flat_map(|b| {
            let text = match b {
                ContentBlock::Text { text } => text.as_str(),
                ContentBlock::ToolUse { input, .. } => input.as_str(),
                ContentBlock::ToolResult { content, .. } => content.as_str(),
            };
            extract_file_paths(text)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(8)
        .collect();
    files.sort();
    files
}

fn extract_file_paths(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let candidate = token.trim_matches(|c: char| {
                matches!(c, ',' | '.' | ':' | ';' | ')' | '(' | '"' | '\'' | '`')
            });
            if candidate.contains('/')
                && Path::new(candidate)
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| {
                        ["rs", "ts", "tsx", "js", "json", "md", "py", "toml"]
                            .contains(&ext.to_ascii_lowercase().as_str())
                    })
            {
                Some(candidate.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut t: String = text.chars().take(max.saturating_sub(1)).collect();
    t.push('…');
    t
}

// ---------------------------------------------------------------------------
// Summary compression (secondary budget control)
// ---------------------------------------------------------------------------

const COMPRESS_MAX_CHARS: usize = 1_200;
const COMPRESS_MAX_LINES: usize = 24;
const COMPRESS_MAX_LINE_CHARS: usize = 160;

/// Compress a summary text to fit within a character and line budget.
///
/// - Collapses inline whitespace
/// - Removes duplicate lines (case-insensitive)
/// - Prioritizes core summary lines over timeline details
/// - Truncates individual lines to `max_line_chars`
#[must_use]
pub fn compress_summary_text(summary: &str) -> String {
    // Normalize: collapse whitespace, dedupe
    let mut seen = BTreeSet::new();
    let mut lines: Vec<String> = Vec::new();

    for raw in summary.lines() {
        let normalized: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            continue;
        }
        let truncated = truncate(&normalized, COMPRESS_MAX_LINE_CHARS);
        let key = truncated.to_ascii_lowercase();
        if seen.insert(key) {
            lines.push(truncated);
        }
    }

    if lines.is_empty() {
        return String::new();
    }

    // Select lines by priority within budget
    let mut selected = BTreeSet::<usize>::new();

    for priority in 0..=3 {
        for (i, line) in lines.iter().enumerate() {
            if selected.contains(&i) || line_priority(line) != priority {
                continue;
            }
            if selected.len() + 1 > COMPRESS_MAX_LINES {
                continue;
            }
            let total_chars: usize = selected
                .iter()
                .map(|idx| lines[*idx].chars().count())
                .sum::<usize>()
                + line.chars().count()
                + selected.len(); // newlines
            if total_chars > COMPRESS_MAX_CHARS {
                continue;
            }
            selected.insert(i);
        }
    }

    let mut result: Vec<String> = selected.iter().map(|i| lines[*i].clone()).collect();
    let omitted = lines.len().saturating_sub(result.len());
    if omitted > 0 {
        let notice = format!("- … {omitted} additional line(s) omitted.");
        let total_chars: usize = result.iter().map(|l| l.chars().count()).sum::<usize>()
            + notice.chars().count()
            + result.len();
        if result.len() < COMPRESS_MAX_LINES && total_chars <= COMPRESS_MAX_CHARS {
            result.push(notice);
        }
    }

    result.join("\n")
}

fn line_priority(line: &str) -> usize {
    if line == "Summary:" || line.starts_with("- Scope:") || line.starts_with("- Current work:") {
        0
    } else if line.starts_with("- Pending work:")
        || line.starts_with("- Key files")
        || line.starts_with("- Tools mentioned:")
        || line.starts_with("- Recent user requests:")
        || line.starts_with("- Previously compacted")
        || line.starts_with("- Newly compacted")
    {
        1
    } else if line.starts_with("- ") || line.starts_with("  - ") {
        2
    } else {
        3
    }
}

// ---------------------------------------------------------------------------
// SessionStore — JSONL persistence with workspace isolation
// ---------------------------------------------------------------------------

/// JSONL-based session persistence with optional workspace fingerprint isolation.
///
/// When created with [`SessionStore::with_workspace`], sessions are stored in
/// a subdirectory keyed by an FNV-1a hash of the workspace path, preventing
/// parallel instances from colliding.
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

    /// Create a session store isolated by workspace fingerprint.
    ///
    /// The on-disk layout becomes `<base_dir>/sessions/<workspace_hash>/`
    /// where the hash is a stable FNV-1a 64-bit digest of the workspace path.
    pub fn with_workspace(
        base_dir: impl AsRef<Path>,
        workspace_root: impl AsRef<Path>,
    ) -> Result<Self, SessionError> {
        let fingerprint = workspace_fingerprint(workspace_root.as_ref());
        let dir = base_dir
            .as_ref()
            .join("sessions")
            .join(fingerprint);
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// The resolved sessions directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Save a session to disk as JSONL.
    pub fn save(&self, session: &Session) -> Result<PathBuf, SessionError> {
        let path = self.session_path(&session.id);
        let file = fs::File::create(&path)?;
        let mut writer = std::io::BufWriter::new(file);

        // Write metadata header
        let meta = serde_json::json!({
            "_type": "session_meta",
            "id": session.id,
            "usage": session.usage,
            "compaction": session.compaction,
            "fork": session.fork,
            "workspace_root": session.workspace_root,
        });
        writeln!(writer, "{}", serde_json::to_string(&meta)?)?;

        // Write messages
        for message in &session.messages {
            let record = serde_json::json!({
                "_type": "message",
                "message": message,
            });
            writeln!(writer, "{}", serde_json::to_string(&record)?)?;
        }

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
        let mut id = session_id.to_string();
        let mut usage = TokenUsage::default();
        let mut compaction = None;
        let mut fork = None;
        let mut workspace_root = None;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(&line)?;

            match value.get("_type").and_then(|v| v.as_str()) {
                Some("session_meta") => {
                    if let Some(v) = value.get("id").and_then(|v| v.as_str()) {
                        id = v.to_string();
                    }
                    if let Some(u) = value.get("usage") {
                        usage = serde_json::from_value(u.clone()).unwrap_or_default();
                    }
                    if let Some(c) = value.get("compaction") {
                        compaction = serde_json::from_value(c.clone()).ok();
                    }
                    if let Some(f) = value.get("fork") {
                        fork = serde_json::from_value(f.clone()).ok();
                    }
                    if let Some(w) = value.get("workspace_root") {
                        workspace_root = serde_json::from_value(w.clone()).ok();
                    }
                }
                Some("message") => {
                    if let Some(m) = value.get("message") {
                        let message: Message = serde_json::from_value(m.clone())?;
                        messages.push(message);
                    }
                }
                // Legacy format: bare message object (no _type field)
                _ => {
                    if value.get("_meta").is_some() {
                        // Legacy meta line
                        if let Some(u) = value.get("usage") {
                            usage = serde_json::from_value(u.clone()).unwrap_or_default();
                        }
                    } else if let Ok(message) = serde_json::from_value::<Message>(value) {
                        messages.push(message);
                    }
                }
            }
        }

        Ok(Session {
            id,
            messages,
            usage,
            compaction,
            fork,
            workspace_root,
        })
    }

    /// List all session IDs in the store, sorted by most recent first.
    pub fn list(&self) -> Result<Vec<String>, SessionError> {
        let mut entries: Vec<(String, std::time::SystemTime)> = Vec::new();

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(id) = name.strip_suffix(".jsonl") {
                let modified = entry.metadata()?.modified().unwrap_or(std::time::UNIX_EPOCH);
                entries.push((id.to_string(), modified));
            }
        }

        entries.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(entries.into_iter().map(|(id, _)| id).collect())
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.jsonl"))
    }
}

/// Stable FNV-1a 64-bit fingerprint of a workspace path.
///
/// Produces a 16-char hex string used to partition on-disk session directories.
#[must_use]
pub fn workspace_fingerprint(workspace_root: &Path) -> String {
    let input = workspace_root.to_string_lossy();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("session_{ts:x}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn structured_compaction_generates_summary() {
        let mut session = Session::new();
        for i in 0..20 {
            session.push_user(format!("message {i}"));
            session.push_assistant(vec![ContentBlock::text(format!("reply {i}"))]);
        }
        assert_eq!(session.messages().len(), 40);

        let result = session.compact(10);
        assert_eq!(result.removed_count, 30);
        // 10 kept + 1 compaction system message
        assert_eq!(session.messages().len(), 11);
        assert_eq!(session.compaction_count(), 1);
        // Verify structured summary content
        assert!(result.summary.contains("Summary:"));
        assert!(result.summary.contains("Scope:"));
        assert!(result.summary.contains("earlier messages compacted"));
    }

    #[test]
    fn compaction_preserves_previous_context() {
        let mut session = Session::new();
        for i in 0..20 {
            session.push_user(format!("msg {i}"));
            session.push_assistant(vec![ContentBlock::text(format!("reply {i}"))]);
        }
        session.compact(6);
        assert_eq!(session.compaction_count(), 1);

        // Add more messages and compact again
        for i in 20..30 {
            session.push_user(format!("msg {i}"));
            session.push_assistant(vec![ContentBlock::text(format!("reply {i}"))]);
        }
        let result = session.compact(6);
        assert_eq!(session.compaction_count(), 2);
        assert!(result.summary.contains("Previously compacted context:"));
        assert!(result.summary.contains("Newly compacted context:"));
    }

    #[test]
    fn compaction_extracts_tool_names() {
        let mut session = Session::new();
        session.push_user("run a command");
        session.push_assistant(vec![ContentBlock::tool_use("t1", "bash", "ls -la")]);
        session.push_tool_result("t1", "bash", "file1.rs\nfile2.rs", false);
        session.push_assistant(vec![ContentBlock::text("done")]);
        // Add enough messages to trigger compaction
        for i in 0..10 {
            session.push_user(format!("more {i}"));
            session.push_assistant(vec![ContentBlock::text(format!("ok {i}"))]);
        }

        let result = session.compact(4);
        assert!(result.summary.contains("bash"));
        assert!(result.summary.contains("Tools mentioned:"));
    }

    #[test]
    fn compaction_extracts_key_files() {
        let mut session = Session::new();
        session.push_user("Update src/main.rs and tests/test.rs");
        for i in 0..10 {
            session.push_assistant(vec![ContentBlock::text(format!("ok {i}"))]);
            session.push_user(format!("more {i}"));
        }

        let result = session.compact(4);
        assert!(result.summary.contains("src/main.rs"));
        assert!(result.summary.contains("tests/test.rs"));
    }

    #[test]
    fn session_fork_records_lineage() {
        let mut session = Session::new();
        session.push_user("test");

        // Small delay to ensure timestamp-based ID differs
        std::thread::sleep(std::time::Duration::from_millis(2));

        let fork = session.fork(Some("feature-branch".to_string()));
        assert_ne!(fork.id(), session.id());
        assert_eq!(fork.messages().len(), 1);
        let info = fork.fork_info().expect("fork info should exist");
        assert_eq!(info.parent_session_id, session.id());
        assert_eq!(info.branch_name.as_deref(), Some("feature-branch"));
    }

    #[test]
    fn workspace_fingerprint_is_deterministic() {
        let path_a = Path::new("/tmp/project-alpha");
        let path_b = Path::new("/tmp/project-beta");

        let fp_a1 = workspace_fingerprint(path_a);
        let fp_a2 = workspace_fingerprint(path_a);
        let fp_b = workspace_fingerprint(path_b);

        assert_eq!(fp_a1, fp_a2);
        assert_ne!(fp_a1, fp_b);
        assert_eq!(fp_a1.len(), 16);
    }

    #[test]
    fn summary_compression_respects_budget() {
        let long_summary = (0..50)
            .map(|i| format!("  - timeline entry {i}: a long description of what happened"))
            .collect::<Vec<_>>()
            .join("\n");
        let full = format!("Summary:\n- Scope: 50 messages compacted.\n- Key timeline:\n{long_summary}");

        let compressed = compress_summary_text(&full);
        assert!(compressed.lines().count() <= COMPRESS_MAX_LINES + 1);
        assert!(compressed.contains("Summary:"));
        assert!(compressed.contains("Scope:"));
    }

    #[test]
    fn store_save_load_with_metadata() {
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
    fn store_workspace_isolation() {
        let base = tempfile::tempdir().unwrap();
        let store_a =
            SessionStore::with_workspace(base.path(), "/tmp/project-a").unwrap();
        let store_b =
            SessionStore::with_workspace(base.path(), "/tmp/project-b").unwrap();

        let session_a = Session::with_id("alpha");
        store_a.save(&session_a).unwrap();

        let session_b = Session::with_id("beta");
        store_b.save(&session_b).unwrap();

        // Each store only sees its own sessions
        assert_eq!(store_a.list().unwrap(), vec!["alpha"]);
        assert_eq!(store_b.list().unwrap(), vec!["beta"]);
        assert_ne!(store_a.dir(), store_b.dir());
    }

    #[test]
    fn store_list_sorted_by_recent() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path()).unwrap();

        for id in ["alpha", "beta", "gamma"] {
            let session = Session::with_id(id);
            store.save(&session).unwrap();
            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let ids = store.list().unwrap();
        // Most recent first
        assert_eq!(ids[0], "gamma");
    }

    #[test]
    fn token_usage_accumulation() {
        let mut usage = TokenUsage::default();
        usage.add(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
        });
        usage.add(TokenUsage {
            input_tokens: 20,
            output_tokens: 15,
        });
        assert_eq!(usage.total(), 50);
    }
}
