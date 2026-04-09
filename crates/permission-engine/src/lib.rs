//! # permission-engine
//!
//! Three-level permission model for AI agent tool execution.
//!
//! Inspired by the `PermissionEnforcer` + `PermissionPolicy` pattern from
//! Claude Code's architecture, this crate provides:
//!
//! - **ReadOnly / WorkspaceWrite / FullAccess** permission levels
//! - Rule-based evaluation with custom deny rules
//! - Interactive prompter trait for user approval
//! - Built-in rules for common scenarios (path boundaries, bash patterns)
//!
//! # Example
//!
//! ```
//! use permission_engine::{PermissionEngine, PermissionLevel, PermissionResult, Rule};
//!
//! let engine = PermissionEngine::new(PermissionLevel::WorkspaceWrite)
//!     .with_rule(Rule::deny_path_outside("/workspace"))
//!     .with_rule(Rule::deny_bash_pattern("rm -rf"));
//!
//! let result = engine.check("write_file", r#"{"path":"/etc/passwd"}"#);
//! assert!(!result.allowed);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The three permission levels, ordered from least to most permissive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PermissionLevel {
    /// Can only read files, search, fetch URLs.
    ReadOnly,
    /// Can also write/edit files within the workspace boundary.
    WorkspaceWrite,
    /// Full access including shell execution, sub-agents, external mutations.
    FullAccess,
}

impl std::fmt::Display for PermissionLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "read-only"),
            Self::WorkspaceWrite => write!(f, "workspace-write"),
            Self::FullAccess => write!(f, "full-access"),
        }
    }
}

/// The result of a permission check.
#[derive(Debug, Clone)]
pub struct PermissionResult {
    /// Whether the operation is allowed.
    pub allowed: bool,
    /// Reason for denial, if denied.
    pub reason: Option<String>,
    /// Whether user approval was requested and granted.
    pub user_approved: bool,
}

impl PermissionResult {
    fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
            user_approved: false,
        }
    }

    fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.into()),
            user_approved: false,
        }
    }
}

/// A rule that can deny a tool invocation.
pub struct Rule {
    name: String,
    evaluator: Box<dyn Fn(&str, &str) -> Option<String> + Send + Sync>,
}

impl Rule {
    /// Create a custom rule with a name and evaluator function.
    ///
    /// The evaluator receives `(tool_name, input_json)` and returns
    /// `Some(reason)` to deny or `None` to allow.
    pub fn custom(
        name: impl Into<String>,
        evaluator: impl Fn(&str, &str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            evaluator: Box::new(evaluator),
        }
    }

    /// Deny file operations targeting paths outside the given root.
    pub fn deny_path_outside(root: &str) -> Self {
        let root = root.to_string();
        Self::custom("path_boundary", move |tool_name, input| {
            if !matches!(tool_name, "write_file" | "edit_file" | "read_file") {
                return None;
            }
            if let Ok(v) = serde_json::from_str::<Value>(input) {
                if let Some(path) = v.get("path").and_then(Value::as_str) {
                    if path.starts_with('/') && !path.starts_with(root.as_str()) {
                        return Some(format!(
                            "path {path:?} is outside workspace boundary {root:?}"
                        ));
                    }
                    if path.contains("..") {
                        return Some(format!("path {path:?} contains traversal sequence"));
                    }
                }
            }
            None
        })
    }

    /// Deny bash commands matching a substring pattern.
    pub fn deny_bash_pattern(pattern: &str) -> Self {
        let pattern = pattern.to_string();
        Self::custom(format!("deny_bash({pattern})"), move |tool_name, input| {
            if tool_name != "bash" {
                return None;
            }
            if let Ok(v) = serde_json::from_str::<Value>(input) {
                if let Some(cmd) = v.get("command").and_then(Value::as_str) {
                    if cmd.contains(pattern.as_str()) {
                        return Some(format!(
                            "bash command contains blocked pattern: {pattern:?}"
                        ));
                    }
                }
            }
            None
        })
    }

    /// Deny bash commands in read-only mode (any write-capable command).
    pub fn deny_bash_in_readonly() -> Self {
        Self::custom("readonly_bash", |tool_name, _input| {
            if tool_name == "bash" {
                Some("bash is not allowed in read-only mode".to_string())
            } else {
                None
            }
        })
    }

    /// Only allow specific tools (allowlist).
    pub fn allowlist(tools: Vec<String>) -> Self {
        Self::custom("allowlist", move |tool_name, _input| {
            if tools.iter().any(|t| t == tool_name) {
                None
            } else {
                Some(format!("tool {tool_name:?} is not in the allowed list"))
            }
        })
    }
}

impl std::fmt::Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rule").field("name", &self.name).finish()
    }
}

/// Trait for interactive user approval prompts.
pub trait PermissionPrompter: Send + Sync {
    /// Ask the user whether to allow a tool invocation.
    ///
    /// Returns `true` if approved, `false` if denied.
    fn prompt(&self, tool_name: &str, input: &str, description: &str) -> bool;
}

/// A prompter that always approves (for full-access / unattended mode).
pub struct AutoApprovePrompter;

impl PermissionPrompter for AutoApprovePrompter {
    fn prompt(&self, _tool_name: &str, _input: &str, _description: &str) -> bool {
        true
    }
}

/// A prompter that always denies (for strict lockdown mode).
pub struct AutoDenyPrompter;

impl PermissionPrompter for AutoDenyPrompter {
    fn prompt(&self, _tool_name: &str, _input: &str, _description: &str) -> bool {
        false
    }
}

/// Tool permission requirement declaration.
#[derive(Debug, Clone)]
pub struct ToolPermission {
    /// Tool name.
    pub tool_name: String,
    /// Minimum permission level required.
    pub required_level: PermissionLevel,
}

/// The permission engine that evaluates tool invocations.
pub struct PermissionEngine {
    /// Current session permission level.
    level: PermissionLevel,
    /// Custom deny rules.
    rules: Vec<Rule>,
    /// Tool-level permission requirements.
    tool_permissions: Vec<ToolPermission>,
    /// Optional prompter for interactive approval.
    prompter: Option<Box<dyn PermissionPrompter>>,
}

impl PermissionEngine {
    /// Create a new engine with the given permission level.
    #[must_use]
    pub fn new(level: PermissionLevel) -> Self {
        Self {
            level,
            rules: Vec::new(),
            tool_permissions: Vec::new(),
            prompter: None,
        }
    }

    /// Add a custom deny rule.
    #[must_use]
    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Register tool permission requirements.
    #[must_use]
    pub fn with_tool_permission(mut self, tool_name: &str, level: PermissionLevel) -> Self {
        self.tool_permissions.push(ToolPermission {
            tool_name: tool_name.to_string(),
            required_level: level,
        });
        self
    }

    /// Set an interactive prompter for escalation.
    #[must_use]
    pub fn with_prompter(mut self, prompter: impl PermissionPrompter + 'static) -> Self {
        self.prompter = Some(Box::new(prompter));
        self
    }

    /// Check whether a tool invocation is permitted.
    pub fn check(&self, tool_name: &str, input: &str) -> PermissionResult {
        // 1. Check tool-level permission requirements
        if let Some(req) = self
            .tool_permissions
            .iter()
            .find(|tp| tp.tool_name == tool_name)
        {
            if self.level < req.required_level {
                // Try prompter for escalation
                if let Some(prompter) = &self.prompter {
                    let desc = format!(
                        "Tool {tool_name:?} requires {required} but current level is {current}",
                        required = req.required_level,
                        current = self.level
                    );
                    if prompter.prompt(tool_name, input, &desc) {
                        // User approved escalation, continue to rules
                    } else {
                        return PermissionResult::deny(format!(
                            "permission level {current} insufficient for {tool_name} (requires {required})",
                            current = self.level,
                            required = req.required_level
                        ));
                    }
                } else {
                    return PermissionResult::deny(format!(
                        "permission level {current} insufficient for {tool_name} (requires {required})",
                        current = self.level,
                        required = req.required_level
                    ));
                }
            }
        }

        // 2. Evaluate custom rules
        for rule in &self.rules {
            if let Some(reason) = (rule.evaluator)(tool_name, input) {
                return PermissionResult::deny(reason);
            }
        }

        PermissionResult::allow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_access_allows_everything() {
        let engine = PermissionEngine::new(PermissionLevel::FullAccess);
        let result = engine.check("bash", r#"{"command":"rm -rf /"}"#);
        assert!(result.allowed);
    }

    #[test]
    fn deny_path_outside_workspace() {
        let engine = PermissionEngine::new(PermissionLevel::WorkspaceWrite)
            .with_rule(Rule::deny_path_outside("/workspace"));

        let inside = engine.check("write_file", r#"{"path":"/workspace/src/main.rs","content":"hi"}"#);
        assert!(inside.allowed);

        let outside = engine.check("write_file", r#"{"path":"/etc/passwd","content":"hacked"}"#);
        assert!(!outside.allowed);
        assert!(outside.reason.unwrap().contains("outside workspace"));
    }

    #[test]
    fn deny_bash_pattern() {
        let engine = PermissionEngine::new(PermissionLevel::FullAccess)
            .with_rule(Rule::deny_bash_pattern("rm -rf"));

        let safe = engine.check("bash", r#"{"command":"ls -la"}"#);
        assert!(safe.allowed);

        let dangerous = engine.check("bash", r#"{"command":"rm -rf /"}"#);
        assert!(!dangerous.allowed);
    }

    #[test]
    fn tool_level_permissions() {
        let engine = PermissionEngine::new(PermissionLevel::ReadOnly)
            .with_tool_permission("bash", PermissionLevel::FullAccess)
            .with_tool_permission("write_file", PermissionLevel::WorkspaceWrite)
            .with_tool_permission("read_file", PermissionLevel::ReadOnly);

        assert!(engine.check("read_file", "{}").allowed);
        assert!(!engine.check("write_file", "{}").allowed);
        assert!(!engine.check("bash", "{}").allowed);
    }

    #[test]
    fn prompter_escalation() {
        let engine = PermissionEngine::new(PermissionLevel::ReadOnly)
            .with_tool_permission("bash", PermissionLevel::FullAccess)
            .with_prompter(AutoApprovePrompter);

        // Prompter auto-approves the escalation
        let result = engine.check("bash", r#"{"command":"echo hi"}"#);
        assert!(result.allowed);
    }

    #[test]
    fn deny_prompter_blocks() {
        let engine = PermissionEngine::new(PermissionLevel::ReadOnly)
            .with_tool_permission("bash", PermissionLevel::FullAccess)
            .with_prompter(AutoDenyPrompter);

        let result = engine.check("bash", r#"{"command":"echo hi"}"#);
        assert!(!result.allowed);
    }

    #[test]
    fn allowlist_rule() {
        let engine = PermissionEngine::new(PermissionLevel::FullAccess)
            .with_rule(Rule::allowlist(vec![
                "read_file".to_string(),
                "grep_search".to_string(),
            ]));

        assert!(engine.check("read_file", "{}").allowed);
        assert!(!engine.check("bash", "{}").allowed);
    }

    #[test]
    fn path_traversal_blocked() {
        let engine = PermissionEngine::new(PermissionLevel::WorkspaceWrite)
            .with_rule(Rule::deny_path_outside("/workspace"));

        let result = engine.check("write_file", r#"{"path":"../../etc/passwd","content":"x"}"#);
        assert!(!result.allowed);
        assert!(result.reason.unwrap().contains("traversal"));
    }
}
