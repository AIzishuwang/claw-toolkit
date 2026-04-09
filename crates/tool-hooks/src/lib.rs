//! # tool-hooks
//!
//! Pre/Post tool execution hook pipeline for AI agent systems.
//!
//! Hooks can intercept tool calls to:
//! - **Deny** execution with a reason
//! - **Modify** tool input before execution
//! - **Inject** feedback messages into tool output
//! - **Audit** all tool activity
//!
//! # Example
//!
//! ```
//! use tool_hooks::{HookPipeline, HookAction, PreHookContext};
//!
//! let mut pipeline = HookPipeline::new();
//!
//! // Block dangerous bash commands
//! pipeline.pre("bash", |ctx| {
//!     if ctx.input.contains("rm -rf") {
//!         HookAction::Deny("dangerous delete blocked".into())
//!     } else {
//!         HookAction::Continue
//!     }
//! });
//!
//! // Log all tool calls
//! pipeline.post("*", |ctx| {
//!     println!("[{}] → {}", ctx.tool_name, ctx.output);
//!     HookAction::Continue
//! });
//! ```



/// The result of evaluating a hook.
#[derive(Debug, Clone)]
pub enum HookAction {
    /// Allow the operation to proceed unchanged.
    Continue,
    /// Allow with modified input (pre-hooks only).
    ModifyInput(String),
    /// Deny the operation with the given reason.
    Deny(String),
    /// Allow but inject additional feedback into the output.
    InjectFeedback(String),
}

/// Context passed to pre-execution hooks.
#[derive(Debug)]
pub struct PreHookContext<'a> {
    /// Name of the tool being invoked.
    pub tool_name: &'a str,
    /// Raw JSON input string for the tool.
    pub input: &'a str,
}

/// Context passed to post-execution hooks.
#[derive(Debug)]
pub struct PostHookContext<'a> {
    /// Name of the tool that was invoked.
    pub tool_name: &'a str,
    /// Raw JSON input string that was used.
    pub input: &'a str,
    /// Tool execution output.
    pub output: &'a str,
    /// Whether the tool execution resulted in an error.
    pub is_error: bool,
}

/// Result from running the pre-hook pipeline.
#[derive(Debug, Clone)]
pub struct PreHookResult {
    /// Whether execution was denied.
    pub denied: bool,
    /// Deny reason if denied.
    pub deny_reason: Option<String>,
    /// Modified input if any hook requested input modification.
    pub modified_input: Option<String>,
    /// Feedback messages to prepend to output.
    pub feedback: Vec<String>,
}

impl PreHookResult {
    fn allow() -> Self {
        Self {
            denied: false,
            deny_reason: None,
            modified_input: None,
            feedback: Vec::new(),
        }
    }
}

/// Result from running the post-hook pipeline.
#[derive(Debug, Clone)]
pub struct PostHookResult {
    /// Whether the post-hook pipeline denied (retroactively marks as error).
    pub denied: bool,
    /// Deny reason.
    pub deny_reason: Option<String>,
    /// Feedback messages to append to output.
    pub feedback: Vec<String>,
}

impl PostHookResult {
    fn allow() -> Self {
        Self {
            denied: false,
            deny_reason: None,
            feedback: Vec::new(),
        }
    }
}

type PreHookFn = Box<dyn Fn(&PreHookContext) -> HookAction + Send + Sync>;
type PostHookFn = Box<dyn Fn(&PostHookContext) -> HookAction + Send + Sync>;

struct RegisteredPreHook {
    pattern: String,
    handler: PreHookFn,
}

struct RegisteredPostHook {
    pattern: String,
    handler: PostHookFn,
}

/// A pipeline of pre and post tool-execution hooks.
///
/// Hooks are evaluated in registration order. The pipeline short-circuits on
/// the first `Deny` action.
pub struct HookPipeline {
    pre_hooks: Vec<RegisteredPreHook>,
    post_hooks: Vec<RegisteredPostHook>,
}

impl Default for HookPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl HookPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pre_hooks: Vec::new(),
            post_hooks: Vec::new(),
        }
    }

    /// Register a pre-execution hook.
    ///
    /// `pattern` can be a specific tool name or `"*"` to match all tools.
    pub fn pre(
        &mut self,
        pattern: &str,
        handler: impl Fn(&PreHookContext) -> HookAction + Send + Sync + 'static,
    ) -> &mut Self {
        self.pre_hooks.push(RegisteredPreHook {
            pattern: pattern.to_string(),
            handler: Box::new(handler),
        });
        self
    }

    /// Register a post-execution hook.
    pub fn post(
        &mut self,
        pattern: &str,
        handler: impl Fn(&PostHookContext) -> HookAction + Send + Sync + 'static,
    ) -> &mut Self {
        self.post_hooks.push(RegisteredPostHook {
            pattern: pattern.to_string(),
            handler: Box::new(handler),
        });
        self
    }

    /// Run pre-execution hooks for a tool call.
    pub fn run_pre(&self, tool_name: &str, input: &str) -> PreHookResult {
        let mut result = PreHookResult::allow();
        let ctx = PreHookContext { tool_name, input };

        for hook in &self.pre_hooks {
            if !pattern_matches(&hook.pattern, tool_name) {
                continue;
            }

            match (hook.handler)(&ctx) {
                HookAction::Continue => {}
                HookAction::ModifyInput(new_input) => {
                    result.modified_input = Some(new_input);
                }
                HookAction::Deny(reason) => {
                    result.denied = true;
                    result.deny_reason = Some(reason);
                    return result; // short-circuit
                }
                HookAction::InjectFeedback(msg) => {
                    result.feedback.push(msg);
                }
            }
        }

        result
    }

    /// Run post-execution hooks for a completed tool call.
    pub fn run_post(
        &self,
        tool_name: &str,
        input: &str,
        output: &str,
        is_error: bool,
    ) -> PostHookResult {
        let mut result = PostHookResult::allow();
        let ctx = PostHookContext {
            tool_name,
            input,
            output,
            is_error,
        };

        for hook in &self.post_hooks {
            if !pattern_matches(&hook.pattern, tool_name) {
                continue;
            }

            match (hook.handler)(&ctx) {
                HookAction::Continue => {}
                HookAction::Deny(reason) => {
                    result.denied = true;
                    result.deny_reason = Some(reason);
                    return result;
                }
                HookAction::InjectFeedback(msg) => {
                    result.feedback.push(msg);
                }
                HookAction::ModifyInput(_) => {
                    // ModifyInput is meaningless in post-hooks, treat as Continue
                }
            }
        }

        result
    }

    /// Merge hook feedback into a tool output string.
    pub fn merge_feedback(output: String, feedback: &[String], is_error: bool) -> String {
        if feedback.is_empty() {
            return output;
        }
        let label = if is_error {
            "Hook feedback (error)"
        } else {
            "Hook feedback"
        };
        let feedback_block = format!("{label}:\n{}", feedback.join("\n"));

        if output.trim().is_empty() {
            feedback_block
        } else {
            format!("{output}\n\n{feedback_block}")
        }
    }
}

fn pattern_matches(pattern: &str, tool_name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if pattern.ends_with('*') {
        let prefix = &pattern[..pattern.len() - 1];
        return tool_name.starts_with(prefix);
    }
    pattern == tool_name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_hook_allows_by_default() {
        let pipeline = HookPipeline::new();
        let result = pipeline.run_pre("bash", r#"{"command":"ls"}"#);
        assert!(!result.denied);
    }

    #[test]
    fn pre_hook_denies_dangerous_command() {
        let mut pipeline = HookPipeline::new();
        pipeline.pre("bash", |ctx| {
            if ctx.input.contains("rm -rf") {
                HookAction::Deny("blocked".into())
            } else {
                HookAction::Continue
            }
        });

        let safe = pipeline.run_pre("bash", r#"{"command":"ls"}"#);
        assert!(!safe.denied);

        let dangerous = pipeline.run_pre("bash", r#"{"command":"rm -rf /"}"#);
        assert!(dangerous.denied);
        assert_eq!(dangerous.deny_reason.unwrap(), "blocked");
    }

    #[test]
    fn pre_hook_modifies_input() {
        let mut pipeline = HookPipeline::new();
        pipeline.pre("*", |_ctx| HookAction::ModifyInput("modified".into()));

        let result = pipeline.run_pre("bash", "original");
        assert!(!result.denied);
        assert_eq!(result.modified_input.unwrap(), "modified");
    }

    #[test]
    fn wildcard_matches_all() {
        let mut pipeline = HookPipeline::new();
        let called = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let called_clone = called.clone();

        pipeline.pre("*", move |_| {
            called_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            HookAction::Continue
        });

        pipeline.run_pre("bash", "");
        pipeline.run_pre("read_file", "");
        pipeline.run_pre("write_file", "");

        assert_eq!(called.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn post_hook_injects_feedback() {
        let mut pipeline = HookPipeline::new();
        pipeline.post("*", |_| HookAction::InjectFeedback("audit: logged".into()));

        let result = pipeline.run_post("bash", "{}", "output", false);
        assert!(!result.denied);
        assert_eq!(result.feedback, vec!["audit: logged"]);
    }

    #[test]
    fn merge_feedback_appends() {
        let output = "tool output".to_string();
        let feedback = vec!["note 1".to_string(), "note 2".to_string()];

        let merged = HookPipeline::merge_feedback(output, &feedback, false);
        assert!(merged.contains("tool output"));
        assert!(merged.contains("Hook feedback:"));
        assert!(merged.contains("note 1"));
    }

    #[test]
    fn prefix_pattern_matching() {
        assert!(pattern_matches("Task*", "TaskCreate"));
        assert!(pattern_matches("Task*", "TaskGet"));
        assert!(!pattern_matches("Task*", "bash"));
    }

    #[test]
    fn deny_short_circuits() {
        let mut pipeline = HookPipeline::new();
        let reached = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reached_clone = reached.clone();

        pipeline.pre("bash", |_| HookAction::Deny("first".into()));
        pipeline.pre("bash", move |_| {
            reached_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            HookAction::Continue
        });

        let result = pipeline.run_pre("bash", "");
        assert!(result.denied);
        assert!(!reached.load(std::sync::atomic::Ordering::SeqCst));
    }
}
