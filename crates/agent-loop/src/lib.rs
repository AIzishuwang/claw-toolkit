//! # agent-loop
//!
//! Core agentic loop framework extracted from Claude Code's `ConversationRuntime`.
//!
//! The loop pattern:
//! 1. User provides input
//! 2. LLM generates a response (possibly including tool calls)
//! 3. Tools are executed with permission checks and hooks
//! 4. Tool results are fed back to the LLM
//! 5. Repeat until the LLM responds with text only (no tool calls)
//!
//! # Example
//!
//! ```no_run
//! use agent_loop::{AgentLoop, AgentConfig, Tool};
//! use llm_client::MockProvider;
//!
//! let config = AgentConfig::default();
//! let provider = MockProvider::text("Hello!");
//! let tools: Vec<Box<dyn Tool>> = vec![];
//!
//! let mut agent = AgentLoop::new(provider, tools, config);
//! let result = agent.run("Say hello").unwrap();
//! println!("{}", result.final_text());
//! ```

use llm_client::{ContentBlock, LlmClient, LlmError, LlmProvider, Message, StreamEvent, ToolDefinition, Usage};
use permission_engine::{PermissionEngine, PermissionLevel};
use session_store::{Session, TokenUsage};
use tool_hooks::HookPipeline;

pub use prompt_memory::{ProjectContext, SystemPromptBuilder, SYSTEM_PROMPT_DYNAMIC_BOUNDARY};

use serde_json::Value;
use thiserror::Error;
use std::path::PathBuf;

/// Errors from the agent loop.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("llm error: {0}")]
    Llm(#[from] LlmError),
    #[error("exceeded maximum iterations ({0})")]
    MaxIterations(usize),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("no response from LLM")]
    EmptyResponse,
}

/// A tool that the agent can invoke.
pub trait Tool: Send + Sync {
    /// Tool name (must be unique).
    fn name(&self) -> &str;
    /// Tool description for the LLM.
    fn description(&self) -> &str;
    /// JSON schema for tool input.
    fn input_schema(&self) -> Value;
    /// Required permission level.
    fn required_permission(&self) -> PermissionLevel;
    /// Execute the tool with the given JSON input.
    fn execute(&self, input: &Value) -> Result<String, String>;
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum loop iterations before aborting. Default: 50.
    pub max_iterations: usize,
    /// Auto-compact session when estimated tokens exceed this. Default: 100_000.
    pub auto_compact_threshold: usize,
    /// Number of recent messages to keep when compacting. Default: 20.
    pub compact_keep_recent: usize,
    /// System prompt lines.
    pub system_prompt: Vec<String>,
    /// Working directory for instruction file discovery. When set,
    /// `prompt-memory` will walk the ancestor chain to find `CLAUDE.md` files
    /// and auto-build the system prompt.
    pub cwd: Option<PathBuf>,
    /// Whether to auto-discover instruction files (CLAUDE.md, etc.).
    /// Requires `cwd` to be set. Default: true.
    pub discover_instructions: bool,
    /// Current date string injected into the prompt environment section.
    pub current_date: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            auto_compact_threshold: 100_000,
            compact_keep_recent: 20,
            system_prompt: Vec::new(),
            cwd: None,
            discover_instructions: true,
            current_date: None,
        }
    }
}

/// Summary of a completed agent turn.
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// All assistant text blocks from the turn.
    pub assistant_texts: Vec<String>,
    /// Tool calls and their results.
    pub tool_results: Vec<ToolCallResult>,
    /// Number of loop iterations.
    pub iterations: usize,
    /// Token usage.
    pub usage: Usage,
    /// Whether auto-compaction was triggered.
    pub compacted: bool,
}

impl TurnResult {
    /// Get the final text response from the assistant.
    #[must_use]
    pub fn final_text(&self) -> String {
        self.assistant_texts.join("")
    }
}

/// Result of a single tool call within a turn.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    pub tool_name: String,
    pub input: Value,
    pub output: String,
    pub is_error: bool,
    pub was_denied: bool,
}

/// The core agentic loop.
///
/// When `AgentConfig::cwd` is set and `discover_instructions` is true,
/// the agent automatically scans the ancestor directory chain for
/// instruction files (`CLAUDE.md`, `.claw/instructions.md`, etc.)
/// and includes them in the system prompt sent to the LLM.
pub struct AgentLoop {
    client: LlmClient,
    tools: Vec<Box<dyn Tool>>,
    config: AgentConfig,
    session: Session,
    hooks: HookPipeline,
    permissions: Option<PermissionEngine>,
    total_usage: Usage,
    /// Rendered system prompt (built from config + discovered instructions).
    system_prompt: Option<String>,
}

impl AgentLoop {
    /// Create a new agent loop.
    ///
    /// If `config.cwd` is set and `config.discover_instructions` is true,
    /// instruction files will be automatically discovered and included in
    /// the system prompt.
    pub fn new(
        provider: impl LlmProvider + 'static,
        tools: Vec<Box<dyn Tool>>,
        config: AgentConfig,
    ) -> Self {
        let system_prompt = build_system_prompt(&config);
        Self {
            client: LlmClient::custom(provider),
            tools,
            config,
            session: Session::new(),
            hooks: HookPipeline::new(),
            permissions: None,
            total_usage: Usage::default(),
            system_prompt,
        }
    }

    /// Set the hook pipeline.
    #[must_use]
    pub fn with_hooks(mut self, hooks: HookPipeline) -> Self {
        self.hooks = hooks;
        self
    }

    /// Set the permission engine.
    #[must_use]
    pub fn with_permissions(mut self, engine: PermissionEngine) -> Self {
        self.permissions = Some(engine);
        self
    }

    /// Set a custom session (e.g., for resume).
    #[must_use]
    pub fn with_session(mut self, session: Session) -> Self {
        self.session = session;
        self
    }

    /// Get a reference to the current session.
    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Get cumulative token usage.
    #[must_use]
    pub fn total_usage(&self) -> Usage {
        self.total_usage
    }

    /// Run a single turn of the agent loop.
    pub fn run(&mut self, user_input: &str) -> Result<TurnResult, AgentError> {
        self.session.push_user(user_input);

        let tool_defs: Vec<ToolDefinition> = self
            .tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect();

        let mut assistant_texts = Vec::new();
        let mut tool_results = Vec::new();
        let mut iterations = 0;
        let mut usage = Usage::default();

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Err(AgentError::MaxIterations(self.config.max_iterations));
            }

            // Build messages for API
            let messages = self.build_messages();

            // Call LLM
            let events = self.client.stream(&messages, &tool_defs)?;
            let (blocks, turn_usage) = Self::parse_events(events);

            usage.input_tokens += turn_usage.input_tokens;
            usage.output_tokens += turn_usage.output_tokens;

            if blocks.is_empty() {
                return Err(AgentError::EmptyResponse);
            }

            // Process response blocks
            let mut pending_tool_calls = Vec::new();
            let mut response_blocks = Vec::new();

            for block in &blocks {
                match block {
                    ContentBlock::Text { text } => {
                        assistant_texts.push(text.clone());
                        response_blocks.push(session_store::ContentBlock::text(text.clone()));
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        pending_tool_calls.push((id.clone(), name.clone(), input.clone()));
                        response_blocks.push(session_store::ContentBlock::tool_use(
                            id.clone(),
                            name.clone(),
                            serde_json::to_string(input).unwrap_or_default(),
                        ));
                    }
                    _ => {}
                }
            }

            self.session.push_assistant(response_blocks);

            // No tool calls → done
            if pending_tool_calls.is_empty() {
                break;
            }

            // Execute tools
            for (tool_id, tool_name, tool_input) in pending_tool_calls {
                let input_str = serde_json::to_string(&tool_input).unwrap_or_default();

                // Pre-hook
                let pre_result = self.hooks.run_pre(&tool_name, &input_str);
                if pre_result.denied {
                    let reason = pre_result.deny_reason.unwrap_or_else(|| "denied by hook".to_string());
                    self.session.push_tool_result(&tool_id, &tool_name, &reason, true);
                    tool_results.push(ToolCallResult {
                        tool_name: tool_name.clone(),
                        input: tool_input,
                        output: reason,
                        is_error: true,
                        was_denied: true,
                    });
                    continue;
                }

                // Permission check
                if let Some(perms) = &self.permissions {
                    let perm_result = perms.check(&tool_name, &input_str);
                    if !perm_result.allowed {
                        let reason = perm_result.reason.unwrap_or_else(|| "permission denied".to_string());
                        self.session.push_tool_result(&tool_id, &tool_name, &reason, true);
                        tool_results.push(ToolCallResult {
                            tool_name: tool_name.clone(),
                            input: tool_input,
                            output: reason,
                            is_error: true,
                            was_denied: true,
                        });
                        continue;
                    }
                }

                // Execute
                let effective_input = pre_result
                    .modified_input
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(tool_input.clone());

                let tool = self.tools.iter().find(|t| t.name() == tool_name);
                let (mut output, mut is_error) = match tool {
                    Some(t) => match t.execute(&effective_input) {
                        Ok(out) => (out, false),
                        Err(err) => (err, true),
                    },
                    None => (format!("unknown tool: {tool_name}"), true),
                };

                // Pre-hook feedback
                if !pre_result.feedback.is_empty() {
                    output = HookPipeline::merge_feedback(output, &pre_result.feedback, false);
                }

                // Post-hook
                let post_result = self.hooks.run_post(&tool_name, &input_str, &output, is_error);
                if post_result.denied {
                    is_error = true;
                }
                if !post_result.feedback.is_empty() {
                    output = HookPipeline::merge_feedback(output, &post_result.feedback, is_error);
                }

                self.session.push_tool_result(&tool_id, &tool_name, &output, is_error);
                tool_results.push(ToolCallResult {
                    tool_name: tool_name.clone(),
                    input: tool_input,
                    output,
                    is_error,
                    was_denied: false,
                });
            }
        }

        // Auto-compact
        let compacted = if self.session.estimate_tokens() > self.config.auto_compact_threshold {
            self.session.compact(self.config.compact_keep_recent);
            true
        } else {
            false
        };

        self.total_usage.input_tokens += usage.input_tokens;
        self.total_usage.output_tokens += usage.output_tokens;
        self.session.record_usage(TokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        });

        Ok(TurnResult {
            assistant_texts,
            tool_results,
            iterations,
            usage,
            compacted,
        })
    }

    fn build_messages(&self) -> Vec<Message> {
        let mut messages: Vec<Message> = self.session
            .messages()
            .iter()
            .map(|m| {
                let role = match m.role {
                    session_store::MessageRole::System => llm_client::Role::User,
                    session_store::MessageRole::User => llm_client::Role::User,
                    session_store::MessageRole::Assistant => llm_client::Role::Assistant,
                };
                let content = if m.blocks.len() == 1 {
                    match &m.blocks[0] {
                        session_store::ContentBlock::Text { text } => {
                            llm_client::MessageContent::Text(text.clone())
                        }
                        _ => llm_client::MessageContent::Text(format!("{:?}", m.blocks[0])),
                    }
                } else {
                    llm_client::MessageContent::Text(
                        m.blocks
                            .iter()
                            .filter_map(|b| match b {
                                session_store::ContentBlock::Text { text } => Some(text.clone()),
                                session_store::ContentBlock::ToolResult { content, .. } => {
                                    Some(content.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    )
                };
                Message { role, content }
            })
            .collect();

        // Prepend system prompt as the first user message if present
        if let Some(ref prompt) = self.system_prompt {
            messages.insert(0, Message {
                role: llm_client::Role::User,
                content: llm_client::MessageContent::Text(
                    format!("[System instructions — do not acknowledge, just follow them]\n\n{prompt}"),
                ),
            });
        }

        messages
    }

    fn parse_events(events: Vec<StreamEvent>) -> (Vec<ContentBlock>, Usage) {
        let mut blocks = Vec::new();
        let mut text = String::new();
        let mut usage = Usage::default();

        for event in events {
            match event {
                StreamEvent::Text(t) => text.push_str(&t),
                StreamEvent::ToolUse { id, name, input } => {
                    if !text.is_empty() {
                        blocks.push(ContentBlock::Text {
                            text: std::mem::take(&mut text),
                        });
                    }
                    blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                StreamEvent::Usage {
                    input_tokens,
                    output_tokens,
                } => {
                    usage.input_tokens = input_tokens;
                    usage.output_tokens = output_tokens;
                }
                StreamEvent::Done => {}
            }
        }
        if !text.is_empty() {
            blocks.push(ContentBlock::Text { text });
        }

        (blocks, usage)
    }
}

/// Build a rendered system prompt from the agent config.
///
/// When `config.cwd` is set and `config.discover_instructions` is true,
/// this function discovers instruction files and builds a full prompt via
/// `prompt_memory::SystemPromptBuilder`.
fn build_system_prompt(config: &AgentConfig) -> Option<String> {
    // If explicit system_prompt lines are provided, use them directly
    if !config.system_prompt.is_empty() {
        return Some(config.system_prompt.join("\n"));
    }

    // Otherwise, try auto-discovery via prompt-memory
    let cwd = config.cwd.as_ref()?;
    if !config.discover_instructions {
        return None;
    }

    let date = config
        .current_date
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    let ctx = prompt_memory::ProjectContext::discover_with_git(cwd, &date).ok()?;

    let prompt = prompt_memory::SystemPromptBuilder::new()
        .with_project_context(ctx)
        .render();

    if prompt.trim().is_empty() {
        None
    } else {
        Some(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_client::MockProvider;
    use tool_hooks::HookAction;
    use serde_json::json;

    #[test]
    fn simple_text_response() {
        let provider = MockProvider::text("Hello world!");
        let mut agent = AgentLoop::new(provider, vec![], AgentConfig::default());

        let result = agent.run("Say hello").unwrap();
        assert_eq!(result.final_text(), "Hello world!");
        assert_eq!(result.iterations, 1);
        assert!(result.tool_results.is_empty());
    }

    struct EchoTool;
    impl Tool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echo the input" }
        fn input_schema(&self) -> Value { json!({"type":"object","properties":{"text":{"type":"string"}}}) }
        fn required_permission(&self) -> PermissionLevel { PermissionLevel::ReadOnly }
        fn execute(&self, input: &Value) -> Result<String, String> {
            Ok(input.get("text").and_then(Value::as_str).unwrap_or("").to_string())
        }
    }

    #[test]
    fn tool_use_loop() {
        let provider = MockProvider::new(vec![
            // Turn 1: model requests tool
            vec![
                StreamEvent::ToolUse {
                    id: "t1".to_string(),
                    name: "echo".to_string(),
                    input: json!({"text": "echoed!"}),
                },
                StreamEvent::Done,
            ],
            // Turn 2: model gives final text after seeing tool result
            vec![
                StreamEvent::Text("The echo said: echoed!".to_string()),
                StreamEvent::Done,
            ],
        ]);

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let mut agent = AgentLoop::new(provider, tools, AgentConfig::default());

        let result = agent.run("echo something").unwrap();
        assert_eq!(result.final_text(), "The echo said: echoed!");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].output, "echoed!");
    }

    #[test]
    fn hook_denies_tool() {
        let provider = MockProvider::new(vec![
            vec![
                StreamEvent::ToolUse {
                    id: "t1".to_string(),
                    name: "echo".to_string(),
                    input: json!({"text": "blocked"}),
                },
                StreamEvent::Done,
            ],
            vec![
                StreamEvent::Text("denied".to_string()),
                StreamEvent::Done,
            ],
        ]);

        let mut hooks = HookPipeline::new();
        hooks.pre("echo", |_| HookAction::Deny("nope".into()));

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let mut agent = AgentLoop::new(provider, tools, AgentConfig::default()).with_hooks(hooks);

        let result = agent.run("try echo").unwrap();
        assert_eq!(result.tool_results.len(), 1);
        assert!(result.tool_results[0].was_denied);
    }

    #[test]
    fn permission_denies_tool() {
        let provider = MockProvider::new(vec![
            vec![
                StreamEvent::ToolUse {
                    id: "t1".to_string(),
                    name: "echo".to_string(),
                    input: json!({}),
                },
                StreamEvent::Done,
            ],
            vec![
                StreamEvent::Text("denied".to_string()),
                StreamEvent::Done,
            ],
        ]);

        let perms = PermissionEngine::new(PermissionLevel::ReadOnly)
            .with_tool_permission("echo", PermissionLevel::FullAccess);

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let mut agent = AgentLoop::new(provider, tools, AgentConfig::default())
            .with_permissions(perms);

        let result = agent.run("try echo").unwrap();
        assert!(result.tool_results[0].was_denied);
    }

    #[test]
    fn max_iterations_guard() {
        // Provider always returns tool calls → infinite loop
        let responses: Vec<Vec<StreamEvent>> = (0..100)
            .map(|i| vec![
                StreamEvent::ToolUse {
                    id: format!("t{i}"),
                    name: "echo".to_string(),
                    input: json!({"text": "loop"}),
                },
                StreamEvent::Done,
            ])
            .collect();

        let provider = MockProvider::new(responses);
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        let mut agent = AgentLoop::new(
            provider,
            tools,
            AgentConfig {
                max_iterations: 3,
                ..Default::default()
            },
        );

        let result = agent.run("infinite");
        assert!(matches!(result, Err(AgentError::MaxIterations(3))));
    }
}
