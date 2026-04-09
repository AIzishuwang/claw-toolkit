//! # llm-client
//!
//! Unified LLM client supporting Anthropic and OpenAI-compatible APIs.
//!
//! # Example
//!
//! ```no_run
//! use llm_client::{LlmClient, AnthropicProvider, Message, Role, StreamEvent};
//!
//! let client = LlmClient::anthropic("sk-ant-...", "claude-sonnet-4-20250514");
//! let messages = vec![Message::user("Hello!")];
//! let events = client.stream(&messages, &[]).unwrap();
//! for event in events {
//!     if let StreamEvent::Text(text) = event {
//!         print!("{text}");
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

/// Errors from LLM client operations.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("stream error: {0}")]
    Stream(String),
    #[error("missing api key")]
    MissingApiKey,
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

/// Message content — either a simple string or structured blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: Value },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl Message {
    /// Create a user message with text content.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(text.into()),
        }
    }

    /// Create an assistant message with text content.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(text.into()),
        }
    }

    /// Create a tool result message.
    #[must_use]
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
            }]),
        }
    }
}

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Streamed events from an LLM response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text content delta.
    Text(String),
    /// Tool use request from the model.
    ToolUse { id: String, name: String, input: Value },
    /// Token usage report.
    Usage { input_tokens: u32, output_tokens: u32 },
    /// Stream completed.
    Done,
}

/// Token usage from a response.
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Model alias resolution.
pub fn resolve_model_alias(alias: &str) -> &str {
    match alias {
        "opus" => "claude-opus-4-6",
        "sonnet" => "claude-sonnet-4-6",
        "haiku" => "claude-haiku-4-5-20251213",
        "gpt4" => "gpt-4o",
        "gpt4o" => "gpt-4o",
        other => other,
    }
}

/// Provider-agnostic LLM client.
pub struct LlmClient {
    provider: Box<dyn LlmProvider>,
}

impl LlmClient {
    /// Create a client for Anthropic's API.
    #[must_use]
    pub fn anthropic(api_key: &str, model: &str) -> Self {
        Self {
            provider: Box::new(AnthropicProvider::new(api_key, model)),
        }
    }

    /// Create a client for an OpenAI-compatible API.
    #[must_use]
    pub fn openai_compat(base_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            provider: Box::new(OpenAiCompatProvider::new(base_url, api_key, model)),
        }
    }

    /// Create a client from a custom provider.
    #[must_use]
    pub fn custom(provider: impl LlmProvider + 'static) -> Self {
        Self {
            provider: Box::new(provider),
        }
    }

    /// Send messages and stream the response.
    pub fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Vec<StreamEvent>, LlmError> {
        self.provider.stream(messages, tools)
    }

    /// Send messages and get a complete (non-streaming) response.
    pub fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Vec<ContentBlock>, Usage), LlmError> {
        let events = self.stream(messages, tools)?;
        let mut blocks = Vec::new();
        let mut text = String::new();
        let mut usage = Usage::default();

        for event in events {
            match event {
                StreamEvent::Text(t) => text.push_str(&t),
                StreamEvent::ToolUse { id, name, input } => {
                    if !text.is_empty() {
                        blocks.push(ContentBlock::Text { text: std::mem::take(&mut text) });
                    }
                    blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                StreamEvent::Usage { input_tokens, output_tokens } => {
                    usage.input_tokens = input_tokens;
                    usage.output_tokens = output_tokens;
                }
                StreamEvent::Done => {}
            }
        }
        if !text.is_empty() {
            blocks.push(ContentBlock::Text { text });
        }

        Ok((blocks, usage))
    }
}

/// Trait for LLM provider implementations.
pub trait LlmProvider: Send + Sync {
    /// Send a request and return streamed events.
    fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Vec<StreamEvent>, LlmError>;
}

/// Anthropic API provider.
pub struct AnthropicProvider {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    client: reqwest::blocking::Client,
}

impl AnthropicProvider {
    #[must_use]
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: resolve_model_alias(model).to_string(),
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            max_tokens: if model.contains("opus") { 32_000 } else { 64_000 },
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Set a custom base URL (for proxies).
    #[must_use]
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    #[must_use]
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }
}

impl LlmProvider for AnthropicProvider {
    fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Vec<StreamEvent>, LlmError> {
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "stream": true,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools.iter().map(|t| json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })).collect::<Vec<_>>());
        }

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()?;

        let status = response.status().as_u16();
        if status >= 400 {
            let message = response.text().unwrap_or_default();
            return Err(LlmError::Api { status, message });
        }

        let raw = response.text()?;
        let mut parser = sse_parser::SseParser::new();
        let sse_events = parser.feed(&raw);

        let mut stream_events = Vec::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input = String::new();

        for sse in sse_events {
            if sse.is_done() {
                stream_events.push(StreamEvent::Done);
                continue;
            }

            let data: Value = match serde_json::from_str(&sse.data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            match sse.event_type.as_str() {
                "content_block_start" => {
                    if let Some(cb) = data.get("content_block") {
                        if cb.get("type").and_then(Value::as_str) == Some("tool_use") {
                            current_tool_id = cb.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                            current_tool_name = cb.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                            current_tool_input.clear();
                        }
                    }
                }
                "content_block_delta" => {
                    if let Some(delta) = data.get("delta") {
                        match delta.get("type").and_then(Value::as_str) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(Value::as_str) {
                                    stream_events.push(StreamEvent::Text(text.to_string()));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                                    current_tool_input.push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => {
                    if !current_tool_name.is_empty() {
                        let input: Value = serde_json::from_str(&current_tool_input).unwrap_or(Value::Null);
                        stream_events.push(StreamEvent::ToolUse {
                            id: std::mem::take(&mut current_tool_id),
                            name: std::mem::take(&mut current_tool_name),
                            input,
                        });
                        current_tool_input.clear();
                    }
                }
                "message_delta" => {
                    if let Some(usage) = data.get("usage") {
                        let input_tokens = usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                        let output_tokens = usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                        stream_events.push(StreamEvent::Usage { input_tokens, output_tokens });
                    }
                }
                "message_start" => {
                    if let Some(msg) = data.get("message") {
                        if let Some(usage) = msg.get("usage") {
                            let input_tokens = usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                            let output_tokens = usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
                            stream_events.push(StreamEvent::Usage { input_tokens, output_tokens });
                        }
                    }
                }
                "message_stop" => {
                    stream_events.push(StreamEvent::Done);
                }
                _ => {}
            }
        }

        Ok(stream_events)
    }
}

/// OpenAI-compatible API provider.
pub struct OpenAiCompatProvider {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl OpenAiCompatProvider {
    #[must_use]
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: resolve_model_alias(model).to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl LlmProvider for OpenAiCompatProvider {
    fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Vec<StreamEvent>, LlmError> {
        // Convert to OpenAI format
        let oai_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                let content = match &m.content {
                    MessageContent::Text(t) => json!(t),
                    MessageContent::Blocks(blocks) => {
                        let parts: Vec<Value> = blocks.iter().map(|b| match b {
                            ContentBlock::Text { text } => json!({"type": "text", "text": text}),
                            ContentBlock::ToolUse { id, name, input } => json!({
                                "type": "function", "id": id, "function": {"name": name, "arguments": input.to_string()}
                            }),
                            ContentBlock::ToolResult { tool_use_id, content, is_error: _ } => json!({
                                "type": "function_result", "tool_call_id": tool_use_id, "content": content
                            }),
                        }).collect();
                        json!(parts)
                    }
                };
                json!({"role": role, "content": content})
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": oai_messages,
            "stream": false,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools.iter().map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            })).collect::<Vec<_>>());
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()?;

        let status = response.status().as_u16();
        if status >= 400 {
            let message = response.text().unwrap_or_default();
            return Err(LlmError::Api { status, message });
        }

        let data: Value = response.json()?;
        let mut events = Vec::new();

        if let Some(choice) = data.get("choices").and_then(|c| c.get(0)) {
            if let Some(content) = choice.get("message").and_then(|m| m.get("content")).and_then(Value::as_str) {
                events.push(StreamEvent::Text(content.to_string()));
            }
            if let Some(tool_calls) = choice.get("message").and_then(|m| m.get("tool_calls")).and_then(Value::as_array) {
                for tc in tool_calls {
                    if let (Some(id), Some(func)) = (tc.get("id").and_then(Value::as_str), tc.get("function")) {
                        let name = func.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                        let args = func.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        let input: Value = serde_json::from_str(args).unwrap_or(Value::Null);
                        events.push(StreamEvent::ToolUse { id: id.to_string(), name, input });
                    }
                }
            }
        }

        if let Some(usage) = data.get("usage") {
            let input_tokens = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
            let output_tokens = usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0) as u32;
            events.push(StreamEvent::Usage { input_tokens, output_tokens });
        }

        events.push(StreamEvent::Done);
        Ok(events)
    }
}

/// Deterministic mock provider for testing.
pub struct MockProvider {
    responses: std::sync::Mutex<Vec<Vec<StreamEvent>>>,
}

impl MockProvider {
    /// Create a mock provider with scripted responses.
    ///
    /// Each call to `stream()` pops the next response from the queue.
    #[must_use]
    pub fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
        // Reverse so we can pop from the end
        let mut r = responses;
        r.reverse();
        Self {
            responses: std::sync::Mutex::new(r),
        }
    }

    /// Create a mock that returns a single text response.
    #[must_use]
    pub fn text(text: &str) -> Self {
        Self::new(vec![vec![
            StreamEvent::Text(text.to_string()),
            StreamEvent::Usage { input_tokens: 10, output_tokens: 5 },
            StreamEvent::Done,
        ]])
    }
}

impl LlmProvider for MockProvider {
    fn stream(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<Vec<StreamEvent>, LlmError> {
        let mut responses = self.responses.lock().unwrap();
        responses.pop().ok_or_else(|| LlmError::Stream("no more mock responses".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_alias_resolution() {
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-6");
        assert_eq!(resolve_model_alias("sonnet"), "claude-sonnet-4-6");
        assert_eq!(resolve_model_alias("custom-model"), "custom-model");
    }

    #[test]
    fn mock_provider_text() {
        let client = LlmClient::custom(MockProvider::text("hello"));
        let (blocks, usage) = client.complete(&[Message::user("hi")], &[]).unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "hello"));
        assert_eq!(usage.input_tokens, 10);
    }

    #[test]
    fn mock_provider_tool_use() {
        let provider = MockProvider::new(vec![vec![
            StreamEvent::ToolUse {
                id: "t1".to_string(),
                name: "read_file".to_string(),
                input: json!({"path": "main.rs"}),
            },
            StreamEvent::Done,
        ]]);

        let client = LlmClient::custom(provider);
        let (blocks, _) = client.complete(&[Message::user("read file")], &[]).unwrap();
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { name, .. } if name == "read_file"));
    }

    #[test]
    fn mock_provider_exhaustion() {
        let provider = MockProvider::new(vec![]);
        let client = LlmClient::custom(provider);
        let result = client.complete(&[Message::user("hi")], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn message_constructors() {
        let user = Message::user("hello");
        assert_eq!(user.role, Role::User);

        let assistant = Message::assistant("hi");
        assert_eq!(assistant.role, Role::Assistant);

        let tool = Message::tool_result("t1", "output", false);
        assert_eq!(tool.role, Role::User);
    }
}
