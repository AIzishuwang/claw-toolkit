//! # mcp-client
//!
//! Model Context Protocol (MCP) client SDK.
//!
//! MCP is a protocol for connecting AI models to external tools and data sources.
//! This crate provides a client that can connect to MCP servers via stdio
//! (subprocess) transport, discover their tools/resources, and invoke them.
//!
//! # Example
//!
//! ```no_run
//! use mcp_client::{McpClient, StdioTransport};
//!
//! let transport = StdioTransport::new("npx", &["-y", "@modelcontextprotocol/server-github"]).unwrap();
//! let mut client = McpClient::new(transport).unwrap();
//!
//! let tools = client.list_tools().unwrap();
//! let result = client.call_tool("search_repos", serde_json::json!({"query": "rust"})).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;


use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// MCP protocol version.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Errors from MCP operations.
#[derive(Debug, Error)]
pub enum McpError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("json-rpc error ({code}): {message}")]
    JsonRpc { code: i64, message: String },
    #[error("server not initialized")]
    NotInitialized,
    #[error("unexpected response: {0}")]
    UnexpectedResponse(String),
    #[error("tool not found: {0}")]
    ToolNotFound(String),
}

/// An MCP tool exposed by a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

/// An MCP resource exposed by a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// Content returned from a tool call or resource read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpContent {
    #[serde(default = "default_content_type")]
    pub r#type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}

fn default_content_type() -> String {
    "text".to_string()
}

/// Result of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<McpContent>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

/// Server info returned during initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
}

/// Transport trait for MCP communication.
pub trait McpTransport: Send {
    /// Send a JSON-RPC message and receive the response.
    fn send(&mut self, request: &Value) -> Result<Value, McpError>;
    /// Close the transport.
    fn close(&mut self) -> Result<(), McpError>;
}

/// Stdio transport — communicates with an MCP server via a subprocess.
pub struct StdioTransport {
    child: Child,
    next_id: AtomicU64,
}

impl StdioTransport {
    /// Spawn a new subprocess MCP server.
    pub fn new(program: &str, args: &[&str]) -> Result<Self, McpError> {
        let child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn {program}: {e}")))?;

        Ok(Self {
            child,
            next_id: AtomicU64::new(1),
        })
    }
}

impl McpTransport for StdioTransport {
    fn send(&mut self, request: &Value) -> Result<Value, McpError> {
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| McpError::Transport("stdin closed".into()))?;

        let mut request = request.clone();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        request["id"] = json!(id);
        request["jsonrpc"] = json!("2.0");

        let line = serde_json::to_string(&request)?;
        writeln!(stdin, "{line}")?;
        stdin.flush()?;

        let stdout = self
            .child
            .stdout
            .as_mut()
            .ok_or_else(|| McpError::Transport("stdout closed".into()))?;

        let mut reader = BufReader::new(stdout);
        let mut response_line = String::new();
        reader.read_line(&mut response_line)?;

        let response: Value = serde_json::from_str(response_line.trim())?;

        // Check for JSON-RPC error
        if let Some(error) = response.get("error") {
            let code = error.get("code").and_then(Value::as_i64).unwrap_or(-1);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_string();
            return Err(McpError::JsonRpc { code, message });
        }

        Ok(response)
    }

    fn close(&mut self) -> Result<(), McpError> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// In-memory mock transport for testing.
pub struct MockTransport {
    responses: std::collections::VecDeque<Value>,
}

impl MockTransport {
    /// Create a mock transport with queued responses.
    #[must_use]
    pub fn new(responses: Vec<Value>) -> Self {
        Self {
            responses: responses.into(),
        }
    }
}

impl McpTransport for MockTransport {
    fn send(&mut self, _request: &Value) -> Result<Value, McpError> {
        self.responses
            .pop_front()
            .ok_or_else(|| McpError::Transport("no more mock responses".into()))
    }

    fn close(&mut self) -> Result<(), McpError> {
        Ok(())
    }
}

/// MCP client for interacting with MCP servers.
pub struct McpClient<T: McpTransport> {
    transport: T,
    server_info: Option<ServerInfo>,
    cached_tools: Option<Vec<McpTool>>,
}

impl<T: McpTransport> McpClient<T> {
    /// Create a new MCP client with the given transport.
    ///
    /// Automatically sends the `initialize` request.
    pub fn new(transport: T) -> Result<Self, McpError> {
        let mut client = Self {
            transport,
            server_info: None,
            cached_tools: None,
        };
        client.initialize()?;
        Ok(client)
    }

    /// Create without auto-initialization (for testing).
    #[must_use]
    pub fn new_uninitialized(transport: T) -> Self {
        Self {
            transport,
            server_info: None,
            cached_tools: None,
        }
    }

    /// Send a JSON-RPC request and check for errors.
    fn send_request(&mut self, request: &Value) -> Result<Value, McpError> {
        let response = self.transport.send(request)?;

        // Check for JSON-RPC error in response
        if let Some(error) = response.get("error") {
            let code = error.get("code").and_then(Value::as_i64).unwrap_or(-1);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
                .to_string();
            return Err(McpError::JsonRpc { code, message });
        }

        Ok(response)
    }

    /// Extract the `result` field from a JSON-RPC response.
    fn extract_result(&mut self, request: &Value) -> Result<Value, McpError> {
        let response = self.send_request(request)?;
        response
            .get("result")
            .cloned()
            .ok_or_else(|| McpError::UnexpectedResponse("missing result".into()))
    }

    /// Initialize the MCP connection.
    pub fn initialize(&mut self) -> Result<ServerInfo, McpError> {
        let request = json!({
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "claw-toolkit",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        });

        let result = self.extract_result(&request)?;

        let info: ServerInfo = serde_json::from_value(
            result
                .get("serverInfo")
                .cloned()
                .unwrap_or(json!({"name": "unknown"})),
        )?;

        self.server_info = Some(info.clone());

        // Send initialized notification (ignore errors)
        let notification = json!({
            "method": "notifications/initialized",
            "params": {}
        });
        let _ = self.transport.send(&notification);

        Ok(info)
    }

    /// List available tools from the server.
    pub fn list_tools(&mut self) -> Result<Vec<McpTool>, McpError> {
        let request = json!({
            "method": "tools/list",
            "params": {}
        });

        let result = self.extract_result(&request)?;
        let tools: Vec<McpTool> = serde_json::from_value(
            result.get("tools").cloned().unwrap_or(json!([])),
        )?;

        self.cached_tools = Some(tools.clone());
        Ok(tools)
    }

    /// Call a tool by name with the given arguments.
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<ToolCallResult, McpError> {
        let request = json!({
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        });

        let result = self.extract_result(&request)?;
        let call_result: ToolCallResult = serde_json::from_value(result)?;
        Ok(call_result)
    }

    /// List available resources from the server.
    pub fn list_resources(&mut self) -> Result<Vec<McpResource>, McpError> {
        let request = json!({
            "method": "resources/list",
            "params": {}
        });

        let result = self.extract_result(&request)?;
        let resources: Vec<McpResource> = serde_json::from_value(
            result.get("resources").cloned().unwrap_or(json!([])),
        )?;

        Ok(resources)
    }

    /// Read a resource by URI.
    pub fn read_resource(&mut self, uri: &str) -> Result<Vec<McpContent>, McpError> {
        let request = json!({
            "method": "resources/read",
            "params": {
                "uri": uri
            }
        });

        let result = self.extract_result(&request)?;
        let contents: Vec<McpContent> = serde_json::from_value(
            result.get("contents").cloned().unwrap_or(json!([])),
        )?;

        Ok(contents)
    }

    /// Close the transport connection.
    pub fn close(&mut self) -> Result<(), McpError> {
        self.transport.close()
    }

    /// Get server info (available after initialization).
    #[must_use]
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_init_response() -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": { "name": "test-server", "version": "1.0" },
                "capabilities": {}
            }
        })
    }

    fn mock_initialized_response() -> Value {
        json!({ "jsonrpc": "2.0", "id": 2, "result": {} })
    }

    #[test]
    fn initialize_and_list_tools() {
        let transport = MockTransport::new(vec![
            mock_init_response(),
            mock_initialized_response(),
            json!({
                "jsonrpc": "2.0", "id": 3,
                "result": {
                    "tools": [
                        { "name": "read_file", "description": "Read a file", "inputSchema": {} },
                        { "name": "search", "description": "Search code" }
                    ]
                }
            }),
        ]);

        let mut client = McpClient::new(transport).unwrap();
        assert_eq!(client.server_info().unwrap().name, "test-server");

        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
    }

    #[test]
    fn call_tool() {
        let transport = MockTransport::new(vec![
            mock_init_response(),
            mock_initialized_response(),
            json!({
                "jsonrpc": "2.0", "id": 3,
                "result": {
                    "content": [{ "type": "text", "text": "file contents here" }],
                    "isError": false
                }
            }),
        ]);

        let mut client = McpClient::new(transport).unwrap();
        let result = client
            .call_tool("read_file", json!({"path": "main.rs"}))
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content[0].text.as_deref(), Some("file contents here"));
    }

    #[test]
    fn list_resources() {
        let transport = MockTransport::new(vec![
            mock_init_response(),
            mock_initialized_response(),
            json!({
                "jsonrpc": "2.0", "id": 3,
                "result": {
                    "resources": [
                        { "uri": "file:///data.json", "name": "data.json", "mimeType": "application/json" }
                    ]
                }
            }),
        ]);

        let mut client = McpClient::new(transport).unwrap();
        let resources = client.list_resources().unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].uri, "file:///data.json");
    }

    #[test]
    fn handle_json_rpc_error() {
        let transport = MockTransport::new(vec![
            mock_init_response(),
            mock_initialized_response(),
            json!({
                "jsonrpc": "2.0", "id": 3,
                "error": { "code": -32601, "message": "Method not found" }
            }),
        ]);

        let mut client = McpClient::new(transport).unwrap();
        let result = client.list_tools();
        assert!(matches!(result, Err(McpError::JsonRpc { code: -32601, .. })));
    }
}
