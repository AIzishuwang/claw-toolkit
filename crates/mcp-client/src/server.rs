//! MCP Server — exposes tools to external MCP clients via JSON-RPC over stdio.
//!
//! Implements a minimal LSP-framed (Content-Length header) JSON-RPC 2.0 server
//! that answers `initialize`, `tools/list`, and `tools/call` requests.
//!
//! # Example
//!
//! ```no_run
//! use mcp_client::server::{McpServer, McpServerSpec, McpServerTool};
//! use serde_json::json;
//!
//! let spec = McpServerSpec {
//!     server_name: "my-agent".into(),
//!     server_version: "1.0.0".into(),
//!     tools: vec![McpServerTool {
//!         name: "greet".into(),
//!         description: Some("Say hello".into()),
//!         input_schema: Some(json!({"type": "object", "properties": {"name": {"type": "string"}}})),
//!     }],
//!     tool_handler: Box::new(|name, args| {
//!         Ok(format!("Hello from {name}: {args}"))
//!     }),
//! };
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Protocol version the server advertises during `initialize`.
pub const MCP_SERVER_PROTOCOL_VERSION: &str = "2025-03-26";

/// Synchronous handler invoked for every `tools/call` request.
///
/// Returning `Ok(text)` yields a single text content block with `isError: false`.
/// Returning `Err(message)` yields a text block with `isError: true`.
pub type ToolCallHandler =
    Box<dyn Fn(&str, &Value) -> Result<String, String> + Send + Sync + 'static>;

/// A tool descriptor exposed by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "inputSchema", skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
}

/// Configuration for an [`McpServer`] instance.
pub struct McpServerSpec {
    /// Name advertised in the `serverInfo` field of the `initialize` response.
    pub server_name: String,
    /// Version advertised in `serverInfo`.
    pub server_version: String,
    /// Tool descriptors returned for `tools/list`.
    pub tools: Vec<McpServerTool>,
    /// Handler invoked for `tools/call`.
    pub tool_handler: ToolCallHandler,
}

// ---------------------------------------------------------------------------
// JSON-RPC types (shared with client, but kept self-contained for the server)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum JsonRpcId {
    Number(u64),
    String(String),
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

// ---------------------------------------------------------------------------
// Server core — dispatch logic (no async I/O, testable in isolation)
// ---------------------------------------------------------------------------

/// Minimal MCP server that dispatches JSON-RPC requests.
///
/// The server does not manage I/O directly. Instead, call [`McpServer::dispatch`]
/// with incoming requests and send the response back through whatever transport
/// you use (stdio, HTTP, WebSocket, etc.).
pub struct McpServer {
    spec: McpServerSpec,
}

impl McpServer {
    /// Create a new MCP server with the given spec.
    #[must_use]
    pub fn new(spec: McpServerSpec) -> Self {
        Self { spec }
    }

    /// Dispatch a single JSON-RPC request and return the response.
    ///
    /// The server handles:
    /// - `initialize` — returns server info and capabilities
    /// - `tools/list` — returns registered tool descriptors
    /// - `tools/call` — invokes the tool handler
    /// - Unknown methods — returns `-32601 Method not found`
    #[must_use]
    pub fn dispatch(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();
        match request.method.as_str() {
            "initialize" => self.handle_initialize(id),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, request.params),
            other => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("method not found: {other}"),
                    data: None,
                }),
            },
        }
    }

    /// Parse a raw JSON string into a request and dispatch it.
    pub fn dispatch_raw(&self, raw: &str) -> Result<JsonRpcResponse, String> {
        let request: JsonRpcRequest =
            serde_json::from_str(raw).map_err(|e| format!("invalid JSON-RPC request: {e}"))?;
        Ok(self.dispatch(request))
    }

    fn handle_initialize(&self, id: JsonRpcId) -> JsonRpcResponse {
        let result = json!({
            "protocolVersion": MCP_SERVER_PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": self.spec.server_name,
                "version": self.spec.server_version,
            }
        });
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn handle_tools_list(&self, id: JsonRpcId) -> JsonRpcResponse {
        let tools = serde_json::to_value(&self.spec.tools).unwrap_or(json!([]));
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({ "tools": tools })),
            error: None,
        }
    }

    fn handle_tools_call(&self, id: JsonRpcId, params: Option<Value>) -> JsonRpcResponse {
        let Some(params) = params else {
            return invalid_params_response(id, "missing params for tools/call");
        };

        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let (text, is_error) = match (self.spec.tool_handler)(name, &arguments) {
            Ok(text) => (text, false),
            Err(message) => (message, true),
        };

        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({
                "content": [{ "type": "text", "text": text }],
                "isError": is_error,
            })),
            error: None,
        }
    }
}

fn invalid_params_response(id: JsonRpcId, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code: -32602,
            message: message.to_string(),
            data: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_server() -> McpServer {
        McpServer::new(McpServerSpec {
            server_name: "test".to_string(),
            server_version: "1.0.0".to_string(),
            tools: vec![McpServerTool {
                name: "echo".to_string(),
                description: Some("Echo tool".to_string()),
                input_schema: Some(json!({"type": "object"})),
            }],
            tool_handler: Box::new(|name, args| Ok(format!("called {name} with {args}"))),
        })
    }

    #[test]
    fn dispatch_initialize_returns_server_info() {
        let server = test_server();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(1),
            method: "initialize".to_string(),
            params: None,
        };
        let response = server.dispatch(request);
        assert_eq!(response.id, JsonRpcId::Number(1));
        assert!(response.error.is_none());
        let result = response.result.expect("initialize result");
        assert_eq!(result["protocolVersion"], MCP_SERVER_PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], "test");
    }

    #[test]
    fn dispatch_tools_list_returns_registered_tools() {
        let server = test_server();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(2),
            method: "tools/list".to_string(),
            params: None,
        };
        let response = server.dispatch(request);
        assert!(response.error.is_none());
        let result = response.result.expect("tools/list result");
        assert_eq!(result["tools"][0]["name"], "echo");
    }

    #[test]
    fn dispatch_tools_call_wraps_handler_output() {
        let server = test_server();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(3),
            method: "tools/call".to_string(),
            params: Some(json!({"name": "echo", "arguments": {"text": "hi"}})),
        };
        let response = server.dispatch(request);
        let result = response.result.expect("tools/call result");
        assert_eq!(result["isError"], false);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("called echo"));
    }

    #[test]
    fn dispatch_tools_call_surfaces_handler_error() {
        let server = McpServer::new(McpServerSpec {
            server_name: "test".to_string(),
            server_version: "0.0.0".to_string(),
            tools: Vec::new(),
            tool_handler: Box::new(|_, _| Err("boom".to_string())),
        });
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(4),
            method: "tools/call".to_string(),
            params: Some(json!({"name": "broken"})),
        };
        let response = server.dispatch(request);
        let result = response.result.expect("tools/call result");
        assert_eq!(result["isError"], true);
        assert_eq!(result["content"][0]["text"], "boom");
    }

    #[test]
    fn dispatch_unknown_method_returns_error() {
        let server = test_server();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(5),
            method: "nonsense".to_string(),
            params: None,
        };
        let response = server.dispatch(request);
        let error = response.error.expect("error payload");
        assert_eq!(error.code, -32601);
    }

    #[test]
    fn dispatch_raw_json() {
        let server = test_server();
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let response = server.dispatch_raw(raw).unwrap();
        assert!(response.error.is_none());
        assert!(response.result.is_some());
    }
}
