//! MCP Tool Registry — thread-safe registry for managing MCP server connections and tools.
//!
//! Tracks connection state, registered tools, and resources for each MCP server.
//! Provides a unified lookup interface for the agent loop to dispatch tool calls
//! to the correct server.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Status of a managed MCP server connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    AuthRequired,
    Error,
}

impl std::fmt::Display for McpConnectionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "disconnected"),
            Self::Connecting => write!(f, "connecting"),
            Self::Connected => write!(f, "connected"),
            Self::AuthRequired => write!(f, "auth_required"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Metadata about an MCP resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceInfo {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// Metadata about an MCP tool exposed by a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<Value>,
}

/// Tracked state of an MCP server connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerState {
    pub server_name: String,
    pub status: McpConnectionStatus,
    pub tools: Vec<McpToolInfo>,
    pub resources: Vec<McpResourceInfo>,
    pub server_info: Option<String>,
    pub error_message: Option<String>,
}

/// Thread-safe registry of MCP server connections and their tools/resources.
///
/// Designed to be shared across the agent loop and tool handlers via `Arc`.
#[derive(Debug, Clone, Default)]
pub struct McpToolRegistry {
    inner: Arc<Mutex<HashMap<String, McpServerState>>>,
}

impl McpToolRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or update a server in the registry.
    pub fn register_server(
        &self,
        server_name: &str,
        status: McpConnectionStatus,
        tools: Vec<McpToolInfo>,
        resources: Vec<McpResourceInfo>,
        server_info: Option<String>,
    ) {
        let mut inner = self.inner.lock().expect("registry lock poisoned");
        inner.insert(
            server_name.to_owned(),
            McpServerState {
                server_name: server_name.to_owned(),
                status,
                tools,
                resources,
                server_info,
                error_message: None,
            },
        );
    }

    /// Get the state of a specific server.
    pub fn get_server(&self, server_name: &str) -> Option<McpServerState> {
        let inner = self.inner.lock().expect("registry lock poisoned");
        inner.get(server_name).cloned()
    }

    /// List all registered servers.
    pub fn list_servers(&self) -> Vec<McpServerState> {
        let inner = self.inner.lock().expect("registry lock poisoned");
        inner.values().cloned().collect()
    }

    /// List resources from a connected server.
    pub fn list_resources(&self, server_name: &str) -> Result<Vec<McpResourceInfo>, String> {
        let inner = self.inner.lock().expect("registry lock poisoned");
        match inner.get(server_name) {
            Some(state) => {
                if state.status != McpConnectionStatus::Connected {
                    return Err(format!(
                        "server '{}' is not connected (status: {})",
                        server_name, state.status
                    ));
                }
                Ok(state.resources.clone())
            }
            None => Err(format!("server '{}' not found", server_name)),
        }
    }

    /// Read a specific resource by URI from a connected server.
    pub fn read_resource(&self, server_name: &str, uri: &str) -> Result<McpResourceInfo, String> {
        let inner = self.inner.lock().expect("registry lock poisoned");
        let state = inner
            .get(server_name)
            .ok_or_else(|| format!("server '{}' not found", server_name))?;

        if state.status != McpConnectionStatus::Connected {
            return Err(format!(
                "server '{}' is not connected (status: {})",
                server_name, state.status
            ));
        }

        state
            .resources
            .iter()
            .find(|r| r.uri == uri)
            .cloned()
            .ok_or_else(|| format!("resource '{}' not found on server '{}'", uri, server_name))
    }

    /// List tools from a connected server.
    pub fn list_tools(&self, server_name: &str) -> Result<Vec<McpToolInfo>, String> {
        let inner = self.inner.lock().expect("registry lock poisoned");
        match inner.get(server_name) {
            Some(state) => {
                if state.status != McpConnectionStatus::Connected {
                    return Err(format!(
                        "server '{}' is not connected (status: {})",
                        server_name, state.status
                    ));
                }
                Ok(state.tools.clone())
            }
            None => Err(format!("server '{}' not found", server_name)),
        }
    }

    /// Update the connection status of a server.
    pub fn set_status(
        &self,
        server_name: &str,
        status: McpConnectionStatus,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock().expect("registry lock poisoned");
        let state = inner
            .get_mut(server_name)
            .ok_or_else(|| format!("server '{}' not found", server_name))?;
        state.status = status;
        Ok(())
    }

    /// Set an error message on a server.
    pub fn set_error(&self, server_name: &str, error: String) -> Result<(), String> {
        let mut inner = self.inner.lock().expect("registry lock poisoned");
        let state = inner
            .get_mut(server_name)
            .ok_or_else(|| format!("server '{}' not found", server_name))?;
        state.status = McpConnectionStatus::Error;
        state.error_message = Some(error);
        Ok(())
    }

    /// Remove a server from the registry.
    pub fn disconnect(&self, server_name: &str) -> Option<McpServerState> {
        let mut inner = self.inner.lock().expect("registry lock poisoned");
        inner.remove(server_name)
    }

    /// Number of registered servers.
    #[must_use]
    pub fn len(&self) -> usize {
        let inner = self.inner.lock().expect("registry lock poisoned");
        inner.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// MCP tool name normalization utilities
// ---------------------------------------------------------------------------

/// Normalize a server name for use as an MCP tool prefix.
///
/// Replaces special characters with underscores to produce a safe identifier.
#[must_use]
pub fn normalize_name(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .collect()
}

/// Build a qualified tool name: `mcp__{server}__{tool}`.
#[must_use]
pub fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "mcp__{}__{}",
        normalize_name(server_name),
        normalize_name(tool_name)
    )
}

/// Build a tool prefix for a server: `mcp__{server}__`.
#[must_use]
pub fn mcp_tool_prefix(server_name: &str) -> String {
    format!("mcp__{}__", normalize_name(server_name))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn register_and_retrieve_server() {
        let registry = McpToolRegistry::new();
        registry.register_server(
            "test-server",
            McpConnectionStatus::Connected,
            vec![McpToolInfo {
                name: "greet".into(),
                description: Some("Greet someone".into()),
                input_schema: None,
            }],
            vec![McpResourceInfo {
                uri: "res://data".into(),
                name: "Data".into(),
                description: None,
                mime_type: Some("application/json".into()),
            }],
            Some("TestServer v1.0".into()),
        );

        let server = registry.get_server("test-server").expect("should exist");
        assert_eq!(server.status, McpConnectionStatus::Connected);
        assert_eq!(server.tools.len(), 1);
        assert_eq!(server.resources.len(), 1);
    }

    #[test]
    fn list_resources_from_connected_server() {
        let registry = McpToolRegistry::new();
        registry.register_server(
            "srv",
            McpConnectionStatus::Connected,
            vec![],
            vec![McpResourceInfo {
                uri: "res://alpha".into(),
                name: "Alpha".into(),
                description: None,
                mime_type: None,
            }],
            None,
        );
        let resources = registry.list_resources("srv").expect("should succeed");
        assert_eq!(resources.len(), 1);
    }

    #[test]
    fn rejects_operations_on_disconnected_server() {
        let registry = McpToolRegistry::new();
        registry.register_server(
            "srv",
            McpConnectionStatus::Disconnected,
            vec![],
            vec![],
            None,
        );
        assert!(registry.list_resources("srv").is_err());
        assert!(registry.list_tools("srv").is_err());
    }

    #[test]
    fn rejects_operations_on_missing_server() {
        let registry = McpToolRegistry::new();
        assert!(registry.list_resources("missing").is_err());
        assert!(registry.read_resource("missing", "uri").is_err());
        assert!(registry.list_tools("missing").is_err());
    }

    #[test]
    fn set_status_and_disconnect() {
        let registry = McpToolRegistry::new();
        registry.register_server(
            "srv",
            McpConnectionStatus::AuthRequired,
            vec![],
            vec![],
            None,
        );
        registry
            .set_status("srv", McpConnectionStatus::Connected)
            .unwrap();
        assert_eq!(
            registry.get_server("srv").unwrap().status,
            McpConnectionStatus::Connected
        );
        let removed = registry.disconnect("srv");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[test]
    fn set_error_updates_status() {
        let registry = McpToolRegistry::new();
        registry.register_server(
            "srv",
            McpConnectionStatus::Connected,
            vec![],
            vec![],
            None,
        );
        registry.set_error("srv", "connection lost".into()).unwrap();
        let state = registry.get_server("srv").unwrap();
        assert_eq!(state.status, McpConnectionStatus::Error);
        assert_eq!(state.error_message.as_deref(), Some("connection lost"));
    }

    #[test]
    fn list_servers_returns_all() {
        let registry = McpToolRegistry::new();
        registry.register_server("a", McpConnectionStatus::Connected, vec![], vec![], None);
        registry.register_server("b", McpConnectionStatus::Connecting, vec![], vec![], None);
        assert_eq!(registry.list_servers().len(), 2);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn list_tools_from_connected_server() {
        let registry = McpToolRegistry::new();
        registry.register_server(
            "srv",
            McpConnectionStatus::Connected,
            vec![McpToolInfo {
                name: "inspect".into(),
                description: Some("Inspect data".into()),
                input_schema: Some(json!({"type": "object"})),
            }],
            vec![],
            None,
        );
        let tools = registry.list_tools("srv").unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "inspect");
    }

    #[test]
    fn connection_status_display() {
        assert_eq!(McpConnectionStatus::Connected.to_string(), "connected");
        assert_eq!(McpConnectionStatus::AuthRequired.to_string(), "auth_required");
        assert_eq!(McpConnectionStatus::Error.to_string(), "error");
    }

    #[test]
    fn normalize_and_tool_names() {
        assert_eq!(normalize_name("github.com"), "github_com");
        assert_eq!(mcp_tool_name("my-server", "search"), "mcp__my-server__search");
        assert_eq!(mcp_tool_prefix("srv"), "mcp__srv__");
    }
}
