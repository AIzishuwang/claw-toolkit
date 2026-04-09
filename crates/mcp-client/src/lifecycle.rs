//! MCP Lifecycle — state machine for MCP server connection phases.
//!
//! Tracks the full lifecycle of an MCP server connection:
//! `ConfigLoad → ServerRegistration → SpawnConnect → InitializeHandshake →
//!  ToolDiscovery → ResourceDiscovery → Ready → Invocation → Shutdown → Cleanup`
//!
//! Validates phase transitions, records errors with context, and produces
//! degraded-mode reports when some servers fail while others succeed.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Phases in the MCP server connection lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpLifecyclePhase {
    ConfigLoad,
    ServerRegistration,
    SpawnConnect,
    InitializeHandshake,
    ToolDiscovery,
    ResourceDiscovery,
    Ready,
    Invocation,
    ErrorSurfacing,
    Shutdown,
    Cleanup,
}

impl McpLifecyclePhase {
    /// All phases in lifecycle order.
    #[must_use]
    pub fn all() -> [Self; 11] {
        [
            Self::ConfigLoad,
            Self::ServerRegistration,
            Self::SpawnConnect,
            Self::InitializeHandshake,
            Self::ToolDiscovery,
            Self::ResourceDiscovery,
            Self::Ready,
            Self::Invocation,
            Self::ErrorSurfacing,
            Self::Shutdown,
            Self::Cleanup,
        ]
    }
}

impl std::fmt::Display for McpLifecyclePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigLoad => write!(f, "config_load"),
            Self::ServerRegistration => write!(f, "server_registration"),
            Self::SpawnConnect => write!(f, "spawn_connect"),
            Self::InitializeHandshake => write!(f, "initialize_handshake"),
            Self::ToolDiscovery => write!(f, "tool_discovery"),
            Self::ResourceDiscovery => write!(f, "resource_discovery"),
            Self::Ready => write!(f, "ready"),
            Self::Invocation => write!(f, "invocation"),
            Self::ErrorSurfacing => write!(f, "error_surfacing"),
            Self::Shutdown => write!(f, "shutdown"),
            Self::Cleanup => write!(f, "cleanup"),
        }
    }
}

/// Structured error with lifecycle context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpErrorSurface {
    pub phase: McpLifecyclePhase,
    pub server_name: Option<String>,
    pub message: String,
    pub context: BTreeMap<String, String>,
    pub recoverable: bool,
    pub timestamp: u64,
}

impl McpErrorSurface {
    #[must_use]
    pub fn new(
        phase: McpLifecyclePhase,
        server_name: Option<String>,
        message: impl Into<String>,
        context: BTreeMap<String, String>,
        recoverable: bool,
    ) -> Self {
        Self {
            phase,
            server_name,
            message: message.into(),
            context,
            recoverable,
            timestamp: now_secs(),
        }
    }
}

impl std::fmt::Display for McpErrorSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MCP lifecycle error during {}: {}",
            self.phase, self.message
        )?;
        if let Some(server_name) = &self.server_name {
            write!(f, " (server: {server_name})")?;
        }
        if self.recoverable {
            write!(f, " [recoverable]")?;
        }
        Ok(())
    }
}

impl std::error::Error for McpErrorSurface {}

/// Result of running a lifecycle phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpPhaseResult {
    Success {
        phase: McpLifecyclePhase,
        duration: Duration,
    },
    Failure {
        phase: McpLifecyclePhase,
        error: McpErrorSurface,
    },
    Timeout {
        phase: McpLifecyclePhase,
        waited: Duration,
        error: McpErrorSurface,
    },
}

impl McpPhaseResult {
    #[must_use]
    pub fn phase(&self) -> McpLifecyclePhase {
        match self {
            Self::Success { phase, .. }
            | Self::Failure { phase, .. }
            | Self::Timeout { phase, .. } => *phase,
        }
    }

    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }
}

/// A server that failed during lifecycle progression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpFailedServer {
    pub server_name: String,
    pub phase: McpLifecyclePhase,
    pub error: McpErrorSurface,
}

/// Report generated when some MCP servers fail while others succeed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpDegradedReport {
    pub working_servers: Vec<String>,
    pub failed_servers: Vec<McpFailedServer>,
    pub available_tools: Vec<String>,
    pub missing_tools: Vec<String>,
}

impl McpDegradedReport {
    #[must_use]
    pub fn new(
        working_servers: Vec<String>,
        failed_servers: Vec<McpFailedServer>,
        available_tools: Vec<String>,
        expected_tools: Vec<String>,
    ) -> Self {
        let working_servers = dedupe_sorted(working_servers);
        let available_tools = dedupe_sorted(available_tools);
        let available_set: BTreeSet<_> = available_tools.iter().cloned().collect();
        let expected = dedupe_sorted(expected_tools);
        let missing_tools = expected
            .into_iter()
            .filter(|t| !available_set.contains(t))
            .collect();

        Self {
            working_servers,
            failed_servers,
            available_tools,
            missing_tools,
        }
    }
}

/// Lifecycle state tracker for a single MCP server connection.
#[derive(Debug, Clone, Default)]
pub struct McpLifecycleState {
    current_phase: Option<McpLifecyclePhase>,
    phase_errors: BTreeMap<McpLifecyclePhase, Vec<McpErrorSurface>>,
    phase_timestamps: BTreeMap<McpLifecyclePhase, u64>,
    phase_results: Vec<McpPhaseResult>,
}

impl McpLifecycleState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn current_phase(&self) -> Option<McpLifecyclePhase> {
        self.current_phase
    }

    #[must_use]
    pub fn errors_for_phase(&self, phase: McpLifecyclePhase) -> &[McpErrorSurface] {
        self.phase_errors
            .get(&phase)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    #[must_use]
    pub fn results(&self) -> &[McpPhaseResult] {
        &self.phase_results
    }

    #[must_use]
    pub fn phase_timestamp(&self, phase: McpLifecyclePhase) -> Option<u64> {
        self.phase_timestamps.get(&phase).copied()
    }

    fn record_phase(&mut self, phase: McpLifecyclePhase) {
        self.current_phase = Some(phase);
        self.phase_timestamps.insert(phase, now_secs());
    }

    fn record_error(&mut self, error: McpErrorSurface) {
        self.phase_errors
            .entry(error.phase)
            .or_default()
            .push(error);
    }

    fn record_result(&mut self, result: McpPhaseResult) {
        self.phase_results.push(result);
    }

    fn can_resume_after_error(&self) -> bool {
        match self.phase_results.last() {
            Some(McpPhaseResult::Failure { error, .. } | McpPhaseResult::Timeout { error, .. }) => {
                error.recoverable
            }
            _ => false,
        }
    }
}

/// Validates and enforces MCP lifecycle phase transitions.
#[derive(Debug, Clone, Default)]
pub struct McpLifecycleValidator {
    state: McpLifecycleState,
}

impl McpLifecycleValidator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn state(&self) -> &McpLifecycleState {
        &self.state
    }

    /// Check whether a transition from `from` to `to` is valid.
    #[must_use]
    pub fn validate_phase_transition(from: McpLifecyclePhase, to: McpLifecyclePhase) -> bool {
        match (from, to) {
            (McpLifecyclePhase::ConfigLoad, McpLifecyclePhase::ServerRegistration)
            | (McpLifecyclePhase::ServerRegistration, McpLifecyclePhase::SpawnConnect)
            | (McpLifecyclePhase::SpawnConnect, McpLifecyclePhase::InitializeHandshake)
            | (McpLifecyclePhase::InitializeHandshake, McpLifecyclePhase::ToolDiscovery)
            | (McpLifecyclePhase::ToolDiscovery, McpLifecyclePhase::ResourceDiscovery)
            | (McpLifecyclePhase::ToolDiscovery, McpLifecyclePhase::Ready)
            | (McpLifecyclePhase::ResourceDiscovery, McpLifecyclePhase::Ready)
            | (McpLifecyclePhase::Ready, McpLifecyclePhase::Invocation)
            | (McpLifecyclePhase::Invocation, McpLifecyclePhase::Ready)
            | (McpLifecyclePhase::ErrorSurfacing, McpLifecyclePhase::Ready)
            | (McpLifecyclePhase::ErrorSurfacing, McpLifecyclePhase::Shutdown)
            | (McpLifecyclePhase::Shutdown, McpLifecyclePhase::Cleanup) => true,
            (_, McpLifecyclePhase::Shutdown) => from != McpLifecyclePhase::Cleanup,
            (_, McpLifecyclePhase::ErrorSurfacing) => {
                from != McpLifecyclePhase::Cleanup && from != McpLifecyclePhase::Shutdown
            }
            _ => false,
        }
    }

    /// Attempt to transition to `phase`. Returns success or a structured failure.
    pub fn run_phase(&mut self, phase: McpLifecyclePhase) -> McpPhaseResult {
        let started = Instant::now();

        if let Some(current) = self.state.current_phase() {
            if current == McpLifecyclePhase::ErrorSurfacing
                && phase == McpLifecyclePhase::Ready
                && !self.state.can_resume_after_error()
            {
                return self.record_failure(McpErrorSurface::new(
                    phase,
                    None,
                    "cannot return to ready after a non-recoverable MCP lifecycle failure",
                    BTreeMap::from([
                        ("from".to_string(), current.to_string()),
                        ("to".to_string(), phase.to_string()),
                    ]),
                    false,
                ));
            }

            if !Self::validate_phase_transition(current, phase) {
                return self.record_failure(McpErrorSurface::new(
                    phase,
                    None,
                    format!("invalid MCP lifecycle transition from {current} to {phase}"),
                    BTreeMap::from([
                        ("from".to_string(), current.to_string()),
                        ("to".to_string(), phase.to_string()),
                    ]),
                    false,
                ));
            }
        } else if phase != McpLifecyclePhase::ConfigLoad {
            return self.record_failure(McpErrorSurface::new(
                phase,
                None,
                format!("invalid initial MCP lifecycle phase {phase}"),
                BTreeMap::from([("phase".to_string(), phase.to_string())]),
                false,
            ));
        }

        self.state.record_phase(phase);
        let result = McpPhaseResult::Success {
            phase,
            duration: started.elapsed(),
        };
        self.state.record_result(result.clone());
        result
    }

    /// Record an explicit failure at a given phase.
    pub fn record_failure(&mut self, error: McpErrorSurface) -> McpPhaseResult {
        let phase = error.phase;
        self.state.record_error(error.clone());
        self.state.record_phase(McpLifecyclePhase::ErrorSurfacing);
        let result = McpPhaseResult::Failure { phase, error };
        self.state.record_result(result.clone());
        result
    }

    /// Record a timeout at a given phase.
    pub fn record_timeout(
        &mut self,
        phase: McpLifecyclePhase,
        waited: Duration,
        server_name: Option<String>,
        mut context: BTreeMap<String, String>,
    ) -> McpPhaseResult {
        context.insert("waited_ms".to_string(), waited.as_millis().to_string());
        let error = McpErrorSurface::new(
            phase,
            server_name,
            format!(
                "MCP lifecycle phase {phase} timed out after {} ms",
                waited.as_millis()
            ),
            context,
            true,
        );
        self.state.record_error(error.clone());
        self.state.record_phase(McpLifecyclePhase::ErrorSurfacing);
        let result = McpPhaseResult::Timeout {
            phase,
            waited,
            error,
        };
        self.state.record_result(result.clone());
        result
    }
}

fn dedupe_sorted(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_lifecycle_succeeds() {
        let mut validator = McpLifecycleValidator::new();
        let phases = [
            McpLifecyclePhase::ConfigLoad,
            McpLifecyclePhase::ServerRegistration,
            McpLifecyclePhase::SpawnConnect,
            McpLifecyclePhase::InitializeHandshake,
            McpLifecyclePhase::ToolDiscovery,
            McpLifecyclePhase::ResourceDiscovery,
            McpLifecyclePhase::Ready,
            McpLifecyclePhase::Invocation,
            McpLifecyclePhase::Ready,
            McpLifecyclePhase::Shutdown,
            McpLifecyclePhase::Cleanup,
        ];
        for phase in phases {
            let result = validator.run_phase(phase);
            assert!(result.is_success(), "phase {phase} should succeed");
        }
        assert_eq!(
            validator.state().current_phase(),
            Some(McpLifecyclePhase::Cleanup)
        );
    }

    #[test]
    fn skip_resource_discovery_is_allowed() {
        let mut validator = McpLifecycleValidator::new();
        for phase in [
            McpLifecyclePhase::ConfigLoad,
            McpLifecyclePhase::ServerRegistration,
            McpLifecyclePhase::SpawnConnect,
            McpLifecyclePhase::InitializeHandshake,
            McpLifecyclePhase::ToolDiscovery,
        ] {
            assert!(validator.run_phase(phase).is_success());
        }
        // Skip ResourceDiscovery → Ready is allowed
        assert!(validator.run_phase(McpLifecyclePhase::Ready).is_success());
    }

    #[test]
    fn invalid_transition_produces_failure() {
        let mut validator = McpLifecycleValidator::new();
        assert!(validator.run_phase(McpLifecyclePhase::ConfigLoad).is_success());
        assert!(validator.run_phase(McpLifecyclePhase::ServerRegistration).is_success());

        let result = validator.run_phase(McpLifecyclePhase::Ready);
        assert!(!result.is_success());
        assert_eq!(
            validator.state().current_phase(),
            Some(McpLifecyclePhase::ErrorSurfacing)
        );
    }

    #[test]
    fn nonrecoverable_failure_blocks_resume() {
        let mut validator = McpLifecycleValidator::new();
        for phase in [
            McpLifecyclePhase::ConfigLoad,
            McpLifecyclePhase::ServerRegistration,
            McpLifecyclePhase::SpawnConnect,
            McpLifecyclePhase::InitializeHandshake,
            McpLifecyclePhase::ToolDiscovery,
            McpLifecyclePhase::Ready,
        ] {
            assert!(validator.run_phase(phase).is_success());
        }
        validator.record_failure(McpErrorSurface::new(
            McpLifecyclePhase::Invocation,
            Some("alpha".into()),
            "fatal",
            BTreeMap::new(),
            false, // NOT recoverable
        ));

        let result = validator.run_phase(McpLifecyclePhase::Ready);
        assert!(!result.is_success());
    }

    #[test]
    fn timeout_is_recoverable() {
        let mut validator = McpLifecycleValidator::new();
        let result = validator.record_timeout(
            McpLifecyclePhase::SpawnConnect,
            Duration::from_millis(250),
            Some("alpha".into()),
            BTreeMap::new(),
        );
        match result {
            McpPhaseResult::Timeout { error, .. } => {
                assert!(error.recoverable);
                assert_eq!(error.server_name.as_deref(), Some("alpha"));
            }
            _ => panic!("expected timeout"),
        }
    }

    #[test]
    fn degraded_report_tracks_missing_tools() {
        let report = McpDegradedReport::new(
            vec!["alpha".into(), "beta".into()],
            vec![McpFailedServer {
                server_name: "broken".into(),
                phase: McpLifecyclePhase::InitializeHandshake,
                error: McpErrorSurface::new(
                    McpLifecyclePhase::InitializeHandshake,
                    Some("broken".into()),
                    "failed",
                    BTreeMap::new(),
                    false,
                ),
            }],
            vec!["alpha.echo".into(), "beta.search".into()],
            vec!["alpha.echo".into(), "beta.search".into(), "broken.fetch".into()],
        );
        assert_eq!(report.missing_tools, vec!["broken.fetch".to_string()]);
        assert_eq!(report.working_servers.len(), 2);
    }

    #[test]
    fn error_surface_display() {
        let error = McpErrorSurface::new(
            McpLifecyclePhase::SpawnConnect,
            Some("alpha".into()),
            "process exited",
            BTreeMap::new(),
            true,
        );
        let rendered = error.to_string();
        assert!(rendered.contains("spawn_connect"));
        assert!(rendered.contains("alpha"));
        assert!(rendered.contains("recoverable"));
    }
}
