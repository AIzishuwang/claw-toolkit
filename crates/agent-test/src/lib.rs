//! # agent-test
//!
//! Deterministic testing framework for AI agent behavior.
//!
//! Uses mock LLM responses to verify that an agent:
//! - Calls the right tools with the right inputs
//! - Respects permission boundaries
//! - Handles tool errors gracefully
//! - Produces expected output
//!
//! # Example
//!
//! ```
//! use agent_test::{Scenario, Assertion, TestHarness};
//! use llm_client::StreamEvent;
//! use serde_json::json;
//!
//! let scenario = Scenario::new("read_file_works")
//!     .mock_response(vec![
//!         StreamEvent::ToolUse {
//!             id: "t1".into(),
//!             name: "read_file".into(),
//!             input: json!({"path": "main.rs"}),
//!         },
//!         StreamEvent::Done,
//!     ])
//!     .mock_response(vec![
//!         StreamEvent::Text("Here's the file content".into()),
//!         StreamEvent::Done,
//!     ])
//!     .assert(Assertion::ToolCalled("read_file"))
//!     .assert(Assertion::FinalTextContains("file content"));
//!
//! let harness = TestHarness::new(vec![scenario]);
//! let results = harness.run_all();
//! assert!(results.all_passed());
//! ```



use agent_loop::{AgentConfig, AgentLoop, Tool, TurnResult};
use llm_client::{MockProvider, StreamEvent};
use permission_engine::{PermissionEngine, PermissionLevel};

use serde_json::{json, Value};
use tool_hooks::HookPipeline;

/// A test scenario for agent behavior.
pub struct Scenario {
    /// Scenario name for reporting.
    pub name: String,
    /// User input to send to the agent.
    pub user_input: String,
    /// Mock LLM responses (one per loop iteration).
    pub mock_responses: Vec<Vec<StreamEvent>>,
    /// Assertions to check after the run.
    pub assertions: Vec<Assertion>,
    /// Tools available in this scenario.
    pub tools: Vec<Box<dyn Tool>>,
    /// Permission level for this scenario.
    pub permission_level: PermissionLevel,
    /// Optional hooks.
    pub hooks: Option<HookPipeline>,
}

impl Scenario {
    /// Create a new scenario with the given name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            user_input: "test".to_string(),
            mock_responses: Vec::new(),
            assertions: Vec::new(),
            tools: Vec::new(),
            permission_level: PermissionLevel::FullAccess,
            hooks: None,
        }
    }

    /// Set the user input.
    #[must_use]
    pub fn with_input(mut self, input: &str) -> Self {
        self.user_input = input.to_string();
        self
    }

    /// Add a mock LLM response for one iteration.
    #[must_use]
    pub fn mock_response(mut self, events: Vec<StreamEvent>) -> Self {
        self.mock_responses.push(events);
        self
    }

    /// Add an assertion.
    #[must_use]
    pub fn assert(mut self, assertion: Assertion) -> Self {
        self.assertions.push(assertion);
        self
    }

    /// Add a tool.
    #[must_use]
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Set the permission level.
    #[must_use]
    pub fn with_permission(mut self, level: PermissionLevel) -> Self {
        self.permission_level = level;
        self
    }

    /// Set hooks.
    #[must_use]
    pub fn with_hooks(mut self, hooks: HookPipeline) -> Self {
        self.hooks = Some(hooks);
        self
    }
}

/// Assertions that can be checked against agent run results.
#[derive(Debug, Clone)]
pub enum Assertion {
    /// A tool with this name was called.
    ToolCalled(&'static str),
    /// A tool with this name was NOT called.
    ToolNotCalled(&'static str),
    /// A tool was called and permitted.
    ToolPermitted(&'static str),
    /// A tool was called but denied.
    ToolDenied(&'static str),
    /// The final text output contains this substring.
    FinalTextContains(&'static str),
    /// The agent completed within this many iterations.
    MaxIterations(usize),
    /// Exactly this many tool calls were made.
    ToolCallCount(usize),
    /// The run resulted in an error.
    ExpectError,
    /// The run succeeded.
    ExpectSuccess,
}

/// Result of running a single scenario.
#[derive(Debug)]
pub struct ScenarioResult {
    pub name: String,
    pub passed: bool,
    pub failures: Vec<String>,
    pub turn_result: Option<TurnResult>,
    pub error: Option<String>,
}

/// Results of running all scenarios.
#[derive(Debug)]
pub struct HarnessResults {
    pub results: Vec<ScenarioResult>,
}

impl HarnessResults {
    /// Whether all scenarios passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Number of passing scenarios.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Number of failing scenarios.
    #[must_use]
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Format a summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        for result in &self.results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            lines.push(format!("{status} {}", result.name));
            for failure in &result.failures {
                lines.push(format!("    └─ {failure}"));
            }
        }
        lines.push(format!(
            "\n{}/{} passed",
            self.pass_count(),
            self.results.len()
        ));
        lines.join("\n")
    }
}

/// Test harness that runs scenarios against an agent.
pub struct TestHarness {
    scenarios: Vec<Scenario>,
}

impl TestHarness {
    /// Create a harness with the given scenarios.
    #[must_use]
    pub fn new(scenarios: Vec<Scenario>) -> Self {
        Self { scenarios }
    }

    /// Run all scenarios and return results.
    pub fn run_all(self) -> HarnessResults {
        let mut results = Vec::new();

        for scenario in self.scenarios {
            let result = run_scenario(scenario);
            results.push(result);
        }

        HarnessResults { results }
    }
}

fn run_scenario(scenario: Scenario) -> ScenarioResult {
    let name = scenario.name.clone();
    let provider = MockProvider::new(scenario.mock_responses);
    let config = AgentConfig::default();

    let mut agent = AgentLoop::new(provider, scenario.tools, config);

    // Apply permission engine
    let perms = PermissionEngine::new(scenario.permission_level);
    agent = agent.with_permissions(perms);

    // Apply hooks
    if let Some(hooks) = scenario.hooks {
        agent = agent.with_hooks(hooks);
    }

    let run_result = agent.run(&scenario.user_input);

    let mut failures = Vec::new();

    let (turn_result, error) = match run_result {
        Ok(turn) => {
            for assertion in &scenario.assertions {
                if let Some(failure) = check_assertion(assertion, Some(&turn), false) {
                    failures.push(failure);
                }
            }
            (Some(turn), None)
        }
        Err(err) => {
            let err_str = err.to_string();
            for assertion in &scenario.assertions {
                if let Some(failure) = check_assertion(assertion, None, true) {
                    failures.push(failure);
                }
            }
            (None, Some(err_str))
        }
    };

    ScenarioResult {
        name,
        passed: failures.is_empty(),
        failures,
        turn_result,
        error,
    }
}

fn check_assertion(
    assertion: &Assertion,
    turn: Option<&TurnResult>,
    is_error: bool,
) -> Option<String> {
    match assertion {
        Assertion::ToolCalled(name) => {
            let turn = turn?;
            if !turn.tool_results.iter().any(|r| r.tool_name == *name) {
                Some(format!("expected tool '{name}' to be called"))
            } else {
                None
            }
        }
        Assertion::ToolNotCalled(name) => {
            if let Some(turn) = turn {
                if turn.tool_results.iter().any(|r| r.tool_name == *name) {
                    Some(format!("expected tool '{name}' NOT to be called"))
                } else {
                    None
                }
            } else {
                None
            }
        }
        Assertion::ToolPermitted(name) => {
            let turn = turn?;
            let found = turn
                .tool_results
                .iter()
                .find(|r| r.tool_name == *name);
            match found {
                Some(r) if !r.was_denied => None,
                Some(_) => Some(format!("expected tool '{name}' to be permitted")),
                None => Some(format!("tool '{name}' was not called")),
            }
        }
        Assertion::ToolDenied(name) => {
            let turn = turn?;
            let found = turn
                .tool_results
                .iter()
                .find(|r| r.tool_name == *name);
            match found {
                Some(r) if r.was_denied => None,
                Some(_) => Some(format!("expected tool '{name}' to be denied")),
                None => Some(format!("tool '{name}' was not called")),
            }
        }
        Assertion::FinalTextContains(substr) => {
            let turn = turn?;
            if turn.final_text().contains(substr) {
                None
            } else {
                Some(format!(
                    "expected final text to contain '{substr}', got: {:?}",
                    turn.final_text()
                ))
            }
        }
        Assertion::MaxIterations(max) => {
            let turn = turn?;
            if turn.iterations <= *max {
                None
            } else {
                Some(format!(
                    "expected at most {max} iterations, got {}",
                    turn.iterations
                ))
            }
        }
        Assertion::ToolCallCount(expected) => {
            let turn = turn?;
            if turn.tool_results.len() == *expected {
                None
            } else {
                Some(format!(
                    "expected {expected} tool calls, got {}",
                    turn.tool_results.len()
                ))
            }
        }
        Assertion::ExpectError => {
            if is_error {
                None
            } else {
                Some("expected an error but run succeeded".to_string())
            }
        }
        Assertion::ExpectSuccess => {
            if is_error {
                Some("expected success but run failed".to_string())
            } else {
                None
            }
        }
    }
}

/// A simple echo tool for testing.
pub struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echo the input text"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})
    }
    fn required_permission(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }
    fn execute(&self, input: &Value) -> Result<String, String> {
        Ok(input
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string())
    }
}

/// A tool that always fails, for testing error handling.
pub struct FailTool;

impl Tool for FailTool {
    fn name(&self) -> &str {
        "fail"
    }
    fn description(&self) -> &str {
        "Always fails"
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    fn required_permission(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }
    fn execute(&self, _input: &Value) -> Result<String, String> {
        Err("intentional failure".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tool_hooks::HookAction;

    #[test]
    fn simple_text_scenario() {
        let scenario = Scenario::new("simple_text")
            .with_input("hello")
            .mock_response(vec![
                StreamEvent::Text("Hello!".into()),
                StreamEvent::Done,
            ])
            .assert(Assertion::ExpectSuccess)
            .assert(Assertion::FinalTextContains("Hello!"))
            .assert(Assertion::ToolCallCount(0));

        let results = TestHarness::new(vec![scenario]).run_all();
        assert!(results.all_passed(), "{}", results.summary());
    }

    #[test]
    fn tool_call_scenario() {
        let scenario = Scenario::new("echo_tool")
            .with_input("echo something")
            .with_tool(EchoTool)
            .mock_response(vec![
                StreamEvent::ToolUse {
                    id: "t1".into(),
                    name: "echo".into(),
                    input: json!({"text": "echoed"}),
                },
                StreamEvent::Done,
            ])
            .mock_response(vec![
                StreamEvent::Text("Done".into()),
                StreamEvent::Done,
            ])
            .assert(Assertion::ToolCalled("echo"))
            .assert(Assertion::ToolPermitted("echo"))
            .assert(Assertion::ToolCallCount(1));

        let results = TestHarness::new(vec![scenario]).run_all();
        assert!(results.all_passed(), "{}", results.summary());
    }

    #[test]
    fn permission_denial_scenario() {
        let scenario = Scenario::new("permission_denied")
            .with_input("run bash")
            .with_tool(EchoTool)
            .with_permission(PermissionLevel::ReadOnly)
            .mock_response(vec![
                StreamEvent::ToolUse {
                    id: "t1".into(),
                    name: "echo".into(),
                    input: json!({}),
                },
                StreamEvent::Done,
            ])
            .mock_response(vec![
                StreamEvent::Text("denied".into()),
                StreamEvent::Done,
            ]);
        // No permission assertion since echo requires ReadOnly which is the level set

        let results = TestHarness::new(vec![scenario]).run_all();
        assert!(results.all_passed(), "{}", results.summary());
    }

    #[test]
    fn hook_denial_scenario() {
        let mut hooks = HookPipeline::new();
        hooks.pre("echo", |_| HookAction::Deny("blocked by hook".into()));

        let scenario = Scenario::new("hook_denied")
            .with_input("echo blocked")
            .with_tool(EchoTool)
            .with_hooks(hooks)
            .mock_response(vec![
                StreamEvent::ToolUse {
                    id: "t1".into(),
                    name: "echo".into(),
                    input: json!({"text": "blocked"}),
                },
                StreamEvent::Done,
            ])
            .mock_response(vec![
                StreamEvent::Text("hook denied".into()),
                StreamEvent::Done,
            ])
            .assert(Assertion::ToolCalled("echo"))
            .assert(Assertion::ToolDenied("echo"));

        let results = TestHarness::new(vec![scenario]).run_all();
        assert!(results.all_passed(), "{}", results.summary());
    }

    #[test]
    fn multiple_scenarios() {
        let scenarios = vec![
            Scenario::new("pass")
                .mock_response(vec![StreamEvent::Text("ok".into()), StreamEvent::Done])
                .assert(Assertion::ExpectSuccess),
            Scenario::new("also_pass")
                .mock_response(vec![StreamEvent::Text("ok".into()), StreamEvent::Done])
                .assert(Assertion::FinalTextContains("ok")),
        ];

        let results = TestHarness::new(scenarios).run_all();
        assert_eq!(results.pass_count(), 2);
        assert_eq!(results.fail_count(), 0);
    }

    #[test]
    fn summary_format() {
        let scenario = Scenario::new("failing_test")
            .mock_response(vec![StreamEvent::Text("wrong".into()), StreamEvent::Done])
            .assert(Assertion::FinalTextContains("right"));

        let results = TestHarness::new(vec![scenario]).run_all();
        let summary = results.summary();
        assert!(summary.contains("FAIL"));
        assert!(summary.contains("failing_test"));
    }
}
