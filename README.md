# 🦀 Claw Toolkit

> AI Agent 开发工具库 — 从 Claude Code 核心架构提取的可复用 Rust 组件

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-75%2F75-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 简介

Claw Toolkit 是一个 Rust workspace，包含 **9 个独立可用的 crate**，封装了构建 AI Agent 系统所需的核心组件：

- 🔄 **Agent Loop** — 完整的 agentic 循环框架
- 🤖 **LLM Client** — 统一的 Anthropic / OpenAI 客户端
- 🔌 **MCP Client** — Model Context Protocol 客户端 SDK
- 🛡️ **Permission Engine** — 三级权限控制引擎
- 🪝 **Tool Hooks** — Pre/Post 工具执行钩子管道
- 📂 **Safe FS** — 防御性文件系统操作
- 💾 **Session Store** — 会话持久化与自动压缩
- 📡 **SSE Parser** — 增量 Server-Sent Events 解析器
- 🧪 **Agent Test** — Agent 行为测试框架

## 🏗️ 架构

```
claw-toolkit/
├── Cargo.toml                 # workspace root
└── crates/
    ├── sse-parser/            # 零依赖 SSE 解析
    ├── safe-fs/               # 沙箱文件操作
    ├── tool-hooks/            # 工具钩子管道
    ├── permission-engine/     # AI 权限引擎
    ├── session-store/         # 会话管理+压缩
    ├── llm-client/            # 统一 LLM 客户端 (依赖 sse-parser)
    ├── agent-loop/            # Agent 循环框架 (依赖以上全部)
    ├── mcp-client/            # MCP 协议客户端
    └── agent-test/            # Agent 测试框架 (依赖 agent-loop)
```

**依赖关系：**

```
agent-test ──► agent-loop ──► llm-client ──► sse-parser
                   │
                   ├──► permission-engine
                   ├──► tool-hooks
                   └──► session-store

mcp-client (独立)
safe-fs    (独立)
```

## 🚀 快速开始

### 构建

```bash
git clone https://github.com/YOUR_USER/claw-toolkit.git
cd claw-toolkit
cargo build
```

### 测试

```bash
cargo test          # 运行全部 75 个测试
cargo test -p sse-parser  # 单独测试某个 crate
```

### 文档

```bash
cargo doc --no-deps --open
```

## 📦 Crate 介绍

### `sse-parser` — SSE 流式解析器

增量解析 Server-Sent Events，正确处理跨网络 chunk 边界的事件。

```rust
use sse_parser::SseParser;

let mut parser = SseParser::new();

// 网络 chunk 1（不完整）
let events = parser.feed("data: hel");
assert!(events.is_empty());

// 网络 chunk 2（补全事件）
let events = parser.feed("lo\n\n");
assert_eq!(events[0].data, "hello");
```

### `llm-client` — 统一 LLM 客户端

支持 Anthropic Messages API 和 OpenAI 兼容 API，内置 SSE 流式解析。

```rust
use llm_client::{LlmClient, Message};

// Anthropic
let client = LlmClient::anthropic("sk-ant-...", "sonnet");
let (blocks, usage) = client.complete(&[Message::user("Hello!")], &[])?;

// OpenAI 兼容（如 ai.zishucode.top）
let client = LlmClient::openai_compat("https://ai.zishucode.top/v1", "sk-...", "gpt-4o");
```

### `permission-engine` — 三级权限引擎

ReadOnly → WorkspaceWrite → FullAccess 三级权限模型，支持自定义规则和交互式审批。

```rust
use permission_engine::{PermissionEngine, PermissionLevel, Rule};

let engine = PermissionEngine::new(PermissionLevel::WorkspaceWrite)
    .with_rule(Rule::deny_path_outside("/workspace"))
    .with_rule(Rule::deny_bash_pattern("rm -rf"));

let result = engine.check("write_file", r#"{"path":"/etc/passwd"}"#);
assert!(!result.allowed); // ❌ 路径不在工作区
```

### `safe-fs` — 安全文件操作

沙箱化的文件系统，防止路径穿越、symlink 逃逸和二进制文件写入。

```rust
use safe_fs::SafeFs;

let fs = SafeFs::new("/workspace")?;
fs.read("src/main.rs")?;            // ✅ 正常读取
fs.read("../../etc/passwd")?;        // ❌ PathTraversal
fs.write("data.bin", "\x00binary")?; // ❌ BinaryContent
fs.grep("TODO", None, false)?;      // ✅ 全文搜索
```

### `agent-loop` — Agent 循环框架

核心 agentic loop：User → LLM → Tool → LLM → ... 自动管理权限、钩子和会话压缩。

```rust
use agent_loop::{AgentLoop, AgentConfig, Tool};
use llm_client::MockProvider;

let provider = MockProvider::text("Hello!");
let mut agent = AgentLoop::new(provider, vec![], AgentConfig::default());
let result = agent.run("Say hello")?;
println!("{}", result.final_text()); // "Hello!"
```

### `tool-hooks` — 工具钩子系统

Pre/Post 钩子管道，支持拦截、修改和审计工具调用。

```rust
use tool_hooks::{HookPipeline, HookAction};

let mut pipeline = HookPipeline::new();
pipeline.pre("bash", |ctx| {
    if ctx.input.contains("rm -rf") {
        HookAction::Deny("dangerous command blocked".into())
    } else {
        HookAction::Continue
    }
});
```

### `mcp-client` — MCP 客户端 SDK

通过 Stdio 子进程与 MCP Server 通信，支持 tools/resources。

```rust
use mcp_client::{McpClient, StdioTransport};

let transport = StdioTransport::new("npx", &["-y", "@mcp/server-github"])?;
let mut client = McpClient::new(transport)?;
let tools = client.list_tools()?;
let result = client.call_tool("search_repos", json!({"query": "rust"}))?;
```

### `agent-test` — Agent 测试框架

场景化测试 Agent 行为，使用 Mock LLM 确保确定性。

```rust
use agent_test::{Scenario, Assertion, TestHarness};
use llm_client::StreamEvent;

let scenario = Scenario::new("echo_works")
    .with_tool(EchoTool)
    .mock_response(vec![
        StreamEvent::ToolUse { id: "t1".into(), name: "echo".into(), input: json!({"text": "hi"}) },
        StreamEvent::Done,
    ])
    .mock_response(vec![StreamEvent::Text("Done".into()), StreamEvent::Done])
    .assert(Assertion::ToolCalled("echo"))
    .assert(Assertion::ToolPermitted("echo"));

let results = TestHarness::new(vec![scenario]).run_all();
assert!(results.all_passed());
```

## 📊 项目统计

| Crate | 代码行数 | 测试数 |
|-------|---------|-------|
| sse-parser | 283 | 12 |
| safe-fs | 646 | 12 |
| tool-hooks | 385 | 8 |
| permission-engine | 392 | 8 |
| session-store | 508 | 6+1 |
| llm-client | 604 | 5+1 |
| agent-loop | 562 | 5+1 |
| mcp-client | 493 | 4+1 |
| agent-test | 547 | 6+1 |
| **Total** | **4,409** | **75** |

## 📄 License

MIT
