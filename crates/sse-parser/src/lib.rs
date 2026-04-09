//! # sse-parser
//!
//! Incremental Server-Sent Events (SSE) parser optimized for LLM API streaming.
//!
//! Unlike naive line-based parsers, this implementation correctly handles:
//! - Events split across multiple network chunks
//! - Multi-line `data:` fields
//! - All SSE field types (`data`, `event`, `id`, `retry`)
//! - Comment lines (`:` prefix)
//!
//! # Example
//!
//! ```
//! use sse_parser::SseParser;
//!
//! let mut parser = SseParser::new();
//!
//! // Simulate chunked network data
//! let events = parser.feed("data: {\"text\": \"hel");
//! assert!(events.is_empty()); // incomplete event
//!
//! let events = parser.feed("lo\"}\n\n");
//! assert_eq!(events.len(), 1);
//! assert_eq!(events[0].data, "{\"text\": \"hello\"}");
//! ```

/// A single parsed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The event type. Defaults to `"message"` if no `event:` field was present.
    pub event_type: String,
    /// The accumulated `data:` field content. Multiple `data:` lines are joined
    /// with newlines.
    pub data: String,
    /// The `id:` field value, if present.
    pub id: Option<String>,
    /// The `retry:` field value in milliseconds, if present.
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Returns `true` if this is the special `[DONE]` sentinel used by many
    /// LLM APIs to signal the end of a stream.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

/// Incremental SSE parser that correctly handles events split across chunk
/// boundaries.
///
/// Feed raw bytes/strings from the network into [`SseParser::feed`] and collect
/// fully parsed [`SseEvent`]s as they become available.
#[derive(Debug, Clone)]
pub struct SseParser {
    buffer: String,
    current_data: Vec<String>,
    current_event_type: Option<String>,
    current_id: Option<String>,
    current_retry: Option<u64>,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    /// Create a new parser with an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_data: Vec::new(),
            current_event_type: None,
            current_id: None,
            current_retry: None,
        }
    }

    /// Feed a chunk of raw SSE text into the parser.
    ///
    /// Returns a `Vec` of fully parsed events. The vec may be empty if the
    /// chunk did not complete any events (e.g., the data was split mid-line).
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        loop {
            // Look for the next line boundary
            let newline_pos = match self.buffer.find('\n') {
                Some(pos) => pos,
                None => break, // incomplete line, wait for more data
            };

            let line = self.buffer[..newline_pos].trim_end_matches('\r').to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                // Empty line = event dispatch
                if let Some(event) = self.dispatch_event() {
                    events.push(event);
                }
            } else {
                self.process_field(&line);
            }
        }

        events
    }

    /// Reset the parser state, discarding any buffered partial data.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_data.clear();
        self.current_event_type = None;
        self.current_id = None;
        self.current_retry = None;
    }

    fn process_field(&mut self, line: &str) {
        // Comment lines start with ':'
        if line.starts_with(':') {
            return;
        }

        let (field, value) = match line.find(':') {
            Some(pos) => {
                let field = &line[..pos];
                let value = line[pos + 1..].strip_prefix(' ').unwrap_or(&line[pos + 1..]);
                (field, value)
            }
            None => (line.as_ref(), ""),
        };

        match field {
            "data" => self.current_data.push(value.to_string()),
            "event" => self.current_event_type = Some(value.to_string()),
            "id" => self.current_id = Some(value.to_string()),
            "retry" => {
                if let Ok(ms) = value.trim().parse::<u64>() {
                    self.current_retry = Some(ms);
                }
            }
            _ => {} // Unknown fields are ignored per spec
        }
    }

    fn dispatch_event(&mut self) -> Option<SseEvent> {
        if self.current_data.is_empty() {
            // No data accumulated — reset fields but don't emit
            self.current_event_type = None;
            self.current_id = None;
            self.current_retry = None;
            return None;
        }

        let event = SseEvent {
            event_type: self
                .current_event_type
                .take()
                .unwrap_or_else(|| "message".to_string()),
            data: std::mem::take(&mut self.current_data).join("\n"),
            id: self.current_id.take(),
            retry: self.current_retry.take(),
        };

        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello world\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello world");
        assert_eq!(events[0].event_type, "message");
    }

    #[test]
    fn parse_event_with_type() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: content_block_delta\ndata: {\"delta\":\"hi\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "content_block_delta");
        assert_eq!(events[0].data, "{\"delta\":\"hi\"}");
    }

    #[test]
    fn handle_cross_chunk_boundaries() {
        let mut parser = SseParser::new();

        let events = parser.feed("data: hel");
        assert!(events.is_empty());

        let events = parser.feed("lo\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn handle_multi_line_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\ndata: line3\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn handle_multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: first\n\ndata: second\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn skip_comment_lines() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\ndata: actual\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "actual");
    }

    #[test]
    fn parse_id_and_retry() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 42\nretry: 3000\ndata: with metadata\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, Some("42".to_string()));
        assert_eq!(events[0].retry, Some(3000));
    }

    #[test]
    fn detect_done_sentinel() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert!(events[0].is_done());
    }

    #[test]
    fn empty_data_dispatch_is_skipped() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: ping\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn field_without_value() {
        let mut parser = SseParser::new();
        let events = parser.feed("data\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: crlf\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "crlf");
    }

    #[test]
    fn reset_clears_state() {
        let mut parser = SseParser::new();
        parser.feed("data: partial");
        parser.reset();
        let events = parser.feed("data: fresh\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "fresh");
    }
}
