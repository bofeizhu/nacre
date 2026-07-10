//! Python `str()`/`repr()` emulation for prompt interpolation.
//!
//! Upstream prompts interpolate some context values bare inside f-strings
//! (`f"{context['entity_types']}"`), which renders lists/dicts with Python
//! `repr` semantics: single-quote-preferring strings, `None`/`True`/`False`,
//! `", "` item separators, and dict insertion order. Recordings and golden
//! traces contain that exact text, so the port reproduces it.
//!
//! Only the value shapes that actually flow through prompt contexts are
//! covered (JSON-representable data); exotic `repr` corners (unprintable
//! unicode categories, float shortest-round-trip) are out of scope and
//! pinned by the prompt-fidelity fixtures.

use serde_json::Value;

/// Interpolate `value` the way a Python f-string does: strings pass through
/// unquoted (`str()` of a `str` is itself); everything else renders with
/// [`py_repr`].
pub fn py_interp(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Render `value` as Python `repr()` would.
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_str_repr(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, val)| format!("{}: {}", py_str_repr(key), py_repr(val)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python string `repr`: prefers single quotes, switches to double quotes
/// when the string contains `'` but no `"`.
fn py_str_repr(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let quote = if has_single && !has_double { '"' } else { '\'' };

    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalars_render_like_python() {
        assert_eq!(py_repr(&json!(null)), "None");
        assert_eq!(py_repr(&json!(true)), "True");
        assert_eq!(py_repr(&json!(42)), "42");
        assert_eq!(py_repr(&json!("plain")), "'plain'");
    }

    #[test]
    fn quote_preference_matches_python() {
        assert_eq!(py_repr(&json!("Nisha's dad")), r#""Nisha's dad""#);
        assert_eq!(py_repr(&json!(r#"say "hi""#)), r#"'say "hi"'"#);
        // Both quote kinds present: single quotes win, apostrophe escaped.
        assert_eq!(py_repr(&json!(r#"it's "big""#)), r#"'it\'s "big"'"#);
    }

    #[test]
    fn containers_render_in_insertion_order() {
        let value = json!({"entity_type_id": 0, "entity_type_name": "Entity", "a": [1, null]});
        assert_eq!(
            py_repr(&value),
            "{'entity_type_id': 0, 'entity_type_name': 'Entity', 'a': [1, None]}"
        );
    }

    #[test]
    fn interp_passes_strings_through_bare() {
        assert_eq!(py_interp(&json!("no quotes")), "no quotes");
        assert_eq!(py_interp(&json!(["x"])), "['x']");
    }

    #[test]
    fn control_chars_escape_like_python() {
        assert_eq!(py_repr(&json!("a\nb\tc\\d")), r"'a\nb\tc\\d'");
        assert_eq!(py_repr(&json!("bell\u{7}")), r"'bell\x07'");
    }
}
