//! Ports `graphiti_core/prompts/prompt_helpers.py`.

use std::io;

use serde::Serialize;
use serde_json::Value;

/// Appended to prompts by some providers to keep unicode readable.
// ports: graphiti_core/prompts/prompt_helpers.py::DO_NOT_ESCAPE_UNICODE
pub const DO_NOT_ESCAPE_UNICODE: &str = "\nDo not escape unicode characters.\n";

/// Serialize data to JSON for use in prompts, matching Python
/// `json.dumps(data, ensure_ascii=False)` byte-for-byte: `", "` and `": "`
/// separators, non-ASCII preserved, dict insertion order kept.
// ports: graphiti_core/prompts/prompt_helpers.py::to_prompt_json
pub fn to_prompt_json(data: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyDumpsFormatter);
    data.serialize(&mut ser)
        .expect("serializing a serde_json::Value is infallible");
    String::from_utf8(out).expect("serde_json emits UTF-8")
}

/// `serde_json` formatter reproducing Python `json.dumps` default
/// separators (`', '` between items, `': '` after keys).
struct PyDumpsFormatter;

impl serde_json::ser::Formatter for PyDumpsFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_python_json_dumps() {
        let value = json!([
            {"content": "Owen: Jordan još vodi radionicu — 陶芸クラス.", "timestamp": null},
            {"n": 3, "ok": true}
        ]);
        assert_eq!(
            to_prompt_json(&value),
            r#"[{"content": "Owen: Jordan još vodi radionicu — 陶芸クラス.", "timestamp": null}, {"n": 3, "ok": true}]"#
        );
    }

    #[test]
    fn strings_escape_like_python() {
        assert_eq!(to_prompt_json(&json!("a\nb \"q\"")), r#""a\nb \"q\"""#);
    }
}
