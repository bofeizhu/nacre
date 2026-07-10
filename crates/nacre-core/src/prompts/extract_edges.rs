//! Ports `graphiti_core/prompts/extract_edges.py` (pinned v0.29.2).
//!
//! Prompt text is byte-verbatim; fidelity is enforced against fixtures
//! rendered from the actual upstream Python — see `tests/prompt_fidelity.rs`.

use serde_json::Value;

use super::helpers::to_prompt_json;
use super::msg;
use super::py::py_interp;
use crate::model::{Message, Role};

// ports: graphiti_core/prompts/extract_edges.py::edge
pub fn edge(context: &Value) -> Vec<Message> {
    let mut edge_types_section = String::new();
    if let Some(edge_types) = context.get("edge_types") {
        // Python truthiness: absent, null, or an empty list produce no section.
        let truthy = match edge_types {
            Value::Null => false,
            Value::Array(items) => !items.is_empty(),
            Value::Object(map) => !map.is_empty(),
            Value::String(s) => !s.is_empty(),
            _ => true,
        };
        if truthy {
            edge_types_section = format!(
                r#"
<FACT_TYPES>
{}
</FACT_TYPES>
"#,
                to_prompt_json(edge_types)
            );
        }
    }

    let sys_prompt = "You are an expert fact extractor that extracts fact triples from text. \
1. Extracted fact triples should also be extracted with relevant date information. \
2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. \
Use each episode's timestamp to resolve temporal references within that episode. \
REFERENCE_TIME is a fallback for when no per-episode timestamp is available.";

    let user_prompt = format!(
        r#"
<PREVIOUS_MESSAGES>
{previous_episodes}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{episode_content}
</CURRENT_MESSAGE>

<ENTITIES>
{nodes}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}  # ISO 8601 (UTC); used to resolve relative time mentions
</REFERENCE_TIME>
{edge_types_section}
# TASK
Extract all factual relationships between the given ENTITIES based on the CURRENT MESSAGE.
Only extract facts that:
- involve two DISTINCT ENTITIES from the ENTITIES list,
- are clearly stated or unambiguously implied in the CURRENT MESSAGE,
    and can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.

You may use information from the PREVIOUS MESSAGES only to disambiguate references or support continuity.


{custom_extraction_instructions}

# EXTRACTION RULES

1. **Entity Name Validation**: `source_entity_name` and `target_entity_name` must use only the `name` values from the ENTITIES list provided above.
   - **CRITICAL**: Using names not in the list will cause the edge to be rejected
2. Each fact must involve two **distinct** entities — `source_entity_name` and `target_entity_name` NEVER refer to the same entity.
3. Prefer facts that involve two distinct entities from the ENTITIES list. When a sentence describes a specific, concrete detail about a single entity (a brand name, a specific item, a physical description, a quantity, a location, a named activity), do NOT drop it. Instead, look for a second entity in the ENTITIES list that the detail relates to and form a proper triple (e.g., Entity -> OWNS -> item-entity, Entity -> LIVES_IN -> place-entity, Entity -> HAS_ATTRIBUTE -> detail-entity). Only skip the fact when no second entity in the ENTITIES list can anchor the detail.
   - BAD: "Alice feels happy" (vague single-entity state with no concrete detail — what is Alice happy about?)
   - GOOD: "Alice feels happy about Bob's promotion" → Alice -> FEELS_HAPPY_ABOUT -> Bob's promotion
   - GOOD: "Nate plays games on a Gamecube" → Nate -> PLAYS_GAMES_ON -> Gamecube (when "Gamecube" is in ENTITIES)
   - GOOD: "Alice congratulated Bob" (relationship between two entities), "Alice lives in Paris" (relationship between entity and place)
4. Do not emit semantically redundant facts, even across episodes within the CURRENT_MESSAGE. However, if a later episode adds specific details to a previously stated fact (e.g., adding a brand name, a count, a color, a location, or any concrete attribute), extract the more detailed version as a NEW fact — it is NOT a duplicate. Only treat facts as duplicates when they convey the same specificity.
   - NOT a duplicate: "user plays video games" (Episode 0) vs. "user plays games on a Gamecube" (Episode 1) → extract the second, more detailed fact.
   - IS a duplicate: "user plays games on a Gamecube" (Episode 0) vs. "user plays Gamecube games" (Episode 1) → extract once, list both episodes in `episode_indices`.
5. The `fact` MUST preserve all specific details from the source text: proper nouns, brand names, product names, model numbers, quantities, counts, colors, materials, physical descriptions, specific items, named locations, and named activities. Paraphrase the sentence structure but NEVER generalize:
   - NEVER generalize "Gamecube" to "gaming console", "Ford Mustang" to "car", "wool coat" to "coat", "red and purple lighting" to "lighting", "cracked windshield" to "car damage", or "three screenplays" to "several screenplays".
   - Do not verbatim quote the original text, but every concrete noun, number, and descriptor in the source should survive into the `fact`.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions (e.g., "last week"). When the CURRENT_MESSAGE contains multiple episodes with per-episode timestamps, prefer the timestamp of the specific episode the fact originates from.
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

# RELATION TYPE RULES

- If FACT_TYPES are provided and the relationship matches one of the types (considering the entity type signature), use that fact_type_name as the `relation_type`.
- Otherwise, derive a `relation_type` from the relationship predicate in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, IS_FRIENDS_WITH).

# DATETIME RULES

- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z).
- If the fact is ongoing (present tense), set `valid_at` to the timestamp of the episode the fact originates from. If no per-episode timestamp is available, use REFERENCE_TIME.
- If a change/termination is expressed, set `invalid_at` to the relevant timestamp.
- Leave both fields `null` if no explicit or resolvable time is stated.
- If only a date is mentioned (no time), assume 00:00:00.
- If only a year is mentioned, use January 1st at 00:00:00.
        "#,
        previous_episodes = to_prompt_json(&context["previous_episodes"]),
        episode_content = py_interp(&context["episode_content"]),
        nodes = to_prompt_json(&context["nodes"]),
        reference_time = py_interp(&context["reference_time"]),
        edge_types_section = edge_types_section,
        custom_extraction_instructions = py_interp(&context["custom_extraction_instructions"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}

// ports: graphiti_core/prompts/extract_edges.py::extract_attributes
pub fn extract_attributes(context: &Value) -> Vec<Message> {
    let sys_prompt = "You are a fact attribute extraction specialist. \
You ONLY emit attribute values that are explicitly stated in the FACT or \
already present in EXISTING ATTRIBUTES. You output strictly the JSON specified \
by the response schema — no reasoning, no explanation, no commentary in any field.";

    let user_prompt = format!(
        r#"Given the following FACT, its REFERENCE TIME, and any EXISTING ATTRIBUTES, update the attributes.

HARD RULES — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from the FACT,
   (b) the existing value already in EXISTING ATTRIBUTES (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.

2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description tells you the FORMAT a real value should take. The
   description text is NEVER itself a value. NEVER copy schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   the FACT, set the field to null (or omit it) — do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes. If you cannot produce a clean value of that type from the FACT, the field is null.

6. Use REFERENCE TIME to resolve any relative temporal expressions in the fact.

7. Preserve existing attribute values unless the FACT explicitly provides a new value.

<FACT>
{fact}
</FACT>

<REFERENCE TIME>
{reference_time}
</REFERENCE TIME>

<EXISTING ATTRIBUTES>
{existing_attributes}
</EXISTING ATTRIBUTES>
"#,
        fact = py_interp(&context["fact"]),
        reference_time = py_interp(&context["reference_time"]),
        existing_attributes = to_prompt_json(&context["existing_attributes"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}

// ports: graphiti_core/prompts/extract_edges.py::extract_timestamps
pub fn extract_timestamps(context: &Value) -> Vec<Message> {
    let sys_prompt = "You extract temporal bounds from facts. NEVER hallucinate dates.";

    let user_prompt = format!(
        r#"Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

<FACT>
{fact}
</FACT>

<REFERENCE TIME>
{reference_time}
</REFERENCE TIME>
"#,
        fact = py_interp(&context["fact"]),
        reference_time = py_interp(&context["reference_time"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}

// ports: graphiti_core/prompts/extract_edges.py::extract_timestamps_batch
pub fn extract_timestamps_batch(context: &Value) -> Vec<Message> {
    let sys_prompt = "You extract temporal bounds from facts. NEVER hallucinate dates.";

    let user_prompt = format!(
        r#"Given a list of FACTS with their REFERENCE TIMES, determine when each fact
became true (valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using each fact's REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to its REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

Return one timestamps entry per fact, in the same order.

<FACTS>
{facts}
</FACTS>
"#,
        facts = to_prompt_json(&context["facts"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}
