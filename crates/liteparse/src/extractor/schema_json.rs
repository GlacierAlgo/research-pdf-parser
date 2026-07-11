//! Standard-JSON-Schema input for extraction: flatten a (possibly nested)
//! JSON Schema into the engine's flat field lists.
//!
//! Users hand us the same JSON Schema they would hand an LLM structured-output
//! call — familiar, and the extraction-specific signal rides on *standard*
//! keywords (`description`, `enum`, `format`), so one schema serves both
//! engines. Handled shapes:
//!   - `object` (and `$ref` to one) → recurse; leaves named by dotted path.
//!   - `anyOf`/`oneOf` nullable unions → first non-null branch.
//!   - `array<scalar>` → one `list` field; each candidate becomes one element.
//!   - `array<object>` (repeated record groups, `line_items[]`) → a
//!     [`FlatObjectArray`] for the row-grouping path; its record sub-fields are
//!     flattened relative to the record root.
//!   - scalar leaf → one retrieval field; `enum`/`format` become value-path
//!     signal; type comes from schema `type`/`format`, then a name hint (many
//!     money/date fields are typed plain `string`).
//!
//! Ported verbatim from the bench bridge (`examples/extract_bench.rs`), which
//! itself mirrors the frozen Python stand-in (`extract_poc/extract_cli.py`
//! `flatten_schema`); the example now consumes this module.

use super::{FieldType, SchemaField};
use serde_json::Value;

/// A scalar leaf pulled out of the nested schema.
#[derive(Debug, Clone)]
pub struct FlatField {
    /// Path from the schema root (e.g. `["vendor", "name"]`). For object-array
    /// sub-fields the path is relative to the record root.
    pub path: Vec<String>,
    /// Machine name = last path component (retrieval-query fallback + label
    /// vocabulary). Deliberately NOT the dotted path — parent-object words in
    /// the query would shift retrieval.
    pub name: String,
    /// Retrieval query source: the leaf's `description`, else the path words.
    pub description: String,
    pub field_type: FieldType,
    /// For `List` leaves: the scalar item type (how each element coerces).
    /// Ignored for non-list leaves.
    pub item_type: FieldType,
    pub choices: Vec<String>,
    pub format: Option<String>,
}

impl FlatField {
    /// Dotted output name (`vendor.address.city`).
    pub fn dotted(&self) -> String {
        self.path.join(".")
    }

    /// The engine-facing field (retrieval + value path).
    pub fn to_schema_field(&self) -> SchemaField {
        SchemaField {
            name: self.name.clone(),
            description: Some(self.description.clone()),
            field_type: self.field_type,
            choices: self.choices.clone(),
            format: self.format.clone(),
        }
    }
}

/// An object-array (repeated record group): the group's path plus the record's
/// leaves (scalar/list sub-fields, pathed relative to the record root).
#[derive(Debug, Clone)]
pub struct FlatObjectArray {
    pub path: Vec<String>,
    pub fields: Vec<FlatField>,
}

impl FlatObjectArray {
    /// Dotted output name for the group (`line_items`).
    pub fn dotted(&self) -> String {
        self.path.join(".")
    }
}

/// A JSON Schema flattened to the engine's shapes.
#[derive(Debug, Clone, Default)]
pub struct FlatSchema {
    pub fields: Vec<FlatField>,
    pub object_arrays: Vec<FlatObjectArray>,
}

/// Flatten a standard JSON Schema. Never fails: unrecognized shapes are
/// skipped (an empty result means nothing in the schema was extractable —
/// callers should surface that, see the bench bridge's warning).
pub fn flatten_json_schema(root: &Value) -> FlatSchema {
    let mut schema = FlatSchema::default();
    flatten(
        root,
        root,
        &mut Vec::new(),
        &mut schema.fields,
        &mut schema.object_arrays,
    );
    schema
}

/// What a node effectively is once `anyOf`/`oneOf`/nullable/`$ref` are collapsed.
enum Kind {
    Object,
    Array,
    Scalar,
}

/// Follow a single `$ref` JSON pointer (`#/$defs/Foo`) against the root schema.
/// Returns the node unchanged if there's no local `$ref` to resolve.
fn deref<'a>(root: &'a Value, node: &'a Value) -> &'a Value {
    if let Some(Value::String(r)) = node.get("$ref")
        && let Some(target) = r.strip_prefix("#/")
    {
        let mut cur = root;
        for seg in target.split('/') {
            match cur.get(seg) {
                Some(next) => cur = next,
                None => return node,
            }
        }
        return cur;
    }
    node
}

/// Collapse `anyOf`/`oneOf`/nullable and classify the node.
fn resolve<'a>(root: &'a Value, node: &'a Value) -> (Kind, &'a Value) {
    let node = deref(root, node);
    // anyOf/oneOf → first non-null branch (nullable unions).
    let node = ["anyOf", "oneOf"]
        .iter()
        .find_map(|k| node.get(k).and_then(Value::as_array))
        .and_then(|branches| {
            branches
                .iter()
                .find(|b| deref(root, b).get("type").and_then(Value::as_str) != Some("null"))
                .or_else(|| branches.first())
        })
        .map(|b| deref(root, b))
        .unwrap_or(node);

    let ty = type_str(node);
    if ty == Some("object") || node.get("properties").is_some() {
        (Kind::Object, node)
    } else if ty == Some("array") {
        (Kind::Array, node)
    } else {
        (Kind::Scalar, node)
    }
}

/// The node's `type`, taking the first non-null when it's a `["string","null"]`
/// union.
fn type_str(node: &Value) -> Option<&str> {
    match node.get("type")? {
        Value::String(s) => Some(s.as_str()),
        Value::Array(xs) => xs.iter().filter_map(Value::as_str).find(|s| *s != "null"),
        _ => None,
    }
}

fn flatten(
    root: &Value,
    node: &Value,
    path: &mut Vec<String>,
    out: &mut Vec<FlatField>,
    object_arrays: &mut Vec<FlatObjectArray>,
) {
    let (kind, node) = resolve(root, node);
    match kind {
        Kind::Object => {
            if let Some(props) = node.get("properties").and_then(Value::as_object) {
                for (name, sub) in props {
                    path.push(name.clone());
                    flatten(root, sub, path, out, object_arrays);
                    path.pop();
                }
            }
        }
        Kind::Array => {
            // Resolve the item schema. Scalar items → a `list` field; object
            // items → a row-grouping group (its record sub-fields flattened
            // relative to the record); deeper array-of-array is dropped.
            let null = Value::Null;
            let (item_kind, item_node) = resolve(root, node.get("items").unwrap_or(&null));
            if let Kind::Object = item_kind {
                // Record sub-fields, pathed relative to the record root. Nested
                // object-arrays inside a record are v1-out-of-scope (ignored).
                let mut fields = Vec::new();
                let mut ignore = Vec::new();
                flatten(root, item_node, &mut Vec::new(), &mut fields, &mut ignore);
                if !fields.is_empty() {
                    object_arrays.push(FlatObjectArray {
                        path: path.clone(),
                        fields,
                    });
                }
            } else if let Kind::Scalar = item_kind {
                let name = path.last().cloned().unwrap_or_default();
                // The array's own description is the retrieval query for the list.
                let description = node
                    .get("description")
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| path.join(" ").replace('_', " "));
                out.push(FlatField {
                    path: path.clone(),
                    field_type: FieldType::List,
                    item_type: leaf_type(item_node, &name),
                    choices: string_list(item_node.get("enum")),
                    format: item_node
                        .get("format")
                        .and_then(Value::as_str)
                        .map(String::from),
                    name,
                    description,
                });
            }
            // Kind::Array item (array-of-array) is dropped — v1-out-of-scope.
        }
        Kind::Scalar => {
            let name = path.last().cloned().unwrap_or_default();
            let description = node
                .get("description")
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(String::from)
                .unwrap_or_else(|| path.join(" ").replace('_', " "));
            out.push(FlatField {
                path: path.clone(),
                field_type: leaf_type(node, &name),
                item_type: FieldType::Str, // unused for scalar leaves
                choices: string_list(node.get("enum")),
                format: node.get("format").and_then(Value::as_str).map(String::from),
                name,
                description,
            });
        }
    }
}

/// Effective value type: explicit schema `type`/`format` first, then a NAME
/// hint (money/date fields are frequently typed as plain `string`).
fn leaf_type(node: &Value, name: &str) -> FieldType {
    if matches!(
        node.get("format").and_then(Value::as_str),
        Some("date" | "date-time")
    ) {
        return FieldType::Date;
    }
    let ty = type_str(node);
    let (date_hint, num_hint) = name_hints(name);
    match ty {
        Some("number") => FieldType::Number,
        Some("integer") => FieldType::Int,
        Some("boolean") => FieldType::Bool,
        _ if date_hint => FieldType::Date,
        Some("string") if num_hint => FieldType::Number, // "tax_total": string → numeric
        Some("string") | None => FieldType::Str,
        _ => FieldType::Str,
    }
}

/// (date_hint, numeric_hint) from the field name's word tokens. Word-boundaried
/// like the Python `\b…\b` regexes: tokenize on non-alnum + camelCase, then set
/// membership (so "no" doesn't fire inside "notes").
///
/// Identifier-ish tokens veto the numeric hint: `invoice_number`, `tax_id`,
/// `case_no` name alphanumeric codes, not amounts, and coercing them through
/// the numeric scanner strips letters/prefixes out of the value. Note
/// "number"/"no" are themselves ID words in field names ("PO number", "case
/// no"), not money words — amounts are named amount/total/price/….
fn name_hints(name: &str) -> (bool, bool) {
    const DATE: &[&str] = &["date", "dob", "issued", "due"];
    const NUM: &[&str] = &[
        "amount", "total", "subtotal", "price", "cost", "qty", "quantity", "rate", "fee",
        "balance", "tax", "discount", "shipping", "paid", "due", "count",
    ];
    const ID: &[&str] = &[
        "id",
        "ids",
        "identifier",
        "code",
        "ref",
        "reference",
        "sku",
        "serial",
        "number",
        "no",
        "num",
        "iban",
        "swift",
        "ssn",
        "ein",
        "vat",
        "phone",
        "fax",
        "zip",
        "postal",
    ];
    let tokens = tokenize(name);
    let date = tokens
        .iter()
        .any(|t| DATE.contains(&t.as_str()) || t.starts_with("expir"));
    let id = tokens.iter().any(|t| ID.contains(&t.as_str()));
    let num = !id && tokens.iter().any(|t| NUM.contains(&t.as_str()));
    (date, num)
}

/// Lowercased alnum tokens, splitting on non-alnum and camelCase boundaries.
fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            if ch.is_uppercase() && prev_lower && !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur)); // camelCase seam
            }
            cur.extend(ch.to_lowercase());
            prev_lower = ch.is_lowercase();
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
            prev_lower = false;
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

fn string_list(node: Option<&Value>) -> Vec<String> {
    node.and_then(Value::as_array)
        .map(|xs| {
            xs.iter()
                .map(|x| match x {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nested_objects_flatten_to_dotted_paths() {
        let schema = json!({
            "type": "object",
            "properties": {
                "vendor": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "vendor legal name"},
                        "address": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}}
                        }
                    }
                }
            }
        });
        let flat = flatten_json_schema(&schema);
        let dotted: Vec<String> = flat.fields.iter().map(|f| f.dotted()).collect();
        assert_eq!(dotted, ["vendor.address.city", "vendor.name"]);
        // Retrieval name stays the leaf word, not the dotted path.
        assert_eq!(flat.fields[1].name, "name");
        assert_eq!(flat.fields[1].description, "vendor legal name");
        // Missing description falls back to the path words.
        assert_eq!(flat.fields[0].description, "vendor address city");
    }

    #[test]
    fn nullable_union_and_name_hints() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tax_total": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "issue_date": {"type": "string"},
                "notes": {"type": "string"},
                "invoice_number": {"type": "string"},
                "tax_id": {"type": "string"},
                "case_no": {"type": "string"}
            }
        });
        let flat = flatten_json_schema(&schema);
        let by_name = |n: &str| flat.fields.iter().find(|f| f.name == n).unwrap();
        assert_eq!(by_name("tax_total").field_type, FieldType::Number); // num hint through the union
        assert_eq!(by_name("issue_date").field_type, FieldType::Date);
        assert_eq!(by_name("notes").field_type, FieldType::Str); // "no" must not fire inside "notes"
        // ID-ish names stay strings — alphanumeric codes must not be digit-mined.
        assert_eq!(by_name("invoice_number").field_type, FieldType::Str);
        assert_eq!(by_name("tax_id").field_type, FieldType::Str);
        assert_eq!(by_name("case_no").field_type, FieldType::Str);
    }

    #[test]
    fn enum_format_and_ref() {
        let schema = json!({
            "$defs": {"Tier": {"type": "string", "enum": ["basic", "premium"]}},
            "type": "object",
            "properties": {
                "plan": {"$ref": "#/$defs/Tier"},
                "contact": {"type": "string", "format": "email"}
            }
        });
        let flat = flatten_json_schema(&schema);
        let by_name = |n: &str| flat.fields.iter().find(|f| f.name == n).unwrap();
        assert_eq!(by_name("plan").choices, ["basic", "premium"]);
        assert_eq!(by_name("contact").format.as_deref(), Some("email"));
    }

    #[test]
    fn arrays_split_scalar_list_vs_object_group() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "amount": {"type": "number"}
                        }
                    }
                }
            }
        });
        let flat = flatten_json_schema(&schema);
        assert_eq!(flat.fields.len(), 1);
        assert_eq!(flat.fields[0].field_type, FieldType::List);
        assert_eq!(flat.fields[0].item_type, FieldType::Str);
        assert_eq!(flat.object_arrays.len(), 1);
        let oa = &flat.object_arrays[0];
        assert_eq!(oa.dotted(), "line_items");
        // Sub-fields are pathed relative to the record root.
        let subs: Vec<String> = oa.fields.iter().map(|f| f.dotted()).collect();
        assert_eq!(subs, ["amount", "description"]);
        assert_eq!(oa.fields[0].field_type, FieldType::Number);
    }
}
