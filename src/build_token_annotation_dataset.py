#!/usr/bin/env python3
"""Build a token-level annotated dataset from `code_snippets`.

Each output row corresponds to one LLM tokenizer token aligned back to the
original code snippet with character offsets. The row is enriched with:
- Pygments lexical labels
- Tree-sitter syntax paths
- snippet metadata required for future joins with LLM activations

The script is designed to be robust on large collections:
- snippet loading is streaming-like and language-filterable
- rows are written in batches
- Tree-sitter parsing errors are logged and preserved in the output
- outputs can be saved as JSONL, Parquet, Arrow IPC, or Hugging Face Dataset

Typical usage:
    python3 src/build_token_annotation_dataset.py \
        --snippets-dir code_snippets \
        --tokenizer-name google/gemma-2-2b \
        --output-path outputs/code_token_annotations.parquet
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound


LOGGER = logging.getLogger("build_token_annotation_dataset")


LANGUAGE_TO_PYGMENTS = {
    "C": "c",
    "C#": "csharp",
    "C++": "cpp",
    "Java": "java",
    "Javascript": "javascript",
    "PHP": "php",
    "Python": "python",
}

TREE_SITTER_LANGUAGE_SPECS = {
    "C": {
        "module": "tree_sitter_c",
        "attrs": ("language", "LANGUAGE"),
    },
    "C#": {
        "module": "tree_sitter_c_sharp",
        "attrs": ("language", "LANGUAGE"),
    },
    "C++": {
        "module": "tree_sitter_cpp",
        "attrs": ("language", "LANGUAGE"),
    },
    "Java": {
        "module": "tree_sitter_java",
        "attrs": ("language", "LANGUAGE"),
    },
    "Javascript": {
        "module": "tree_sitter_javascript",
        "attrs": ("language", "LANGUAGE"),
    },
    "PHP": {
        "module": "tree_sitter_php",
        "attrs": ("language", "language_php", "language_php_only", "LANGUAGE"),
    },
    "Python": {
        "module": "tree_sitter_python",
        "attrs": ("language", "LANGUAGE"),
    },
}

TREE_SITTER_UNKNOWN = ""
TREE_SITTER_SPECIAL = "special"
PYGMENTS_SPECIAL = "special"
NAME_LIKE_PYGMENTS_LABELS = {
    "name",
    "variable",
    "function",
    "class",
    "namespace",
    "builtin",
    "decorator",
}
IDENTIFIER_LIKE_NODE_TYPES = {
    "identifier",
    "name",
    "variable_name",
    "simple_identifier",
}
SKIP_DECL_IDENTIFIER_NODE_TYPES = {
    "field_identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "type_identifier",
    "primitive_type",
}


@dataclass(frozen=True)
class SnippetRecord:
    snippet_id: str
    global_idx: str
    language: str
    code: str
    file_path: str
    source_snippet_ids: List[str]
    source_file_paths: List[str]
    num_fragments: int


@dataclass(frozen=True)
class TreeSitterParseResult:
    char_paths: List[str]
    char_leaf_types: List[str]
    char_identifier_names: List[str]
    char_identifier_access_kinds: List[str]
    status: str
    error: str
    scopes: List["ScopeRecord"]
    declarations: List["ScopeDeclaration"]
    usages: List["NameUsage"]


@dataclass(frozen=True)
class ScopeDeclaration:
    name: str
    kind: str
    start_char: int
    end_char: int
    scope_id: int
    scope_depth: int


@dataclass(frozen=True)
class NameUsage:
    name: str
    access_kind: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ScopeRecord:
    scope_id: int
    parent_scope_id: Optional[int]
    node_type: str
    start_char: int
    end_char: int
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an annotated token-level dataset from code_snippets.",
    )
    parser.add_argument("--snippets-dir", default="code_snippets")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--output-format",
        choices=["auto", "jsonl", "parquet", "arrow", "hf"],
        default="auto",
        help="If 'auto', infer the format from --output-path.",
    )
    parser.add_argument(
        "--tokenizer-name",
        required=True,
        help="Tokenizer name/path loadable by transformers.AutoTokenizer.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional subset of languages to keep, e.g. Python Java C++.",
    )
    parser.add_argument("--code-field", default="code")
    parser.add_argument(
        "--group-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, merge snippet fragments sharing the same base id inside a language, "
            "for example 1800-1 ... 1800-5."
        ),
    )
    parser.add_argument(
        "--group-separator",
        default="",
        help="String inserted between grouped code fragments.",
    )
    parser.add_argument("--max-snippets", type=int, default=None)
    parser.add_argument("--skip-snippets", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--truncation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable tokenizer truncation.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add tokenizer special tokens. Disabled by default for offset fidelity.",
    )
    parser.add_argument(
        "--include-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Duplicate the raw code snippet in every output row.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Passed to transformers.AutoTokenizer.from_pretrained.",
    )
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=5000,
        help="Number of token rows accumulated before flushing to disk.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N processed snippets.",
    )
    parser.add_argument(
        "--fail-on-tree-sitter-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop immediately on Tree-sitter parser/import errors.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_global_idx(idx: str) -> str:
    return re.sub(r"-[^-]+-", "-", idx)


def iter_snippets(
    directory: Path,
    languages: Optional[Iterable[str]] = None,
    code_field: str = "code",
    group_variants: bool = True,
    group_separator: str = "",
    skip_snippets: int = 0,
    max_snippets: Optional[int] = None,
) -> Iterator[SnippetRecord]:
    selected_languages = set(languages) if languages is not None else None
    emitted = 0
    seen = 0

    for language_dir in sorted(directory.iterdir()):
        if not language_dir.is_dir():
            continue

        language = language_dir.name
        if selected_languages is not None and language not in selected_languages:
            continue

        loaded_items: List[Dict[str, str]] = []
        for file_path in sorted(language_dir.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if code_field not in payload:
                raise KeyError(f"Missing code field '{code_field}' in {file_path}")

            loaded_items.append(
                {
                    "idx": payload["idx"],
                    "global_idx": normalize_global_idx(payload["idx"]),
                    "language": language,
                    "code": payload[code_field],
                    "file_path": str(file_path),
                }
            )

        if group_variants:
            grouped_items: Dict[str, List[Dict[str, str]]] = {}
            for item in loaded_items:
                group_key = group_key_from_idx(item["global_idx"])
                grouped_items.setdefault(group_key, []).append(item)
            iterable_items = [
                combine_grouped_snippets(language, group_key, items, group_separator)
                for group_key, items in sorted(grouped_items.items())
            ]
        else:
            iterable_items = [
                SnippetRecord(
                    snippet_id=item["idx"],
                    global_idx=item["global_idx"],
                    language=item["language"],
                    code=item["code"],
                    file_path=item["file_path"],
                    source_snippet_ids=[item["idx"]],
                    source_file_paths=[item["file_path"]],
                    num_fragments=1,
                )
                for item in loaded_items
            ]

        for snippet in iterable_items:
            if seen < skip_snippets:
                seen += 1
                continue

            yield snippet
            emitted += 1
            seen += 1

            if max_snippets is not None and emitted >= max_snippets:
                return


def extract_variant_number(idx: str) -> int:
    try:
        return int(idx.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def group_key_from_idx(normalized_idx: str) -> str:
    if "-" not in normalized_idx:
        return normalized_idx
    return normalized_idx.rsplit("-", 1)[0]


def combine_grouped_snippets(
    language: str,
    group_key: str,
    items: Sequence[Dict[str, str]],
    separator: str,
) -> SnippetRecord:
    ordered_items = sorted(
        items,
        key=lambda item: (extract_variant_number(item["idx"]), item["file_path"]),
    )
    return SnippetRecord(
        snippet_id=group_key,
        global_idx=group_key,
        language=language,
        code=separator.join(item["code"] for item in ordered_items),
        file_path=ordered_items[0]["file_path"],
        source_snippet_ids=[item["idx"] for item in ordered_items],
        source_file_paths=[item["file_path"] for item in ordered_items],
        num_fragments=len(ordered_items),
    )


def load_tokenizer(tokenizer_name: str, trust_remote_code: bool = False) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install it with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "The tokenizer must be a Hugging Face fast tokenizer because "
            "`return_offsets_mapping=True` is required."
        )
    return tokenizer


def tokenize_llm_code(
    code: str,
    tokenizer: Any,
    add_special_tokens: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "add_special_tokens": add_special_tokens,
        "truncation": truncation,
        "return_offsets_mapping": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length

    encoding = tokenizer(code, **kwargs)
    if "offset_mapping" not in encoding:
        raise ValueError(
            "The tokenizer does not expose `offset_mapping`. "
            "Use a Hugging Face fast tokenizer."
        )

    input_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer returned inconsistent `input_ids` and `offset_mapping` lengths.")

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokenizer_tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
    else:
        tokenizer_tokens = [str(token_id) for token_id in input_ids]

    if hasattr(tokenizer, "decode"):
        decoded_tokens = [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in input_ids
        ]
    else:
        decoded_tokens = tokenizer_tokens[:]

    return {
        "input_ids": input_ids,
        "offset_mapping": offsets,
        "tokenizer_tokens": tokenizer_tokens,
        "decoded_tokens": decoded_tokens,
    }


def simplify_token_type(token_type: Any) -> str:
    token_str = str(token_type)
    mapping = {
        "Token.Keyword": "keyword",
        "Token.Name.Function": "function",
        "Token.Name.Class": "class",
        "Token.Name.Namespace": "namespace",
        "Token.Name.Builtin": "builtin",
        "Token.Name.Decorator": "decorator",
        "Token.Name.Variable": "variable",
        "Token.Name": "name",
        "Token.Literal.String": "string",
        "Token.Literal.Number": "number",
        "Token.Operator": "operator",
        "Token.Punctuation": "punctuation",
        "Token.Comment": "comment",
        "Token.Text.Whitespace": "whitespace",
        "Token.Text": "text",
    }

    for key in sorted(mapping.keys(), key=len, reverse=True):
        if token_str.startswith(key):
            return mapping[key]
    return "other"


def get_pygments_lexer(language: str):
    pygments_language = LANGUAGE_TO_PYGMENTS.get(language, language.lower())
    try:
        return get_lexer_by_name(pygments_language)
    except ClassNotFound as exc:
        raise ValueError(
            f"Could not find a Pygments lexer for language '{language}'."
        ) from exc


def build_pygments_char_labels(code: str, language: str) -> tuple[List[str], List[str]]:
    if not code:
        return [], []

    lexer = get_pygments_lexer(language)
    full_labels = ["other"] * len(code)
    simple_labels = ["other"] * len(code)

    for token_start, token_type, token_value in lexer.get_tokens_unprocessed(code):
        if not token_value:
            continue
        token_end = min(token_start + len(token_value), len(code))
        full_type = str(token_type)
        simple_type = simplify_token_type(token_type)
        for char_index in range(token_start, token_end):
            full_labels[char_index] = full_type
            simple_labels[char_index] = simple_type

    return full_labels, simple_labels


def build_byte_to_char_map(text: str) -> List[int]:
    mapping: List[int] = []
    char_index = 0
    for char in text:
        encoded = char.encode("utf-8")
        mapping.extend([char_index] * len(encoded))
        char_index += 1
    mapping.append(len(text))
    return mapping


def byte_span_to_char_span(byte_to_char: Sequence[int], start_byte: int, end_byte: int) -> tuple[int, int]:
    if start_byte < 0 or end_byte < start_byte:
        raise ValueError(f"Invalid byte span: {(start_byte, end_byte)}")
    if start_byte >= len(byte_to_char):
        return len(byte_to_char) - 1, len(byte_to_char) - 1
    if end_byte >= len(byte_to_char):
        end_byte = len(byte_to_char) - 1
    return byte_to_char[start_byte], byte_to_char[end_byte]


class TreeSitterParserRegistry:
    def __init__(self) -> None:
        self._parsers: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}

    def get_parser(self, language: str) -> Any:
        if language in self._parsers:
            return self._parsers[language]
        if language in self._errors:
            raise RuntimeError(self._errors[language])

        spec = TREE_SITTER_LANGUAGE_SPECS.get(language)
        if spec is None:
            message = f"No Tree-sitter configuration registered for language '{language}'."
            self._errors[language] = message
            raise RuntimeError(message)

        try:
            tree_sitter_module = importlib.import_module("tree_sitter")
            grammar_module = importlib.import_module(spec["module"])
        except ImportError as exc:
            message = (
                f"Tree-sitter dependency missing for '{language}'. "
                f"Install `tree_sitter` and `{spec['module']}`."
            )
            self._errors[language] = message
            raise RuntimeError(message) from exc

        language_obj = None
        for attr_name in spec["attrs"]:
            attr_value = getattr(grammar_module, attr_name, None)
            if attr_value is None:
                continue
            candidate = attr_value() if callable(attr_value) else attr_value
            try:
                language_obj = tree_sitter_module.Language(candidate)
                break
            except Exception:
                continue

        if language_obj is None:
            message = (
                f"Could not build a Tree-sitter Language object for '{language}' "
                f"from module '{spec['module']}'."
            )
            self._errors[language] = message
            raise RuntimeError(message)

        try:
            parser = tree_sitter_module.Parser(language_obj)
        except TypeError:
            parser = tree_sitter_module.Parser()
            parser.set_language(language_obj)

        self._parsers[language] = parser
        return parser


def child_by_field_name_safe(node: Any, field_name: str) -> Optional[Any]:
    try:
        return node.child_by_field_name(field_name)
    except Exception:
        return None


def node_type_contains_any(node_type: str, keywords: Sequence[str]) -> bool:
    return any(keyword in node_type for keyword in keywords)


def is_identifier_like_node(node: Any) -> bool:
    node_type = getattr(node, "type", "")
    return is_identifier_like_type(node_type)


def is_identifier_like_type(node_type: str) -> bool:
    if node_type in SKIP_DECL_IDENTIFIER_NODE_TYPES:
        return False
    if node_type in IDENTIFIER_LIKE_NODE_TYPES:
        return True
    return node_type.endswith("identifier") and node_type not in SKIP_DECL_IDENTIFIER_NODE_TYPES


def node_byte_text(code_bytes: bytes, node: Any) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def node_char_span(byte_to_char: Sequence[int], node: Any) -> tuple[int, int]:
    return byte_span_to_char_span(byte_to_char, node.start_byte, node.end_byte)


def node_contains_node(container: Any, candidate: Any) -> bool:
    return (
        getattr(container, "start_byte", -1) <= getattr(candidate, "start_byte", -2)
        and getattr(candidate, "end_byte", -1) <= getattr(container, "end_byte", -2)
    )


def collect_identifier_leaf_nodes(node: Any) -> List[Any]:
    results: List[Any] = []

    if is_identifier_like_node(node):
        results.append(node)
        return results

    for child in getattr(node, "children", []):
        results.extend(collect_identifier_leaf_nodes(child))

    return results


def is_scope_node(node: Any, is_root: bool = False) -> bool:
    if is_root:
        return True

    node_type = getattr(node, "type", "")
    exact_types = {
        "module",
        "program",
        "translation_unit",
        "block",
        "compound_statement",
        "statement_block",
        "declaration_list",
        "class_body",
        "enum_body",
        "namespace_body",
        "catch_clause",
        "except_clause",
    }
    if node_type in exact_types:
        return True
    if node_type.endswith("_block"):
        return True
    return node_type_contains_any(
        node_type,
        (
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "constructor_declaration",
            "class_definition",
            "class_declaration",
            "namespace_definition",
            "lambda",
        ),
    )


def scope_decl_kind(language: str, node_type: str) -> Optional[str]:
    function_decl_types_by_language = {
        "C": {
            "function_definition",
            "function_declarator",
        },
        "C++": {
            "function_definition",
            "function_declarator",
            "template_declaration",
        },
        "Java": {
            "method_declaration",
            "constructor_declaration",
        },
        "C#": {
            "method_declaration",
            "constructor_declaration",
        },
        "Javascript": {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
        },
        "PHP": {
            "function_definition",
            "method_declaration",
        },
        "Python": {
            "function_definition",
        },
    }

    class_decl_types_by_language = {
        "C++": {"class_specifier", "struct_specifier"},
        "Java": {"class_declaration", "interface_declaration", "enum_declaration"},
        "C#": {"class_declaration", "interface_declaration", "struct_declaration", "enum_declaration"},
        "Javascript": {"class_declaration"},
        "PHP": {"class_declaration", "interface_declaration", "trait_declaration"},
        "Python": {"class_definition"},
    }

    namespace_decl_types_by_language = {
        "C++": {"namespace_definition"},
        "C#": {"namespace_declaration"},
        "PHP": {"namespace_definition"},
    }

    if node_type in function_decl_types_by_language.get(language, set()):
        return "function"

    if node_type in class_decl_types_by_language.get(language, set()):
        return "class"

    if node_type in namespace_decl_types_by_language.get(language, set()):
        return "namespace"

    return None


def declaration_kind_for_node_type(language: str, node_type: str) -> Optional[str]:
    if "parameter" in node_type:
        return "parameter"

    if node_type in {"variable_declarator", "init_declarator"}:
        return "variable"

    if node_type in {"with_item"}:
        return "with_alias"

    if node_type in {"catch_clause", "except_clause"}:
        return "exception_variable"

    if node_type_contains_any(
        node_type,
        ("for_statement", "foreach_statement", "enhanced_for_statement"),
    ):
        return "loop_variable"

    # Python : une assignation peut introduire un nom.
    if language == "Python" and node_type == "assignment":
        return "implicit_variable"

    # PHP : selon ton objectif, tu peux considérer $x = ... comme introduction.
    if language == "PHP" and node_type == "assignment_expression":
        return "implicit_variable"

    # JavaScript : seulement let/const/var devraient déclarer.
    # Pas x = 1.
    return None


def declaration_candidate_subtrees(language: str, node: Any) -> List[Any]:
    node_type = getattr(node, "type", "")

    if node_type in {"variable_declarator", "init_declarator"}:
        return [
            child for field in ("name", "declarator")
            if (child := child_by_field_name_safe(node, field)) is not None
        ]

    if "parameter" in node_type:
        return [
            child for field in ("name", "declarator", "pattern")
            if (child := child_by_field_name_safe(node, field)) is not None
        ] or list(getattr(node, "children", []))

    if language == "Python" and node_type == "assignment":
        target = (
            child_by_field_name_safe(node, "left")
            or child_by_field_name_safe(node, "target")
        )
        return [target] if target is not None else []

    if language == "PHP" and node_type == "assignment_expression":
        left = child_by_field_name_safe(node, "left")
        return [left] if left is not None else []

    if node_type in {"with_item"}:
        alias = child_by_field_name_safe(node, "alias")
        return [alias] if alias is not None else []

    if node_type in {"catch_clause", "except_clause"}:
        return list(getattr(node, "children", []))

    if node_type_contains_any(
        node_type,
        ("for_statement", "foreach_statement", "enhanced_for_statement"),
    ):
        left = (
            child_by_field_name_safe(node, "left")
            or child_by_field_name_safe(node, "name")
            or child_by_field_name_safe(node, "pattern")
        )
        return [left] if left is not None else []

    return []

def is_descendant_of(node, ancestor) -> bool:
    cur = node
    while cur is not None:
        if cur == ancestor:
            return True
        cur = cur.parent
    return False

def is_direct_assignment_target(identifier_node: Any, lhs: Any) -> bool:
    """
    True pour x dans x = ...
    False pour obj dans obj.x = ...
    False pour i dans a[i] = ...
    """
    if lhs is None:
        return False

    if is_identifier_like_node(lhs):
        return lhs == identifier_node

    # patterns simples / parenthèses / destructuring : à adapter selon langage
    lhs_type = getattr(lhs, "type", "")
    if lhs_type in {
        "tuple_pattern",
        "list_pattern",
        "object_pattern",
        "array_pattern",
        "formal_parameter",
        "parenthesized_expression",
        "parenthesized_pattern",
    }:
        return node_contains_node(lhs, identifier_node)

    return False

def classify_identifier_access(identifier_node: Any, ancestors: Sequence[Any]) -> str:
    for ancestor in reversed(ancestors):
        t = getattr(ancestor, "type", "")

        if t in {"update_expression", "unary_update_expression"}:
            return "read_write"

        if t in {"augmented_assignment", "augmented_assignment_expression"}:
            left = child_by_field_name_safe(ancestor, "left")
            if left and is_direct_assignment_target(identifier_node, left):
                return "read_write"
            return "read"

        if t in {"assignment", "assignment_expression"}:
            left = child_by_field_name_safe(ancestor, "left")
            right = child_by_field_name_safe(ancestor, "right")

            if left and is_direct_assignment_target(identifier_node, left):
                return "write"

            if right and is_descendant_of(identifier_node, right):
                return "read"

            # identifiant dans un lhs complexe : obj.x, a[i], *p
            if left and is_descendant_of(identifier_node, left):
                return "read"

            return "read"

        if t in {"variable_declarator", "init_declarator"}:
            name = (
                child_by_field_name_safe(ancestor, "name")
                or child_by_field_name_safe(ancestor, "declarator")
            )
            value = (
                child_by_field_name_safe(ancestor, "value")
                or child_by_field_name_safe(ancestor, "initializer")
            )

            if name and is_descendant_of(identifier_node, name):
                return "write"

            if value and is_descendant_of(identifier_node, value):
                return "read"

        if "parameter" in t:
            name = (
                child_by_field_name_safe(ancestor, "name")
                or child_by_field_name_safe(ancestor, "declarator")
            )
            if name and is_descendant_of(identifier_node, name):
                return "write"

    return "read"

def is_property_or_field_identifier(node: Any, ancestors: Sequence[Any]) -> bool:
    node_type = getattr(node, "type", "")

    if node_type in {
        "field_identifier",
        "property_identifier",
        "shorthand_property_identifier",
    }:
        return True

    for ancestor in reversed(ancestors):
        ancestor_type = getattr(ancestor, "type", "")
        if ancestor_type in {
            "field_expression",
            "member_expression",
            "attribute",
            "subscript_expression",
        }:
            # obj.x : x n'est pas forcément une variable locale
            prop = (
                child_by_field_name_safe(ancestor, "property")
                or child_by_field_name_safe(ancestor, "field")
                or child_by_field_name_safe(ancestor, "attribute")
            )
            if prop and is_descendant_of(node, prop):
                return True

    return False


def collect_identifier_usages(
    code_bytes: bytes,
    byte_to_char: Sequence[int],
    node: Any,
    ancestors: Optional[List[Any]] = None,
    results: Optional[List[NameUsage]] = None,
) -> List[NameUsage]:
    if ancestors is None:
        ancestors = []
    if results is None:
        results = []

    if is_identifier_like_node(node):
        access_kind = classify_identifier_access(node, ancestors)
        if is_property_or_field_identifier(node, ancestors):
            if is_property_or_field_identifier(node, ancestors):
                access_kind = "property"
        name = node_byte_text(code_bytes, node)
        start_char, end_char = node_char_span(byte_to_char, node)
        results.append(
            NameUsage(
                name=name,
                access_kind=access_kind,
                start_char=start_char,
                end_char=end_char,
            )
        )
        return results

    next_ancestors = ancestors + [node]
    for child in getattr(node, "children", []):
        collect_identifier_usages(code_bytes, byte_to_char, child, next_ancestors, results)

    return results


def add_declaration_from_node(
    declarations: List[ScopeDeclaration],
    seen: set[tuple[int, str, int, int, str]],
    code_bytes: bytes,
    byte_to_char: Sequence[int],
    scope: ScopeRecord,
    name_node: Any,
    kind: str,
) -> None:
    name = node_byte_text(code_bytes, name_node)
    if not name:
        return

    start_char, end_char = node_char_span(byte_to_char, name_node)
    key = (scope.scope_id, kind, start_char, end_char, name)
    if key in seen:
        return

    seen.add(key)
    declarations.append(
        ScopeDeclaration(
            name=name,
            kind="variable" if kind == "implicit_variable" else kind,
            start_char=start_char,
            end_char=end_char,
            scope_id=scope.scope_id,
            scope_depth=scope.depth,
        )
    )


def declaration_exists_before_position(
    declarations: Sequence[ScopeDeclaration],
    scopes: Sequence[ScopeRecord],
    scope: ScopeRecord,
    name: str,
    position: int,
) -> bool:
    return any(
        declaration.name == name
        for declaration in resolve_visible_declarations(scopes, declarations, scope, position)
    )


def build_tree_sitter_scope_data(
    code: str,
    language: str,
    root_node: Any,
    byte_to_char: Sequence[int],
) -> tuple[List[ScopeRecord], List[ScopeDeclaration]]:
    code_bytes = code.encode("utf-8")
    scopes: List[ScopeRecord] = [
        ScopeRecord(
            scope_id=0,
            parent_scope_id=None,
            node_type=getattr(root_node, "type", "root"),
            start_char=0,
            end_char=len(code),
            depth=0,
        )
    ]
    declarations: List[ScopeDeclaration] = []
    seen_declarations: set[tuple[int, str, int, int, str]] = set()
    next_scope_id = 1

    def walk(node: Any, active_scope: ScopeRecord) -> None:
        nonlocal next_scope_id
        node_type = getattr(node, "type", "")
        current_scope = active_scope

        decl_kind = scope_decl_kind(language, node_type)
        if decl_kind is not None:
            name_field = (
                child_by_field_name_safe(node, "name")
                or child_by_field_name_safe(node, "declarator")
            )
            if name_field is not None:
                identifier_nodes = collect_identifier_leaf_nodes(name_field)
                for identifier_node in identifier_nodes[:1]:
                    add_declaration_from_node(
                        declarations=declarations,
                        seen=seen_declarations,
                        code_bytes=code_bytes,
                        byte_to_char=byte_to_char,
                        scope=active_scope,
                        name_node=identifier_node,
                        kind=decl_kind,
                    )

        if is_scope_node(node, is_root=False):
            start_char, end_char = node_char_span(byte_to_char, node)
            current_scope = ScopeRecord(
                scope_id=next_scope_id,
                parent_scope_id=active_scope.scope_id,
                node_type=node_type,
                start_char=start_char,
                end_char=end_char,
                depth=active_scope.depth + 1,
            )
            next_scope_id += 1
            scopes.append(current_scope)

        inline_decl_kind = declaration_kind_for_node_type(language, node_type)
        if inline_decl_kind is not None:
            for candidate in declaration_candidate_subtrees(language, node):
                for identifier_node in collect_identifier_leaf_nodes(candidate):
                    identifier_name = node_byte_text(code_bytes, identifier_node)
                    identifier_start_char, _ = node_char_span(byte_to_char, identifier_node)
                    if (
                        inline_decl_kind in {"implicit_variable", "loop_variable"}
                        and declaration_exists_before_position(
                            declarations=declarations,
                            scopes=scopes,
                            scope=current_scope,
                            name=identifier_name,
                            position=identifier_start_char,
                        )
                    ):
                        continue
                    add_declaration_from_node(
                        declarations=declarations,
                        seen=seen_declarations,
                        code_bytes=code_bytes,
                        byte_to_char=byte_to_char,
                        scope=current_scope,
                        name_node=identifier_node,
                        kind=inline_decl_kind,
                    )

        for child in getattr(node, "children", []):
            walk(child, current_scope)

    for child in getattr(root_node, "children", []):
        walk(child, scopes[0])

    return scopes, declarations


def resolve_scope_at_char(scopes: Sequence[ScopeRecord], position: int) -> ScopeRecord:
    containing = [
        scope
        for scope in scopes
        if scope.start_char <= position <= scope.end_char
    ]
    if not containing:
        return scopes[0]
    return max(containing, key=lambda scope: (scope.depth, -(scope.end_char - scope.start_char)))


def resolve_visible_declarations(
    scopes: Sequence[ScopeRecord],
    declarations: Sequence[ScopeDeclaration],
    scope: ScopeRecord,
    position: int,
) -> List[ScopeDeclaration]:
    declarations_by_scope: Dict[int, List[ScopeDeclaration]] = {}
    for declaration in declarations:
        declarations_by_scope.setdefault(declaration.scope_id, []).append(declaration)

    scope_by_id = {candidate.scope_id: candidate for candidate in scopes}
    visible: Dict[str, ScopeDeclaration] = {}
    current_scope_id: Optional[int] = scope.scope_id

    while current_scope_id is not None:
        candidates = declarations_by_scope.get(current_scope_id, [])
        for declaration in sorted(candidates, key=lambda item: item.start_char, reverse=True):
            # A declaration becomes visible only after its full textual span has
            # been emitted. This prevents later sub-tokens of a split identifier
            # like `maxsize` -> `max` + `size` from being marked in-scope while
            # the declaration token itself is still being generated.
            if declaration.end_char <= position and declaration.name not in visible:
                visible[declaration.name] = declaration
        parent_scope_id = scope_by_id[current_scope_id].parent_scope_id
        current_scope_id = parent_scope_id

    return sorted(
        visible.values(),
        key=lambda item: (-item.scope_depth, item.start_char, item.name),
    )


def collect_tree_sitter_leaf_paths(code_bytes: bytes, node: Any, ancestors: Optional[List[str]] = None, results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if ancestors is None:
        ancestors = []
    if results is None:
        results = []

    current_path = ancestors + [node.type]
    if getattr(node, "child_count", 0) == 0:
        if node.end_byte > node.start_byte:
            results.append(
                {
                    "type": node.type,
                    "path": " > ".join(current_path),
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                }
            )
        return results

    for child in getattr(node, "children", []):
        collect_tree_sitter_leaf_paths(code_bytes, child, current_path, results)
    return results


def build_tree_sitter_char_labels(
    code: str,
    language: str,
    registry: TreeSitterParserRegistry,
    fail_on_error: bool = False,
) -> TreeSitterParseResult:
    char_paths = [TREE_SITTER_UNKNOWN] * len(code)
    char_leaf_types = [TREE_SITTER_UNKNOWN] * len(code)
    char_identifier_names = [""] * len(code)
    char_identifier_access_kinds = [""] * len(code)

    if not code:
        return TreeSitterParseResult(
            char_paths=char_paths,
            char_leaf_types=char_leaf_types,
            char_identifier_names=char_identifier_names,
            char_identifier_access_kinds=char_identifier_access_kinds,
            status="empty",
            error="",
            scopes=[],
            declarations=[],
            usages=[],
        )

    try:
        parser = registry.get_parser(language)
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        root_node = tree.root_node
        byte_to_char = build_byte_to_char_map(code)
        scopes, declarations = build_tree_sitter_scope_data(code, language, root_node, byte_to_char)
        usages = collect_identifier_usages(code_bytes, byte_to_char, root_node)

        for leaf in collect_tree_sitter_leaf_paths(code_bytes, root_node):
            start_char, end_char = byte_span_to_char_span(
                byte_to_char,
                leaf["start_byte"],
                leaf["end_byte"],
            )
            leaf_text = code[start_char:end_char]
            for char_index in range(start_char, min(end_char, len(code))):
                char_paths[char_index] = leaf["path"]
                char_leaf_types[char_index] = leaf["type"]
                if is_identifier_like_type(leaf["type"]):
                    char_identifier_names[char_index] = leaf_text

        for usage in usages:
            for char_index in range(usage.start_char, min(usage.end_char, len(code))):
                char_identifier_access_kinds[char_index] = usage.access_kind

        has_error = getattr(root_node, "has_error", False)
        return TreeSitterParseResult(
            char_paths=char_paths,
            char_leaf_types=char_leaf_types,
            char_identifier_names=char_identifier_names,
            char_identifier_access_kinds=char_identifier_access_kinds,
            status="parsed_with_errors" if has_error else "ok",
            error="",
            scopes=scopes,
            declarations=declarations,
            usages=usages,
        )
    except Exception as exc:
        if fail_on_error:
            raise
        LOGGER.warning("Tree-sitter failed for language=%s: %s", language, exc)
        return TreeSitterParseResult(
            char_paths=char_paths,
            char_leaf_types=char_leaf_types,
            char_identifier_names=char_identifier_names,
            char_identifier_access_kinds=char_identifier_access_kinds,
            status="error",
            error=str(exc),
            scopes=[],
            declarations=[],
            usages=[],
        )


def majority_vote(labels: Sequence[str], default: str) -> str:
    filtered = [label for label in labels if label]
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


def select_pygments_labels(simple_slice: Sequence[str], full_slice: Sequence[str]) -> tuple[str, str]:
    if not simple_slice or not full_slice:
        return PYGMENTS_SPECIAL, PYGMENTS_SPECIAL

    weak_labels = {"whitespace", "text"}
    informative_indices = [
        index for index, simple_label in enumerate(simple_slice) if simple_label not in weak_labels
    ]
    if informative_indices:
        simple_candidates = [simple_slice[index] for index in informative_indices]
        full_candidates = [full_slice[index] for index in informative_indices]
        return (
            majority_vote(simple_candidates, default="other"),
            majority_vote(full_candidates, default="other"),
        )

    return (
        majority_vote(simple_slice, default="other"),
        majority_vote(full_slice, default="other"),
    )


def select_tree_sitter_labels(path_slice: Sequence[str], leaf_slice: Sequence[str]) -> tuple[str, str]:
    if not path_slice or not leaf_slice:
        return TREE_SITTER_SPECIAL, TREE_SITTER_SPECIAL

    informative_indices = [
        index for index, path in enumerate(path_slice) if path not in {"", TREE_SITTER_UNKNOWN}
    ]
    if informative_indices:
        path_candidates = [path_slice[index] for index in informative_indices]
        leaf_candidates = [leaf_slice[index] for index in informative_indices]
        return (
            majority_vote(path_candidates, default=TREE_SITTER_UNKNOWN),
            majority_vote(leaf_candidates, default=TREE_SITTER_UNKNOWN),
        )

    return TREE_SITTER_UNKNOWN, TREE_SITTER_UNKNOWN


def find_token_pos_for_char(offsets: Sequence[tuple[int, int]], char_position: int) -> Optional[int]:
    for token_pos, (start, end) in enumerate(offsets):
        if start <= char_position < end:
            return token_pos
    for token_pos, (start, end) in enumerate(offsets):
        if end > char_position and start != end:
            return token_pos
    return None


def find_declaration_covering_token(
    declarations: Sequence[ScopeDeclaration],
    name: str,
    start: int,
    end: int,
) -> Optional[ScopeDeclaration]:
    for declaration in declarations:
        if (
            declaration.name == name
            and start < declaration.end_char
            and declaration.start_char < end
        ):
            return declaration
    return None


def find_usage_covering_token(
    usages: Sequence[NameUsage],
    name: str,
    start: int,
    end: int,
) -> Optional[NameUsage]:
    for usage in usages:
        if usage.name == name and start < usage.end_char and usage.start_char < end:
            return usage
    return None


def declaration_key(declaration: Optional[ScopeDeclaration], fallback_name: str) -> tuple[Any, ...]:
    if declaration is None:
        return ("unresolved", fallback_name)
    return (
        declaration.scope_id,
        declaration.name,
        declaration.start_char,
        declaration.end_char,
        declaration.kind,
    )

def span_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def build_rows_for_snippet(
    snippet: SnippetRecord,
    tokenizer: Any,
    parser_registry: TreeSitterParserRegistry,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    llm_tokens = tokenize_llm_code(
        snippet.code,
        tokenizer=tokenizer,
        add_special_tokens=args.add_special_tokens,
        truncation=args.truncation,
        max_length=args.max_length,
    )
    pygments_full, pygments_simple = build_pygments_char_labels(snippet.code, snippet.language)
    tree_sitter = build_tree_sitter_char_labels(
        snippet.code,
        snippet.language,
        registry=parser_registry,
        fail_on_error=args.fail_on_tree_sitter_error,
    )
    offsets = llm_tokens["offset_mapping"]
    declaration_token_pos_by_key = {
        declaration_key(declaration, declaration.name): find_token_pos_for_char(offsets, declaration.start_char)
        for declaration in tree_sitter.declarations
    }
    last_write_token_pos_by_key: Dict[tuple[Any, ...], int] = {}

    rows: List[Dict[str, Any]] = []
    for token_pos, (token_id, offsets, tokenizer_token, decoded_token) in enumerate(
        zip(
            llm_tokens["input_ids"],
            llm_tokens["offset_mapping"],
            llm_tokens["tokenizer_tokens"],
            llm_tokens["decoded_tokens"],
        )
    ):
        start, end = offsets
        scope_position = start if start < len(snippet.code) else max(len(snippet.code) - 1, 0)
        active_scope = resolve_scope_at_char(tree_sitter.scopes, scope_position) if tree_sitter.scopes else ScopeRecord(
            scope_id=0,
            parent_scope_id=None,
            node_type="root",
            start_char=0,
            end_char=len(snippet.code),
            depth=0,
        )
        visible_declarations = resolve_visible_declarations(
            tree_sitter.scopes if tree_sitter.scopes else [active_scope],
            tree_sitter.declarations,
            active_scope,
            scope_position,
        )

        if start == end:
            token_text = ""
            pygments_full_label = PYGMENTS_SPECIAL
            pygments_simple_label = PYGMENTS_SPECIAL
            tree_sitter_path = TREE_SITTER_SPECIAL
            tree_sitter_leaf = TREE_SITTER_SPECIAL
        else:
            token_text = snippet.code[start:end]
            pygments_simple_label, pygments_full_label = select_pygments_labels(
                pygments_simple[start:end],
                pygments_full[start:end],
            )
            tree_sitter_path, tree_sitter_leaf = select_tree_sitter_labels(
                tree_sitter.char_paths[start:end],
                tree_sitter.char_leaf_types[start:end],
            )

        identifier_name = (
            majority_vote(
                [name for name in tree_sitter.char_identifier_names[start:end] if name],
                default="",
            )
            if start != end
            else ""
        )
        visible_names = [declaration.name for declaration in visible_declarations]
        visible_kinds = [declaration.kind for declaration in visible_declarations]
        visible_decl_starts = [declaration.start_char for declaration in visible_declarations]
        visible_by_name = {declaration.name: declaration for declaration in visible_declarations}
        is_name_like_token = bool(identifier_name) or pygments_simple_label in NAME_LIKE_PYGMENTS_LABELS
        matched_name = identifier_name or token_text.strip()
        current_declaration = (
            find_declaration_covering_token(tree_sitter.declarations, matched_name, start, end)
            if is_name_like_token and matched_name and start != end
            else None
        )
        current_usage = (
            find_usage_covering_token(tree_sitter.usages, matched_name, start, end)
            if is_name_like_token and matched_name and start != end
            else None
        )
        matched_declaration = visible_by_name.get(matched_name) if is_name_like_token and matched_name else None
        token_name_is_decl = (
            current_declaration is not None
            and span_overlaps(
                start,
                end,
                current_declaration.start_char,
                current_declaration.end_char,
            )
        )
        effective_declaration = current_declaration if token_name_is_decl else matched_declaration
        effective_declaration_key = declaration_key(effective_declaration, matched_name) if matched_name else None
        declaration_token_pos = (
            declaration_token_pos_by_key.get(effective_declaration_key)
            if effective_declaration_key is not None
            else None
        )
        last_write_token_pos = (
            last_write_token_pos_by_key.get(effective_declaration_key)
            if effective_declaration_key is not None
            else None
        )
        access_kind = current_usage.access_kind if current_usage is not None else None
        is_read = access_kind in {"read", "read_write"}
        is_write = access_kind in {"write", "read_write"}
        is_read_write = access_kind == "read_write"

        row = {
            "snippet_id": snippet.snippet_id,
            "global_idx": snippet.global_idx,
            "language": snippet.language,
            "file_path": snippet.file_path,
            "source_snippet_ids": json.dumps(snippet.source_snippet_ids, ensure_ascii=False),
            "source_file_paths": json.dumps(snippet.source_file_paths, ensure_ascii=False),
            "num_fragments": snippet.num_fragments,
            "token_id": int(token_id),
            "token_pos": token_pos,
            "token_text": token_text,
            "tokenizer_token": tokenizer_token,
            "token_decoded": decoded_token,
            "token_offset_start": start,
            "token_offset_end": end,
            "pygments_label": pygments_full_label,
            "pygments_simple_label": pygments_simple_label,
            "tree_sitter_path": tree_sitter_path,
            "tree_sitter_leaf_type": tree_sitter_leaf,
            "tree_sitter_status": tree_sitter.status,
            "tree_sitter_error": tree_sitter.error,
            "scope_depth": active_scope.depth,
            "scope_node_type": active_scope.node_type,
            "num_in_scope_names": len(visible_names),
            "in_scope_names": json.dumps(visible_names, ensure_ascii=False),
            "in_scope_name_kinds": json.dumps(visible_kinds, ensure_ascii=False),
            "in_scope_decl_starts": json.dumps(visible_decl_starts, ensure_ascii=False),
            "is_name_like_token": is_name_like_token,
            "token_identifier_name": identifier_name or None,
            "token_name_is_in_scope": matched_declaration is not None,
            "token_name_is_decl": token_name_is_decl,
            "token_name_scope_kind": matched_declaration.kind if matched_declaration is not None else None,
            "token_variable_access_kind": access_kind,
            "token_variable_is_read": is_read,
            "token_variable_is_write": is_write,
            "token_variable_is_read_write": is_read_write,
            "token_variable_usage_start": current_usage.start_char if current_usage is not None else None,
            "token_variable_usage_end": current_usage.end_char if current_usage is not None else None,
            "token_name_declaration_token_pos": declaration_token_pos,
            "token_name_distance_to_declaration_tokens": (
                token_pos - declaration_token_pos if declaration_token_pos is not None else None
            ),
            "token_name_last_write_token_pos": last_write_token_pos,
            "token_name_distance_to_last_write_tokens": (
                token_pos - last_write_token_pos if last_write_token_pos is not None else None
            ),
        }
        if args.include_code:
            row["code"] = snippet.code
        rows.append(row)

        if (
            is_write
            and current_usage is not None
            and effective_declaration_key is not None
            and end >= current_usage.end_char
        ):
            last_write_token_pos_by_key[effective_declaration_key] = token_pos

    return rows


def infer_output_format(output_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if output_path.suffix == ".jsonl":
        return "jsonl"
    if output_path.suffix == ".parquet":
        return "parquet"
    if output_path.suffix == ".arrow":
        return "arrow"
    if output_path.suffix == "":
        return "hf"
    raise ValueError(
        "Could not infer output format from path. Pass --output-format explicitly."
    )


class BaseWriter:
    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class JsonlWriter(BaseWriter):
    def __init__(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = output_path.open("w", encoding="utf-8")

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        for row in rows:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._handle.close()


class PyArrowTableWriter(BaseWriter):
    def __init__(self, output_path: Path, file_format: str) -> None:
        try:
            import pyarrow as pa
            import pyarrow.ipc as pa_ipc
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                "pyarrow is required for Arrow/Parquet outputs. Install it with `pip install pyarrow`."
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._pa = pa
        self._pa_ipc = pa_ipc
        self._pq = pq
        self._file_format = file_format
        self._schema = pa.schema(
            [
                pa.field("snippet_id", pa.string()),
                pa.field("global_idx", pa.string()),
                pa.field("language", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("source_snippet_ids", pa.string()),
                pa.field("source_file_paths", pa.string()),
                pa.field("num_fragments", pa.int64()),
                pa.field("token_id", pa.int64()),
                pa.field("token_pos", pa.int64()),
                pa.field("token_text", pa.string()),
                pa.field("tokenizer_token", pa.string()),
                pa.field("token_decoded", pa.string()),
                pa.field("token_offset_start", pa.int64()),
                pa.field("token_offset_end", pa.int64()),
                pa.field("pygments_label", pa.string()),
                pa.field("pygments_simple_label", pa.string()),
                pa.field("tree_sitter_path", pa.string()),
                pa.field("tree_sitter_leaf_type", pa.string()),
                pa.field("tree_sitter_status", pa.string()),
                pa.field("tree_sitter_error", pa.string()),
                pa.field("scope_depth", pa.int64()),
                pa.field("scope_node_type", pa.string()),
                pa.field("num_in_scope_names", pa.int64()),
                pa.field("in_scope_names", pa.string()),
                pa.field("in_scope_name_kinds", pa.string()),
                pa.field("in_scope_decl_starts", pa.string()),
                pa.field("is_name_like_token", pa.bool_()),
                pa.field("token_identifier_name", pa.string()),
                pa.field("token_name_is_in_scope", pa.bool_()),
                pa.field("token_name_is_decl", pa.bool_()),
                pa.field("token_name_scope_kind", pa.string()),
                pa.field("token_variable_access_kind", pa.string()),
                pa.field("token_variable_is_read", pa.bool_()),
                pa.field("token_variable_is_write", pa.bool_()),
                pa.field("token_variable_is_read_write", pa.bool_()),
                pa.field("token_variable_usage_start", pa.int64()),
                pa.field("token_variable_usage_end", pa.int64()),
                pa.field("token_name_declaration_token_pos", pa.int64()),
                pa.field("token_name_distance_to_declaration_tokens", pa.int64()),
                pa.field("token_name_last_write_token_pos", pa.int64()),
                pa.field("token_name_distance_to_last_write_tokens", pa.int64()),
                pa.field("code", pa.string()),
            ]
        )
        self._output_path = output_path
        self._sink = pa.OSFile(str(output_path), "wb")
        self._writer = None

        if file_format == "parquet":
            self._writer = pq.ParquetWriter(self._sink, self._schema)
        elif file_format == "arrow":
            self._writer = pa_ipc.new_file(self._sink, self._schema)
        else:
            raise ValueError(f"Unsupported pyarrow format: {file_format}")

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        normalized_rows = []
        for row in rows:
            normalized = dict(row)
            normalized.setdefault("code", None)
            normalized.setdefault("token_identifier_name", None)
            normalized.setdefault("token_name_scope_kind", None)
            normalized.setdefault("token_variable_access_kind", None)
            normalized.setdefault("token_variable_usage_start", None)
            normalized.setdefault("token_variable_usage_end", None)
            normalized.setdefault("token_name_declaration_token_pos", None)
            normalized.setdefault("token_name_distance_to_declaration_tokens", None)
            normalized.setdefault("token_name_last_write_token_pos", None)
            normalized.setdefault("token_name_distance_to_last_write_tokens", None)
            normalized_rows.append(normalized)
        table = self._pa.Table.from_pylist(normalized_rows, schema=self._schema)
        if self._file_format == "parquet":
            self._writer.write_table(table)
        else:
            self._writer.write(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._sink.close()


class HuggingFaceDatasetWriter(BaseWriter):
    def __init__(self, output_path: Path) -> None:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise SystemExit(
                "datasets is required for Hugging Face dataset output. Install it with `pip install datasets pyarrow`."
            ) from exc

        self._dataset_cls = Dataset
        self._output_path = output_path
        self._temp_dir = Path(tempfile.mkdtemp(prefix="token_annotations_"))
        self._parquet_path = self._temp_dir / "data.parquet"
        self._parquet_writer = PyArrowTableWriter(self._parquet_path, file_format="parquet")

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        self._parquet_writer.write_rows(rows)

    def close(self) -> None:
        try:
            self._parquet_writer.close()
            dataset = self._dataset_cls.from_parquet(str(self._parquet_path))
            self._output_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(self._output_path))
        finally:
            shutil.rmtree(self._temp_dir, ignore_errors=True)


def create_writer(output_path: Path, output_format: str) -> BaseWriter:
    if output_format == "jsonl":
        return JsonlWriter(output_path)
    if output_format == "parquet":
        return PyArrowTableWriter(output_path, file_format="parquet")
    if output_format == "arrow":
        return PyArrowTableWriter(output_path, file_format="arrow")
    if output_format == "hf":
        return HuggingFaceDatasetWriter(output_path)
    raise ValueError(f"Unsupported output format: {output_format}")


def save_metadata(
    output_path: Path,
    output_format: str,
    args: argparse.Namespace,
    summary: Dict[str, Any],
) -> None:
    metadata_path = output_path / "metadata.json" if output_format == "hf" else output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "snippets_dir": args.snippets_dir,
        "output_format": output_format,
        "tokenizer_name": args.tokenizer_name,
        "languages": args.languages,
        "code_field": args.code_field,
        "group_variants": args.group_variants,
        "group_separator": args.group_separator,
        "max_snippets": args.max_snippets,
        "skip_snippets": args.skip_snippets,
        "max_length": args.max_length,
        "truncation": args.truncation,
        "add_special_tokens": args.add_special_tokens,
        "include_code": args.include_code,
        "writer_batch_size": args.writer_batch_size,
        **summary,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    snippets_dir = Path(args.snippets_dir)
    output_path = Path(args.output_path)
    output_format = infer_output_format(output_path, args.output_format)

    LOGGER.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = load_tokenizer(
        args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    parser_registry = TreeSitterParserRegistry()
    writer = create_writer(output_path, output_format)

    processed_snippets = 0
    failed_snippets = 0
    written_rows = 0
    buffer: List[Dict[str, Any]] = []
    tree_sitter_status_counter: Counter[str] = Counter()

    try:
        for snippet in iter_snippets(
            snippets_dir,
            languages=args.languages,
            code_field=args.code_field,
            group_variants=args.group_variants,
            group_separator=args.group_separator,
            skip_snippets=args.skip_snippets,
            max_snippets=args.max_snippets,
        ):
            try:
                rows = build_rows_for_snippet(
                    snippet,
                    tokenizer=tokenizer,
                    parser_registry=parser_registry,
                    args=args,
                )
            except Exception:
                failed_snippets += 1
                LOGGER.exception(
                    "Failed to process snippet_id=%s language=%s file=%s",
                    snippet.snippet_id,
                    snippet.language,
                    snippet.file_path,
                )
                if args.fail_on_tree_sitter_error:
                    raise
                continue

            if rows:
                tree_sitter_status_counter.update(row["tree_sitter_status"] for row in rows)
                buffer.extend(rows)
                written_rows += len(rows)

            processed_snippets += 1

            if len(buffer) >= args.writer_batch_size:
                writer.write_rows(buffer)
                buffer.clear()

            if processed_snippets % args.log_every == 0:
                LOGGER.info(
                    "Processed %d snippets | failed=%d | written_rows=%d",
                    processed_snippets,
                    failed_snippets,
                    written_rows,
                )

        if buffer:
            writer.write_rows(buffer)
            buffer.clear()
    finally:
        writer.close()

    summary = {
        "processed_snippets": processed_snippets,
        "failed_snippets": failed_snippets,
        "written_rows": written_rows,
        "tree_sitter_status_counts": dict(tree_sitter_status_counter),
    }
    save_metadata(output_path, output_format, args, summary)

    LOGGER.info(
        "Finished. processed_snippets=%d failed_snippets=%d written_rows=%d output=%s",
        processed_snippets,
        failed_snippets,
        written_rows,
        output_path,
    )


if __name__ == "__main__":
    main()
