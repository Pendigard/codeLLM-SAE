import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from torch.utils.data import DataLoader, Dataset


LANGUAGE_TO_PYGMENTS = {
    "C": "c",
    "C#": "csharp",
    "C++": "cpp",
    "Java": "java",
    "Javascript": "javascript",
    "PHP": "php",
    "Python": "python",
}

SIMPLIFIED_TOKEN_TYPES = [
    "special",
    "whitespace",
    "keyword",
    "function",
    "class",
    "namespace",
    "builtin",
    "decorator",
    "variable",
    "name",
    "string",
    "number",
    "operator",
    "punctuation",
    "comment",
    "text",
    "other",
]

TOKEN_TYPE_TO_ID = {token_type: idx for idx, token_type in enumerate(SIMPLIFIED_TOKEN_TYPES)}


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


def _normalize_global_idx(idx: str) -> str:
    return re.sub(r"-[^-]+-", "-", idx)


def _load_snippets(directory: Path, languages: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    selected_languages = set(languages) if languages is not None else None
    snippets: List[Dict[str, Any]] = []

    for language_dir in sorted(directory.iterdir()):
        if not language_dir.is_dir():
            continue

        language = language_dir.name
        if selected_languages is not None and language not in selected_languages:
            continue

        for file_path in sorted(language_dir.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as handle:
                item = json.load(handle)

            item["language"] = language
            item["global_idx"] = _normalize_global_idx(item["idx"])
            item["file_path"] = str(file_path)
            snippets.append(item)

    return snippets


def _get_lexer(language: str):
    pygments_language = LANGUAGE_TO_PYGMENTS.get(language, language.lower())
    try:
        return get_lexer_by_name(pygments_language)
    except ClassNotFound as exc:
        raise ValueError(
            f"Impossible de trouver un lexer Pygments pour le langage '{language}'."
        ) from exc


def _char_level_pygments_labels(code: str, language: str) -> Tuple[List[str], List[str]]:
    lexer = _get_lexer(language)

    if not code:
        return [], []

    full_labels = ["other"] * len(code)
    simple_labels = ["other"] * len(code)
    for token_start, token_type, token_value in lexer.get_tokens_unprocessed(code):
        if not token_value:
            continue

        token_full_type = str(token_type)
        token_simple_type = simplify_token_type(token_type)
        token_end = token_start + len(token_value)

        for char_index in range(token_start, min(token_end, len(code))):
            full_labels[char_index] = token_full_type
            simple_labels[char_index] = token_simple_type

    return full_labels, simple_labels


def _majority_vote(labels: Sequence[str]) -> str:
    if not labels:
        return "special"

    counts: Dict[str, int] = {}
    best_label = labels[0]
    best_count = 0

    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]

    return best_label


def _select_token_labels(
    simple_slice: Sequence[str],
    full_slice: Sequence[str],
) -> Tuple[str, str]:
    if not simple_slice or not full_slice:
        return "special", "special"

    # If a tokenizer token spans both whitespace/text and a real syntax token
    # (e.g. "▁="), prefer the informative syntax label.
    weak_simple_labels = {"whitespace", "text"}
    informative_indices = [
        i for i, simple_label in enumerate(simple_slice) if simple_label not in weak_simple_labels
    ]

    if informative_indices:
        informative_simple = [simple_slice[i] for i in informative_indices]
        informative_full = [full_slice[i] for i in informative_indices]
        return _majority_vote(informative_simple), _majority_vote(informative_full)

    return _majority_vote(simple_slice), _majority_vote(full_slice)


def _token_text_from_offset(code: str, start: int, end: int) -> str:
    if start == end:
        return ""
    return code[start:end]


class CodeLLMPygmentsDataset(Dataset):
    def __init__(
        self,
        directory: str,
        tokenizer: Any,
        languages: Optional[Iterable[str]] = None,
        code_field: str = "code",
        truncation: bool = True,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        label_pad_token_id: int = -100,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.directory = Path(directory)
        self.tokenizer = tokenizer
        self.languages = sorted(languages) if languages is not None else None
        self.code_field = code_field
        self.truncation = truncation
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.label_pad_token_id = label_pad_token_id
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.samples = _load_snippets(self.directory, self.languages)
        self.token_type_to_id = TOKEN_TYPE_TO_ID
        self.id_to_token_type = {idx: token_type for token_type, idx in TOKEN_TYPE_TO_ID.items()}

        if not self.samples:
            raise ValueError(f"Aucun snippet charge depuis '{self.directory}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        code = sample[self.code_field]
        language = sample["language"]

        full_char_labels, simple_char_labels = _char_level_pygments_labels(code, language)
        encoding = self._tokenize_code(code)
        offsets = encoding.pop("offset_mapping")

        token_simple_types: List[str] = []
        token_full_types: List[str] = []
        pygments_label_ids: List[int] = []
        token_texts: List[str] = []

        for start, end in offsets:
            if start == end:
                simple_label = "special"
                full_label = "special"
            else:
                simple_slice = simple_char_labels[start:end]
                full_slice = full_char_labels[start:end]
                simple_label, full_label = _select_token_labels(simple_slice, full_slice)

            token_simple_types.append(simple_label)
            token_full_types.append(full_label)
            pygments_label_ids.append(self.token_type_to_id[simple_label])
            token_texts.append(_token_text_from_offset(code, start, end))

        item = {
            "idx": sample["idx"],
            "global_idx": sample["global_idx"],
            "language": language,
            "file_path": sample.get("file_path"),
            "code": code,
            "snippet": sample.get("snippet"),
            "comment": sample.get("comment"),
            "problem_desc": sample.get("problem_desc"),
            "tokens": sample.get("tokens"),
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": offsets,
            "token_texts": token_texts,
            "pygments_types": token_full_types,
            "pygments_simple_types": token_simple_types,
            "pygments_label_ids": pygments_label_ids,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"]

        return item

    def _tokenize_code(self, code: str) -> Dict[str, List[int]]:
        tokenizer_kwargs = {
            "add_special_tokens": self.add_special_tokens,
            "truncation": self.truncation,
            "return_offsets_mapping": True,
            **self.tokenizer_kwargs,
        }

        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length

        encoding = self.tokenizer(code, **tokenizer_kwargs)

        if "offset_mapping" not in encoding:
            raise ValueError(
                "Le tokenizer doit supporter `return_offsets_mapping=True`. "
                "Utilise de preference un fast tokenizer Hugging Face."
            )

        return {
            key: value
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "token_type_ids", "offset_mapping"}
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs: List[Dict[str, List[int]]] = []
        max_seq_len = max(len(item["pygments_label_ids"]) for item in batch)

        for item in batch:
            encoded_input = {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            if "token_type_ids" in item:
                encoded_input["token_type_ids"] = item["token_type_ids"]
            encoded_inputs.append(encoded_input)

        padded_inputs = self.tokenizer.pad(
            encoded_inputs,
            padding=True,
            return_tensors="pt",
        )

        padded_labels = torch.full(
            (len(batch), max_seq_len),
            fill_value=self.label_pad_token_id,
            dtype=torch.long,
        )
        for row_idx, item in enumerate(batch):
            labels = torch.tensor(item["pygments_label_ids"], dtype=torch.long)
            padded_labels[row_idx, : len(labels)] = labels

        metadata = {
            "idx": [item["idx"] for item in batch],
            "global_idx": [item["global_idx"] for item in batch],
            "language": [item["language"] for item in batch],
            "file_path": [item["file_path"] for item in batch],
            "code": [item["code"] for item in batch],
            "snippet": [item["snippet"] for item in batch],
            "comment": [item["comment"] for item in batch],
            "problem_desc": [item["problem_desc"] for item in batch],
            "tokens": [item["tokens"] for item in batch],
            "offset_mapping": [item["offset_mapping"] for item in batch],
            "token_texts": [item["token_texts"] for item in batch],
            "pygments_types": [item["pygments_types"] for item in batch],
            "pygments_simple_types": [item["pygments_simple_types"] for item in batch],
        }

        padded_inputs["pygments_label_ids"] = padded_labels
        padded_inputs.update(metadata)
        return padded_inputs


def build_code_dataloader(
    directory: str,
    tokenizer: Any,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    languages: Optional[Iterable[str]] = None,
    max_length: Optional[int] = None,
    truncation: bool = True,
    add_special_tokens: bool = True,
    label_pad_token_id: int = -100,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **dataloader_kwargs: Any,
) -> Tuple[CodeLLMPygmentsDataset, DataLoader]:
    dataset = CodeLLMPygmentsDataset(
        directory=directory,
        tokenizer=tokenizer,
        languages=languages,
        max_length=max_length,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        label_pad_token_id=label_pad_token_id,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        **dataloader_kwargs,
    )

    return dataset, dataloader


if __name__ == "__main__":
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Installe `transformers` pour lancer l'exemple: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    dataset, dataloader = build_code_dataloader(
        directory="code_snippets",
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        languages=["Python", "Java", "Javascript", "C++", "C"],
        max_length=256,
    )

    batch = next(iter(dataloader))
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch languages: {batch['language']}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"pygments_label_ids shape: {tuple(batch['pygments_label_ids'].shape)}")
