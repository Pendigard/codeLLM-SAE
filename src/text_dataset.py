from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from torch.utils.data import Dataset


TEXT_SIMPLIFIED_TYPES = [
    "special",
    "whitespace",
    "word",
    "number",
    "punctuation",
    "mixed",
    "other",
]

TEXT_TOKEN_TYPE_TO_ID = {
    token_type: idx for idx, token_type in enumerate(TEXT_SIMPLIFIED_TYPES)
}


@dataclass
class TextSample:
    idx: str
    global_idx: str
    language: str
    file_path: Optional[str]
    text: str
    source_dataset: str
    source_split: str
    text_label: Optional[str]
    text_label_id: Optional[int]


def _char_type(character: str) -> tuple[str, str]:
    if character.isspace():
        return "Text.Whitespace", "whitespace"
    if character.isalpha():
        return "Text.Word", "word"
    if character.isdigit():
        return "Text.Number", "number"
    if not character.isalnum():
        return "Text.Punctuation", "punctuation"
    return "Text.Other", "other"


def _char_level_text_labels(text: str) -> tuple[list[str], list[str]]:
    full_labels: list[str] = []
    simple_labels: list[str] = []

    for character in text:
        full_label, simple_label = _char_type(character)
        full_labels.append(full_label)
        simple_labels.append(simple_label)

    return full_labels, simple_labels


def _majority_vote(labels: list[str]) -> str:
    if not labels:
        return "special"

    counts: dict[str, int] = {}
    best_label = labels[0]
    best_count = 0

    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]

    return best_label


def _select_token_labels(simple_slice: list[str], full_slice: list[str]) -> tuple[str, str]:
    if not simple_slice or not full_slice:
        return "special", "special"

    weak_simple_labels = {"whitespace", "other"}
    informative_indices = [
        idx for idx, simple_label in enumerate(simple_slice) if simple_label not in weak_simple_labels
    ]

    if informative_indices:
        informative_simple = [simple_slice[idx] for idx in informative_indices]
        informative_full = [full_slice[idx] for idx in informative_indices]
        simple_label = _majority_vote(informative_simple)
        full_label = _majority_vote(informative_full)
    else:
        simple_label = _majority_vote(simple_slice)
        full_label = _majority_vote(full_slice)

    if len(set(simple_slice)) > 1 and simple_label not in {"special", "whitespace"}:
        simple_label = "mixed"
        full_label = "Text.Mixed"

    return simple_label, full_label


def _token_text_from_offset(text: str, start: int, end: int) -> str:
    if start == end:
        return ""
    return text[start:end]


def _label_names(dataset_split: Any, label_field: str) -> Optional[list[str]]:
    if label_field not in dataset_split.features:
        return None

    feature = dataset_split.features[label_field]
    names = getattr(feature, "names", None)
    if names is None:
        return None
    return list(names)


def load_short_text_samples(
    tokenizer: Any,
    target_token_rows: int,
    dataset_specs: Optional[Iterable[dict[str, Any]]] = None,
    min_chars: int = 12,
    max_chars: int = 240,
    shuffle_seed: int = 0,
    max_samples_per_dataset: int = 10000,
) -> list[TextSample]:
    """Load a diverse short-text pool and stop near a target token budget."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required for text extraction. Install it with `pip install datasets`."
        ) from exc

    if dataset_specs is None:
        dataset_specs = [
            {
                "path": "ag_news",
                "config": None,
                "split": "train",
                "text_field": "text",
                "label_field": "label",
            },
            {
                "path": "glue",
                "config": "sst2",
                "split": "train",
                "text_field": "sentence",
                "label_field": "label",
            },
            {
                "path": "trec",
                "config": None,
                "split": "train",
                "text_field": "text",
                "label_field": "coarse_label",
            },
            {
                "path": "rotten_tomatoes",
                "config": None,
                "split": "train",
                "text_field": "text",
                "label_field": "label",
            },
            {
                "path": "emotion",
                "config": None,
                "split": "train",
                "text_field": "text",
                "label_field": "label",
            },
        ]

    samples: list[TextSample] = []
    total_token_rows = 0

    for spec in dataset_specs:
        dataset = load_dataset(
            spec["path"],
            spec.get("config"),
            split=spec["split"],
        )
        dataset = dataset.shuffle(seed=shuffle_seed)

        label_names = _label_names(dataset, spec.get("label_field", "label"))
        dataset_name = (
            spec["path"]
            if spec.get("config") is None
            else f"{spec['path']}:{spec['config']}"
        )

        added = 0
        for example_idx, example in enumerate(dataset):
            text = str(example[spec["text_field"]]).strip()
            if len(text) < min_chars or len(text) > max_chars:
                continue

            tokenized = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                return_attention_mask=False,
            )
            token_count = len(tokenized["input_ids"])
            if token_count == 0:
                continue

            label_id = example.get(spec.get("label_field", "label"))
            label_text = None
            if label_names is not None and label_id is not None and not isinstance(label_id, str):
                if 0 <= int(label_id) < len(label_names):
                    label_text = label_names[int(label_id)]
            elif label_id is not None:
                label_text = str(label_id)

            sample_idx = f"{dataset_name}-{spec['split']}-{example_idx}"
            samples.append(
                TextSample(
                    idx=sample_idx,
                    global_idx=sample_idx,
                    language="text",
                    file_path=None,
                    text=text,
                    source_dataset=dataset_name,
                    source_split=spec["split"],
                    text_label=label_text,
                    text_label_id=None if label_id is None else int(label_id),
                )
            )
            total_token_rows += token_count
            added += 1

            enough_global = target_token_rows > 0 and total_token_rows >= target_token_rows
            enough_local = added >= max_samples_per_dataset
            if enough_global or enough_local:
                break

        if target_token_rows > 0 and total_token_rows >= target_token_rows:
            break

    return samples


class TextTokenDataset(Dataset):
    def __init__(
        self,
        samples: list[TextSample],
        tokenizer: Any,
        truncation: bool = True,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        if not self.samples:
            raise ValueError("No text samples were loaded.")

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_text(self, text: str) -> dict[str, list[int]]:
        tokenizer_kwargs = {
            "add_special_tokens": self.add_special_tokens,
            "truncation": self.truncation,
            "return_offsets_mapping": True,
            **self.tokenizer_kwargs,
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length

        encoding = self.tokenizer(text, **tokenizer_kwargs)
        if "offset_mapping" not in encoding:
            raise ValueError(
                "The tokenizer must support `return_offsets_mapping=True`."
            )

        return {
            key: value
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "token_type_ids", "offset_mapping"}
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        text = sample.text

        full_char_labels, simple_char_labels = _char_level_text_labels(text)
        encoding = self._tokenize_text(text)
        offsets = encoding.pop("offset_mapping")

        token_simple_types: list[str] = []
        token_full_types: list[str] = []
        token_texts: list[str] = []
        token_type_ids: list[int] = []

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
            token_texts.append(_token_text_from_offset(text, start, end))
            token_type_ids.append(TEXT_TOKEN_TYPE_TO_ID.get(simple_label, TEXT_TOKEN_TYPE_TO_ID["other"]))

        item = {
            "idx": sample.idx,
            "global_idx": sample.global_idx,
            "language": sample.language,
            "file_path": sample.file_path,
            "text": text,
            "source_dataset": sample.source_dataset,
            "source_split": sample.source_split,
            "text_label": sample.text_label,
            "text_label_id": sample.text_label_id,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": offsets,
            "token_texts": token_texts,
            "pygments_types": token_full_types,
            "pygments_simple_types": token_simple_types,
            "pygments_label_ids": token_type_ids,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"]

        return item
