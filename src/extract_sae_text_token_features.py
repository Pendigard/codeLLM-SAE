#!/usr/bin/env python3
"""Extract per-token SAE feature summaries for short natural-language texts.

This mirrors `extract_sae_token_features.py` as closely as possible so that the
resulting DataFrame can be compared with the code-token DataFrame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.extract_sae_token_features import (
    activation_stats,
    feature_threshold_column_name,
    load_model,
    load_sae,
    maybe_prepend_bos,
    parse_dtype,
    save_dataframe,
    token_ids_to_decoded_strings,
    token_ids_to_strings,
)
from text_dataset import TextTokenDataset, load_short_text_samples


DEFAULT_REFERENCE_DF_NAMES = [
    "code_sae_token_features.pkl",
    "code_sae_token_features.parquet",
    "code_sae_token_features.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a per-token DataFrame with top-K SAE features for short text.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--hook-name", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--only-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, top-K is computed on positive activations only.",
    )
    parser.add_argument(
        "--prepend-bos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, prepend the tokenizer BOS token before the model forward pass. "
            "This shifts token positions by one and inserts a synthetic BOS row."
        ),
    )
    parser.add_argument(
        "--include-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, duplicate the full raw text in every token row.",
    )
    parser.add_argument(
        "--feature-thresholds",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 1.0],
        help="Thresholds used to count active SAE features per token.",
    )
    parser.add_argument(
        "--save-metadata-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a sidecar JSON with extraction arguments and summary stats.",
    )
    parser.add_argument(
        "--target-token-rows",
        type=int,
        default=None,
        help="Approximate number of token rows to extract for the text DataFrame.",
    )
    parser.add_argument(
        "--target-rows-like",
        default=None,
        help=(
            "Optional reference DataFrame path. If provided, the text extraction "
            "targets approximately the same number of rows."
        ),
    )
    parser.add_argument(
        "--dataset-specs-json",
        default=None,
        help="Optional JSON file describing datasets to load instead of the default mix.",
    )
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=240)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-samples-per-dataset", type=int, default=10000)
    return parser.parse_args()


def load_dataframe_for_count(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported reference extension `{path.suffix}`. Use .pkl, .pickle, .parquet, or .csv."
    )


def infer_target_token_rows(target_rows_like: str | None, target_token_rows: int | None) -> int:
    if target_token_rows is not None:
        return int(target_token_rows)

    candidate_paths: list[Path] = []
    if target_rows_like is not None:
        candidate_paths.append(Path(target_rows_like))
    else:
        candidate_paths.extend(Path(name) for name in DEFAULT_REFERENCE_DF_NAMES)

    for candidate in candidate_paths:
        if candidate.exists():
            reference_df = load_dataframe_for_count(candidate)
            return int(len(reference_df))

    raise ValueError(
        "Could not infer a target number of token rows. Pass --target-token-rows or "
        "--target-rows-like /path/to/code_dataframe.pkl."
    )


def load_dataset_specs(dataset_specs_json: str | None) -> list[dict[str, Any]] | None:
    if dataset_specs_json is None:
        return None

    path = Path(dataset_specs_json)
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, list):
        raise ValueError("`--dataset-specs-json` must contain a JSON list of dataset specs.")

    return loaded


def build_rows_for_text_sample(
    sample: dict[str, Any],
    string_tokens: list[str],
    decoded_tokens: list[str],
    features: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seq_len = features.shape[0]
    text = sample["text"]

    for pos in range(seq_len):
        feature_vector = features[pos]
        values_for_topk = torch.clamp(feature_vector, min=0) if args.only_positive else feature_vector
        k_eff = min(args.top_k, values_for_topk.shape[0])
        top_vals_tensor, top_ids_tensor = torch.topk(values_for_topk, k=k_eff)

        top_vals = top_vals_tensor.tolist()
        top_ids = [int(idx) for idx in top_ids_tensor.tolist()]

        if args.only_positive:
            filtered = [(idx, float(val)) for idx, val in zip(top_ids, top_vals) if val > 0]
            top_ids = [idx for idx, _ in filtered]
            top_vals = [val for _, val in filtered]
        else:
            top_vals = [float(val) for val in top_vals]

        char_start, char_end = sample["offset_mapping"][pos]
        token_text = sample["token_texts"][pos]
        token_str = string_tokens[pos] if pos < len(string_tokens) else None
        token_decoded = decoded_tokens[pos] if pos < len(decoded_tokens) else None
        token_id = int(sample["input_ids"][pos])

        row = {
            "idx": sample["idx"],
            "global_idx": sample["global_idx"],
            "language": sample["language"],
            "modality": "text",
            "file_path": sample["file_path"],
            "problem_desc": None,
            "comment": None,
            "source_dataset": sample["source_dataset"],
            "source_split": sample["source_split"],
            "text_label": sample["text_label"],
            "text_label_id": sample["text_label_id"],
            "token_pos": pos,
            "seq_len": seq_len,
            "code_char_len": len(text),
            "text_char_len": len(text),
            "token_id": token_id,
            "token_str": token_str,
            "token_decoded": token_decoded,
            "token_text": token_text,
            "char_start": int(char_start),
            "char_end": int(char_end),
            "char_len": int(char_end - char_start),
            "is_special_token": bool(char_start == char_end),
            "is_whitespace_token": bool(token_text != "" and token_text.isspace()),
            "prev_token_str": string_tokens[pos - 1] if pos > 0 else None,
            "next_token_str": string_tokens[pos + 1] if pos + 1 < len(string_tokens) else None,
            "pygments_type": sample["pygments_types"][pos],
            "pygments_simple_type": sample["pygments_simple_types"][pos],
            "pygments_label_id": sample["pygments_label_ids"][pos],
            "top_k_feature_ids": top_ids,
            "top_k_feature_activations": top_vals,
            "top_k_found": len(top_ids),
            "model_name": args.model_name,
            "sae_release": args.sae_release,
            "sae_id": args.sae_id,
            "hook_name": args.hook_name,
        }
        row.update(activation_stats(feature_vector, top_ids, top_vals, args.feature_thresholds))

        for rank in range(args.top_k):
            row[f"top_{rank + 1}_feature_id"] = top_ids[rank] if rank < len(top_ids) else None
            row[f"top_{rank + 1}_feature_activation"] = top_vals[rank] if rank < len(top_vals) else None

        if args.include_text:
            row["text"] = text

        rows.append(row)

    return rows


def extract_dataframe(
    dataset: TextTokenDataset,
    model: Any,
    sae: Any,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sae_device = next(sae.parameters()).device

    for sample_idx in tqdm(range(len(dataset)), desc="Extracting SAE token features (text)"):
        sample = dataset[sample_idx]

        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long).unsqueeze(0)
        (
            input_ids,
            attention_mask,
            offset_mapping,
            token_texts,
            pygments_types,
            pygments_simple_types,
            pygments_label_ids,
        ) = maybe_prepend_bos(
            input_ids=input_ids,
            attention_mask=attention_mask,
            offset_mapping=list(sample["offset_mapping"]),
            token_texts=list(sample["token_texts"]),
            pygments_types=list(sample["pygments_types"]),
            pygments_simple_types=list(sample["pygments_simple_types"]),
            pygments_label_ids=list(sample["pygments_label_ids"]),
            tokenizer=model.tokenizer,
            prepend_bos=args.prepend_bos,
        )

        prepared_sample = dict(sample)
        prepared_sample["offset_mapping"] = offset_mapping
        prepared_sample["token_texts"] = token_texts
        prepared_sample["pygments_types"] = pygments_types
        prepared_sample["pygments_simple_types"] = pygments_simple_types
        prepared_sample["pygments_label_ids"] = pygments_label_ids
        prepared_sample["input_ids"] = input_ids.squeeze(0).tolist()

        with torch.inference_mode():
            input_ids = input_ids.to(model.cfg.device)
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[args.hook_name],
            )
            activations = cache[args.hook_name][0]
            features = sae.encode(activations.to(sae_device)).detach().cpu()

        string_tokens = token_ids_to_strings(model.tokenizer, prepared_sample["input_ids"])
        decoded_tokens = token_ids_to_decoded_strings(model.tokenizer, prepared_sample["input_ids"])
        rows.extend(build_rows_for_text_sample(prepared_sample, string_tokens, decoded_tokens, features, args))

        del cache

    return pd.DataFrame(rows)


def save_metadata(
    df: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    target_token_rows: int,
) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "args": vars(args),
        "target_token_rows": int(target_token_rows),
        "num_rows": int(len(df)),
        "num_texts": int(df["global_idx"].nunique()) if not df.empty else 0,
        "languages": sorted(df["language"].dropna().unique().tolist()) if not df.empty else [],
        "source_datasets": sorted(df["source_dataset"].dropna().unique().tolist()) if not df.empty else [],
        "columns": df.columns.tolist(),
        "feature_threshold_columns": [
            feature_threshold_column_name(threshold) for threshold in args.feature_thresholds
        ],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    output_path = Path(args.output_path)
    target_token_rows = infer_target_token_rows(args.target_rows_like, args.target_token_rows)
    dataset_specs = load_dataset_specs(args.dataset_specs_json)

    print(f"Loading model `{args.model_name}` on {args.device}...")
    model = load_model(args.model_name, args.device, dtype)

    print(f"Loading SAE `{args.sae_release}` / `{args.sae_id}`...")
    sae = load_sae(args.sae_release, args.sae_id, args.device)

    print(f"Selecting short-text samples to target about {target_token_rows} token rows...")
    samples = load_short_text_samples(
        tokenizer=model.tokenizer,
        target_token_rows=target_token_rows,
        dataset_specs=dataset_specs,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        shuffle_seed=args.shuffle_seed,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    dataset = TextTokenDataset(
        samples=samples,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False,
    )

    print(f"Loaded {len(dataset)} texts. Extracting top-{args.top_k} features per token...")
    df = extract_dataframe(dataset=dataset, model=model, sae=sae, args=args)

    print(f"Saving DataFrame to `{output_path}`...")
    save_dataframe(df, output_path)

    if args.save_metadata_json:
        save_metadata(df, output_path, args, target_token_rows)

    print(f"Done. Saved {len(df)} rows from {len(dataset)} texts.")


if __name__ == "__main__":
    main()
