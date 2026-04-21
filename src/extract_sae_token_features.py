#!/usr/bin/env python3
"""Extract per-token SAE feature summaries for code snippets.

This script builds a DataFrame with one row per LLM token. Each row contains:
- snippet metadata (`idx`, `global_idx`, `language`, `file_path`, ...)
- token metadata (token id, token string, char span, Pygments labels, ...)
- SAE statistics for the token activation vector
- the top-K most activated SAE features for that token

The output is saved to disk as a pickle/parquet/csv file depending on the
extension passed to `--output-path`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm

from code_dataset import CodeLLMPygmentsDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a per-token DataFrame with top-K SAE features for code snippets.",
    )
    parser.add_argument("--snippets-dir", default="code_snippets")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--hook-name", required=True)
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional subset of languages to keep, e.g. Python Java C++.",
    )
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
        "--include-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, duplicate the raw code in every token row.",
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
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def load_model(model_name: str, device: str, dtype: torch.dtype):
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit(
            "transformer_lens is not installed. Install it with `pip install transformer-lens`."
        ) from exc

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        center_writing_weights=False,
    )
    model.eval()
    return model


def load_sae(sae_release: str, sae_id: str, device: str):
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise SystemExit(
            "sae_lens is not installed. Install it with `pip install sae-lens`."
        ) from exc

    loaded = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()
    return sae


def token_ids_to_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return list(tokenizer.convert_ids_to_tokens(list(token_ids)))
    return [str(token_id) for token_id in token_ids]


def token_ids_to_decoded_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    if hasattr(tokenizer, "decode"):
        return [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in token_ids
        ]
    return [str(token_id) for token_id in token_ids]


def maybe_prepend_bos(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    token_texts: list[str],
    pygments_types: list[str],
    pygments_simple_types: list[str],
    pygments_label_ids: list[int],
    tokenizer: Any,
    prepend_bos: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[str], list[str], list[str], list[int]]:
    if not prepend_bos:
        return (
            input_ids,
            attention_mask,
            offset_mapping,
            token_texts,
            pygments_types,
            pygments_simple_types,
            pygments_label_ids,
        )

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise ValueError("`--prepend-bos` was set but the tokenizer has no `bos_token_id`.")

    bos_tensor = torch.tensor([[bos_token_id]], dtype=input_ids.dtype, device=input_ids.device)
    bos_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

    return (
        torch.cat([bos_tensor, input_ids], dim=1),
        torch.cat([bos_mask, attention_mask], dim=1),
        [(0, 0)] + offset_mapping,
        [""] + token_texts,
        ["special"] + pygments_types,
        ["special"] + pygments_simple_types,
        [-100] + pygments_label_ids,
    )


def feature_threshold_column_name(threshold: float) -> str:
    cleaned = str(threshold).replace("-", "neg_").replace(".", "_")
    return f"sae_active_ge_{cleaned}"


def activation_stats(
    feature_vector: torch.Tensor,
    top_ids: list[int],
    top_vals: list[float],
    thresholds: Iterable[float],
) -> dict[str, Any]:
    positive = torch.clamp(feature_vector, min=0)
    l0_positive = int((positive > 0).sum().item())
    l1_positive = float(positive.sum().item())
    l2 = float(torch.linalg.vector_norm(feature_vector, ord=2).item())
    max_activation = float(feature_vector.max().item())
    mean_activation = float(feature_vector.mean().item())
    std_activation = float(feature_vector.std(unbiased=False).item())
    top_k_sum = float(sum(top_vals))
    top_k_fraction = float(top_k_sum / l1_positive) if l1_positive > 0 else 0.0

    stats = {
        "sae_width": int(feature_vector.shape[0]),
        "sae_l0_positive": l0_positive,
        "sae_l1_positive": l1_positive,
        "sae_l2": l2,
        "sae_max_activation": max_activation,
        "sae_mean_activation": mean_activation,
        "sae_std_activation": std_activation,
        "top_k_activation_sum": top_k_sum,
        "top_k_fraction_of_l1": top_k_fraction,
        "top_feature_id": top_ids[0] if top_ids else None,
        "top_feature_activation": top_vals[0] if top_vals else None,
    }

    for threshold in thresholds:
        stats[feature_threshold_column_name(threshold)] = int((feature_vector >= threshold).sum().item())

    return stats


def build_rows_for_sample(
    sample: dict[str, Any],
    string_tokens: list[str],
    decoded_tokens: list[str],
    features: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seq_len = features.shape[0]

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
        simple_type = sample["pygments_simple_types"][pos]
        full_type = sample["pygments_types"][pos]

        row = {
            "idx": sample["idx"],
            "global_idx": sample["global_idx"],
            "language": sample["language"],
            "file_path": sample["file_path"],
            "problem_desc": sample["problem_desc"],
            "comment": sample["comment"],
            "token_pos": pos,
            "seq_len": seq_len,
            "code_char_len": len(sample["code"]),
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
            "pygments_type": full_type,
            "pygments_simple_type": simple_type,
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

        if args.include_code:
            row["code"] = sample["code"]

        rows.append(row)

    return rows


def extract_dataframe(
    dataset: CodeLLMPygmentsDataset,
    model: Any,
    sae: Any,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sae_device = next(sae.parameters()).device

    for sample_idx in tqdm(range(len(dataset)), desc="Extracting SAE token features"):
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
        rows.extend(build_rows_for_sample(prepared_sample, string_tokens, decoded_tokens, features, args))

        del cache

    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_path)
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension `{output_path.suffix}`. "
            "Use .pkl, .pickle, .parquet, or .csv."
        )


def save_metadata(df: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "args": vars(args),
        "num_rows": int(len(df)),
        "num_snippets": int(df["global_idx"].nunique()) if not df.empty else 0,
        "languages": sorted(df["language"].dropna().unique().tolist()) if not df.empty else [],
        "columns": df.columns.tolist(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    output_path = Path(args.output_path)

    print(f"Loading model `{args.model_name}` on {args.device}...")
    model = load_model(args.model_name, args.device, dtype)

    print(f"Loading SAE `{args.sae_release}` / `{args.sae_id}`...")
    sae = load_sae(args.sae_release, args.sae_id, args.device)

    dataset = CodeLLMPygmentsDataset(
        directory=args.snippets_dir,
        tokenizer=model.tokenizer,
        languages=args.languages,
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False,
    )

    print(f"Loaded {len(dataset)} snippets. Extracting top-{args.top_k} features per token...")
    df = extract_dataframe(dataset=dataset, model=model, sae=sae, args=args)

    print(f"Saving DataFrame to `{output_path}`...")
    save_dataframe(df, output_path)

    if args.save_metadata_json:
        save_metadata(df, output_path, args)

    print(f"Done. Saved {len(df)} rows.")


if __name__ == "__main__":
    main()
