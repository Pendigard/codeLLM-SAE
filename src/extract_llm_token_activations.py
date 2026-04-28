#!/usr/bin/env python3
"""Extract per-token LLM activations for concatenated code snippets.

This script mirrors the grouping logic used for token annotations:
- snippet fragments like 1800-1 ... 1800-5 can be concatenated first
- the concatenated code is fed to the model once
- one output row is written per tokenizer token

Each row contains enough metadata to join with the token-label dataset:
- snippet_id
- global_idx
- language
- token_id
- token_pos

The activation vector for the requested layer/hook is stored in each row.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.build_token_annotation_dataset import iter_snippets

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


LOGGER = logging.getLogger("extract_llm_token_activations")


class _NullProgress:
    def __init__(self, iterable: Sequence[Any]) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, **_: Any) -> None:
        return


def make_progress(iterable: Sequence[Any], desc: str, unit: str):
    if tqdm is None:
        LOGGER.warning("tqdm is not installed; falling back to console logs without a progress bar.")
        return _NullProgress(iterable)
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one activation vector per token for grouped code snippets.",
    )
    parser.add_argument("--snippets-dir", default="code_snippets")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--output-format",
        choices=["auto", "jsonl", "parquet", "arrow", "hf"],
        default="auto",
    )
    parser.add_argument("--model-name", required=True)
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
        help="Merge snippet fragments sharing the same base id before the forward pass.",
    )
    parser.add_argument(
        "--group-separator",
        default="",
        help="String inserted between grouped code fragments.",
    )
    parser.add_argument("--max-snippets", type=int, default=None)
    parser.add_argument("--skip-snippets", type=int, default=0)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--activation-kind",
        default="resid_post",
        choices=["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"],
        help="Hook family used to extract activations if --hook-name is not given.",
    )
    parser.add_argument(
        "--hook-name",
        default=None,
        help="Optional explicit TransformerLens hook name. Overrides --layer/--activation-kind.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--storage-dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Datatype used to store activation vectors on disk.",
    )
    parser.add_argument(
        "--add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default so token positions match the label dataset.",
    )
    parser.add_argument(
        "--truncation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable tokenizer truncation.",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=256,
        help="Number of token rows buffered before flushing to disk.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--save-metadata-json",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def parse_torch_dtype(name: str):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required. Install it with `pip install torch`.") from exc

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


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


def build_hook_name(layer: int, activation_kind: str, explicit_hook_name: Optional[str]) -> str:
    if explicit_hook_name:
        return explicit_hook_name
    return f"blocks.{layer}.hook_{activation_kind}"


def load_model(model_name: str, device: str, dtype_name: str, trust_remote_code: bool = False) -> Any:
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit(
            "transformer_lens is required. Install it with `pip install transformer-lens`."
        ) from exc

    dtype = parse_torch_dtype(dtype_name)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        center_writing_weights=False,
    )
    model.eval()
    LOGGER.info("Model loaded successfully")
    return model


def tokenize_code(
    tokenizer: Any,
    code: str,
    add_special_tokens: bool,
    truncation: bool,
    max_length: Optional[int],
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
            "Use a fast Hugging Face tokenizer."
        )

    input_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
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


def enforce_context_limit(seq_len: int, model: Any, snippet_id: str, truncation: bool) -> None:
    model_ctx = getattr(getattr(model, "cfg", None), "n_ctx", None)
    if model_ctx is None:
        return
    if seq_len <= model_ctx:
        return
    if truncation:
        LOGGER.warning(
            "Snippet %s produced %d tokens which exceeds model context %d. "
            "Tokenizer truncation is enabled, so later tokens may be dropped.",
            snippet_id,
            seq_len,
            model_ctx,
        )
        return
    raise ValueError(
        f"Snippet {snippet_id} produced {seq_len} tokens which exceeds model context "
        f"{model_ctx}. Re-run with --truncation and optionally --max-length."
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


class PyArrowActivationWriter(BaseWriter):
    def __init__(self, output_path: Path, file_format: str, storage_dtype: str) -> None:
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
        value_type = pa.float16() if storage_dtype == "float16" else pa.float32()
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
                pa.field("is_special_token", pa.bool_()),
                pa.field("activation", pa.list_(value_type)),
                pa.field("activation_dim", pa.int64()),
                pa.field("model_name", pa.string()),
                pa.field("layer", pa.int64()),
                pa.field("activation_kind", pa.string()),
                pa.field("hook_name", pa.string()),
                pa.field("code", pa.string()),
            ]
        )
        self._sink = pa.OSFile(str(output_path), "wb")
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
            normalized_rows.append(normalized)
        table = self._pa.Table.from_pylist(normalized_rows, schema=self._schema)
        if self._file_format == "parquet":
            self._writer.write_table(table)
        else:
            self._writer.write(table)

    def close(self) -> None:
        self._writer.close()
        self._sink.close()


class HuggingFaceDatasetWriter(BaseWriter):
    def __init__(self, output_path: Path, storage_dtype: str) -> None:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise SystemExit(
                "datasets is required for Hugging Face dataset output. Install it with `pip install datasets pyarrow`."
            ) from exc

        self._dataset_cls = Dataset
        self._output_path = output_path
        self._temp_dir = Path(tempfile.mkdtemp(prefix="llm_activations_"))
        self._parquet_path = self._temp_dir / "data.parquet"
        self._parquet_writer = PyArrowActivationWriter(
            self._parquet_path,
            file_format="parquet",
            storage_dtype=storage_dtype,
        )

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


def create_writer(output_path: Path, output_format: str, storage_dtype: str) -> BaseWriter:
    if output_format == "jsonl":
        return JsonlWriter(output_path)
    if output_format == "parquet":
        return PyArrowActivationWriter(output_path, file_format="parquet", storage_dtype=storage_dtype)
    if output_format == "arrow":
        return PyArrowActivationWriter(output_path, file_format="arrow", storage_dtype=storage_dtype)
    if output_format == "hf":
        return HuggingFaceDatasetWriter(output_path, storage_dtype=storage_dtype)
    raise ValueError(f"Unsupported output format: {output_format}")


def maybe_cast_activation(activation_vector: Any, storage_dtype: str) -> List[float]:
    if storage_dtype == "float16":
        activation_vector = activation_vector.to(dtype=parse_torch_dtype("float16"))
    else:
        activation_vector = activation_vector.to(dtype=parse_torch_dtype("float32"))
    return activation_vector.tolist()


def build_rows_for_snippet(
    snippet: Any,
    tokenized: Dict[str, Any],
    activations: Any,
    args: argparse.Namespace,
    hook_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seq_len = activations.shape[0]

    if seq_len != len(tokenized["input_ids"]):
        raise ValueError(
            f"Activation length mismatch for snippet {snippet.snippet_id}: "
            f"{seq_len} activations vs {len(tokenized['input_ids'])} tokens."
        )

    for token_pos, token_id in enumerate(tokenized["input_ids"]):
        start, end = tokenized["offset_mapping"][token_pos]
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
            "token_text": snippet.code[start:end] if start != end else "",
            "tokenizer_token": tokenized["tokenizer_tokens"][token_pos],
            "token_decoded": tokenized["decoded_tokens"][token_pos],
            "token_offset_start": int(start),
            "token_offset_end": int(end),
            "is_special_token": bool(start == end),
            "activation": maybe_cast_activation(activations[token_pos], args.storage_dtype),
            "activation_dim": int(activations.shape[-1]),
            "model_name": args.model_name,
            "layer": args.layer,
            "activation_kind": args.activation_kind,
            "hook_name": hook_name,
        }
        if args.include_code:
            row["code"] = snippet.code
        rows.append(row)

    return rows


def extract_rows(args: argparse.Namespace, model: Any, hook_name: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required. Install it with `pip install torch`.") from exc

    processed_snippets = 0
    failed_snippets = 0
    written_rows = 0
    activation_dim: Optional[int] = None
    buffer: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    snippets = list(
        iter_snippets(
            Path(args.snippets_dir),
            languages=args.languages,
            code_field=args.code_field,
            group_variants=args.group_variants,
            group_separator=args.group_separator,
            skip_snippets=args.skip_snippets,
            max_snippets=args.max_snippets,
        )
    )

    LOGGER.info(
        "Prepared %d grouped snippets for extraction%s",
        len(snippets),
        f" across languages={args.languages}" if args.languages else "",
    )

    writer = create_writer(
        Path(args.output_path),
        infer_output_format(Path(args.output_path), args.output_format),
        args.storage_dtype,
    )
    LOGGER.info("Writing activations to %s", args.output_path)
    try:
        progress = make_progress(
            snippets,
            desc="Extracting LLM activations",
            unit="snippet",
        )
        for snippet in progress:
            try:
                tokenized = tokenize_code(
                    model.tokenizer,
                    snippet.code,
                    add_special_tokens=args.add_special_tokens,
                    truncation=args.truncation,
                    max_length=args.max_length,
                )
                enforce_context_limit(
                    seq_len=len(tokenized["input_ids"]),
                    model=model,
                    snippet_id=snippet.snippet_id,
                    truncation=args.truncation,
                )

                input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=model.cfg.device).unsqueeze(0)
                with torch.inference_mode():
                    _, cache = model.run_with_cache(
                        input_ids,
                        names_filter=[hook_name],
                    )
                    activations = cache[hook_name][0].detach().cpu()
                del cache

                if activation_dim is None:
                    activation_dim = int(activations.shape[-1])

                rows = build_rows_for_snippet(
                    snippet=snippet,
                    tokenized=tokenized,
                    activations=activations,
                    args=args,
                    hook_name=hook_name,
                )
                buffer.extend(rows)
                written_rows += len(rows)
                processed_snippets += 1

                if len(buffer) >= args.writer_batch_size:
                    LOGGER.info("Flushing %d token rows to disk", len(buffer))
                    writer.write_rows(buffer)
                    buffer.clear()

                progress.set_postfix(
                    rows=written_rows,
                    failed=failed_snippets,
                    tokens=len(tokenized["input_ids"]),
                )

                if processed_snippets % args.log_every == 0:
                    LOGGER.info(
                        "Processed %d/%d snippets | failed=%d | written_rows=%d | last_snippet=%s | last_tokens=%d",
                        processed_snippets,
                        len(snippets),
                        failed_snippets,
                        written_rows,
                        snippet.snippet_id,
                        len(tokenized["input_ids"]),
                    )
            except Exception:
                failed_snippets += 1
                LOGGER.exception(
                    "Failed to process snippet_id=%s language=%s file=%s",
                    snippet.snippet_id,
                    snippet.language,
                    snippet.file_path,
                )
                progress.set_postfix(
                    rows=written_rows,
                    failed=failed_snippets,
                )

        if buffer:
            LOGGER.info("Final flush of %d token rows to disk", len(buffer))
            writer.write_rows(buffer)
            buffer.clear()
    finally:
        writer.close()

    summary = {
        "processed_snippets": processed_snippets,
        "failed_snippets": failed_snippets,
        "written_rows": written_rows,
        "activation_dim": activation_dim,
    }
    return summary, []


def save_metadata(output_path: Path, output_format: str, args: argparse.Namespace, summary: Dict[str, Any], hook_name: str) -> None:
    metadata_path = output_path / "metadata.json" if output_format == "hf" else output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "snippets_dir": args.snippets_dir,
        "output_format": output_format,
        "model_name": args.model_name,
        "languages": args.languages,
        "code_field": args.code_field,
        "group_variants": args.group_variants,
        "group_separator": args.group_separator,
        "layer": args.layer,
        "activation_kind": args.activation_kind,
        "hook_name": hook_name,
        "device": args.device,
        "model_dtype": args.model_dtype,
        "storage_dtype": args.storage_dtype,
        "add_special_tokens": args.add_special_tokens,
        "truncation": args.truncation,
        "max_length": args.max_length,
        "include_code": args.include_code,
        "writer_batch_size": args.writer_batch_size,
        **summary,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_path = Path(args.output_path)
    output_format = infer_output_format(output_path, args.output_format)
    hook_name = build_hook_name(args.layer, args.activation_kind, args.hook_name)

    LOGGER.info("Loading model %s on %s", args.model_name, args.device)
    model = load_model(
        model_name=args.model_name,
        device=args.device,
        dtype_name=args.model_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    LOGGER.info(
        "Starting activation extraction | hook=%s | layer=%d | activation_kind=%s | storage_dtype=%s",
        hook_name,
        args.layer,
        args.activation_kind,
        args.storage_dtype,
    )
    summary, _ = extract_rows(args=args, model=model, hook_name=hook_name)

    if args.save_metadata_json:
        save_metadata(output_path, output_format, args, summary, hook_name)

    LOGGER.info(
        "Finished. processed_snippets=%d failed_snippets=%d written_rows=%d output=%s",
        summary["processed_snippets"],
        summary["failed_snippets"],
        summary["written_rows"],
        output_path,
    )


if __name__ == "__main__":
    main()
