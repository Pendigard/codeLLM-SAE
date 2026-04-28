#!/usr/bin/env python3
"""Extract joinable token-level SAE features from a precomputed activation dataset.

The input is the dataset produced by `extract_llm_token_activations.py`.
Each input row already corresponds to one tokenizer token and contains a dense
activation vector. This script:
- loads those dense activations from disk in batches
- encodes them with a sparse autoencoder
- writes one output row per token with top-k sparse features

The output stays joinable with the labels / raw activations datasets and is
compatible with `sae_analysis.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.extract_llm_token_activations import (
    configure_logging,
    infer_output_format,
    make_progress,
    parse_torch_dtype,
)


LOGGER = logging.getLogger("extract_joinable_sae_token_features")


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


class PyArrowSAEWriter(BaseWriter):
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
        self._schema = None
        self._sink = pa.OSFile(str(output_path), "wb")
        self._writer = None

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        normalized_rows = []
        for row in rows:
            normalized = dict(row)
            normalized.setdefault("code", None)
            normalized_rows.append(normalized)

        if self._schema is None:
            table = self._pa.Table.from_pylist(normalized_rows)
            self._schema = table.schema
            if self._file_format == "parquet":
                self._writer = self._pq.ParquetWriter(self._sink, self._schema)
            elif self._file_format == "arrow":
                self._writer = self._pa_ipc.new_file(self._sink, self._schema)
            else:
                raise ValueError(f"Unsupported pyarrow format: {self._file_format}")
        else:
            aligned_rows = []
            schema_names = set(self._schema.names)
            for row in normalized_rows:
                aligned_rows.append({name: row.get(name) for name in self._schema.names})
                extra_names = set(row) - schema_names
                if extra_names:
                    raise ValueError(
                        "Encountered new columns after writer schema was fixed: "
                        f"{sorted(extra_names)}. Ensure top-k and threshold settings stay constant within one run."
                    )
            table = self._pa.Table.from_pylist(aligned_rows, schema=self._schema)
        if self._file_format == "parquet":
            self._writer.write_table(table)
        else:
            self._writer.write(table)

    def close(self) -> None:
        if self._writer is not None:
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
        self._temp_dir = Path(tempfile.mkdtemp(prefix="sae_token_features_"))
        self._parquet_path = self._temp_dir / "data.parquet"
        self._parquet_writer = PyArrowSAEWriter(
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
        return PyArrowSAEWriter(output_path, file_format="parquet", storage_dtype=storage_dtype)
    if output_format == "arrow":
        return PyArrowSAEWriter(output_path, file_format="arrow", storage_dtype=storage_dtype)
    if output_format == "hf":
        return HuggingFaceDatasetWriter(output_path, storage_dtype=storage_dtype)
    raise ValueError(f"Unsupported output format: {output_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract joinable token-level SAE sparse features from a precomputed activation dataset.",
    )
    parser.add_argument("--activations-path", required=True)
    parser.add_argument(
        "--activations-format",
        choices=["auto", "jsonl", "parquet", "arrow", "hf"],
        default="auto",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--output-format",
        choices=["auto", "jsonl", "parquet", "arrow", "hf"],
        default="auto",
    )
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional subset of languages to keep, e.g. Python Java C++.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--only-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, top-k is computed on positive SAE activations only.",
    )
    parser.add_argument(
        "--feature-thresholds",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 1.0],
        help="Thresholds used to count active SAE features per token.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--storage-dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Datatype used to store scalar feature activations on disk.",
    )
    parser.add_argument(
        "--include-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--writer-batch-size", type=int, default=1024)
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


def load_sae(sae_release: str, sae_id: str, device: str) -> Any:
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise SystemExit(
            "sae_lens is required. Install it with `pip install sae-lens`."
        ) from exc

    loaded = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()
    LOGGER.info("SAE loaded successfully")
    return sae


def infer_input_format(input_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if input_path.suffix == ".jsonl":
        return "jsonl"
    if input_path.suffix == ".parquet":
        return "parquet"
    if input_path.suffix == ".arrow":
        return "arrow"
    if input_path.suffix == "":
        return "hf"
    raise ValueError(
        "Could not infer input format from path. Pass --activations-format explicitly."
    )


def metadata_path_for_dataset(dataset_path: Path, dataset_format: str) -> Path:
    if dataset_format == "hf":
        return dataset_path / "metadata.json"
    return dataset_path.with_suffix(dataset_path.suffix + ".meta.json")


def load_activation_metadata(dataset_path: Path, dataset_format: str) -> Dict[str, Any]:
    metadata_path = metadata_path_for_dataset(dataset_path, dataset_format)
    if not metadata_path.exists():
        LOGGER.warning("No activation metadata file found at %s", metadata_path)
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def filter_activation_row(row: Dict[str, Any], languages: Optional[Sequence[str]]) -> bool:
    if languages is not None and row.get("language") not in set(languages):
        return False
    if "activation" not in row:
        raise KeyError("Input activation row is missing the `activation` column.")
    return True


def iter_activation_batches(
    dataset_path: Path,
    dataset_format: str,
    batch_size: int,
    languages: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
):
    yielded_rows = 0

    if dataset_format == "jsonl":
        batch: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not filter_activation_row(row, languages):
                    continue
                batch.append(row)
                yielded_rows += 1
                if max_rows is not None and yielded_rows >= max_rows:
                    yield batch
                    break
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
        return

    if dataset_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                "pyarrow is required to read parquet activations. Install it with `pip install pyarrow`."
            ) from exc

        parquet_file = pq.ParquetFile(dataset_path)
        batch: List[Dict[str, Any]] = []
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in record_batch.to_pylist():
                if not filter_activation_row(row, languages):
                    continue
                batch.append(row)
                yielded_rows += 1
                if max_rows is not None and yielded_rows >= max_rows:
                    yield batch
                    return
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
        return

    if dataset_format == "arrow":
        try:
            import pyarrow.ipc as pa_ipc
        except ImportError as exc:
            raise SystemExit(
                "pyarrow is required to read arrow activations. Install it with `pip install pyarrow`."
            ) from exc

        with dataset_path.open("rb") as handle:
            reader = pa_ipc.open_file(handle)
            batch: List[Dict[str, Any]] = []
            for record_batch in reader:
                for row in record_batch.to_pylist():
                    if not filter_activation_row(row, languages):
                        continue
                    batch.append(row)
                    yielded_rows += 1
                    if max_rows is not None and yielded_rows >= max_rows:
                        yield batch
                        return
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch
        return

    if dataset_format == "hf":
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise SystemExit(
                "datasets is required to read Hugging Face activations. Install it with `pip install datasets pyarrow`."
            ) from exc

        dataset = load_from_disk(str(dataset_path))
        batch: List[Dict[str, Any]] = []
        for row in dataset:
            if not filter_activation_row(row, languages):
                continue
            batch.append(row)
            yielded_rows += 1
            if max_rows is not None and yielded_rows >= max_rows:
                yield batch
                return
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        return

    raise ValueError(f"Unsupported activation input format: {dataset_format}")


def maybe_cast_float_list(values: Sequence[float], storage_dtype: str) -> List[float]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required. Install it with `pip install torch`.") from exc

    tensor = torch.tensor(list(values), dtype=parse_torch_dtype(storage_dtype))
    return tensor.tolist()


def feature_threshold_column_name(threshold: float) -> str:
    cleaned = str(threshold).replace("-", "neg_").replace(".", "_")
    return f"sae_active_ge_{cleaned}"


def activation_stats(
    feature_vector: Any,
    top_ids: List[int],
    top_vals: List[float],
    thresholds: Iterable[float],
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required. Install it with `pip install torch`.") from exc

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


def compute_top_k_sparse_features(
    feature_vector: Any,
    top_k: int,
    only_positive: bool,
) -> tuple[List[int], List[float]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required. Install it with `pip install torch`.") from exc

    values_for_topk = torch.clamp(feature_vector, min=0) if only_positive else feature_vector
    k_eff = min(top_k, values_for_topk.shape[0])
    top_vals_tensor, top_ids_tensor = torch.topk(values_for_topk, k=k_eff)

    top_vals = [float(val) for val in top_vals_tensor.tolist()]
    top_ids = [int(idx) for idx in top_ids_tensor.tolist()]

    if only_positive:
        filtered = [(idx, val) for idx, val in zip(top_ids, top_vals) if val > 0]
        top_ids = [idx for idx, _ in filtered]
        top_vals = [val for _, val in filtered]

    return top_ids, top_vals


def build_joinable_sae_rows_from_activation_rows(
    activation_rows: Sequence[Dict[str, Any]],
    features: Any,
    args: argparse.Namespace,
    source_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seq_len = features.shape[0]

    if seq_len != len(activation_rows):
        raise ValueError(
            f"Feature length mismatch: {seq_len} features vs {len(activation_rows)} activation rows."
        )

    for row_index, activation_row in enumerate(activation_rows):
        feature_vector = features[row_index]
        top_ids, top_vals = compute_top_k_sparse_features(
            feature_vector=feature_vector,
            top_k=args.top_k,
            only_positive=args.only_positive,
        )
        stored_top_vals = maybe_cast_float_list(top_vals, args.storage_dtype)
        char_start = activation_row.get("char_start", activation_row.get("token_offset_start"))
        char_end = activation_row.get("char_end", activation_row.get("token_offset_end"))
        if char_start is None or char_end is None:
            raise KeyError(
                "Activation row must contain either (`char_start`, `char_end`) "
                "or (`token_offset_start`, `token_offset_end`)."
            )
        token_text = activation_row.get("token_text", "")
        token_pos = int(activation_row["token_pos"])

        row = {
            "snippet_id": activation_row["snippet_id"],
            "global_idx": activation_row["global_idx"],
            "language": activation_row["language"],
            "file_path": activation_row.get("file_path"),
            "source_snippet_ids": activation_row.get("source_snippet_ids"),
            "source_file_paths": activation_row.get("source_file_paths"),
            "num_fragments": activation_row.get("num_fragments"),
            "token_id": int(activation_row["token_id"]),
            "token_pos": token_pos,
            "seq_len": activation_row.get("seq_len"),
            "code_char_len": activation_row.get("code_char_len"),
            "tokenizer_token": activation_row.get("tokenizer_token", activation_row.get("token_str")),
            "token_decoded": activation_row.get("token_decoded"),
            "token_text": token_text,
            "char_start": int(char_start),
            "char_end": int(char_end),
            "char_len": int(char_end - char_start),
            "token_offset_start": int(activation_row.get("token_offset_start", char_start)),
            "token_offset_end": int(activation_row.get("token_offset_end", char_end)),
            "is_special_token": bool(
                activation_row.get("is_special_token", int(char_start) == int(char_end))
            ),
            "is_whitespace_token": bool(
                activation_row.get("is_whitespace_token", token_text != "" and token_text.isspace())
            ),
            "prev_token_str": activation_row.get("prev_token_str"),
            "next_token_str": activation_row.get("next_token_str"),
            "top_k_feature_ids": top_ids,
            "top_k_feature_activations": stored_top_vals,
            "top_k_found": len(top_ids),
            "selected_features": top_ids,
            "model_name": activation_row.get("model_name", source_metadata.get("model_name")),
            "sae_release": args.sae_release,
            "sae_id": args.sae_id,
            "layer": activation_row.get("layer", source_metadata.get("layer")),
            "activation_kind": activation_row.get("activation_kind", source_metadata.get("activation_kind")),
            "hook_name": activation_row.get("hook_name", source_metadata.get("hook_name")),
        }
        row.update(activation_stats(feature_vector, top_ids, stored_top_vals, args.feature_thresholds))

        for rank in range(args.top_k):
            row[f"top_{rank + 1}_feature_id"] = top_ids[rank] if rank < len(top_ids) else None
            row[f"top_{rank + 1}_feature_activation"] = stored_top_vals[rank] if rank < len(stored_top_vals) else None

        if args.include_code:
            row["code"] = activation_row.get("code")

        rows.append(row)

    return rows


def extract_rows(
    args: argparse.Namespace,
    sae: Any,
    activations_path: Path,
    activations_format: str,
    source_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required. Install it with `pip install torch`.") from exc

    processed_batches = 0
    failed_batches = 0
    written_rows = 0
    sae_width: Optional[int] = None

    output_path = Path(args.output_path)
    output_format = infer_output_format(output_path, args.output_format)
    writer: BaseWriter = create_writer(output_path, output_format, args.storage_dtype)
    LOGGER.info("Writing SAE token features to %s", args.output_path)

    sae_device = next(sae.parameters()).device

    try:
        activation_batches = list(
            iter_activation_batches(
                dataset_path=activations_path,
                dataset_format=activations_format,
                batch_size=args.writer_batch_size,
                languages=args.languages,
                max_rows=args.max_rows,
            )
        )
        LOGGER.info(
            "Prepared %d activation batches from %s%s",
            len(activation_batches),
            activations_path,
            f" with languages={args.languages}" if args.languages else "",
        )
        progress = make_progress(
            activation_batches,
            desc="Encoding SAE features",
            unit="batch",
        )
        for activation_batch in progress:
            try:
                dense_activations = torch.tensor(
                    [row["activation"] for row in activation_batch],
                    dtype=parse_torch_dtype("float32"),
                    device=sae_device,
                )
                with torch.inference_mode():
                    features = sae.encode(dense_activations).detach().cpu()

                if sae_width is None:
                    sae_width = int(features.shape[-1])

                rows = build_joinable_sae_rows_from_activation_rows(
                    activation_rows=activation_batch,
                    features=features,
                    args=args,
                    source_metadata=source_metadata,
                )
                writer.write_rows(rows)
                written_rows += len(rows)
                processed_batches += 1

                progress.set_postfix(
                    rows=written_rows,
                    failed=failed_batches,
                    batch_rows=len(activation_batch),
                )

                if processed_batches % args.log_every == 0:
                    LOGGER.info(
                        "Processed %d/%d batches | failed=%d | written_rows=%d | last_batch_rows=%d",
                        processed_batches,
                        len(activation_batches),
                        failed_batches,
                        written_rows,
                        len(activation_batch),
                    )
            except Exception:
                failed_batches += 1
                LOGGER.exception(
                    "Failed to process activation batch starting at snippet_id=%s token_pos=%s",
                    activation_batch[0].get("snippet_id") if activation_batch else None,
                    activation_batch[0].get("token_pos") if activation_batch else None,
                )
                progress.set_postfix(
                    rows=written_rows,
                    failed=failed_batches,
                )
    finally:
        writer.close()

    return {
        "processed_batches": processed_batches,
        "failed_batches": failed_batches,
        "written_rows": written_rows,
        "sae_width": sae_width,
    }


def save_metadata(
    output_path: Path,
    output_format: str,
    args: argparse.Namespace,
    summary: Dict[str, Any],
    source_metadata: Dict[str, Any],
) -> None:
    metadata_path = output_path / "metadata.json" if output_format == "hf" else output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "activations_path": args.activations_path,
        "activations_format": args.activations_format,
        "output_format": output_format,
        "model_name": source_metadata.get("model_name"),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "languages": args.languages,
        "layer": source_metadata.get("layer"),
        "activation_kind": source_metadata.get("activation_kind"),
        "hook_name": source_metadata.get("hook_name"),
        "top_k": args.top_k,
        "only_positive": args.only_positive,
        "feature_thresholds": args.feature_thresholds,
        "device": args.device,
        "storage_dtype": args.storage_dtype,
        "max_rows": args.max_rows,
        "include_code": args.include_code,
        "writer_batch_size": args.writer_batch_size,
        "source_activation_metadata": source_metadata,
        **summary,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    activations_path = Path(args.activations_path)
    activations_format = infer_input_format(activations_path, args.activations_format)
    output_path = Path(args.output_path)
    output_format = infer_output_format(output_path, args.output_format)
    source_metadata = load_activation_metadata(activations_path, activations_format)

    LOGGER.info("Loading SAE %s / %s", args.sae_release, args.sae_id)
    sae = load_sae(
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )

    LOGGER.info(
        "Starting SAE feature extraction from activations | input=%s | top_k=%d | storage_dtype=%s",
        activations_path,
        args.top_k,
        args.storage_dtype,
    )
    summary = extract_rows(
        args=args,
        sae=sae,
        activations_path=activations_path,
        activations_format=activations_format,
        source_metadata=source_metadata,
    )

    if args.save_metadata_json:
        save_metadata(output_path, output_format, args, summary, source_metadata)

    LOGGER.info(
        "Finished. processed_batches=%d failed_batches=%d written_rows=%d output=%s",
        summary["processed_batches"],
        summary["failed_batches"],
        summary["written_rows"],
        output_path,
    )


if __name__ == "__main__":
    main()
