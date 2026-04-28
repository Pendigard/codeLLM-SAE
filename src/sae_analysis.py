from __future__ import annotations

import ast
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import json
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colors as mcolors

from circuitsvis.tokens import colored_tokens


def _normalize_top_feature_list(value):
    """Normalize a feature-id field into a list[int]."""
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value).reshape(-1)
        return [int(x) for x in arr if pd.notna(x)]

    # si jamais la colonne contient parfois une string du type "[1, 2, 3]"
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                arr = np.asarray(parsed).reshape(-1)
                return [int(x) for x in arr if pd.notna(x)]
            return [int(parsed)]
        except (ValueError, SyntaxError):
            return [int(value)]

    return [int(value)]


def _normalize_numeric_list(value):
    """Normalize a numeric field into a list[float]."""
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value).reshape(-1)
        return [float(x) for x in arr if pd.notna(x)]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                arr = np.asarray(parsed).reshape(-1)
                return [float(x) for x in arr if pd.notna(x)]
            return [float(parsed)]
        except (ValueError, SyntaxError):
            return [float(value)]

    return [float(value)]


def _extract_top_k_features(row: pd.Series, top_k: int) -> list[int]:
    if "top_k_feature_ids" in row and row["top_k_feature_ids"] is not None:
        feats = _normalize_top_feature_list(row["top_k_feature_ids"])
        return feats[:top_k]

    feats: list[int] = []
    for rank in range(1, top_k + 1):
        col = f"top_{rank}_feature_id"
        if col not in row:
            break
        value = row[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        feats.append(int(value))
    return feats


def _resolve_snippet_id_col(df: pd.DataFrame, snippet_id_col: Optional[str] = None) -> str:
    if snippet_id_col is not None:
        if snippet_id_col not in df.columns:
            raise ValueError(f"Missing snippet id column: {snippet_id_col}")
        return snippet_id_col

    for candidate in ("global_idx", "code_id", "idx"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not infer the snippet id column. Pass `snippet_id_col` explicitly."
    )


def build_language_token_feature_sets(
    df: pd.DataFrame,
    top_k: int = 5,
    language_col: str = "language",
    token_col: str = "token_text",
    drop_empty_tokens: bool = True,
) -> pd.DataFrame:
    """Build one feature set per (language, token).

    Features come from token-level SAE activations, truncated to `top_k`.
    All rows sharing the same (language, token) are merged by union.
    """
    required = {language_col, token_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["selected_features"] = work.apply(lambda row: _extract_top_k_features(row, top_k=top_k), axis=1)

    if drop_empty_tokens:
        work = work[work[token_col].astype(str).str.len() > 0]

    grouped = (
        work.groupby([language_col, token_col], dropna=False)["selected_features"]
        .agg(lambda col: set(fid for feats in col for fid in feats))
        .reset_index(name="feature_set")
    )
    return grouped


def build_snippet_feature_sets(
    df: pd.DataFrame,
    top_k: int = 5,
    language_col: str = "language",
    snippet_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build one feature set per (snippet, language).

    This mirrors the old notebook logic more closely: each snippet is
    represented by the union of the top-k features seen across its tokens.
    """
    snippet_id_col = _resolve_snippet_id_col(df, snippet_id_col=snippet_id_col)
    required = {snippet_id_col, language_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["selected_features"] = work.apply(
        lambda row: _extract_top_k_features(row, top_k=top_k),
        axis=1,
    )

    grouped = (
        work.groupby([snippet_id_col, language_col], dropna=False)["selected_features"]
        .agg(lambda col: set(fid for feats in col for fid in feats))
        .reset_index(name="feature_set")
    )
    return grouped


def _prepare_selected_features_column(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    work = df.copy()

    if "top_k_feature_ids" in work.columns:
        work["selected_features"] = work["top_k_feature_ids"].map(
            lambda value: _normalize_top_feature_list(value)[:top_k]
        )
    else:
        work["selected_features"] = work.apply(
            lambda row: _extract_top_k_features(row, top_k=top_k),
            axis=1,
        )

    return work


def _prepare_selected_feature_pairs_column(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    work = _prepare_selected_features_column(df, top_k=top_k)

    if "top_k_feature_activations" in work.columns:
        work["selected_activations"] = work["top_k_feature_activations"].map(
            lambda value: _normalize_numeric_list(value)[:top_k]
        )
    else:
        work["selected_activations"] = work["selected_features"].map(
            lambda feats: [np.nan] * len(feats)
        )

    work["selected_feature_pairs"] = work.apply(
        lambda row: list(zip(row["selected_features"], row["selected_activations"])),
        axis=1,
    )
    return work


def _build_snippet_feature_indicator_matrix(
    df: pd.DataFrame,
    top_k: int,
    language_col: str,
    snippet_id_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return metadata table and dense binary matrix snippet x feature."""
    work = _prepare_selected_features_column(df, top_k=top_k)

    snippet_feat = (
        work[[snippet_id_col, language_col, "selected_features"]]
        .explode("selected_features")
        .dropna(subset=["selected_features"])
        .drop_duplicates()
        .assign(value=1)
    )

    if snippet_feat.empty:
        metadata = (
            work[[snippet_id_col, language_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return metadata, np.zeros((len(metadata), 0), dtype=np.uint8), np.array([], dtype=object)

    snippet_feat["selected_features"] = snippet_feat["selected_features"].astype(int)

    X_df = snippet_feat.pivot_table(
        index=[snippet_id_col, language_col],
        columns="selected_features",
        values="value",
        fill_value=0,
        aggfunc="max",
    )

    metadata = X_df.index.to_frame(index=False)
    X = X_df.to_numpy(dtype=np.uint8)
    feature_ids = X_df.columns.to_numpy()
    return metadata, X, feature_ids


def _normalize_positive_values(positive_values: object | Sequence[object]) -> set[object]:
    if isinstance(positive_values, (str, bytes)):
        return {positive_values}
    if isinstance(positive_values, Sequence):
        return set(positive_values)
    return {positive_values}


def discriminative_features(
    df: pd.DataFrame,
    group_col: str,
    positive_values: object | Sequence[object],
    top_k: int = 5,
    unit_col: Optional[str] = None,
    min_in_group_count: int = 1,
    sort_by: str = "rate_diff",
    ascending: bool = False,
) -> pd.DataFrame:
    """Identify features that discriminate a target group from the rest.

    Typical usage:
    - `group_col="language", positive_values=["Python"]`
    - `group_col="pygments_simple_type", positive_values=["keyword"]`

    Rates are computed over units (tokens by default, or `unit_col` if passed).
    A feature is considered present in a unit if it appears in that unit's top-k
    feature list.
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing grouping column: {group_col}")

    work = _prepare_selected_feature_pairs_column(df, top_k=top_k).copy()
    positive_set = _normalize_positive_values(positive_values)

    if unit_col is None:
        work = work.reset_index(drop=False).rename(columns={"index": "_unit_id"})
        unit_col = "_unit_id"
    elif unit_col not in work.columns:
        raise ValueError(f"Missing unit column: {unit_col}")

    work["is_positive_group"] = work[group_col].isin(positive_set)

    unit_meta = work[[unit_col, "is_positive_group"]].drop_duplicates()
    n_pos_units = int(unit_meta["is_positive_group"].sum())
    n_neg_units = int((~unit_meta["is_positive_group"]).sum())

    exploded = (
        work[[unit_col, "is_positive_group", "selected_feature_pairs"]]
        .explode("selected_feature_pairs")
        .dropna(subset=["selected_feature_pairs"])
        .copy()
    )

    if exploded.empty:
        return pd.DataFrame(
            columns=[
                "feature_id",
                "in_group_count",
                "out_group_count",
                "in_group_rate",
                "out_group_rate",
                "rate_diff",
                "lift",
                "in_group_mean_activation",
                "out_group_mean_activation",
                "activation_diff",
            ]
        )

    exploded[["feature_id", "activation"]] = pd.DataFrame(
        exploded["selected_feature_pairs"].tolist(),
        index=exploded.index,
    )
    exploded["feature_id"] = exploded["feature_id"].astype(int)
    exploded["activation"] = exploded["activation"].astype(float)

    # Presence is binary per unit-feature pair.
    unit_feature = exploded[[unit_col, "is_positive_group", "feature_id"]].drop_duplicates()
    counts = (
        unit_feature.groupby(["feature_id", "is_positive_group"])[unit_col]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={True: "in_group_count", False: "out_group_count"})
    )

    if "in_group_count" not in counts.columns:
        counts["in_group_count"] = 0
    if "out_group_count" not in counts.columns:
        counts["out_group_count"] = 0

    activation_means = (
        exploded.groupby(["feature_id", "is_positive_group"])["activation"]
        .mean()
        .unstack()
        .rename(columns={True: "in_group_mean_activation", False: "out_group_mean_activation"})
    )

    result = counts.join(activation_means, how="left").fillna(
        {
            "in_group_mean_activation": np.nan,
            "out_group_mean_activation": np.nan,
        }
    )
    result = result.reset_index()

    result["in_group_count"] = result["in_group_count"].astype(int)
    result["out_group_count"] = result["out_group_count"].astype(int)
    result["in_group_rate"] = (
        result["in_group_count"] / n_pos_units if n_pos_units > 0 else np.nan
    )
    result["out_group_rate"] = (
        result["out_group_count"] / n_neg_units if n_neg_units > 0 else np.nan
    )
    result["rate_diff"] = result["in_group_rate"] - result["out_group_rate"]
    result["lift"] = np.divide(
        result["in_group_rate"],
        result["out_group_rate"],
        out=np.full(len(result), np.inf, dtype=float),
        where=result["out_group_rate"] > 0,
    )
    result["activation_diff"] = (
        result["in_group_mean_activation"] - result["out_group_mean_activation"]
    )
    result["in_group_total_units"] = n_pos_units
    result["out_group_total_units"] = n_neg_units
    result["group_col"] = group_col
    result["positive_values"] = [sorted(str(v) for v in positive_set)] * len(result)

    result = result[result["in_group_count"] >= min_in_group_count]
    result = result.sort_values(
        by=[sort_by, "in_group_rate", "in_group_count"],
        ascending=[ascending, False, False],
    ).reset_index(drop=True)
    return result


def filter_tokens_by_feature_subset(
    df: pd.DataFrame,
    feature_ids: Sequence[int],
    top_k: int = 20,
    match_mode: str = "subset",
    keep_selected_features_col: bool = False,
) -> pd.DataFrame:
    """Filter rows whose top-k feature ids match a target feature pattern.

    Parameters
    ----------
    feature_ids:
        Target feature ids to look for.
    match_mode:
        - "subset": all `feature_ids` must be present in the row's top-k set
        - "exact": the row's top-k set must equal `feature_ids`
        - "any": at least one feature id must be present
    """
    target = {int(fid) for fid in feature_ids}
    if not target:
        raise ValueError("`feature_ids` must contain at least one feature id.")

    work = _prepare_selected_features_column(df, top_k=top_k).copy()
    work["selected_feature_set"] = work["selected_features"].map(set)

    if match_mode == "subset":
        mask = work["selected_feature_set"].map(lambda feature_set: target.issubset(feature_set))
    elif match_mode == "exact":
        mask = work["selected_feature_set"].map(lambda feature_set: feature_set == target)
    elif match_mode == "any":
        mask = work["selected_feature_set"].map(lambda feature_set: len(target & feature_set) > 0)
    else:
        raise ValueError("Unsupported match_mode. Use 'subset', 'exact', or 'any'.")

    filtered = work.loc[mask].copy()

    if not keep_selected_features_col:
        filtered = filtered.drop(columns=["selected_features", "selected_feature_set"])

    return filtered.reset_index(drop=True)


def _mean_jaccard_between_sets(
    left_sets: Sequence[set[int]],
    right_sets: Sequence[set[int]],
    same_language: bool = False,
    left_tokens: Optional[Sequence[str]] = None,
    right_tokens: Optional[Sequence[str]] = None,
    exclude_same_token_on_diagonal: bool = True,
) -> float:
    if not left_sets or not right_sets:
        return np.nan

    scores: list[float] = []

    if same_language:
        # Upper triangle only, and optionally skip identical token labels.
        for i, j in combinations(range(len(left_sets)), 2):
            if exclude_same_token_on_diagonal and left_tokens is not None and right_tokens is not None:
                if left_tokens[i] == right_tokens[j]:
                    continue
            f1 = left_sets[i]
            f2 = right_sets[j]
            union = f1 | f2
            score = (len(f1 & f2) / len(union)) if union else 0.0
            scores.append(score)
    else:
        for i, f1 in enumerate(left_sets):
            for j, f2 in enumerate(right_sets):
                union = f1 | f2
                score = (len(f1 & f2) / len(union)) if union else 0.0
                scores.append(score)

    return float(np.mean(scores)) if scores else np.nan


def _mean_jaccard_between_snippets(
    left_sets: Sequence[set[int]],
    right_sets: Sequence[set[int]],
    left_ids: Sequence[object],
    right_ids: Sequence[object],
    exclude_same_snippet_id_pairs: bool = True,
) -> float:
    if not left_sets or not right_sets:
        return np.nan

    scores: list[float] = []

    for i, f1 in enumerate(left_sets):
        for j, f2 in enumerate(right_sets):
            if exclude_same_snippet_id_pairs and left_ids[i] == right_ids[j]:
                continue

            union = f1 | f2
            score = (len(f1 & f2) / len(union)) if union else 0.0
            scores.append(score)

    return float(np.mean(scores)) if scores else np.nan


def concept_overlap_matrix(
    df: pd.DataFrame,
    top_k: int = 5,
    language_col: str = "language",
    snippet_id_col: Optional[str] = None,
    exclude_same_snippet_id_pairs: bool = True,
) -> pd.DataFrame:
    """Language x language matrix of concept overlap.

    This version is snippet-weighted and is intended to stay close to the
    original `code_analysis.ipynb` logic:
    - union the top-k token features inside each snippet
    - compute pairwise Jaccard overlap between snippets
    - average overlaps by language x language block
    - optionally exclude pairs that share the same snippet id
    """
    snippet_id_col = _resolve_snippet_id_col(df, snippet_id_col=snippet_id_col)
    metadata, X, _ = _build_snippet_feature_indicator_matrix(
        df=df,
        top_k=top_k,
        language_col=language_col,
        snippet_id_col=snippet_id_col,
    )

    languages = sorted(metadata[language_col].dropna().unique().tolist())
    matrix = pd.DataFrame(index=languages, columns=languages, dtype=float)

    if X.shape[0] == 0:
        return matrix

    languages_arr = metadata[language_col].to_numpy()
    snippet_ids_arr = metadata[snippet_id_col].to_numpy()

    inter = X @ X.T
    row_sums = X.sum(axis=1, keepdims=True, dtype=np.int32)
    union = row_sums + row_sums.T - inter

    jaccard = np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union != 0,
    )

    invalid_mask = np.zeros_like(jaccard, dtype=bool)
    if exclude_same_snippet_id_pairs:
        invalid_mask |= snippet_ids_arr[:, None] == snippet_ids_arr[None, :]
    else:
        np.fill_diagonal(invalid_mask, True)

    jaccard = jaccard.astype(np.float32, copy=False)
    jaccard[invalid_mask] = np.nan

    for lang1 in languages:
        mask1 = languages_arr == lang1
        for lang2 in languages:
            mask2 = languages_arr == lang2
            block = jaccard[np.ix_(mask1, mask2)]
            matrix.loc[lang1, lang2] = np.nanmean(block) if block.size > 0 else np.nan

    return matrix


def concept_overlap_matrix_token_level(
    df: pd.DataFrame,
    top_k: int = 5,
    language_col: str = "language",
    token_col: str = "token_text",
    exclude_same_token_on_diagonal: bool = True,
    drop_empty_tokens: bool = True,
) -> pd.DataFrame:
    """Language x language matrix based on token-text aggregates.

    This preserves the first implementation that compares aggregated token
    feature sets inside each language.
    """
    token_sets = build_language_token_feature_sets(
        df=df,
        top_k=top_k,
        language_col=language_col,
        token_col=token_col,
        drop_empty_tokens=drop_empty_tokens,
    )

    languages = sorted(token_sets[language_col].dropna().unique().tolist())
    matrix = pd.DataFrame(index=languages, columns=languages, dtype=float)

    by_lang: dict[str, pd.DataFrame] = {
        lang: token_sets[token_sets[language_col] == lang].reset_index(drop=True)
        for lang in languages
    }

    for lang1 in languages:
        left_df = by_lang[lang1]
        left_sets = left_df["feature_set"].tolist()
        left_tokens = left_df[token_col].astype(str).tolist()

        for lang2 in languages:
            right_df = by_lang[lang2]
            right_sets = right_df["feature_set"].tolist()
            right_tokens = right_df[token_col].astype(str).tolist()

            value = _mean_jaccard_between_sets(
                left_sets=left_sets,
                right_sets=right_sets,
                same_language=(lang1 == lang2),
                left_tokens=left_tokens,
                right_tokens=right_tokens,
                exclude_same_token_on_diagonal=exclude_same_token_on_diagonal,
            )
            matrix.loc[lang1, lang2] = value

    return matrix


def feature_token_weights(
    df: pd.DataFrame,
    feature_id: int,
    activation_threshold: float = 0.0,
    token_col: str = "token_str",
    top_k: int = 20,
    weight_mode: str = "sum_activation",
    drop_empty_tokens: bool = True,
) -> pd.Series:
    """Aggregate token weights for one feature above an activation threshold.

    Parameters
    ----------
    weight_mode:
        - "count": number of qualifying occurrences
        - "sum_activation": sum of activations across qualifying occurrences
        - "mean_activation": mean activation per token
    """
    if token_col not in df.columns:
        raise ValueError(f"Missing token column: {token_col}")

    work = _prepare_selected_feature_pairs_column(df, top_k=top_k).copy()
    exploded = (
        work[[token_col, "selected_feature_pairs"]]
        .explode("selected_feature_pairs")
        .dropna(subset=["selected_feature_pairs"])
        .copy()
    )

    if exploded.empty:
        return pd.Series(dtype=float, name="weight")

    exploded[["feature_id", "activation"]] = pd.DataFrame(
        exploded["selected_feature_pairs"].tolist(),
        index=exploded.index,
    )
    exploded["feature_id"] = exploded["feature_id"].astype(int)
    exploded["activation"] = exploded["activation"].astype(float)
    exploded[token_col] = exploded[token_col].fillna("").astype(str)

    filtered = exploded[
        (exploded["feature_id"] == int(feature_id))
        & (exploded["activation"] >= activation_threshold)
    ].copy()

    if drop_empty_tokens:
        filtered = filtered[filtered[token_col].str.len() > 0]

    if filtered.empty:
        return pd.Series(dtype=float, name="weight")

    if weight_mode == "count":
        weights = filtered.groupby(token_col).size().astype(float)
    elif weight_mode == "sum_activation":
        weights = filtered.groupby(token_col)["activation"].sum()
    elif weight_mode == "mean_activation":
        weights = filtered.groupby(token_col)["activation"].mean()
    else:
        raise ValueError(
            "Unsupported weight_mode. Use 'count', 'sum_activation', or 'mean_activation'."
        )

    weights = weights.sort_values(ascending=False)
    weights.name = "weight"
    return weights


def _sanitize_token_for_wordcloud(token: str) -> str:
    token = str(token)

    if token == "":
        return "<empty>"
    if token.isspace():
        parts: list[str] = []
        newline_count = token.count("\n")
        tab_count = token.count("\t")
        space_count = token.count(" ")
        if newline_count:
            parts.append("\\n" * newline_count)
        if tab_count:
            parts.append("\\t" * tab_count)
        if space_count:
            parts.append("<space>" if space_count == 1 else f"<space>x{space_count}")
        return "".join(parts) if parts else "<whitespace>"

    token = token.replace("\n", "\\n")
    token = token.replace("\t", "\\t")
    return token


def plot_feature_token_wordcloud(
    df: pd.DataFrame,
    feature_id: int,
    activation_threshold: float = 0.0,
    token_col: str = "token_str",
    top_k: int = 20,
    weight_mode: str = "sum_activation",
    max_words: int = 100,
    background_color: str = "white",
    colormap: str = "viridis",
    width: int = 1200,
    height: int = 600,
    ax=None,
):
    """Plot a wordcloud of tokens associated with one feature.

    Only occurrences where `feature_id` appears in the token's top-k list and
    has activation >= `activation_threshold` are kept.

    Returns
    -------
    tuple
        `(ax, weights)` where `weights` is the token-weight Series used to
        generate the wordcloud.
    """
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError(
            "wordcloud is required for plot_feature_token_wordcloud. "
            "Install it with `pip install wordcloud`."
        ) from exc

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_feature_token_wordcloud. "
            "Install it with `pip install matplotlib`."
        ) from exc

    weights = feature_token_weights(
        df=df,
        feature_id=feature_id,
        activation_threshold=activation_threshold,
        token_col=token_col,
        top_k=top_k,
        weight_mode=weight_mode,
    )

    if weights.empty:
        raise ValueError(
            f"No tokens found for feature_id={feature_id} with activation >= {activation_threshold}."
        )

    safe_weights = weights.groupby(weights.index.map(_sanitize_token_for_wordcloud)).sum()

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        collocations=False,
    ).generate_from_frequencies(safe_weights.to_dict())

    if ax is None:
        _, ax = plt.subplots(figsize=(width / 100, height / 100))

    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    ax.set_title(
        f"Feature {feature_id} token wordcloud\n"
        f"threshold >= {activation_threshold}, mode={weight_mode}"
    )
    return ax, safe_weights

def get_feature_activation(row, feature_id, activation_threshold=0.0, top_k=None):
    feature_ids = row["top_k_feature_ids"]
    activations = row["top_k_feature_activations"]

    if top_k is not None:
        feature_ids = feature_ids[:top_k]
        activations = activations[:top_k]

    for fid, act in zip(feature_ids, activations):
        if fid == feature_id and act >= activation_threshold:
            return float(act)

    return 0.0


def visualize_feature_on_snippet_circuitsvis(
    df: pd.DataFrame,
    global_idx,
    language: str,
    feature_id: int,
    activation_threshold: float = 0.0,
    top_k=None,
    token_col: str = "token_text",
):
    """
    Visualize feature activations token-by-token using circuitsvis.

    Parameters
    ----------
    df:
        DataFrame containing one row per token.

    global_idx:
        Snippet identifier.

    language:
        Language of the snippet.

    feature_id:
        SAE feature id to visualize.

    activation_threshold:
        Minimum activation required to display the feature as active.

    top_k:
        If not None, only search inside the first top_k features
        of each token.

    token_col:
        Column used for display, e.g. "token_text",
        "token_decoded", or "tokenizer_token".
    """

    snippet_df = df[
        (df["global_idx"].astype(str) == str(global_idx))
        & (df["language"] == language)
    ].copy()

    if snippet_df.empty:
        raise ValueError(
            f"No snippet found for global_idx={global_idx}, language={language}"
        )

    snippet_df = snippet_df.sort_values("token_pos")

    snippet_df["feature_activation"] = snippet_df.apply(
        lambda row: get_feature_activation(
            row,
            feature_id=feature_id,
            activation_threshold=activation_threshold,
            top_k=top_k,
        ),
        axis=1,
    )

    tokens = snippet_df[token_col].astype(str).tolist()
    values = snippet_df["feature_activation"].astype(float).tolist()

    return colored_tokens(tokens, values)

def compute_PMI(
    df: pd.DataFrame,
    feature_id: int,
    mask_label: pd.Series,
    activation_threshold: float = 0.0,
    ):

    mask_feature = df['top_k_feature_ids'].apply(lambda x: feature_id in x)

    exploded_inter = df[mask_feature & mask_label].explode(['top_k_feature_ids', 'top_k_feature_activations'])

    P_inter = len(exploded_inter[(exploded_inter['top_k_feature_ids'] == feature_id) & (exploded_inter['top_k_feature_activations'] >= activation_threshold)]) / len(df)

    exploded_feature = df[mask_feature].explode(['top_k_feature_ids', 'top_k_feature_activations'])
    P_feature = len(exploded_feature[(exploded_feature['top_k_feature_ids'] == feature_id) & (exploded_feature['top_k_feature_activations'] >= activation_threshold)]) / len(df)

    P_label = mask_label.mean()

    PMI = np.log2(P_inter / (P_feature * P_label)) if P_inter > 0 and P_feature > 0 and P_label > 0 else float('-inf')
    return PMI


def add_neighbor_tokens(df, relative_positions=[-2, -1, 1, 2], col = "token_str"):
    df = df.copy()

    # Une seule ligne par token réel
    token_df = (
        df.sort_values(["global_idx", "language", "token_pos"])
          .drop_duplicates(subset=["global_idx", "language", "token_pos"])
          .copy()
    )

    new_cols = []

    # Tokens voisins par snippet
    for rel_pos in relative_positions:
        col_prefix = "previous" if rel_pos < 0 else "next"
        col_suffix = abs(rel_pos)
        col_name = f"{col_prefix}_{col}_{col_suffix}"
        token_df[col_name] = token_df.groupby(["global_idx", "language"])[col].shift(-rel_pos)
        new_cols.append(col_name)

    # Reprojection sur le df original
    cols_to_merge = [
        "global_idx", "language", "token_pos",
        *new_cols
    ]

    df = df.merge(
        token_df[cols_to_merge],
        on=["global_idx", "language", "token_pos"],
        how="left"
    )

    return df

def get_description(feature_id : int, models = ["gpt-4o-mini", "gemini-flash-2.0"], folder_path = "/Users/celian/Documents/Fac/M2/RIKEN Traineeship/projects/labeled_features/"):
    for model in models:
        path = os.path.join(folder_path, f"explanations-{model}")
        for file in os.listdir(path):
            # check file is a jsonl
            if file.endswith(".jsonl"):
                with open(os.path.join(path, file), "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["index"] == str(feature_id):
                            print(f"Model: {model}, Explanation: {data['description']}")



def compute_all_features_pmi(
    df: pd.DataFrame,
    mask_label: pd.Series,
    activation_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute binary PMI and activation-weighted PMI for every feature.

    Binary PMI:
        feature is present if activation >= activation_threshold

    Weighted PMI:
        uses activation mass instead of binary presence.

    Required columns:
        - top_k_feature_ids
        - top_k_feature_activations
    """

    if len(mask_label) != len(df):
        raise ValueError("mask_label must have the same length as df")

    mask_label = mask_label.astype(bool)
    N = len(df)
    P_label = mask_label.mean()

    output_cols = [
        "feature_id",
        "count_feature",
        "count_intersection",
        "P_feature",
        "P_label",
        "P_inter",
        "PMI",
        "activation_feature",
        "activation_intersection",
        "P_weighted_feature",
        "P_weighted_label",
        "P_weighted_inter",
        "weighted_PMI",
    ]

    if P_label == 0 or N == 0:
        return pd.DataFrame(columns=output_cols)

    tmp = df[["top_k_feature_ids", "top_k_feature_activations"]].copy()
    tmp = tmp.reset_index(names="token_idx")
    tmp["mask_label"] = mask_label.to_numpy()

    exploded = tmp.explode(
        ["top_k_feature_ids", "top_k_feature_activations"],
        ignore_index=True
    )

    exploded = exploded.rename(
        columns={
            "top_k_feature_ids": "feature_id",
            "top_k_feature_activations": "activation",
        }
    )

    exploded["activation"] = pd.to_numeric(exploded["activation"], errors="coerce")
    exploded = exploded.dropna(subset=["feature_id", "activation"])

    # On garde les activations >= threshold pour définir la présence
    # et pour la masse d'activation pondérée
    exploded = exploded[exploded["activation"] >= activation_threshold].copy()

    if exploded.empty:
        return pd.DataFrame(columns=output_cols)

    # Une feature ne compte qu'une fois par token.
    # Si doublon rare, on garde l'activation max.
    exploded = (
        exploded
        .groupby(["token_idx", "feature_id"], as_index=False)
        .agg(
            activation=("activation", "max"),
            mask_label=("mask_label", "first")
        )
    )

    # =========================
    # Binary PMI
    # =========================

    count_feature = (
        exploded
        .groupby("feature_id")
        .size()
        .rename("count_feature")
    )

    count_intersection = (
        exploded[exploded["mask_label"]]
        .groupby("feature_id")
        .size()
        .rename("count_intersection")
    )

    result = pd.concat([count_feature, count_intersection], axis=1).fillna(0)

    result["count_feature"] = result["count_feature"].astype(int)
    result["count_intersection"] = result["count_intersection"].astype(int)

    result["P_feature"] = result["count_feature"] / N
    result["P_label"] = P_label
    result["P_inter"] = result["count_intersection"] / N

    result["PMI"] = np.where(
        result["count_intersection"] > 0,
        np.log2(result["P_inter"] / (result["P_feature"] * result["P_label"])),
        float("-inf")
    )

    # =========================
    # Weighted PMI
    # =========================

    total_activation = exploded["activation"].sum()

    activation_feature = (
        exploded
        .groupby("feature_id")["activation"]
        .sum()
        .rename("activation_feature")
    )

    activation_intersection = (
        exploded[exploded["mask_label"]]
        .groupby("feature_id")["activation"]
        .sum()
        .rename("activation_intersection")
    )

    result = pd.concat(
        [result, activation_feature, activation_intersection],
        axis=1
    ).fillna(0)

    label_activation_mass = exploded.loc[
        exploded["mask_label"], "activation"
    ].sum()

    result["P_weighted_feature"] = result["activation_feature"] / total_activation
    result["P_weighted_label"] = label_activation_mass / total_activation
    result["P_weighted_inter"] = result["activation_intersection"] / total_activation

    result["weighted_PMI"] = np.where(
        result["activation_intersection"] > 0,
        np.log2(
            result["P_weighted_inter"]
            / (result["P_weighted_feature"] * result["P_weighted_label"])
        ),
        float("-inf")
    )

    return (
        result
        .reset_index()
        .sort_values("PMI", ascending=False)
        .reset_index(drop=True)
    )


def plot_language_feature_distribution(
    df: pd.DataFrame,
    feature_id: int,
    language_col: str = "language",
    ax=None,
):
    feature_mask = df['top_k_feature_ids'].apply(lambda x: feature_id in x)
    counts = df[feature_mask][language_col].value_counts()

    if ax is None:
        _, ax = plt.subplots(dpi=300)

    counts.plot.bar(ax=ax)
    ax.set_title(f"Language distribution for feature {feature_id}")
    ax.set_xlabel(language_col)
    ax.set_ylabel("Count")
    return ax

def pie_plot_per_feature(df: pd.DataFrame, feature_id: int, label_col: str, ax=None):

    feature_mask = df['top_k_feature_ids'].apply(lambda x: feature_id in x)
    counts = df[feature_mask][label_col].value_counts()

    if ax is None:
        _, ax = plt.subplots()

    counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Distribution of {label_col} for feature {feature_id}")
    ax.set_ylabel("")
    return ax

def get_snippets_with_feature(
    df: pd.DataFrame,
    feature_id: int,
    activation_threshold: float = 0.1,
    top_k: int = 5,
):
    """
    Return rows where `feature_id` is present among the top_k features
    of the token, with activation >= activation_threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - top_k_feature_ids
            - top_k_feature_activations

    feature_id : int
        Feature to search.

    activation_threshold : float
        Minimum activation required.

    top_k : int
        Only search inside the first top_k ranked features.

    Returns
    -------
    pd.DataFrame
        Matching rows sorted by feature activation descending.
    """

    def get_feature_activation(row):
        ids = row["top_k_feature_ids"][:top_k]
        acts = row["top_k_feature_activations"][:top_k]

        for fid, act in zip(ids, acts):
            if fid == feature_id and act >= activation_threshold:
                return act

        return None

    result = df.copy()

    result["feature_activation"] = result.apply(
        get_feature_activation,
        axis=1
    )

    result = result[result["feature_activation"].notna()]

    result = result.sort_values(
        "feature_activation",
        ascending=False
    ).reset_index(drop=True)

    return result
