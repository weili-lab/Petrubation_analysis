"""Fast exact single-label PS scores from backed AnnData streams.

This is Path A from `ALGOS.md`: one perturbation label per cell, optional
streamed histogram quantile clipping, no dense cell by gene matrix, and no dense
cell by perturbation score matrix. The core idea is to stream log-normalized
chunks to collect Welch t-test and ridge sufficient statistics, then stream again
to compute only each cell's observed-perturbation score.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

from .stats import extract_anndata_matrix, get_obs_column, top_k_indices, validate_layer, welch_t_scores_from_stats
from .types import StreamFeatureStats


DEFAULT_TARGET_SUM = 1e4
DEFAULT_CLIP_BINS = 2048


@dataclass(frozen=True)
class ExactFastPsResult:
    scores: np.ndarray
    valid_mask: np.ndarray
    beta: np.ndarray
    union_gene_indices: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExactFastMultiLabelPsResult:
    scores: np.ndarray
    cell_indices: np.ndarray
    perturbation_indices: np.ndarray
    perturbations: list[str]
    valid_mask: np.ndarray
    beta: np.ndarray
    union_gene_indices: np.ndarray
    metadata: dict[str, Any]


def run_ps_score_exact_fast_anndata(
    adata: Any,
    *,
    perturb_column: str,
    ctrl_name: str,
    layer: str | None = None,
    perturbations: Sequence[str] | None = None,
    null_labels: Sequence[str] | None = None,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_max: int = 500,
    chunk_size: int = 8192,
    lr_lambda: float = 0.01,
    score_lambda: float = 0.0,
    scale_factor: float = 3.0,
    target_sum: float = DEFAULT_TARGET_SUM,
    rank_by_abs_t: bool = True,
    scale_score: bool = True,
    clip_quantile: float | None = None,
    clip_bins: int = DEFAULT_CLIP_BINS,
) -> ExactFastPsResult:
    """Run streaming exact PS scores for single-label perturbation datasets."""

    validate_layer(layer)
    _validate_target_genes(target_genes)
    clip_quantile = _validate_clip_quantile(clip_quantile)
    clip_bins = _validate_clip_bins(clip_bins)
    labels = np.asarray(get_obs_column(adata.obs, perturb_column), dtype=object)
    var_names = np.asarray(adata.var_names, dtype=object)
    matrix = extract_anndata_matrix(adata, layer=layer)

    null_label_set = set() if null_labels is None else {str(label) for label in null_labels}
    selected = _resolve_selected_perturbations(
        labels,
        control_label=ctrl_name,
        perturbations=perturbations,
        null_labels=null_label_set,
    )
    label_to_row = {ctrl_name: 0, **{perturbation: index + 1 for index, perturbation in enumerate(selected)}}
    perturbation_codes = np.asarray(
        [label_to_row.get(str(label), -1) - 1 for label in labels],
        dtype=np.int32,
    )

    stage_start = time.perf_counter()

    # Pass 1: one streamed pass collects all target-selection and ridge RHS stats.
    pass1_start = time.perf_counter()
    sums = np.zeros((len(selected) + 1, var_names.shape[0]), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    counts = np.zeros(len(selected) + 1, dtype=np.int64)
    for start in range(0, labels.shape[0], chunk_size):
        stop = min(start + chunk_size, labels.shape[0])
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk_labels = labels[start:stop]
        for label_key in _iter_unique_label_keys(chunk_labels):
            row = label_to_row.get(label_key)
            if row is None:
                continue
            mask = chunk_labels == label_key
            _add_group_stats(chunk[mask], row=row, sums=sums, squared_sums=squared_sums, counts=counts)
    pass1_seconds = time.perf_counter() - pass1_start

    control_stats = StreamFeatureStats(count=int(counts[0]), sums=sums[0], squared_sums=squared_sums[0])
    target_gene_indices_by_perturbation: dict[str, np.ndarray] = {}
    target_gene_metadata: dict[str, dict[str, Any]] = {}
    gene_lookup = {str(gene): index for index, gene in enumerate(var_names.astype(str))}
    if target_genes is not None:
        target_gene_indices_by_perturbation = _resolve_provided_target_genes(
            target_genes=target_genes,
            selected_perturbations=selected,
            gene_lookup=gene_lookup,
            target_gene_min=1,
            target_gene_max=target_gene_max,
        )
        target_gene_source_detail = {"mode": "provided"}
    else:
        for perturbation_index, perturbation in enumerate(selected, start=1):
            perturb_stats = StreamFeatureStats(
                count=int(counts[perturbation_index]),
                sums=sums[perturbation_index],
                squared_sums=squared_sums[perturbation_index],
            )
            t_scores = welch_t_scores_from_stats(perturb_stats, control_stats)
            gene_indices = top_k_indices(
                t_scores,
                min(target_gene_max, t_scores.shape[0]),
                absolute=rank_by_abs_t,
            )
            target_gene_indices_by_perturbation[perturbation] = gene_indices.astype(np.int64, copy=False)
        target_gene_source_detail = {"mode": "streamed_welch", "rank_by_abs_t": bool(rank_by_abs_t)}

    for perturbation_index, perturbation in enumerate(selected, start=1):
        gene_indices = target_gene_indices_by_perturbation[perturbation]
        target_gene_metadata[perturbation] = {
            "cell_count": int(counts[perturbation_index]),
            "selected_gene_count": int(gene_indices.shape[0]),
            "selected_genes": [str(var_names[index]) for index in gene_indices],
        }

    union_gene_indices = _ordered_union_indices(target_gene_indices_by_perturbation.values())
    clip_values: np.ndarray | None = None
    clip_threshold_seconds = 0.0
    clipped_stats_seconds = 0.0

    if clip_quantile is None:
        beta_sums = sums[:, union_gene_indices]
    else:
        clip_start = time.perf_counter()
        clip_values = _estimate_histogram_clip_values(
            matrix,
            labels=labels,
            label_to_row=label_to_row,
            union_gene_indices=union_gene_indices,
            model_cell_count=int(counts.sum()),
            chunk_size=chunk_size,
            target_sum=target_sum,
            quantile=clip_quantile,
            bins=clip_bins,
        )
        clip_threshold_seconds = time.perf_counter() - clip_start

        clipped_stats_start = time.perf_counter()
        beta_sums = np.zeros((len(selected) + 1, union_gene_indices.shape[0]), dtype=np.float64)
        beta_squared_sums = np.zeros_like(beta_sums)
        beta_counts = np.zeros(len(selected) + 1, dtype=np.int64)
        for start in range(0, labels.shape[0], chunk_size):
            stop = min(start + chunk_size, labels.shape[0])
            chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
            chunk = chunk[:, union_gene_indices]
            chunk = _clip_matrix_columns(chunk, clip_values)
            chunk_labels = labels[start:stop]
            for label_key in _iter_unique_label_keys(chunk_labels):
                row = label_to_row.get(label_key)
                if row is None:
                    continue
                mask = chunk_labels == label_key
                _add_group_stats(chunk[mask], row=row, sums=beta_sums, squared_sums=beta_squared_sums, counts=beta_counts)
        clipped_stats_seconds = time.perf_counter() - clipped_stats_start

    # Ridge solve: use the single-label Gram structure instead of materializing X.
    ridge_start = time.perf_counter()
    perturbation_counts = counts[1:].astype(np.float64, copy=False)
    total_rhs = beta_sums.sum(axis=0)
    perturbation_rhs = beta_sums[1:]
    beta = _solve_single_label_ridge(
        total_rhs=total_rhs,
        perturbation_rhs=perturbation_rhs,
        perturbation_counts=perturbation_counts,
        model_cell_count=float(counts.sum()),
        lr_lambda=float(lr_lambda),
    )
    ridge_seconds = time.perf_counter() - ridge_start

    beta_norm_sq = np.einsum("ij,ij->i", beta[1:], beta[1:])
    baseline_projection = beta[1:] @ beta[0]

    # Pass 2: stream again and compute only the observed perturbation score.
    scores = np.zeros((labels.shape[0], 1), dtype=np.float32)
    valid_mask = np.zeros(labels.shape[0], dtype=bool)
    max_score_by_perturbation = np.zeros(len(selected), dtype=np.float64)

    pass2_start = time.perf_counter()
    for start in range(0, labels.shape[0], chunk_size):
        stop = min(start + chunk_size, labels.shape[0])
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk = chunk[:, union_gene_indices]
        if clip_values is not None:
            chunk = _clip_matrix_columns(chunk, clip_values)
        chunk_labels = labels[start:stop]
        row_indices = np.arange(start, stop, dtype=np.int64)
        for perturbation in _iter_unique_label_keys(chunk_labels):
            perturbation_index = label_to_row.get(perturbation, 0) - 1
            if perturbation_index < 0:
                continue
            mask = chunk_labels == perturbation
            denominator = beta_norm_sq[perturbation_index]
            if denominator <= 0.0:
                continue
            projected = _matvec(chunk[mask], beta[perturbation_index + 1])
            raw = (projected - baseline_projection[perturbation_index] - score_lambda) / denominator
            clipped = np.clip(raw, 0.0, scale_factor) / scale_factor
            selected_rows = row_indices[mask]
            scores[selected_rows, 0] = clipped.astype(np.float32, copy=False)
            valid_mask[selected_rows] = True
            if clipped.size:
                max_score_by_perturbation[perturbation_index] = max(
                    max_score_by_perturbation[perturbation_index],
                    float(np.max(clipped)),
                )
    pass2_seconds = time.perf_counter() - pass2_start

    if scale_score:
        valid_rows = valid_mask & (perturbation_codes >= 0)
        valid_indices = np.flatnonzero(valid_rows)
        row_max = max_score_by_perturbation[perturbation_codes[valid_indices]]
        nonzero = row_max > 0.0
        scores[valid_indices[nonzero], 0] /= row_max[nonzero].astype(np.float32, copy=False)
        scores[valid_indices[~nonzero], 0] = 0.0

    for index, perturbation in enumerate(selected):
        target_gene_metadata[perturbation]["beta_norm_sq"] = float(beta_norm_sq[index])
        target_gene_metadata[perturbation]["max_score_before_column_scale"] = float(
            max_score_by_perturbation[index]
        )
        target_gene_metadata[perturbation]["column_scaled"] = bool(
            scale_score and max_score_by_perturbation[index] > 0.0
        )

    metadata = {
        "algorithm": "ps_score_exact_fast",
        "input_type": "anndata-single-label-backed-stream",
        "layer": layer,
        "perturb_column": perturb_column,
        "control_label": ctrl_name,
        "target_gene_source": target_gene_source_detail["mode"],
        "target_gene_source_detail": target_gene_source_detail,
        "target_gene_max": int(target_gene_max),
        "rank_by_abs_t": bool(rank_by_abs_t),
        "quantile_clip": clip_quantile is not None,
        "clip_quantile": None if clip_quantile is None else float(clip_quantile),
        "clip_method": None if clip_quantile is None else "streaming_histogram",
        "clip_bins": None if clip_quantile is None else int(clip_bins),
        "clip_value_summary": _summarize_clip_values(clip_values),
        "clip_values": None if clip_values is None else clip_values.tolist(),
        "chunk_size": int(chunk_size),
        "target_sum": float(target_sum),
        "lr_lambda": float(lr_lambda),
        "score_lambda": float(score_lambda),
        "scale_factor": float(scale_factor),
        "scale_score": bool(scale_score),
        "score_vector_shape": (int(labels.shape[0]), 1),
        "selected_perturbations": list(selected),
        "control_cell_count": int(counts[0]),
        "perturbation_cell_counts": {perturbation: int(counts[index + 1]) for index, perturbation in enumerate(selected)},
        "union_target_gene_count": int(union_gene_indices.shape[0]),
        "union_target_genes": [str(var_names[index]) for index in union_gene_indices],
        "target_gene_metadata": target_gene_metadata,
        "beta_shape": tuple(int(value) for value in beta.shape),
        "valid_scored_cell_count": int(np.count_nonzero(valid_mask)),
        "timings": {
            "pass1_sufficient_stats_seconds": float(pass1_seconds),
            "clip_threshold_seconds": float(clip_threshold_seconds),
            "clipped_sufficient_stats_seconds": float(clipped_stats_seconds),
            "ridge_solve_seconds": float(ridge_seconds),
            "pass2_scoring_seconds": float(pass2_seconds),
            "total_seconds": float(time.perf_counter() - stage_start),
        },
        "max_rss_kb": _max_rss_kb(),
    }
    return ExactFastPsResult(
        scores=scores,
        valid_mask=valid_mask,
        beta=beta,
        union_gene_indices=union_gene_indices,
        metadata=metadata,
    )


def run_ps_score_exact_fast_multilabel_anndata(
    adata: Any,
    *,
    guide_matrix: Any,
    perturbation_names: Sequence[str],
    layer: str | None = None,
    perturbations: Sequence[str] | None = None,
    control_mask: Sequence[bool] | np.ndarray | None = None,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_max: int = 500,
    chunk_size: int = 8192,
    lr_lambda: float = 0.01,
    score_lambda: float = 0.0,
    scale_factor: float = 3.0,
    target_sum: float = DEFAULT_TARGET_SUM,
    rank_by_abs_t: bool = True,
    scale_score: bool = True,
    clip_quantile: float | None = None,
    clip_bins: int = DEFAULT_CLIP_BINS,
) -> ExactFastMultiLabelPsResult:
    """Run additive exact-fast PS scores from a multi-label guide matrix.

    `guide_matrix` is interpreted as cell by perturbation presence. Positive
    values are binarized. Rows with no selected perturbations are controls unless
    `control_mask` is supplied.
    """

    validate_layer(layer)
    _validate_target_genes(target_genes)
    clip_quantile = _validate_clip_quantile(clip_quantile)
    clip_bins = _validate_clip_bins(clip_bins)
    var_names = np.asarray(adata.var_names, dtype=object)
    matrix = extract_anndata_matrix(adata, layer=layer)
    guides, selected = _prepare_multilabel_guides(
        guide_matrix=guide_matrix,
        perturbation_names=perturbation_names,
        perturbations=perturbations,
        n_obs=int(adata.n_obs),
    )
    control_rows = _resolve_multilabel_control_mask(control_mask=control_mask, guides=guides)
    active_counts_per_cell = np.asarray(guides.getnnz(axis=1)).ravel().astype(np.int64, copy=False)
    model_rows = control_rows | (active_counts_per_cell > 0)
    model_cell_count = int(np.count_nonzero(model_rows))
    if model_cell_count == 0:
        raise ValueError("At least one control or perturbation-positive cell is required")

    stage_start = time.perf_counter()
    pass1_start = time.perf_counter()
    sums = np.zeros((len(selected) + 1, var_names.shape[0]), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    counts = np.zeros(len(selected) + 1, dtype=np.int64)
    for start in range(0, guides.shape[0], chunk_size):
        stop = min(start + chunk_size, guides.shape[0])
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk_guides = guides[start:stop]
        chunk_control = control_rows[start:stop]
        if np.any(chunk_control):
            _add_group_stats(chunk[chunk_control], row=0, sums=sums, squared_sums=squared_sums, counts=counts)
        _add_multilabel_group_stats(chunk, chunk_guides, sums=sums, squared_sums=squared_sums, counts=counts)
    pass1_seconds = time.perf_counter() - pass1_start

    control_stats = StreamFeatureStats(count=int(counts[0]), sums=sums[0], squared_sums=squared_sums[0])
    if control_stats.count == 0:
        raise ValueError("Control cells are required for multi-label exact-fast scoring")

    target_gene_indices_by_perturbation: dict[str, np.ndarray] = {}
    target_gene_metadata: dict[str, dict[str, Any]] = {}
    gene_lookup = {str(gene): index for index, gene in enumerate(var_names.astype(str))}
    if target_genes is not None:
        target_gene_indices_by_perturbation = _resolve_provided_target_genes(
            target_genes=target_genes,
            selected_perturbations=selected,
            gene_lookup=gene_lookup,
            target_gene_min=1,
            target_gene_max=target_gene_max,
        )
        target_gene_source_detail = {"mode": "provided"}
    else:
        for perturbation_index, perturbation in enumerate(selected, start=1):
            perturb_stats = StreamFeatureStats(
                count=int(counts[perturbation_index]),
                sums=sums[perturbation_index],
                squared_sums=squared_sums[perturbation_index],
            )
            t_scores = welch_t_scores_from_stats(perturb_stats, control_stats)
            target_gene_indices_by_perturbation[perturbation] = top_k_indices(
                t_scores,
                min(target_gene_max, t_scores.shape[0]),
                absolute=rank_by_abs_t,
            ).astype(np.int64, copy=False)
        target_gene_source_detail = {"mode": "streamed_welch", "rank_by_abs_t": bool(rank_by_abs_t)}

    for perturbation_index, perturbation in enumerate(selected, start=1):
        gene_indices = target_gene_indices_by_perturbation[perturbation]
        target_gene_metadata[perturbation] = {
            "cell_count": int(counts[perturbation_index]),
            "selected_gene_count": int(gene_indices.shape[0]),
            "selected_genes": [str(var_names[index]) for index in gene_indices],
        }

    union_gene_indices = _ordered_union_indices(target_gene_indices_by_perturbation.values())
    clip_values: np.ndarray | None = None
    clip_threshold_seconds = 0.0
    if clip_quantile is not None:
        clip_start = time.perf_counter()
        clip_values = _estimate_multilabel_histogram_clip_values(
            matrix,
            guides=guides,
            control_mask=control_rows,
            union_gene_indices=union_gene_indices,
            model_cell_count=model_cell_count,
            chunk_size=chunk_size,
            target_sum=target_sum,
            quantile=clip_quantile,
            bins=clip_bins,
        )
        clip_threshold_seconds = time.perf_counter() - clip_start

    ridge_stats_start = time.perf_counter()
    xtx, xty = _collect_multilabel_ridge_stats(
        matrix,
        guides=guides,
        control_mask=control_rows,
        union_gene_indices=union_gene_indices,
        clip_values=clip_values,
        chunk_size=chunk_size,
        target_sum=target_sum,
    )
    ridge_stats_seconds = time.perf_counter() - ridge_stats_start

    ridge_start = time.perf_counter()
    ridge_system = xtx + sparse.eye(xtx.shape[0], format="csc", dtype=np.float64) * float(lr_lambda)
    beta = np.asarray(spsolve(ridge_system, xty), dtype=np.float64)
    if beta.ndim == 1:
        beta = beta[:, None]
    ridge_seconds = time.perf_counter() - ridge_start

    beta_norm_sq = np.einsum("ij,ij->i", beta[1:], beta[1:])
    score_values: list[np.ndarray] = []
    cell_index_values: list[np.ndarray] = []
    perturbation_index_values: list[np.ndarray] = []
    max_score_by_perturbation = np.zeros(len(selected), dtype=np.float64)
    valid_mask = np.zeros(guides.shape[0], dtype=bool)

    scoring_start = time.perf_counter()
    for start in range(0, guides.shape[0], chunk_size):
        stop = min(start + chunk_size, guides.shape[0])
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk = chunk[:, union_gene_indices]
        if clip_values is not None:
            chunk = _clip_matrix_columns(chunk, clip_values)
        chunk_guides = guides[start:stop]
        row_indices = np.arange(start, stop, dtype=np.int64)
        for active_set, local_rows in _group_rows_by_active_set(chunk_guides).items():
            active_indices = np.asarray(active_set, dtype=np.int64)
            active_beta = beta[active_indices + 1]
            gram = active_beta @ active_beta.T
            rhs = _matmat(chunk[local_rows], active_beta.T) - (active_beta @ beta[0])[None, :]
            bounded_scores = _solve_bounded_quadratic_scores(
                gram=gram,
                rhs=rhs,
                linear_penalty=float(score_lambda),
                upper=float(scale_factor),
            )
            normalized = bounded_scores / float(scale_factor)
            global_rows = row_indices[local_rows]
            valid_mask[global_rows] = True
            for offset, perturbation_index in enumerate(active_indices):
                values = normalized[:, offset].astype(np.float32, copy=False)
                score_values.append(values)
                cell_index_values.append(global_rows.copy())
                perturbation_index_values.append(
                    np.full(values.shape[0], int(perturbation_index), dtype=np.int32)
                )
                if values.size:
                    max_score_by_perturbation[perturbation_index] = max(
                        max_score_by_perturbation[perturbation_index],
                        float(np.max(values)),
                    )
    scoring_seconds = time.perf_counter() - scoring_start

    if score_values:
        scores = np.concatenate(score_values).astype(np.float32, copy=False)
        cell_indices = np.concatenate(cell_index_values).astype(np.int64, copy=False)
        perturbation_indices = np.concatenate(perturbation_index_values).astype(np.int32, copy=False)
    else:
        scores = np.zeros(0, dtype=np.float32)
        cell_indices = np.zeros(0, dtype=np.int64)
        perturbation_indices = np.zeros(0, dtype=np.int32)

    if scale_score and scores.size:
        row_max = max_score_by_perturbation[perturbation_indices]
        nonzero = row_max > 0.0
        scores[nonzero] /= row_max[nonzero].astype(np.float32, copy=False)
        scores[~nonzero] = 0.0

    for index, perturbation in enumerate(selected):
        target_gene_metadata[perturbation]["beta_norm_sq"] = float(beta_norm_sq[index])
        target_gene_metadata[perturbation]["max_score_before_column_scale"] = float(
            max_score_by_perturbation[index]
        )
        target_gene_metadata[perturbation]["column_scaled"] = bool(
            scale_score and max_score_by_perturbation[index] > 0.0
        )

    metadata = {
        "algorithm": "ps_score_exact_fast_multilabel",
        "input_type": "anndata-multilabel-guide-matrix-backed-stream",
        "layer": layer,
        "guide_matrix_binary": True,
        "control_source": "provided_mask" if control_mask is not None else "zero_selected_guides",
        "target_gene_source": target_gene_source_detail["mode"],
        "target_gene_source_detail": target_gene_source_detail,
        "target_gene_max": int(target_gene_max),
        "rank_by_abs_t": bool(rank_by_abs_t),
        "quantile_clip": clip_quantile is not None,
        "clip_quantile": None if clip_quantile is None else float(clip_quantile),
        "clip_method": None if clip_quantile is None else "streaming_histogram",
        "clip_bins": None if clip_quantile is None else int(clip_bins),
        "clip_value_summary": _summarize_clip_values(clip_values),
        "chunk_size": int(chunk_size),
        "target_sum": float(target_sum),
        "lr_lambda": float(lr_lambda),
        "score_lambda": float(score_lambda),
        "scale_factor": float(scale_factor),
        "scale_score": bool(scale_score),
        "score_output_format": "long",
        "score_count": int(scores.shape[0]),
        "selected_perturbations": list(selected),
        "control_cell_count": int(counts[0]),
        "model_cell_count": int(model_cell_count),
        "perturbation_cell_counts": {perturbation: int(counts[index + 1]) for index, perturbation in enumerate(selected)},
        "guide_multiplicity": _summarize_guide_multiplicity(active_counts_per_cell),
        "union_target_gene_count": int(union_gene_indices.shape[0]),
        "union_target_genes": [str(var_names[index]) for index in union_gene_indices],
        "target_gene_metadata": target_gene_metadata,
        "beta_shape": tuple(int(value) for value in beta.shape),
        "valid_scored_cell_count": int(np.count_nonzero(valid_mask)),
        "timings": {
            "pass1_sufficient_stats_seconds": float(pass1_seconds),
            "clip_threshold_seconds": float(clip_threshold_seconds),
            "ridge_sufficient_stats_seconds": float(ridge_stats_seconds),
            "ridge_solve_seconds": float(ridge_seconds),
            "scoring_seconds": float(scoring_seconds),
            "total_seconds": float(time.perf_counter() - stage_start),
        },
        "max_rss_kb": _max_rss_kb(),
    }
    return ExactFastMultiLabelPsResult(
        scores=scores,
        cell_indices=cell_indices,
        perturbation_indices=perturbation_indices,
        perturbations=list(selected),
        valid_mask=valid_mask,
        beta=beta,
        union_gene_indices=union_gene_indices,
        metadata=metadata,
    )


def run_ps_score_exact_fast_dataset(
    dataset_path: str | Path,
    *,
    output_dir: str | Path,
    perturb_column: str,
    ctrl_name: str,
    layer: str | None = None,
    perturbations: Sequence[str] | None = None,
    null_labels: Sequence[str] | None = None,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_max: int = 500,
    chunk_size: int = 8192,
    lr_lambda: float = 0.01,
    score_lambda: float = 0.0,
    scale_factor: float = 3.0,
    target_sum: float = DEFAULT_TARGET_SUM,
    rank_by_abs_t: bool = True,
    scale_score: bool = True,
    clip_quantile: float | None = None,
    clip_bins: int = DEFAULT_CLIP_BINS,
) -> dict[str, Any]:
    adata = ad.read_h5ad(Path(dataset_path), backed="r")
    try:
        result = run_ps_score_exact_fast_anndata(
            adata,
            perturb_column=perturb_column,
            ctrl_name=ctrl_name,
            layer=layer,
            perturbations=perturbations,
            null_labels=null_labels,
            target_genes=target_genes,
            target_gene_max=target_gene_max,
            chunk_size=chunk_size,
            lr_lambda=lr_lambda,
            score_lambda=score_lambda,
            scale_factor=scale_factor,
            target_sum=target_sum,
            rank_by_abs_t=rank_by_abs_t,
            scale_score=scale_score,
            clip_quantile=clip_quantile,
            clip_bins=clip_bins,
        )
    finally:
        _close_adata(adata)
    return write_ps_score_exact_fast_output(result, output_dir=output_dir, dataset_path=dataset_path)


def write_ps_score_exact_fast_output(
    result: ExactFastPsResult,
    *,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    score_path = output_path / "ps-score-exact-fast.npy"
    valid_mask_path = output_path / "ps-score-exact-fast-valid-mask.npy"
    target_gene_path = output_path / "ps-score-exact-fast-target-genes.json"
    manifest_path = output_path / "ps-score-exact-fast-manifest.json"

    np.save(score_path, result.scores)
    np.save(valid_mask_path, result.valid_mask)
    with target_gene_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _to_jsonable(
                {
                    "union_target_genes": result.metadata["union_target_genes"],
                    "target_gene_metadata": result.metadata["target_gene_metadata"],
                }
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    manifest = {
        "algorithm": result.metadata["algorithm"],
        "dataset_path": None if dataset_path is None else str(dataset_path),
        "perturbation_column": result.metadata["perturb_column"],
        "control_label": result.metadata["control_label"],
        "target_gene_source": result.metadata["target_gene_source"],
        "target_gene_max": result.metadata["target_gene_max"],
        "union_target_gene_count": result.metadata["union_target_gene_count"],
        "quantile_clip": result.metadata["quantile_clip"],
        "clip_quantile": result.metadata["clip_quantile"],
        "clip_method": result.metadata["clip_method"],
        "clip_bins": result.metadata["clip_bins"],
        "clip_value_summary": result.metadata["clip_value_summary"],
        "chunk_size": result.metadata["chunk_size"],
        "target_sum": result.metadata["target_sum"],
        "lr_lambda": result.metadata["lr_lambda"],
        "score_lambda": result.metadata["score_lambda"],
        "scale_factor": result.metadata["scale_factor"],
        "scale_score": result.metadata["scale_score"],
        "score_vector_shape": result.metadata["score_vector_shape"],
        "valid_scored_cell_count": result.metadata["valid_scored_cell_count"],
        "score_output_paths": {
            "normalized_scores": str(score_path),
            "valid_mask": str(valid_mask_path),
            "target_gene_metadata": str(target_gene_path),
        },
        "timings": result.metadata["timings"],
        "max_rss_kb": result.metadata["max_rss_kb"],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perturb-column", required=True)
    parser.add_argument("--ctrl-name", required=True)
    parser.add_argument("--layer")
    parser.add_argument("--target-gene-tsv")
    parser.add_argument("--target-gene-max", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--lr-lambda", type=float, default=0.01)
    parser.add_argument("--score-lambda", type=float, default=0.0)
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument("--target-sum", type=float, default=DEFAULT_TARGET_SUM)
    parser.add_argument("--clip-quantile", type=float)
    parser.add_argument("--clip-bins", type=int, default=DEFAULT_CLIP_BINS)
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument("--null-label", action="append", dest="null_labels")
    parser.add_argument("--rank-by-signed-t", action="store_true")
    parser.add_argument("--no-scale-score", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    target_genes = None
    if args.target_gene_tsv:
        target_genes = _load_target_gene_tsv(args.target_gene_tsv)
    return run_ps_score_exact_fast_dataset(
        args.dataset_path,
        output_dir=args.output_dir,
        perturb_column=args.perturb_column,
        ctrl_name=args.ctrl_name,
        layer=args.layer,
        perturbations=args.perturbations,
        null_labels=args.null_labels,
        target_genes=target_genes,
        target_gene_max=args.target_gene_max,
        chunk_size=args.chunk_size,
        lr_lambda=args.lr_lambda,
        score_lambda=args.score_lambda,
        scale_factor=args.scale_factor,
        target_sum=args.target_sum,
        rank_by_abs_t=not args.rank_by_signed_t,
        scale_score=not args.no_scale_score,
        clip_quantile=args.clip_quantile,
        clip_bins=args.clip_bins,
    )


def _resolve_selected_perturbations(
    labels: Sequence[Any],
    *,
    control_label: str,
    perturbations: Sequence[str] | None,
    null_labels: set[str],
) -> list[str]:
    available: list[str] = []
    available_set: set[str] = set()
    for label in labels:
        if _is_missing_label(label) or label == control_label or label in null_labels:
            continue
        key = str(label)
        if key in available_set:
            continue
        available.append(key)
        available_set.add(key)
    if perturbations is None:
        return available
    selected: list[str] = []
    seen: set[str] = set()
    for perturbation in perturbations:
        key = str(perturbation)
        if key in available_set and key not in seen:
            selected.append(key)
            seen.add(key)
    return selected


def _validate_target_genes(
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
) -> None:
    if target_genes is None:
        return
    if isinstance(target_genes, str):
        raise TypeError("target_genes must be a mapping or sequence of gene names, not a string")
    if isinstance(target_genes, Mapping):
        for key, genes in target_genes.items():
            if not isinstance(key, str) or not key:
                raise TypeError("target_genes mapping keys must be non-empty strings")
            _normalize_gene_names(genes)
        return
    _normalize_gene_names(target_genes)


def _validate_clip_quantile(value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or value <= 0.0 or value > 1.0:
        raise ValueError("clip_quantile must be in (0, 1]")
    return float(value)


def _validate_clip_bins(value: int) -> int:
    if not isinstance(value, int) or value < 2:
        raise ValueError("clip_bins must be an integer >= 2")
    return int(value)


def _load_target_gene_tsv(path: str | Path) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "perturbation" not in reader.fieldnames or "gene" not in reader.fieldnames:
            raise ValueError("target gene TSV must contain 'perturbation' and 'gene' columns")
        for row in reader:
            perturbation = row.get("perturbation")
            gene = row.get("gene")
            if not perturbation or not gene:
                continue
            resolved.setdefault(str(perturbation), []).append(str(gene))
    return {perturbation: _normalize_gene_names(genes) for perturbation, genes in resolved.items()}


def _resolve_provided_target_genes(
    *,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    selected_perturbations: Sequence[str],
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    target_gene_max: int,
) -> dict[str, np.ndarray]:
    resolved: dict[str, np.ndarray] = {}
    for perturbation in selected_perturbations:
        provided = _get_provided_genes(target_genes, perturbation=perturbation)
        if provided is None:
            raise ValueError(f"No target genes were provided for perturbation {perturbation!r}")
        genes = _normalize_gene_names(provided)
        if len(genes) > target_gene_max:
            genes = genes[:target_gene_max]
        if len(genes) < target_gene_min:
            raise ValueError(
                f"Need at least {target_gene_min} target genes for perturbation {perturbation!r}"
            )
        missing = [gene for gene in genes if gene not in gene_lookup]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(
                f"Unknown target genes requested for perturbation {perturbation!r}: {joined}"
            )
        resolved[str(perturbation)] = np.asarray([gene_lookup[gene] for gene in genes], dtype=np.int64)
    return resolved


def _get_provided_genes(
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    *,
    perturbation: str,
) -> Sequence[str] | None:
    if target_genes is None:
        return None
    if isinstance(target_genes, Mapping):
        if perturbation in target_genes:
            return target_genes[perturbation]
        return target_genes.get(str(perturbation))
    return target_genes


def _normalize_gene_names(genes: Sequence[str]) -> list[str]:
    if isinstance(genes, str):
        raise TypeError("target gene lists must be sequences of gene names, not strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        if not isinstance(gene, str) or not gene:
            raise TypeError("target gene names must be non-empty strings")
        if gene in seen:
            continue
        normalized.append(gene)
        seen.add(gene)
    return normalized


def _log_normalize_chunk(matrix: Any, *, target_sum: float) -> Any:
    if sparse.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float64)
        totals = np.asarray(work.sum(axis=1)).ravel()
        scales = np.zeros(work.shape[0], dtype=np.float64)
        nonzero = totals > 0
        scales[nonzero] = target_sum / totals[nonzero]
        work = work.multiply(scales[:, None]).tocsr()
        work.data = np.log1p(work.data)
        return work

    dense = np.asarray(matrix, dtype=np.float64).copy()
    totals = dense.sum(axis=1, keepdims=True)
    nonzero = totals[:, 0] > 0
    dense[nonzero] *= target_sum / totals[nonzero]
    dense[~nonzero] = 0.0
    return np.log1p(dense)


def _estimate_histogram_clip_values(
    matrix: Any,
    *,
    labels: np.ndarray,
    label_to_row: Mapping[str, int],
    union_gene_indices: np.ndarray,
    model_cell_count: int,
    chunk_size: int,
    target_sum: float,
    quantile: float,
    bins: int,
) -> np.ndarray:
    if model_cell_count <= 0:
        raise ValueError("Cannot estimate clip values without model cells")
    max_value = float(np.log1p(target_sum))
    hist = np.zeros((union_gene_indices.shape[0], bins), dtype=np.uint32)
    nonzero_counts = np.zeros(union_gene_indices.shape[0], dtype=np.int64)

    for start in range(0, labels.shape[0], chunk_size):
        stop = min(start + chunk_size, labels.shape[0])
        chunk_labels = labels[start:stop]
        model_mask = _model_label_mask(chunk_labels, label_to_row)
        if not np.any(model_mask):
            continue
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk = chunk[model_mask][:, union_gene_indices]
        _accumulate_nonzero_histogram(
            chunk,
            hist=hist,
            nonzero_counts=nonzero_counts,
            max_value=max_value,
        )

    zero_counts = np.full(union_gene_indices.shape[0], model_cell_count, dtype=np.int64) - nonzero_counts
    return _histogram_quantiles(
        hist,
        zero_counts=zero_counts,
        total_count=model_cell_count,
        quantile=quantile,
        max_value=max_value,
    )


def _model_label_mask(labels: Sequence[Any], label_to_row: Mapping[str, int]) -> np.ndarray:
    return np.asarray(
        [False if _is_missing_label(label) else str(label) in label_to_row for label in labels],
        dtype=bool,
    )


def _accumulate_nonzero_histogram(
    matrix: Any,
    *,
    hist: np.ndarray,
    nonzero_counts: np.ndarray,
    max_value: float,
) -> None:
    if sparse.issparse(matrix):
        coo = matrix.tocoo()
        positive = coo.data > 0.0
        if not np.any(positive):
            return
        columns = coo.col[positive]
        nonzero_counts += np.bincount(columns, minlength=hist.shape[0]).astype(np.int64, copy=False)
        bin_indices = _histogram_bin_indices(coo.data[positive], bins=hist.shape[1], max_value=max_value)
        np.add.at(hist, (columns, bin_indices), 1)
        return

    dense = np.asarray(matrix, dtype=np.float64)
    nonzero_rows, nonzero_cols = np.nonzero(dense > 0.0)
    if nonzero_cols.size == 0:
        return
    nonzero_counts += np.bincount(nonzero_cols, minlength=hist.shape[0]).astype(np.int64, copy=False)
    bin_indices = _histogram_bin_indices(dense[nonzero_rows, nonzero_cols], bins=hist.shape[1], max_value=max_value)
    np.add.at(hist, (nonzero_cols, bin_indices), 1)


def _histogram_bin_indices(values: np.ndarray, *, bins: int, max_value: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max_value
    indices = np.floor(scaled * bins).astype(np.int64, copy=False)
    return np.clip(indices, 0, bins - 1)


def _histogram_quantiles(
    hist: np.ndarray,
    *,
    zero_counts: np.ndarray,
    total_count: int,
    quantile: float,
    max_value: float,
) -> np.ndarray:
    edges = (np.arange(1, hist.shape[1] + 1, dtype=np.float64) * max_value) / float(hist.shape[1])
    position = float(total_count - 1) * quantile
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    fraction = position - float(lower_rank)

    clip_values = np.zeros(hist.shape[0], dtype=np.float64)
    for gene_index in range(hist.shape[0]):
        lower = _histogram_value_at_rank(hist[gene_index], zero_count=int(zero_counts[gene_index]), rank=lower_rank, edges=edges)
        upper = _histogram_value_at_rank(hist[gene_index], zero_count=int(zero_counts[gene_index]), rank=upper_rank, edges=edges)
        clip_values[gene_index] = lower + fraction * (upper - lower)
    return clip_values


def _histogram_value_at_rank(hist_row: np.ndarray, *, zero_count: int, rank: int, edges: np.ndarray) -> float:
    if rank < zero_count:
        return 0.0
    nonzero_rank = rank - zero_count
    cumulative = np.cumsum(hist_row, dtype=np.int64)
    if cumulative.size == 0 or cumulative[-1] == 0:
        return 0.0
    bin_index = int(np.searchsorted(cumulative, nonzero_rank + 1, side="left"))
    if bin_index >= edges.shape[0]:
        bin_index = edges.shape[0] - 1
    return float(edges[bin_index])


def _clip_matrix_columns(matrix: Any, clip_values: np.ndarray) -> Any:
    if sparse.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float64)
        if work.data.size:
            work.data = np.minimum(work.data, clip_values[work.indices])
            work.eliminate_zeros()
        return work

    dense = np.asarray(matrix, dtype=np.float64).copy()
    return np.minimum(dense, clip_values[None, :])


def _summarize_clip_values(clip_values: np.ndarray | None) -> dict[str, Any] | None:
    if clip_values is None:
        return None
    return {
        "count": int(clip_values.shape[0]),
        "min": float(np.min(clip_values)) if clip_values.size else 0.0,
        "max": float(np.max(clip_values)) if clip_values.size else 0.0,
        "mean": float(np.mean(clip_values)) if clip_values.size else 0.0,
        "zero_count": int(np.count_nonzero(clip_values == 0.0)),
    }


def _add_group_stats(
    matrix: Any,
    *,
    row: int,
    sums: np.ndarray,
    squared_sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    counts[row] += int(matrix.shape[0])
    if sparse.issparse(matrix):
        sums[row] += np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
        squared_sums[row] += np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(np.float64, copy=False)
        return
    dense = np.asarray(matrix, dtype=np.float64)
    sums[row] += dense.sum(axis=0)
    squared_sums[row] += np.square(dense).sum(axis=0)


def _solve_single_label_ridge(
    *,
    total_rhs: np.ndarray,
    perturbation_rhs: np.ndarray,
    perturbation_counts: np.ndarray,
    model_cell_count: float,
    lr_lambda: float,
) -> np.ndarray:
    perturbation_denominator = perturbation_counts + lr_lambda
    weighted_rhs = ((perturbation_counts / perturbation_denominator)[:, None] * perturbation_rhs).sum(axis=0)
    intercept_denominator = (model_cell_count + lr_lambda) - np.sum(
        perturbation_counts * perturbation_counts / perturbation_denominator
    )
    beta0 = (total_rhs - weighted_rhs) / intercept_denominator
    perturbation_beta = (perturbation_rhs - perturbation_counts[:, None] * beta0[None, :]) / perturbation_denominator[:, None]
    return np.vstack([beta0[None, :], perturbation_beta])


def _prepare_multilabel_guides(
    *,
    guide_matrix: Any,
    perturbation_names: Sequence[str],
    perturbations: Sequence[str] | None,
    n_obs: int,
) -> tuple[sparse.csr_matrix, list[str]]:
    names = [str(name) for name in perturbation_names]
    if not names:
        raise ValueError("perturbation_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("perturbation_names must be unique")
    if sparse.issparse(guide_matrix):
        guides = guide_matrix.tocsr(copy=True)
    else:
        guides = sparse.csr_matrix(np.asarray(guide_matrix))
    if guides.ndim != 2:
        raise ValueError("guide_matrix must be two-dimensional")
    if guides.shape[0] != n_obs:
        raise ValueError("guide_matrix row count must match adata.n_obs")
    if guides.shape[1] != len(names):
        raise ValueError("guide_matrix column count must match perturbation_names")
    if guides.data.size:
        guides.data = (guides.data > 0).astype(np.float64, copy=False)
        guides.eliminate_zeros()
    observed = np.asarray(guides.getnnz(axis=0)).ravel() > 0
    if perturbations is None:
        selected_columns = [index for index, is_observed in enumerate(observed) if is_observed]
    else:
        name_to_column = {name: index for index, name in enumerate(names)}
        missing = [str(name) for name in perturbations if str(name) not in name_to_column]
        if missing:
            raise ValueError("Unknown perturbation names requested: " + ", ".join(sorted(missing)))
        selected_columns = [name_to_column[str(name)] for name in perturbations if observed[name_to_column[str(name)]]]
    if not selected_columns:
        raise ValueError("No selected perturbations have positive guide entries")
    selected = [names[index] for index in selected_columns]
    selected_guides = guides[:, selected_columns].tocsr(copy=True)
    selected_guides.sort_indices()
    return selected_guides, selected


def _resolve_multilabel_control_mask(
    *,
    control_mask: Sequence[bool] | np.ndarray | None,
    guides: sparse.csr_matrix,
) -> np.ndarray:
    active = np.asarray(guides.getnnz(axis=1)).ravel() > 0
    if control_mask is None:
        resolved = ~active
    else:
        resolved = np.asarray(control_mask, dtype=bool)
        if resolved.shape != (guides.shape[0],):
            raise ValueError("control_mask must have one value per observation")
        if np.any(resolved & active):
            raise ValueError("control_mask rows cannot contain selected perturbation guides")
    if not np.any(resolved):
        raise ValueError("At least one control cell is required")
    return resolved


def _add_multilabel_group_stats(
    matrix: Any,
    guides: sparse.csr_matrix,
    *,
    sums: np.ndarray,
    squared_sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    coo = guides.tocoo()
    if coo.nnz == 0:
        return
    order = np.argsort(coo.col, kind="stable")
    columns = coo.col[order]
    rows = coo.row[order]
    for column in np.unique(columns):
        selected_rows = rows[columns == column]
        _add_group_stats(matrix[selected_rows], row=int(column) + 1, sums=sums, squared_sums=squared_sums, counts=counts)


def _estimate_multilabel_histogram_clip_values(
    matrix: Any,
    *,
    guides: sparse.csr_matrix,
    control_mask: np.ndarray,
    union_gene_indices: np.ndarray,
    model_cell_count: int,
    chunk_size: int,
    target_sum: float,
    quantile: float,
    bins: int,
) -> np.ndarray:
    max_value = float(np.log1p(target_sum))
    hist = np.zeros((union_gene_indices.shape[0], bins), dtype=np.uint32)
    nonzero_counts = np.zeros(union_gene_indices.shape[0], dtype=np.int64)
    for start in range(0, guides.shape[0], chunk_size):
        stop = min(start + chunk_size, guides.shape[0])
        active = np.asarray(guides[start:stop].getnnz(axis=1)).ravel() > 0
        model_mask = control_mask[start:stop] | active
        if not np.any(model_mask):
            continue
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk = chunk[model_mask][:, union_gene_indices]
        _accumulate_nonzero_histogram(chunk, hist=hist, nonzero_counts=nonzero_counts, max_value=max_value)
    zero_counts = np.full(union_gene_indices.shape[0], model_cell_count, dtype=np.int64) - nonzero_counts
    return _histogram_quantiles(
        hist,
        zero_counts=zero_counts,
        total_count=model_cell_count,
        quantile=quantile,
        max_value=max_value,
    )


def _collect_multilabel_ridge_stats(
    matrix: Any,
    *,
    guides: sparse.csr_matrix,
    control_mask: np.ndarray,
    union_gene_indices: np.ndarray,
    clip_values: np.ndarray | None,
    chunk_size: int,
    target_sum: float,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    perturbation_count = guides.shape[1]
    xty = np.zeros((perturbation_count + 1, union_gene_indices.shape[0]), dtype=np.float64)
    intercept_count = 0.0
    perturbation_counts = np.zeros(perturbation_count, dtype=np.float64)
    cooccurrence = sparse.csr_matrix((perturbation_count, perturbation_count), dtype=np.float64)

    for start in range(0, guides.shape[0], chunk_size):
        stop = min(start + chunk_size, guides.shape[0])
        chunk_guides = guides[start:stop]
        active = np.asarray(chunk_guides.getnnz(axis=1)).ravel() > 0
        model_mask = control_mask[start:stop] | active
        if not np.any(model_mask):
            continue
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk = chunk[:, union_gene_indices]
        if clip_values is not None:
            chunk = _clip_matrix_columns(chunk, clip_values)
        chunk = chunk[model_mask]
        chunk_guides = chunk_guides[model_mask]
        intercept_count += float(chunk.shape[0])
        xty[0] += _column_sums(chunk)
        perturbation_counts += np.asarray(chunk_guides.sum(axis=0)).ravel().astype(np.float64, copy=False)
        cooccurrence = cooccurrence + (chunk_guides.T @ chunk_guides).tocsr()
        _add_multilabel_xty(chunk, chunk_guides, xty=xty)

    top = sparse.hstack(
        [
            sparse.csr_matrix([[intercept_count]], dtype=np.float64),
            sparse.csr_matrix(perturbation_counts[None, :]),
        ],
        format="csr",
    )
    bottom = sparse.hstack(
        [sparse.csr_matrix(perturbation_counts[:, None]), cooccurrence],
        format="csr",
    )
    xtx = sparse.vstack([top, bottom], format="csc")
    return xtx, xty


def _add_multilabel_xty(matrix: Any, guides: sparse.csr_matrix, *, xty: np.ndarray) -> None:
    coo = guides.tocoo()
    if coo.nnz == 0:
        return
    order = np.argsort(coo.col, kind="stable")
    columns = coo.col[order]
    rows = coo.row[order]
    for column in np.unique(columns):
        selected_rows = rows[columns == column]
        xty[int(column) + 1] += _column_sums(matrix[selected_rows])


def _column_sums(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64).sum(axis=0)


def _group_rows_by_active_set(guides: sparse.csr_matrix) -> dict[tuple[int, ...], np.ndarray]:
    groups: dict[tuple[int, ...], list[int]] = {}
    indptr = guides.indptr
    indices = guides.indices
    for row_index in range(guides.shape[0]):
        active = tuple(int(index) for index in indices[indptr[row_index] : indptr[row_index + 1]])
        if not active:
            continue
        groups.setdefault(active, []).append(row_index)
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()}


def _solve_bounded_quadratic_scores(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
    upper: float,
) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    variable_count = gram.shape[0]
    if variable_count == 1:
        denominator = float(gram[0, 0])
        if denominator <= 0.0:
            return np.zeros((rhs.shape[0], 1), dtype=np.float64)
        return np.clip((rhs[:, [0]] - linear_penalty) / denominator, 0.0, upper)
    if variable_count <= 4:
        return _solve_bounded_quadratic_scores_active_set(
            gram=gram,
            rhs=rhs,
            linear_penalty=linear_penalty,
            upper=upper,
        )
    return _solve_bounded_quadratic_scores_lbfgsb(
        gram=gram,
        rhs=rhs,
        linear_penalty=linear_penalty,
        upper=upper,
    )


def _solve_bounded_quadratic_scores_active_set(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
    upper: float,
) -> np.ndarray:
    cell_count, variable_count = rhs.shape
    best = np.zeros((cell_count, variable_count), dtype=np.float64)
    best_objective = np.full(cell_count, np.inf, dtype=np.float64)
    for states in product((0, 1, 2), repeat=variable_count):
        states_array = np.asarray(states, dtype=np.int8)
        free = states_array == 0
        lower = states_array == 1
        upper_fixed = states_array == 2
        candidate = np.zeros((cell_count, variable_count), dtype=np.float64)
        if np.any(upper_fixed):
            candidate[:, upper_fixed] = upper
        if np.any(free):
            free_rhs = rhs[:, free] - linear_penalty
            fixed = lower | upper_fixed
            if np.any(fixed):
                free_rhs -= candidate[:, fixed] @ gram[np.ix_(fixed, free)]
            gram_free = gram[np.ix_(free, free)]
            try:
                candidate[:, free] = np.linalg.solve(gram_free, free_rhs.T).T
            except np.linalg.LinAlgError:
                candidate[:, free] = np.linalg.lstsq(gram_free, free_rhs.T, rcond=None)[0].T
        feasible = np.all(candidate >= -1e-9, axis=1) & np.all(candidate <= upper + 1e-9, axis=1)
        if not np.any(feasible):
            continue
        candidate = np.clip(candidate, 0.0, upper)
        objective = _bounded_quadratic_objective(candidate, gram=gram, rhs=rhs, linear_penalty=linear_penalty)
        update = feasible & (objective < best_objective)
        if np.any(update):
            best[update] = candidate[update]
            best_objective[update] = objective[update]
    return best


def _solve_bounded_quadratic_scores_lbfgsb(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
    upper: float,
) -> np.ndarray:
    scores = np.zeros_like(rhs, dtype=np.float64)
    bounds = [(0.0, upper)] * rhs.shape[1]
    for row_index, row_rhs in enumerate(rhs):
        start = np.zeros(rhs.shape[1], dtype=np.float64)

        def objective(value: np.ndarray) -> float:
            return float(0.5 * value @ gram @ value - row_rhs @ value + linear_penalty * np.sum(value))

        def gradient(value: np.ndarray) -> np.ndarray:
            return gram @ value - row_rhs + linear_penalty

        result = minimize(objective, start, jac=gradient, bounds=bounds, method="L-BFGS-B")
        scores[row_index] = np.clip(result.x, 0.0, upper)
    return scores


def _bounded_quadratic_objective(
    scores: np.ndarray,
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
) -> np.ndarray:
    return 0.5 * np.sum((scores @ gram) * scores, axis=1) - np.sum(rhs * scores, axis=1) + linear_penalty * np.sum(scores, axis=1)


def _summarize_guide_multiplicity(active_counts: np.ndarray) -> dict[str, Any]:
    return {
        "min": int(np.min(active_counts)) if active_counts.size else 0,
        "max": int(np.max(active_counts)) if active_counts.size else 0,
        "mean": float(np.mean(active_counts)) if active_counts.size else 0.0,
        "zero_count": int(np.count_nonzero(active_counts == 0)),
        "single_count": int(np.count_nonzero(active_counts == 1)),
        "multi_count": int(np.count_nonzero(active_counts >= 2)),
    }


def _matvec(matrix: Any, vector: np.ndarray) -> np.ndarray:
    value = matrix @ vector
    return np.asarray(value, dtype=np.float64).ravel()


def _matmat(matrix: Any, other: np.ndarray) -> np.ndarray:
    value = matrix @ other
    return np.asarray(value, dtype=np.float64)


def _ordered_union_indices(groups: Any) -> np.ndarray:
    union: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for index in group:
            key = int(index)
            if key in seen:
                continue
            union.append(key)
            seen.add(key)
    return np.asarray(union, dtype=np.int64)


def _iter_unique_label_keys(labels: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        if _is_missing_label(label):
            continue
        key = str(label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _is_missing_label(label: Any) -> bool:
    if label is None:
        return True
    if isinstance(label, (float, np.floating)):
        return bool(np.isnan(label))
    return False


def _close_adata(adata: Any) -> None:
    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ExactFastMultiLabelPsResult",
    "ExactFastPsResult",
    "build_parser",
    "main",
    "run_ps_score_exact_fast_anndata",
    "run_ps_score_exact_fast_dataset",
    "run_ps_score_exact_fast_multilabel_anndata",
    "write_ps_score_exact_fast_output",
]


if __name__ == "__main__":
    print(json.dumps(_to_jsonable(main()), indent=2, sort_keys=True))
