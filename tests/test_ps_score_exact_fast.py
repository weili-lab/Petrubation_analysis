from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.optimize import minimize
from scipy import sparse

from perturb_effects.ps_score_exact import run_ps_score_exact_anndata
from perturb_effects.ps_score_exact_fast import run_ps_score_exact_fast_anndata, run_ps_score_exact_fast_multilabel_anndata


def test_exact_fast_histogram_clip_matches_dense_exact_clip() -> None:
    counts = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [20.0, 1.0, 0.0],
            [24.0, 2.0, 0.0],
            [28.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["control", "control", "control", "pertA", "pertA", "pertA"], dtype=object)
    obs_names = [f"cell_{index}" for index in range(counts.shape[0])]
    var_names = ["g1", "g2", "g3"]
    library_sizes = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p((counts / library_sizes) * 1e4)

    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels}, index=obs_names),
        var=pd.DataFrame(index=var_names),
    )
    adata.layers["lognorm"] = lognorm

    exact = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="lognorm",
        counts_layer=None,
        perturbations=["pertA"],
        target_genes={"pertA": ["g1", "g2"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=True,
        clip_quantile=0.5,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        scale_score=True,
        return_wide=True,
    )
    exact_fast = run_ps_score_exact_fast_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        perturbations=["pertA"],
        target_genes={"pertA": ["g1", "g2"]},
        target_gene_max=2,
        chunk_size=2,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        target_sum=1e4,
        scale_score=True,
        clip_quantile=0.5,
        clip_bins=200000,
    )

    perturbed = labels == "pertA"
    exact_scores = exact.loc[np.asarray(obs_names, dtype=object)[perturbed], "pertA"].to_numpy(dtype=float)
    fast_scores = exact_fast.scores[perturbed, 0].astype(float)

    assert exact_fast.metadata["quantile_clip"] is True
    assert exact_fast.metadata["clip_method"] == "streaming_histogram"
    assert exact_fast.metadata["clip_bins"] == 200000
    assert np.allclose(fast_scores, exact_scores, atol=1e-3)


def test_exact_fast_multilabel_matches_single_label_when_one_guide_per_cell() -> None:
    counts = np.asarray(
        [
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [1.0, 6.0, 0.0],
            [0.0, 1.0, 5.0],
            [0.0, 1.0, 6.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["control", "control", "pertA", "pertA", "pertB", "pertB"], dtype=object)
    guide_matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=np.float64,
        )
    )
    obs_names = [f"cell_{index}" for index in range(counts.shape[0])]
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels}, index=obs_names),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    target_genes = {"pertA": ["g1", "g2", "g3"], "pertB": ["g1", "g2", "g3"]}

    single = run_ps_score_exact_fast_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        perturbations=["pertA", "pertB"],
        target_genes=target_genes,
        target_gene_max=3,
        chunk_size=2,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        target_sum=1e4,
        scale_score=False,
    )
    multi = run_ps_score_exact_fast_multilabel_anndata(
        adata,
        guide_matrix=guide_matrix,
        perturbation_names=["pertA", "pertB"],
        target_genes=target_genes,
        target_gene_max=3,
        chunk_size=2,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        target_sum=1e4,
        scale_score=False,
    )

    long_scores = {
        (int(cell), multi.perturbations[int(perturbation)]): float(score)
        for cell, perturbation, score in zip(multi.cell_indices, multi.perturbation_indices, multi.scores, strict=False)
    }
    for cell_index, label in enumerate(labels):
        if label == "control":
            continue
        assert np.isclose(long_scores[(cell_index, str(label))], float(single.scores[cell_index, 0]))


def test_exact_fast_multilabel_two_guides_matches_masked_lbfgsb() -> None:
    counts = np.asarray(
        [
            [8.0, 1.0, 1.0],
            [7.0, 1.0, 1.0],
            [1.0, 8.0, 1.0],
            [1.0, 7.0, 1.0],
            [1.0, 1.0, 8.0],
            [1.0, 1.0, 7.0],
            [1.0, 7.0, 7.0],
        ],
        dtype=np.float64,
    )
    guide_matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 1],
            ],
            dtype=np.float64,
        )
    )
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"cell_{index}" for index in range(counts.shape[0])]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    result = run_ps_score_exact_fast_multilabel_anndata(
        adata,
        guide_matrix=guide_matrix,
        perturbation_names=["pertA", "pertB"],
        target_genes={"pertA": ["g1", "g2", "g3"], "pertB": ["g1", "g2", "g3"]},
        target_gene_max=3,
        chunk_size=3,
        lr_lambda=0.01,
        score_lambda=0.2,
        scale_factor=10.0,
        target_sum=1e4,
        scale_score=False,
    )

    multi_cell = counts.shape[0] - 1
    observed = {
        result.perturbations[int(perturbation)]: float(score) * 10.0
        for cell, perturbation, score in zip(result.cell_indices, result.perturbation_indices, result.scores, strict=False)
        if int(cell) == multi_cell
    }
    library_size = counts[multi_cell].sum()
    y = np.log1p((counts[multi_cell] / library_size) * 1e4)[result.union_gene_indices]
    centered = y - result.beta[0]
    active_beta = result.beta[[1, 2]]

    def objective(value: np.ndarray) -> float:
        residual = value @ active_beta - centered
        return float(0.5 * residual @ residual + 0.2 * np.sum(value))

    def gradient(value: np.ndarray) -> np.ndarray:
        residual = value @ active_beta - centered
        return residual @ active_beta.T + 0.2

    expected = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        jac=gradient,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        method="L-BFGS-B",
    ).x

    assert set(observed) == {"pertA", "pertB"}
    assert np.allclose([observed["pertA"], observed["pertB"]], expected, atol=1e-6)
    assert result.metadata["score_output_format"] == "long"
    assert result.metadata["guide_multiplicity"]["multi_count"] == 1
