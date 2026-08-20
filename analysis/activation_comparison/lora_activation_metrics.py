"""Metrics for comparing LoRA adapter structure across methods."""
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats


def svd_spectrum(delta_w: torch.Tensor) -> np.ndarray:
    """Compute singular values of ΔW."""
    U, S, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)
    return S.cpu().numpy()


def svd_spectrum_per_layer(delta_w_dict: OrderedDict) -> Dict[str, np.ndarray]:
    """Compute SVD spectrum for each LoRA layer."""
    return {base: svd_spectrum(dw) for base, dw in delta_w_dict.items()}


def aggregated_svd_spectrum(
    spectra: Dict[str, np.ndarray], max_components: int = 64
) -> np.ndarray:
    """Mean SVD spectrum across layers, truncated to max_components."""
    truncated = []
    for s in spectra.values():
        n = min(len(s), max_components)
        padded = np.zeros(max_components)
        padded[:n] = s[:n]
        truncated.append(padded)
    return np.mean(truncated, axis=0)


def spectral_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """Cosine similarity of normalized SV vectors."""
    n = min(len(s1), len(s2))
    v1, v2 = s1[:n].copy(), s2[:n].copy()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return float(np.dot(v1 / norm1, v2 / norm2))


def grassmann_distance(
    delta_w_1: torch.Tensor, delta_w_2: torch.Tensor, k: int = 10
) -> float:
    """Chordal Grassmann distance between top-k left singular subspaces."""
    U1, _, _ = torch.linalg.svd(delta_w_1.float(), full_matrices=False)
    U2, _, _ = torch.linalg.svd(delta_w_2.float(), full_matrices=False)
    k1 = min(k, U1.shape[1])
    k2 = min(k, U2.shape[1])
    k_eff = min(k1, k2)
    if k_eff == 0:
        return 1.0
    U1k = U1[:, :k_eff]
    U2k = U2[:, :k_eff]
    _, sigma, _ = torch.linalg.svd(U1k.T @ U2k)
    sigma = torch.clamp(sigma, -1.0, 1.0)
    theta = torch.acos(sigma)
    return float(torch.norm(theta).item())


def pairwise_grassmann(
    all_delta_w: Dict[str, OrderedDict], k: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """Pairwise mean Grassmann distance across all layers.

    Args:
        all_delta_w: method_name -> OrderedDict[layer -> ΔW]
        k: subspace dimension

    Returns:
        (distance_matrix, method_names)
    """
    names = list(all_delta_w.keys())
    n = len(names)
    D = np.zeros((n, n))

    common_layers = None
    for dw_dict in all_delta_w.values():
        keys = set(dw_dict.keys())
        common_layers = keys if common_layers is None else common_layers & keys
    common_layers = sorted(common_layers) if common_layers else []

    for i in range(n):
        for j in range(i + 1, n):
            dists = []
            for layer in common_layers:
                d = grassmann_distance(
                    all_delta_w[names[i]][layer],
                    all_delta_w[names[j]][layer],
                    k=k,
                )
                dists.append(d)
            mean_d = np.mean(dists) if dists else 0.0
            D[i, j] = mean_d
            D[j, i] = mean_d

    return D, names


def frobenius_per_layer(delta_w_dict: OrderedDict) -> OrderedDict:
    """||ΔW||_F per layer."""
    result = OrderedDict()
    for base, dw in delta_w_dict.items():
        result[base] = float(torch.norm(dw.float(), p="fro").item())
    return result


def frobenius_vector(delta_w_dict: OrderedDict) -> np.ndarray:
    """Vector of ||ΔW||_F values in layer order."""
    return np.array(list(frobenius_per_layer(delta_w_dict).values()))


def capacity_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity of per-layer Frobenius norm vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))


def component_scores_from_pairs(
    pairs: OrderedDict, max_rank: Optional[int] = None
) -> np.ndarray:
    """Compute per-component importance scores: score[p] = Σ_layers(||A_row_p|| * ||B_col_p||)."""
    detected_rank = None
    for ab in pairs.values():
        r = min(ab["A"].shape[0], ab["B"].shape[1])
        if detected_rank is None:
            detected_rank = r
        else:
            detected_rank = max(detected_rank, r)

    if max_rank is not None:
        r_eff = min(max_rank, detected_rank) if detected_rank else max_rank
    else:
        r_eff = detected_rank or 0

    if r_eff == 0:
        return np.array([])

    scores = np.zeros(r_eff)
    for ab in pairs.values():
        A = ab["A"].float()
        B = ab["B"].float()
        r_layer = min(A.shape[0], B.shape[1], r_eff)
        a_norms = torch.norm(A[:r_layer, :], p=2, dim=1).cpu().numpy()
        b_norms = torch.norm(B[:, :r_layer], p=2, dim=0).cpu().numpy()
        scores[:r_layer] += a_norms * b_norms

    return scores


def component_rank_correlation(
    scores_ref: np.ndarray, scores_other: np.ndarray, top_k: Optional[int] = None
) -> float:
    """Spearman rank correlation of component importance scores vs reference."""
    n = min(len(scores_ref), len(scores_other))
    if n < 3:
        return 0.0
    s1, s2 = scores_ref[:n], scores_other[:n]
    if top_k is not None:
        top_k = min(top_k, n)
        idx = np.argsort(s1)[::-1][:top_k]
        s1, s2 = s1[idx], s2[idx]
    rho, _ = scipy_stats.spearmanr(s1, s2)
    return float(rho) if not np.isnan(rho) else 0.0


def component_retention_fraction(
    scores_ref: np.ndarray, scores_other: np.ndarray, top_k: int = 20
) -> float:
    """Fraction of reference's top-k components that appear in other's top-k."""
    n = min(len(scores_ref), len(scores_other))
    k = min(top_k, n)
    if k == 0:
        return 0.0
    ref_topk = set(np.argsort(scores_ref[:n])[::-1][:k])
    other_topk = set(np.argsort(scores_other[:n])[::-1][:k])
    return len(ref_topk & other_topk) / k


def compute_all_metrics(
    all_data: Dict[str, dict],
    reference: str = "FedIT",
    k_values: List[int] = [10, 20, 40],
) -> Dict[str, dict]:
    """Compute all metrics for all methods relative to a reference.

    Args:
        all_data: method_name -> {"pairs": ..., "delta_w": ..., "state_dict": ...}
        reference: reference method name
        k_values: subspace dimensions for Grassmann distance

    Returns:
        method_name -> {metric_name: value}
    """
    if reference not in all_data:
        raise ValueError(f"Reference method '{reference}' not found in data")

    ref = all_data[reference]
    ref_spectra = svd_spectrum_per_layer(ref["delta_w"])
    ref_agg_spectrum = aggregated_svd_spectrum(ref_spectra)
    ref_frob = frobenius_vector(ref["delta_w"])
    ref_scores = component_scores_from_pairs(ref["pairs"])

    results = {}
    for name, data in all_data.items():
        m = {}
        spectra = svd_spectrum_per_layer(data["delta_w"])
        agg = aggregated_svd_spectrum(spectra)
        m["spectral_similarity"] = spectral_similarity(ref_agg_spectrum, agg)

        for k in k_values:
            _, dist_names = pairwise_grassmann(
                {reference: ref["delta_w"], name: data["delta_w"]}, k=k
            )
            idx_ref = dist_names.index(reference)
            idx_m = dist_names.index(name)
            D, _ = pairwise_grassmann(
                {reference: ref["delta_w"], name: data["delta_w"]}, k=k
            )
            m[f"grassmann_k{k}"] = D[idx_ref, idx_m]

        frob = frobenius_vector(data["delta_w"])
        m["capacity_similarity"] = capacity_similarity(ref_frob, frob)

        scores = component_scores_from_pairs(data["pairs"])
        m["component_spearman"] = component_rank_correlation(ref_scores, scores)
        for topk in [10, 20, 40]:
            m[f"retention_top{topk}"] = component_retention_fraction(
                ref_scores, scores, top_k=topk
            )

        results[name] = m

    return results
