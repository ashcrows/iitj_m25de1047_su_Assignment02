"""
Evaluation metrics for all pipeline stages.
Author: Rohit (M25DE1047)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Word Error Rate (WER)
# ─────────────────────────────────────────────

def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Compute Word Error Rate using dynamic programming.
    WER = (S + D + I) / N
        S = substitutions, D = deletions, I = insertions, N = ref word count
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    n = len(ref_words)
    m = len(hyp_words)

    # DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # deletion
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j - 1],   # substitution
                )
    if n == 0:
        return 0.0
    return dp[n][m] / n


# ─────────────────────────────────────────────
# Mel-Cepstral Distortion (MCD)
# ─────────────────────────────────────────────

def compute_mcd(mfcc_ref: np.ndarray, mfcc_syn: np.ndarray) -> float:
    """
    Mel-Cepstral Distortion between reference and synthesized MFCC sequences.
    MCD (dB) = (10 / ln(10)) * sqrt(2 * sum((c_ref - c_syn)^2))
    Uses Dynamic Time Warping alignment before computing distortion.

    Args:
        mfcc_ref: (T_ref, D) numpy array
        mfcc_syn: (T_syn, D) numpy array
    Returns:
        MCD in dB
    """
    from scipy.spatial.distance import cdist

    # DTW alignment
    dist_matrix = cdist(mfcc_ref, mfcc_syn, metric="euclidean")
    path = _dtw_path(dist_matrix)

    # Aligned frames
    aligned_ref = mfcc_ref[[p[0] for p in path]]
    aligned_syn = mfcc_syn[[p[1] for p in path]]

    # MCD formula (exclude C0)
    diff = aligned_ref[:, 1:] - aligned_syn[:, 1:]   # exclude C0
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(np.sum(diff ** 2, axis=1)))
    return float(mcd)


def _dtw_path(dist_matrix: np.ndarray) -> List[tuple]:
    """Simple DTW path finding."""
    n, m = dist_matrix.shape
    cost = np.full((n, m), np.inf)
    cost[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + dist_matrix[i, 0]
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + dist_matrix[0, j]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = dist_matrix[i, j] + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )

    # Traceback
    path = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    path.append((0, 0))
    return list(reversed(path))


# ─────────────────────────────────────────────
# Equal Error Rate (EER) for Anti-Spoofing
# ─────────────────────────────────────────────

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    EER is the point where FAR (False Accept Rate) == FRR (False Reject Rate).
    Lower EER → better system.

    Args:
        bonafide_scores: scores for genuine (real) utterances (higher = more genuine)
        spoof_scores: scores for spoofed utterances (lower = more genuine)
    Returns:
        EER as a fraction [0, 1]
    """
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    all_labels = np.concatenate([
        np.ones(len(bonafide_scores)),
        np.zeros(len(spoof_scores)),
    ])

    thresholds = np.sort(all_scores)
    far_arr, frr_arr = [], []

    for thresh in thresholds:
        decisions = (all_scores >= thresh).astype(int)
        # FAR: fraction of spoof accepted as bonafide
        spoof_mask = all_labels == 0
        far = np.mean(decisions[spoof_mask] == 1)
        # FRR: fraction of bonafide rejected
        bonafide_mask = all_labels == 1
        frr = np.mean(decisions[bonafide_mask] == 0)
        far_arr.append(far)
        frr_arr.append(frr)

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)
    # EER at crossover
    idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[idx] + frr_arr[idx]) / 2.0
    return float(eer)


# ─────────────────────────────────────────────
# LID Timestamp Precision
# ─────────────────────────────────────────────

def compute_switching_accuracy(
    predicted_switches: List[float],
    reference_switches: List[float],
    tolerance_ms: float = 200.0,
) -> float:
    """
    Compute fraction of predicted language-switch timestamps within
    ±tolerance_ms of a reference switch.

    Args:
        predicted_switches: list of predicted switch timestamps in seconds
        reference_switches: list of reference switch timestamps in seconds
        tolerance_ms: tolerance window in milliseconds
    Returns:
        accuracy as a fraction [0, 1]
    """
    tolerance_sec = tolerance_ms / 1000.0
    correct = 0
    for pred in predicted_switches:
        if any(abs(pred - ref) <= tolerance_sec for ref in reference_switches):
            correct += 1
    if not predicted_switches:
        return 0.0
    return correct / len(predicted_switches)


# ─────────────────────────────────────────────
# Confusion Matrix for Code-Switching Boundaries
# ─────────────────────────────────────────────

def compute_lid_confusion_matrix(
    predicted_labels: List[int],
    reference_labels: List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute confusion matrix for binary LID (English=0, Hindi=1).
    Returns dict with matrix and per-class F1 scores.
    """
    if label_names is None:
        label_names = ["English", "Hindi"]

    pred = np.array(predicted_labels)
    ref = np.array(reference_labels)
    classes = sorted(set(ref.tolist() + pred.tolist()))

    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for r, p in zip(ref, pred):
        cm[r][p] += 1

    f1_scores = {}
    for i, name in enumerate(label_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_scores[name] = float(f1)

    return {
        "matrix": cm.tolist(),
        "label_names": label_names,
        "f1_scores": f1_scores,
        "macro_f1": float(np.mean(list(f1_scores.values()))),
    }


# ─────────────────────────────────────────────
# Full Pipeline Evaluation Report
# ─────────────────────────────────────────────

def evaluate_pipeline(results: Dict[str, Any], output_dir: str):
    """Generate a summary evaluation report across all parts."""
    report = {}

    # Part 1 metrics
    if "part1" in results:
        lid = results["part1"].get("lid_results", {})
        report["LID_F1_English"] = lid.get("f1_en", "N/A")
        report["LID_F1_Hindi"] = lid.get("f1_hi", "N/A")
        report["LID_Macro_F1"] = lid.get("macro_f1", "N/A")
        report["WER_target"] = "<15% English, <25% Hindi"

    # Part 4 metrics
    if "part4" in results:
        report["AntiSpoof_EER"] = results["part4"].get("eer", "N/A")
        adv = results["part4"].get("adv_results", {})
        report["FGSM_min_epsilon"] = adv.get("min_epsilon", "N/A")
        report["FGSM_achieved_SNR_dB"] = adv.get("snr_db", "N/A")

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation Report:")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"Report saved to: {report_path}")
