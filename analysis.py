"""
AEGIS-MG: Experiment Analysis and Results Reporting
Engineering Applications of Artificial Intelligence (EAAI)

Implements:
  - Wilcoxon signed-rank test over n = num_seeds x num_datasets paired
    observations -- 5 seeds x 5 datasets = 25 (n >= 10 already suffices for
    p < 0.01; the optional "Combined" union as a 6th setting restores 30).
    This fixes Reviewer 1's point: with n = 6 pairs the minimum two-sided p
    is ~0.03125, so the original "p < 0.01 with n = 6" claim was impossible.
  - Effect size r = |Z| / sqrt(N) with N = number of pairs.
  - Certification analysis: certified accuracy, abstention rate, radius
    distribution from Cohen CERTIFY output (Reviewers 1 and 4).
  - False-alert rate with the static-dataset rate and the projected
    operational rate clearly SEPARATED (Reviewer 4's conflation fix).
  - Corrected baseline provenance notes.
  - Zero-day holdout analysis.

Author: [Redacted for double-blind review]
Version: 2.3.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from scipy import stats as sp_stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score,
)

logger = logging.getLogger("aegis.analysis")


# =========================================================================
# Configuration
# =========================================================================

class AnalysisConfig:
    def __init__(self,
                 results_dir: str = "./results",
                 significance_level: float = 0.01,
                 n_bootstrap: int = 10000,
                 dataset_names: Optional[List[str]] = None,
                 num_seeds: int = 5):
        self.results_dir = results_dir
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        # The five evaluated datasets (data.DATASET_REGISTRY order). "Combined"
        # (the union of all five) is an OPTIONAL 6th setting -- add it here only
        # if it is actually built in data.py.
        self.dataset_names = dataset_names or [
            "Ethereum-S", "Ethereum-P", "Bitcoin-M", "Bitcoin-L", "BSC-Cross",
        ]
        self.num_seeds = num_seeds


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================================
# 1. Statistical Testing (Reviewer 1 fix)
# =========================================================================

def wilcoxon_multi_seed(
    aegis_scores: np.ndarray,
    baseline_scores: np.ndarray,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test over paired (seed, dataset) observations.

    Parameters
    ----------
    aegis_scores, baseline_scores : arrays of shape
        (num_seeds, num_datasets); flattened to paired observations.

    Notes
    -----
    - Requires n >= 10 pairs (5 seeds x >= 2 datasets); the protocol here is
      5 seeds x 5 datasets = 25 pairs (a 6th "Combined" setting -> 30).
    - scipy raises if ALL differences are zero; that degenerate case is
      handled explicitly (no detectable difference -> p = 1).
    - Z is recovered from the normal approximation of W so the effect
      size r = |Z| / sqrt(N) is well-defined and reproducible.
    """
    a_arr = np.asarray(aegis_scores, dtype=np.float64)
    b_arr = np.asarray(baseline_scores, dtype=np.float64)
    if a_arr.ndim > 1:
        shape_note = (f"{a_arr.shape[0]} seeds x {a_arr.shape[-1]} "
                      f"datasets")
    else:
        shape_note = "pre-flattened input"
    a = a_arr.flatten()
    b = b_arr.flatten()
    n = len(a)
    assert n == len(b), f"Unequal lengths: {n} vs {len(b)}"
    assert n >= 10, (
        f"n={n} pairs is too small for the reported significance level. "
        f"Run >= 5 seeds x >= 2 datasets (the paper uses 5 x 5 = 25)."
    )

    diffs = a - b
    if np.allclose(diffs, 0.0):
        return {
            "test": "Wilcoxon signed-rank (two-sided)",
            "n_pairs": n, "W_statistic": float("nan"),
            "Z_approx": 0.0, "p_value": 1.0, "effect_size_r": 0.0,
            "effect_size_interpretation": "none",
            "significant": False, "alpha": alpha,
            "note": "All paired differences are zero.",
        }

    stat, p_value = sp_stats.wilcoxon(a, b, alternative="two-sided")

    # Normal approximation: E[W] = n(n+1)/4, Var[W] = n(n+1)(2n+1)/24
    # (n here = number of NON-ZERO differences, per Wilcoxon convention;
    # no tie correction is applied to the variance — with many tied
    # absolute differences the |Z| and hence r are slightly
    # conservative).  The p-value itself comes from scipy, which uses
    # the exact distribution where applicable.
    nz = int(np.sum(diffs != 0))
    expected = nz * (nz + 1) / 4
    var = nz * (nz + 1) * (2 * nz + 1) / 24
    z_score = (stat - expected) / np.sqrt(max(var, 1e-12))
    effect_r = abs(z_score) / np.sqrt(n)

    return {
        "test": "Wilcoxon signed-rank (two-sided)",
        "n_pairs": n,
        "n_nonzero_pairs": nz,
        "W_statistic": float(stat),
        "Z_approx": float(z_score),
        "p_value": float(p_value),
        "effect_size_r": float(effect_r),
        "effect_size_interpretation": _interpret_r(effect_r),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "note": (
            f"n={n} paired observations ({shape_note}).  "
            f"Effect size r = |Z| / sqrt(N), N = {n}."
        ),
    }


def _interpret_r(r: float) -> str:
    """Cohen (1988) conventions for r."""
    if r >= 0.5:
        return "large"
    if r >= 0.3:
        return "medium"
    if r >= 0.1:
        return "small"
    return "none"


def run_all_baseline_comparisons(
    aegis_scores: np.ndarray,
    baseline_dict: Dict[str, np.ndarray],
    alpha: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    """Wilcoxon test for AEGIS-MG vs each baseline (same shapes)."""
    return {name: wilcoxon_multi_seed(aegis_scores, scores, alpha)
            for name, scores in baseline_dict.items()}


# =========================================================================
# 2. Certification Analysis (Cohen CERTIFY output)
# =========================================================================

def analyze_certification(
    certify_output: Dict[str, Any],
    labels: np.ndarray,
    epsilons: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Analyze CertifiedDefenseLayer.certify() output.

    `certify_output` carries its own `node_idx` (the certified subset),
    so labels are indexed with it directly — no separate test mask needed.

    Reports certified accuracy per epsilon, abstention rate, and the
    radius distribution (what Reviewers 1 and 4 ask for).
    """
    if epsilons is None:
        epsilons = [0.05, 0.10, 0.20, 0.30, 0.50]

    node_idx = _to_numpy(certify_output["node_idx"]).astype(int)
    pred_class = _to_numpy(certify_output["predicted_class"])
    radius = _to_numpy(certify_output["certified_radius"])
    certified_mask = _to_numpy(certify_output["certified_mask"]).astype(bool)
    true_labels = np.asarray(labels)[node_idx]
    n = len(true_labels)

    correct = pred_class == true_labels
    abstention_rate = float((~certified_mask).sum()) / max(n, 1)

    cert_acc = {}
    for eps in epsilons:
        ok = correct & certified_mask & (radius >= eps)
        cert_acc[f"eps_{eps:.2f}"] = {
            "certified_accuracy": float(ok.sum()) / max(n, 1),
            "nodes_certified_above_eps": int(
                (certified_mask & (radius >= eps)).sum()),
        }

    cert_radii = radius[certified_mask]
    radius_stats = {}
    if len(cert_radii) > 0:
        radius_stats = {
            "mean": float(cert_radii.mean()),
            "median": float(np.median(cert_radii)),
            "std": float(cert_radii.std()),
            "min": float(cert_radii.min()),
            "max": float(cert_radii.max()),
            "percentile_25": float(np.percentile(cert_radii, 25)),
            "percentile_75": float(np.percentile(cert_radii, 75)),
            "histogram_bins": np.histogram(
                cert_radii, bins=20, range=(0, 1.0))[0].tolist(),
        }

    return {
        "n_nodes_certification_run_on": n,
        "n_certified": int(certified_mask.sum()),
        "n_abstained": int((~certified_mask).sum()),
        "abstention_rate": abstention_rate,
        "certified_accuracy_by_epsilon": cert_acc,
        "radius_distribution": radius_stats,
        "note": (
            "Certified accuracy = fraction of evaluated nodes that are "
            "correctly classified AND certified with radius >= eps "
            "(abstentions count against accuracy, per Cohen et al. 2019)."
        ),
    }


# =========================================================================
# 3. False-Alert Rate (Reviewer 4 correction)
# =========================================================================

def compute_false_alert_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    operational_tps: float,
    legitimate_fraction: float,
) -> Dict[str, Any]:
    """Static test-set rate and projected operational rate, SEPARATED.

    Reviewer 4: the original "~310 false alerts/hour" divided 552,000
    static test-set false positives by the 24-month window (~17,520 h),
    then attributed the figure to live throughput of 17,847 TPS — a
    different denominator.  At full throughput with FPR f on legitimate
    traffic, the live burden is
        TPS x legitimate_fraction x f x 3600 per hour,
    which can exceed a million alerts/hour.  Both quantities are
    reported, clearly labeled, so the paper cannot conflate them again.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_legitimate = int((np.asarray(y_true) == 0).sum())
    n_illicit = int((np.asarray(y_true) == 1).sum())

    fpr = fp / max(n_legitimate, 1)
    fnr = fn / max(n_illicit, 1)

    legitimate_tps = operational_tps * legitimate_fraction
    false_alerts_per_second = legitimate_tps * fpr
    false_alerts_per_hour = false_alerts_per_second * 3600

    return {
        "static": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "fpr": float(fpr),
            "fnr": float(fnr),
        },
        "operational": {
            "throughput_tps": float(operational_tps),
            "legitimate_fraction": float(legitimate_fraction),
            "legitimate_tps": float(legitimate_tps),
            "fpr": float(fpr),
            "false_alerts_per_second": float(false_alerts_per_second),
            "false_alerts_per_hour": float(false_alerts_per_hour),
            "false_alerts_per_day": float(false_alerts_per_hour * 24),
            "note": (
                "Projected at full throughput; NOT the static test-set "
                "figure.  A tiered review system is essential: "
                "high-confidence (p > 0.95) -> automatic action; "
                "moderate (0.70 < p < 0.95) -> human review."
            ),
        },
    }


# =========================================================================
# 4. Per-Class and Confusion Matrix Analysis
# =========================================================================

def compute_full_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """All test metrics for a single (seed, dataset) evaluation.

    MCC and Cohen's kappa via sklearn (the original code used an
    incorrect closed-form approximation).
    """
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    two_class = len(np.unique(y_true)) > 1

    return {
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="binary",
                                     zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary",
                               zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob) if two_class else 0.0,
        "auc_pr": (average_precision_score(y_true, y_prob)
                   if two_class else 0.0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
    }


# =========================================================================
# 5. Multi-Seed Aggregation
# =========================================================================

def aggregate_multi_seed_results(
    per_seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mean ± std for each metric across seeds, plus the raw arrays."""
    metric_keys = ["f1", "precision", "recall", "auc_roc", "auc_pr",
                   "mcc", "cohen_kappa", "fpr"]
    arrays = {k: [] for k in metric_keys}
    for result in per_seed_results:
        test = result["test"]
        for k in metric_keys:
            arrays[k].append(test.get(k, 0.0))

    agg = {}
    for k in metric_keys:
        arr = np.array(arrays[k], dtype=np.float64)
        agg[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                  "values": arr.tolist()}
    agg["num_seeds"] = len(per_seed_results)
    return agg


# =========================================================================
# 6. Cross-Chain Transfer Analysis
# =========================================================================

def analyze_cross_chain(transfer_results: Dict[str, Any]) -> Dict[str, Any]:
    """Repackage cross-chain results with the Reviewer-1 control noted."""
    return {
        "transfer_f1": transfer_results["transfer_f1"],
        "zero_shot_f1": transfer_results["zero_shot_f1"],
        "direct_f1": transfer_results["direct_f1"],
        "abs_degradation": transfer_results["abs_degradation"],
        "rel_degradation_pct": transfer_results["rel_degradation_pct"],
        "note": (
            "Degradation = F1_direct - F1_transfer; direct-training F1 is "
            "the in-domain reference Reviewer 1 requested.  ShadowEyes "
            "(Che et al. 2025) reports FINE-TUNED transfer numbers; "
            "zero-shot is reported separately here."
        ),
    }


# =========================================================================
# 7. Ablation Analysis with Security Metrics (Reviewer 4)
# =========================================================================

def analyze_ablation(
    ablation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute deltas and assemble the security-aware ablation table."""
    full = next(r for r in ablation_results if r["name"] == "full")
    rows = []
    for r in ablation_results:
        rows.append({
            "name": r["name"],
            "test_f1": r["test_f1"],
            "delta_f1": r.get("delta_f1",
                              r["test_f1"] - full["test_f1"]),
            "test_auc_roc": r.get("test_auc_roc"),
            "certified_accuracy_0.20": r.get("certified_accuracy_0.20"),
            "pgd_accuracy_0.20": r.get("pgd_accuracy_0.20"),
            "abstention_rate": r.get("abstention_rate"),
            "mean_certified_radius": r.get("mean_certified_radius"),
            "inference_latency_ms": r.get("inference_latency_ms"),
        })
    return {"components": rows}


# =========================================================================
# 8. Zero-Day Holdout Analysis (Reviewer 4)
# =========================================================================

def analyze_zero_day(
    holdout_results: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Per-category zero-day holdout summary.

    `holdout_results`: dict[category] -> metrics from a model trained
    WITHOUT that category's illicit labels and tested on them
    (data.prepare_chain_dataset(holdout_category=...)).
    """
    analysis: Dict[str, Any] = {
        "per_category": {}, "overall_mean_f1": 0.0,
        "worst_category": "", "worst_f1": 1.0,
    }
    f1s = []
    for category, metrics in holdout_results.items():
        f1 = metrics.get("f1", 0.0)
        f1s.append(f1)
        analysis["per_category"][category] = metrics
        if f1 < analysis["worst_f1"]:
            analysis["worst_f1"] = f1
            analysis["worst_category"] = category
    analysis["overall_mean_f1"] = float(np.mean(f1s)) if f1s else 0.0
    analysis["note"] = (
        "Zero-day protocol: for each category, every illicit node of that "
        "category is removed from train/val labels and forms the positive "
        "test class.  Worst case: "
        f"{analysis['worst_category']} at F1={analysis['worst_f1']:.4f}.  "
        "This is a significant operational limitation (paper §6.2)."
    )
    return analysis


# =========================================================================
# 9. Corrected Baseline Provenance (verified against original papers)
# =========================================================================

CORRECTED_BASELINES = {
    "GCN": {"source": "reimplemented",
            "reference": "Kipf and Welling, ICLR 2017"},
    "GAT": {"source": "reimplemented",
            "reference": "Veličković et al., ICLR 2018"},
    "GraphSAGE": {"source": "reimplemented",
                  "reference": "Hamilton et al., NeurIPS 2017"},
    "FA-GNN": {"source": "official_code",
               "reference": "Liu et al., TNSE 2022"},
    "PEAE-GNN": {
        "source": "official_code",
        "reference": "Huang et al., TCSS 2024",
        "corrected_note": (
            "Original paper's best F1 = 93.05% (EthereumD1, RTM readout) "
            "or 93.33% (EthereumD2) — NOT 92.35% as previously cited.  "
            "PEAE-GNN evaluates GRAPH-level classification on balanced "
            "(undersampled) ego-graphs; our table reports our run of "
            "their code on our node-level data, with the protocol "
            "difference stated."
        ),
    },
    "DIAM": {
        "source": "official_code",
        "reference": "Ding et al., CIKM 2024",
        "note": ("github.com/TommyDzh/DIAM; 2:1:1 split over labeled "
                 "nodes, 5 seeds, best-val-F1 model selection."),
    },
    "2DynEthNet": {
        "source": "official_code",
        "reference": "Yang et al., TIFS 2024",
        "corrected_note": (
            "Original paper's F1 ≈ 87.4% (per-task mean, Table IV) or "
            "92.39% (ablation Table V on ETGraph labels) — NOT 96.8% as "
            "previously cited."
        ),
    },
    "ShadowEyes": {
        "source": "official_code",
        "reference": "Che et al., TIFS 2025",
        "note": ("Cross-platform numbers are FINE-TUNED (not zero-shot): "
                 "BTC→ETH 90.64%, ETH→BTC 87.99%."),
    },
    "AdaLipGNN": {
        "source": "reimplemented",
        "reference": "Singh et al., TSP 2025",
        "corrected_note": (
            "Original paper reports ACCURACY on citation networks (Cora, "
            "CiteSeer, PubMed) and German Credit — never F1, never on "
            "cryptocurrency data.  Any F1 in our table is from OUR "
            "reimplementation on OUR data; its certificate is a "
            "deterministic margin/Lipschitz bound, distinct from "
            "randomized smoothing."
        ),
    },
    "SpaTeD": {
        "source": "official_code",
        "reference": "Ghosh et al., TCSS 2025",
        "category": "non_private_baseline",
        "corrected_note": (
            "SpaTeD is NOT a differential-privacy method (no (eps, delta) "
            "guarantee, no noise injection).  Re-categorized from "
            "'privacy baselines' to 'non-private baselines'."
        ),
    },
}


def get_baseline_info(name: str) -> Dict[str, str]:
    """Corrected baseline citation info (raises on unknown name)."""
    if name not in CORRECTED_BASELINES:
        raise KeyError(
            f"Unknown baseline '{name}'.  Known: "
            f"{sorted(CORRECTED_BASELINES)}")
    return CORRECTED_BASELINES[name]


# =========================================================================
# 10. Full Report
# =========================================================================

def analyze_adversarial_table(adv_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble Table 5 from run_adversarial output.

    Produces, per epsilon: clean accuracy, FGSM robust accuracy, PGD
    robust accuracy, certified accuracy, abstention rate.  Certified
    accuracy counts abstentions as failures (Cohen convention).
    """
    clean = adv_results.get("clean_accuracy")
    eps_keys = sorted(adv_results.get("pgd", {}),
                      key=lambda k: float(k))
    rows = {}
    for e in eps_keys:
        rows[e] = {
            "clean_accuracy": clean,
            "fgsm_robust_accuracy":
                adv_results["fgsm"][e].get("robust_accuracy"),
            "pgd_robust_accuracy":
                adv_results["pgd"][e].get("robust_accuracy"),
            "certified_accuracy":
                adv_results["certified"][e].get("certified_accuracy"),
            "abstention_rate":
                adv_results["certified"][e].get("abstention_rate"),
        }
    return {"clean_accuracy": clean, "by_epsilon": rows,
            "note": ("Certified accuracy counts abstentions as failures "
                     "(Cohen et al. 2019); report the abstention row "
                     "alongside it.")}


def analyze_privacy_table(privacy_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble Table 6 from run_privacy output.

    Utility loss is measured relative to the eps_inf (no-DP) F1.  Reports
    realized epsilon (not just the target) and the MIA success rate; the
    no-DP row should show the HIGHEST MIA success, quantifying the
    privacy benefit.
    """
    sweep = privacy_results["privacy_sweep"]
    ref = sweep.get("eps_inf", {}).get("test", {})
    ref_f1 = ref.get("f1", 0.0)
    rows = {}
    for key, blob in sweep.items():
        f1 = blob["test"].get("f1", 0.0)
        rows[key] = {
            "target_epsilon": (None if key == "eps_inf"
                               else float(key.split("_")[1])),
            "realized_epsilon": blob.get("dp_report", {}).get("epsilon"),
            "f1": f1,
            "auc_roc": blob["test"].get("auc_roc"),
            "utility_loss_pct": (0.0 if ref_f1 == 0
                                 else (ref_f1 - f1) / ref_f1 * 100.0),
            "mia_success_rate": blob.get("mia", {}).get("attack_success_rate"),
            "mia_advantage": blob.get("mia", {}).get("advantage"),
        }
    return {"by_epsilon": rows, "reference_f1_no_dp": ref_f1,
            "note": ("Utility loss relative to no-DP F1.  Realized epsilon "
                     "is from the accountant; calibrate noise per target "
                     "for the sweep to be meaningful (see models.py).")}


def analyze_scalability_table(scal_results: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through assembly for Table 7 with a consistency reminder."""
    return {
        "rows": scal_results.get("rows", []),
        "available_nodes": scal_results.get("available_nodes"),
        "note": ("Throughput defined as nodes / mean inference latency. "
                 "The headline TPS in the abstract MUST use this same "
                 "definition (Reviewer 1)."),
    }


def generate_full_report(
    multi_seed_results: Dict[str, Any],
    baseline_comparisons: Dict[str, Dict[str, Any]],
    certification_analysis: Dict[str, Any],
    false_alert_analysis: Dict[str, Any],
    cross_chain_analysis: Optional[Dict[str, Any]] = None,
    ablation_analysis: Optional[Dict[str, Any]] = None,
    zero_day_analysis: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble all analyses into one JSON-serializable report."""
    report = {
        "multi_seed_aggregate": multi_seed_results,
        "statistical_tests": baseline_comparisons,
        "certification": certification_analysis,
        "false_alert_rate": false_alert_analysis,
        "corrected_baselines": CORRECTED_BASELINES,
    }
    if cross_chain_analysis:
        report["cross_chain"] = cross_chain_analysis
    if ablation_analysis:
        report["ablation"] = ablation_analysis
    if zero_day_analysis:
        report["zero_day"] = zero_day_analysis

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_json_convert)
        logger.info("Report saved to %s.", path)
    return report


def _json_convert(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")
