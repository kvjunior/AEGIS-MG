#!/usr/bin/env python3
"""
AEGIS-MG: Main Experiment Orchestration
Engineering Applications of Artificial Intelligence (EAAI)

Entry point for all experiments reported in the paper:
  --experiment standard     : Table 4 (multi-seed detection performance)
  --experiment adversarial  : Table 5 (adversarial robustness + certification)
  --experiment privacy      : Table 6 (DP analysis at varying epsilon)
  --experiment cross_chain  : Table 8 (cross-chain transfer + direct control)
  --experiment ablation     : Table 9 (component ablation + security metrics)
  --experiment zero_day     : Section 6.2 (REAL per-category holdout;
                              requires a `category` column in the label
                              files — refuses to run otherwise)
  --experiment all          : run all of the above

Usage:
  python aegis-mg.py --config config.yaml --experiment all --dataset ethereum_s
  python aegis-mg.py --config config.yaml --experiment standard --dataset bitcoin_l
  python aegis-mg.py --config config.yaml --experiment cross_chain \\
         --source ethereum_s --target bitcoin_m

Dataset ids (see data.DATASET_REGISTRY): ethereum_s, ethereum_p, bitcoin_m,
bitcoin_l, bsc.  The four cryptocurrency benchmarks load from the published
DIAM *_graph_dict.npz files (validated against DIAM Table 1); bsc is built
from raw BSCScan parquet exports.

Key behaviors (camera-ready fixes):
  * Each dataset is loaded ONCE per experiment (no duplicate loads).
  * The model's number of graph views K and the edge-attribute dimension
    are set FROM THE DATA, so a dataset without contract data runs honestly
    with K = 1 instead of a duplicated view.
  * Checkpoint directories are unique per (experiment, dataset, seed).
  * Ablation component names are taken from config `disable_module`
    fields and validated against train.ABLATION_HANDLERS — an unknown
    name aborts instead of silently evaluating the full model.

Author: [Redacted for double-blind review]
Version: 2.3.0
"""

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import yaml

# ---- Project imports ----
from data import DataConfig, prepare_chain_dataset
from models import ModelConfig, DifferentialPrivacyModule, create_model
from train import (
    TrainConfig, Trainer,
    run_multi_seed_experiment,
    run_cross_chain_transfer,
    run_ablation,
    membership_inference_attack,
    scalability_benchmark,
    save_experiment_results,
)
from analysis import analyze_ablation, analyze_zero_day

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("aegis")


# =========================================================================
# Config loading
# =========================================================================

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s.", path)
    return cfg


def make_data_config(cfg: Dict) -> DataConfig:
    d = cfg["data"]
    labels = d.get("labels", {})
    return DataConfig(
        data_root=d.get("data_root", "./data"),
        bitcoin_subdir=d.get("bitcoin_subdir", "bitcoin"),
        ethereum_subdir=d.get("ethereum_subdir", "ethereum"),
        bsc_subdir=d.get("bsc_subdir", "bsc"),
        # Label files are used only on the parquet path (bsc); the four DIAM
        # benchmarks carry their ground-truth y inside the *_graph_dict.npz.
        bitcoin_labels_file=labels.get(
            "bitcoin", "labels/bitcoin_walletexplorer.csv"),
        ethereum_labels_file=labels.get(
            "ethereum", "labels/ethereum_etherscan_phishing.csv"),
        bsc_labels_file=labels.get(
            "bsc", "labels/bsc_community_reports.csv"),
        max_node_features=d.get("max_node_features", 32),
        max_edge_features=d.get("max_edge_features", 8),
        use_provided_node_features=d.get("use_provided_node_features", False),
        graph_types=d.get("graph_types", ["tx", "contract"]),
        temporal_split=d.get("temporal_split", True),
        train_ratio=d.get("train_ratio", 0.50),
        val_ratio=d.get("val_ratio", 0.25),
        test_ratio=d.get("test_ratio", 0.25),
        # Explicit dataset ids (NOT bare families) so the five datasets are
        # never silently collapsed; resolved via data.DATASET_REGISTRY.
        datasets=d.get("datasets",
                       ["ethereum_s", "ethereum_p", "bitcoin_m",
                        "bitcoin_l", "bsc"]),
        npz_files=d.get("npz_files", {}),
        categories_files=d.get("categories_files", {}),
        validate_reference=d.get("validate_reference", True),
        batch_size=d.get("batch_size", 256),
        num_workers=d.get("num_workers", 8),
        pin_memory=d.get("pin_memory", True),
        chunk_size=d.get("chunk_size", 100000),
        random_seed=cfg["system"].get("random_seed", 42),
    )


def make_model_config(cfg: Dict, num_graph_types: int,
                      edge_attr_dim: int) -> ModelConfig:
    """Build a ModelConfig with K and edge_attr_dim taken FROM THE DATA."""
    m = cfg["model"]
    a, t, f, s = (m["architecture"], m["temporal"],
                  m["cross_graph_fusion"], m["security"])
    if f.get("num_graph_types") not in (None, num_graph_types):
        logger.warning(
            "config num_graph_types=%s but the data provides %d genuine "
            "view(s); using %d.  (The pipeline never duplicates views.)",
            f.get("num_graph_types"), num_graph_types, num_graph_types,
        )
    return ModelConfig(
        input_dim=a["input_dim"],
        hidden_dim=a["hidden_dim"],
        output_dim=a["output_dim"],
        num_layers=a["num_layers"],
        num_heads=a["num_heads"],
        dropout=a["dropout"],
        temporal_encoding_dim=t["encoding_dim"],
        num_fourier_components=t["num_fourier_components"],
        max_temporal_distance=t["max_temporal_distance"],
        edge_attr_dim=edge_attr_dim,
        num_graph_types=num_graph_types,
        cross_graph_heads=f["cross_graph_attention_heads"],
        discrepancy_temperature=f["discrepancy_temperature"],
        contrastive_loss_weight=f["contrastive_loss_weight"],
        enable_adversarial_training=s["adversarial"]["enable"],
        adv_epsilon=s["adversarial"]["epsilon"],
        adv_steps=s["adversarial"]["steps"],
        adv_step_size=s["adversarial"]["step_size"],
        adv_norm=s["adversarial"]["norm"],
        adv_loss_weight=s["adversarial"]["loss_weight"],
        enable_certified_defense=s["certified_defense"]["enable"],
        smoothing_sigma=s["certified_defense"]["smoothing_sigma"],
        selection_samples=s["certified_defense"]["selection_samples"],
        certification_samples=s["certified_defense"]["certification_samples"],
        confidence_alpha=s["certified_defense"]["confidence_alpha"],
        adaptive_variance=s["certified_defense"]["adaptive_variance"],
        adaptive_lambda=s["certified_defense"]["adaptive_lambda"],
        eval_smoothing_samples=s["certified_defense"].get(
            "eval_smoothing_samples", 50),
        enable_dp=s["differential_privacy"]["enable"],
        dp_epsilon=s["differential_privacy"]["target_epsilon"],
        dp_delta=s["differential_privacy"]["delta"],
        dp_max_grad_norm=s["differential_privacy"]["max_grad_norm"],
        dp_noise_multiplier=s["differential_privacy"]["noise_multiplier"],
        dp_rdp_alphas=s["differential_privacy"]["rdp_alphas"],
        use_spectral_norm=m.get("engineering", {}).get(
            "use_spectral_norm", True),
        gradient_checkpointing=m.get("engineering", {}).get(
            "gradient_checkpointing", False),
    )


def make_train_config(cfg: Dict, run_tag: str) -> TrainConfig:
    tr = cfg["training"]
    s = cfg["model"]["security"]
    exp = cfg["experiments"]
    ev = cfg["evaluation"]
    return TrainConfig(
        epochs=tr["epochs"],
        learning_rate=tr["optimizer"]["learning_rate"],
        weight_decay=tr["optimizer"]["weight_decay"],
        gradient_clip_norm=tr["gradient_clip_norm"],
        accumulation_steps=tr["gradient_accumulation_steps"],
        warmup_epochs=tr["scheduler"]["warmup_epochs"],
        min_lr=tr["scheduler"]["min_lr"],
        adversarial_training=s["adversarial"]["enable"],
        adv_warmup_epochs=s["adversarial"].get("warmup_epochs", 10),
        adv_loss_weight=s["adversarial"]["loss_weight"],
        disc_loss_weight=cfg["model"]["cross_graph_fusion"][
            "contrastive_loss_weight"],
        enable_dp=s["differential_privacy"]["enable"],
        dp_max_grad_norm=s["differential_privacy"]["max_grad_norm"],
        dp_noise_multiplier=s["differential_privacy"]["noise_multiplier"],
        dp_target_epsilon=s["differential_privacy"]["target_epsilon"],
        dp_delta=s["differential_privacy"]["delta"],
        patience=tr["early_stopping"]["patience"],
        min_delta=1e-4,
        monitor="val_f1",
        restore_best_weights=tr["early_stopping"].get(
            "restore_best_weights", True),
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        run_tag=run_tag,
        mixed_precision=tr["mixed_precision"]["enable"],
        seeds=exp["seeds"],
        max_certify_nodes=ev["security"]["certification"].get(
            "max_certify_nodes", 2000),
    )


# =========================================================================
# Factories
# =========================================================================

def _make_model_and_dp(model_cfg: ModelConfig):
    model = create_model(vars(model_cfg))
    dp = None
    if model_cfg.enable_dp:
        dp = DifferentialPrivacyModule(
            max_grad_norm=model_cfg.dp_max_grad_norm,
            noise_multiplier=model_cfg.dp_noise_multiplier,
            target_epsilon=model_cfg.dp_epsilon,
            delta=model_cfg.dp_delta,
            rdp_alphas=model_cfg.dp_rdp_alphas,
        )
    return model, dp


def load_chain_once(data_cfg: DataConfig, chain: str,
                    holdout_category: Optional[str] = None
                    ) -> Tuple[Any, List[Dict], Dict]:
    """Single load per chain (split + leakage-safe normalization)."""
    return prepare_chain_dataset(data_cfg, chain,
                                 holdout_category=holdout_category)


def derive_model_cfg(cfg: Dict, views: List[Dict]) -> ModelConfig:
    """ModelConfig with K and edge_attr_dim taken from prepared data."""
    edge_attr_dim = int(views[0]["edge_attr"].shape[1]) if views else 0
    return make_model_config(cfg, num_graph_types=len(views),
                             edge_attr_dim=edge_attr_dim)


# =========================================================================
# Experiment runners
# =========================================================================

def run_standard(cfg, data_cfg, chain, device, out_dir):
    """Table 4: multi-seed detection performance."""
    logger.info("=== Standard detection experiment (Table 4) ===")
    data, views, meta = load_chain_once(data_cfg, chain)
    model_cfg = derive_model_cfg(cfg, views)
    train_cfg = make_train_config(cfg, run_tag=f"standard_{chain}")

    def data_factory(seed=None):
        """Temporal split: deterministic data, reused across seeds (only
        initialization varies).  Random split: re-split per seed (DIAM
        protocol: 5 random 2:1:1 splits)."""
        if data_cfg.temporal_split or seed is None:
            return data, views, meta
        seeded_cfg = replace(data_cfg, random_seed=int(seed))
        return load_chain_once(seeded_cfg, chain)

    eval_eps = cfg["evaluation"]["security"]["certification"][
        "epsilon_values"]
    results = run_multi_seed_experiment(
        model_factory=lambda: _make_model_and_dp(model_cfg),
        data_factory=data_factory,
        config=train_cfg,
        device=device,
        eval_epsilons=eval_eps,
    )
    results["dataset_metadata"] = meta
    save_experiment_results(results, str(out_dir / f"standard_{chain}.json"))
    return results


def run_adversarial(cfg, data_cfg, chain, device, out_dir):
    """Table 5: adversarial robustness + certified accuracy."""
    logger.info("=== Adversarial robustness experiment (Table 5) ===")
    data, views, meta = load_chain_once(data_cfg, chain)
    model_cfg = derive_model_cfg(cfg, views)
    train_cfg = make_train_config(cfg, run_tag=f"adversarial_{chain}")

    model, dp = _make_model_and_dp(model_cfg)
    trainer = Trainer(model, dp, train_cfg, device)
    trainer.fit(data, views)

    epsilons = cfg["evaluation"]["security"]["certification"]["epsilon_fine"]
    pgd_steps = cfg["evaluation"]["security"]["adversarial"].get(
        "pgd_steps", 20)
    results: Dict[str, Any] = {"chain": chain, "clean_accuracy": None,
                               "fgsm": {}, "pgd": {}, "certified": {}}
    cert_out = None  # certification re-thresholded across epsilons
    for eps in epsilons:
        fgsm = trainer.evaluate_adversarial(data, views, epsilon=eps,
                                            attack="fgsm")
        pgd = trainer.evaluate_adversarial(data, views, epsilon=eps,
                                            steps=pgd_steps, attack="pgd")
        cert, cert_out = trainer.evaluate_certified(
            data, views, epsilon=eps, certify_output=cert_out)
        # Clean accuracy is attack-independent; record once.
        if results["clean_accuracy"] is None:
            results["clean_accuracy"] = pgd["clean_accuracy"]
        results["fgsm"][f"{eps:.2f}"] = fgsm
        results["pgd"][f"{eps:.2f}"] = pgd
        results["certified"][f"{eps:.2f}"] = cert

    save_experiment_results(results,
                            str(out_dir / f"adversarial_{chain}.json"))
    return results


def run_privacy(cfg, data_cfg, chain, device, out_dir):
    """Table 6: DP analysis at varying epsilon."""
    logger.info("=== Differential privacy experiment (Table 6) ===")
    data, views, meta = load_chain_once(data_cfg, chain)
    base_model_cfg = derive_model_cfg(cfg, views)

    dp_epsilons = cfg["evaluation"]["security"]["privacy"]["epsilon_values"]
    results: Dict[str, Any] = {"chain": chain, "privacy_sweep": {}}

    for target_eps in dp_epsilons:
        logger.info("Training with target epsilon = %.1f", target_eps)
        mc = ModelConfig(**{**vars(base_model_cfg),
                            "dp_epsilon": float(target_eps)})
        tc = make_train_config(
            cfg, run_tag=f"privacy_{chain}_eps{target_eps}")
        tc.dp_target_epsilon = float(target_eps)

        model, dp = _make_model_and_dp(mc)
        trainer = Trainer(model, dp, tc, device)
        trainer.fit(data, views)
        test = trainer.evaluate_test(data, views)
        mia = membership_inference_attack(trainer, data, views)
        results["privacy_sweep"][f"eps_{target_eps}"] = {
            "test": {k: v for k, v in test.items()
                     if k not in ("y_true", "y_pred", "y_prob")},
            "dp_report": dp.get_privacy_report() if dp else {},
            "mia": mia,
        }

    # Non-private control (epsilon = infinity)
    mc_nodp = ModelConfig(**{**vars(base_model_cfg), "enable_dp": False})
    tc_nodp = make_train_config(cfg, run_tag=f"privacy_{chain}_nodp")
    tc_nodp.enable_dp = False
    model, _ = _make_model_and_dp(mc_nodp)
    trainer = Trainer(model, None, tc_nodp, device)
    trainer.fit(data, views)
    test_nodp = trainer.evaluate_test(data, views)
    mia_nodp = membership_inference_attack(trainer, data, views)
    results["privacy_sweep"]["eps_inf"] = {
        "test": {k: v for k, v in test_nodp.items()
                 if k not in ("y_true", "y_pred", "y_prob")},
        "dp_report": {},
        "mia": mia_nodp,
    }

    save_experiment_results(results, str(out_dir / f"privacy_{chain}.json"))
    return results


def run_cross_chain_exp(cfg, data_cfg, source, target, device, out_dir):
    """Table 8: cross-chain transfer with direct-training control."""
    logger.info("=== Cross-chain %s -> %s (Table 8) ===", source, target)
    src = load_chain_once(data_cfg, source)
    tgt = load_chain_once(data_cfg, target)
    # K must match for weight transfer; derive from source.
    model_cfg = derive_model_cfg(cfg, src[1])
    train_cfg = make_train_config(
        cfg, run_tag=f"xchain_{source}_to_{target}")

    ev = cfg["evaluation"]["cross_chain"]
    results = run_cross_chain_transfer(
        model_factory=lambda: _make_model_and_dp(model_cfg),
        source_data_factory=lambda: src,
        target_data_factory=lambda: tgt,
        config=train_cfg,
        device=device,
        fine_tune_lr=ev["fine_tune_lr"],
        fine_tune_epochs=ev["fine_tune_epochs"],
    )
    save_experiment_results(
        results, str(out_dir / f"cross_chain_{source}_to_{target}.json"))
    return results


def run_ablation_exp(cfg, data_cfg, chain, device, out_dir):
    """Table 9: component ablation with security metrics."""
    logger.info("=== Ablation study (Table 9) ===")
    data, views, meta = load_chain_once(data_cfg, chain)
    model_cfg = derive_model_cfg(cfg, views)
    train_cfg = make_train_config(cfg, run_tag=f"ablation_{chain}")

    # Pass name + disable_module dicts; train.run_ablation validates the
    # disable_module against its handler set and ABORTS on unknown names.
    components = cfg["experiments"]["ablation"]["components"]

    results = run_ablation(
        model_factory=lambda: _make_model_and_dp(model_cfg),
        data_factory=lambda: (data, views, meta),
        config=train_cfg,
        device=device,
        components=components,
    )
    abl_analysis = analyze_ablation(results)
    save_experiment_results(abl_analysis,
                            str(out_dir / f"ablation_{chain}.json"))
    return abl_analysis


def run_scalability_exp(cfg, data_cfg, chain, device, out_dir):
    """Table 7: throughput / latency / memory across graph sizes.

    Sizes are SUBSETS of the real chain graph (never synthetic padding);
    sizes exceeding the available node count are skipped with a warning.
    Trains once, then benchmarks inference on induced subgraphs.
    """
    logger.info("=== Scalability benchmark (Table 7) ===")
    data, views, meta = load_chain_once(data_cfg, chain)
    model_cfg = derive_model_cfg(cfg, views)
    train_cfg = make_train_config(cfg, run_tag=f"scalability_{chain}")

    model, dp = _make_model_and_dp(model_cfg)
    trainer = Trainer(model, dp, train_cfg, device)
    trainer.fit(data, views)

    sizes = cfg["experiments"]["scalability"]["graph_sizes"]
    rows = scalability_benchmark(model, data, views, device,
                                 node_subset_sizes=sizes)
    results = {
        "chain": chain,
        "available_nodes": int(data.x.size(0)),
        "rows": rows,
        "note": ("Throughput = nodes / mean inference latency on the "
                 "induced subgraph (batched full-graph forward).  State "
                 "this methodology in the caption so the headline TPS and "
                 "this table use ONE definition (Reviewer 1)."),
    }
    save_experiment_results(results,
                            str(out_dir / f"scalability_{chain}.json"))
    return results


def run_zero_day_exp(cfg, data_cfg, chain, device, out_dir):
    """Section 6.2: REAL zero-day holdout.

    For each fraud category C: reload the chain with
    holdout_category=C (data.py removes C's illicit nodes from the
    train/val labels and builds a test set of those nodes plus ordinary
    legitimate test nodes), train from scratch, and evaluate.  If the
    label files have no `category` column, prepare_chain_dataset raises
    — this experiment NEVER emits placeholder numbers.
    """
    logger.info("=== Zero-day holdout experiment ===")
    holdout_cats = cfg["experiments"]["zero_day"]["holdout_categories"]
    holdout_results: Dict[str, Dict[str, float]] = {}

    for cat in holdout_cats:
        logger.info("Holding out category: %s", cat)
        data, views, meta = load_chain_once(data_cfg, chain,
                                            holdout_category=cat)
        model_cfg = derive_model_cfg(cfg, views)
        train_cfg = make_train_config(
            cfg, run_tag=f"zeroday_{chain}_{cat}")

        model, dp = _make_model_and_dp(model_cfg)
        trainer = Trainer(model, dp, train_cfg, device)
        trainer.fit(data, views)
        test = trainer.evaluate_test(data, views)
        holdout_results[cat] = {
            "f1": test["f1"],
            "precision": test["precision"],
            "recall": test["recall"],
            "n_holdout_test_nodes": meta["split"]["test_nodes"],
        }

    zd_analysis = analyze_zero_day(holdout_results)
    save_experiment_results(zd_analysis,
                            str(out_dir / f"zero_day_{chain}.json"))
    return zd_analysis


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="AEGIS-MG Experiment Runner")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--experiment", type=str, default="standard",
                        choices=["all", "standard", "adversarial", "privacy",
                                 "cross_chain", "ablation", "scalability",
                                 "zero_day"])
    parser.add_argument("--dataset", "--chain", dest="chain", type=str,
                        default="ethereum_s",
                        help="dataset id (see data.DATASET_REGISTRY)")
    parser.add_argument("--source", type=str, default="ethereum_s",
                        help="cross-chain source dataset id")
    parser.add_argument("--target", type=str, default="bitcoin_m",
                        help="cross-chain target dataset id")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = make_data_config(cfg)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.output_dir or cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["system"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg["system"].get("deterministic", False) and torch.cuda.is_available():
        # Reproducibility contract from config.yaml.  Full determinism on
        # GPU also requires CUBLAS_WORKSPACE_CONFIG=:4096:8 in the
        # environment; cudnn.benchmark trades a little speed for it.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time.time()

    if args.experiment in ("all", "standard"):
        run_standard(cfg, data_cfg, args.chain, device, out_dir)
    if args.experiment in ("all", "adversarial"):
        run_adversarial(cfg, data_cfg, args.chain, device, out_dir)
    if args.experiment in ("all", "privacy"):
        run_privacy(cfg, data_cfg, args.chain, device, out_dir)
    if args.experiment == "cross_chain":
        run_cross_chain_exp(cfg, data_cfg, args.source, args.target,
                            device, out_dir)
    elif args.experiment == "all":
        for pair in cfg["evaluation"]["cross_chain"]["transfer_pairs"]:
            run_cross_chain_exp(cfg, data_cfg, pair["source"],
                                pair["target"], device, out_dir)
    if args.experiment in ("all", "ablation"):
        run_ablation_exp(cfg, data_cfg, args.chain, device, out_dir)
    if args.experiment in ("all", "scalability"):
        run_scalability_exp(cfg, data_cfg, args.chain, device, out_dir)
    if args.experiment in ("all", "zero_day"):
        run_zero_day_exp(cfg, data_cfg, args.chain, device, out_dir)

    logger.info("Total time: %.1f minutes.", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
