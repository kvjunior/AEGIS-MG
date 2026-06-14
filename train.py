"""
AEGIS-MG: Training Pipeline
Engineering Applications of Artificial Intelligence (EAAI)

Implements:
  - Full-graph transductive training with node masks (train/val/test).
  - Multi-seed experiment runner (5 seeds -> valid Wilcoxon, Reviewer 1).
  - PGD adversarial training restricted to TRAINING nodes (no val/test
    label leakage into attack generation or the adversarial loss).
  - DP-SGD-style integration (approximate accounting; see models.py).
  - Certified evaluation via Cohen CERTIFY on TEST nodes only.
  - Inverse-frequency class weighting (Eq. 8 of paper).
  - Cross-chain transfer with direct-training control (Reviewer 1).
  - Ablation with security metrics and config-name mapping (Reviewer 4).

Author: [Redacted for double-blind review]
Version: 2.3.0
"""

import json
import time
import inspect
import logging
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

logger = logging.getLogger("aegis.train")


def _make_grad_scaler():
    """GradScaler across PyTorch versions.

    `torch.amp.GradScaler("cuda")` exists only from PyTorch 2.3; the
    paper's pinned environment (PyTorch 2.1) exposes
    `torch.cuda.amp.GradScaler`.  Both are supported here.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:  # older signature
            pass
    return torch.cuda.amp.GradScaler()


def _autocast(enabled: bool):
    """Autocast context across PyTorch versions (no-op when disabled)."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class TrainConfig:
    """Training configuration matching config.yaml."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    # Full-graph transductive training performs ONE optimizer step per
    # epoch; gradient accumulation is meaningless in this mode.  The
    # field is retained for config compatibility but is NOT used.
    accumulation_steps: int = 4
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Adversarial training (Algorithm 2)
    adversarial_training: bool = True
    adv_warmup_epochs: int = 10
    adv_loss_weight: float = 0.5    # lambda_adv in Eq. 7
    disc_loss_weight: float = 0.1   # lambda_disc in Eq. 7

    # Differential privacy
    enable_dp: bool = True
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.1   # fallback sigma when calibration is off
    dp_target_epsilon: float = 1.0
    dp_delta: float = 1e-5
    # Calibrate the noise multiplier to dp_target_epsilon over the full step
    # budget (Table 6), instead of a fixed sigma that trips the budget
    # mid-training.  Set False to use dp_noise_multiplier verbatim.
    dp_calibrate_noise: bool = True

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    monitor: str = "val_f1"
    restore_best_weights: bool = True

    # Checkpoints
    checkpoint_dir: str = "./checkpoints"
    run_tag: str = "default"   # unique per run to avoid collisions
    save_every: int = 5
    keep_best_k: int = 3

    # Mixed precision (CUDA only; silently disabled on CPU)
    mixed_precision: bool = True

    # Multi-seed (5 seeds x D datasets -> Wilcoxon pairs; 5 x 5 = 25, n >= 10)
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    # Certification evaluation budget
    max_certify_nodes: int = 2000  # cap; CERTIFY costs N draws per node

    # Logging
    log_every: int = 10
    tensorboard_dir: str = "./runs"


# =========================================================================
# Class weights (Eq. 8)
# =========================================================================

def compute_class_weights(labels: torch.Tensor, num_classes: int = 2
                          ) -> torch.Tensor:
    """Inverse-frequency weighting: w_c = N / (C * n_c)  (Eq. 8)."""
    counts = torch.zeros(num_classes, device=labels.device)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    return labels.size(0) / (num_classes * counts.clamp(min=1))


# =========================================================================
# Differential-privacy noise calibration (Table 6)
# =========================================================================

def calibrate_noise_multiplier(
    target_epsilon: float,
    *,
    steps: int,
    sample_rate: float,
    delta: float,
    alphas: Optional[List[float]] = None,
    sigma_bounds: Tuple[float, float] = (0.05, 256.0),
    tol: float = 1e-3,
    max_iter: int = 64,
) -> Tuple[float, float]:
    """Find the DP-SGD noise multiplier sigma that achieves ~target_epsilon.

    Fixing sigma and letting the privacy budget trip mid-training makes the
    privacy/utility curve a function of *training length* rather than the noise
    level.  Instead we solve for the sigma whose RDP accountant reports
    target_epsilon after `steps` optimizer steps at the given `sample_rate`
    and `delta` -- the standard calibration (cf. Opacus get_noise_multiplier),
    which is what makes the Table 6 epsilon sweep meaningful.

    The SAME accountant used during training (models.DifferentialPrivacyModule)
    evaluates epsilon for each candidate sigma, so the calibrated sigma and the
    epsilon the trainer later reports are mutually consistent.

    epsilon is monotonically DECREASING in sigma, so the bisection is exact to
    `tol`.  Returns (sigma, achieved_epsilon).
    """
    from models import DifferentialPrivacyModule  # single source of accounting

    if steps <= 0:
        raise ValueError("DP calibration needs steps > 0.")
    if target_epsilon <= 0:
        raise ValueError("DP calibration needs target_epsilon > 0.")
    q = min(1.0, max(float(sample_rate), 0.0))
    eff_alphas = [a for a in (alphas or [2, 5, 10, 25, 50, 100]) if a > 1]

    def epsilon_for(sigma: float) -> float:
        acct = DifferentialPrivacyModule(
            max_grad_norm=1.0, noise_multiplier=sigma,
            target_epsilon=target_epsilon, delta=delta, rdp_alphas=eff_alphas)
        for _ in range(int(steps)):
            acct._account_step(q)          # accumulates per-step RDP (no side effects)
        return float(acct.get_epsilon())

    lo, hi = sigma_bounds                  # lo: low noise/high eps; hi: high noise/low eps
    eps_lo, eps_hi = epsilon_for(lo), epsilon_for(hi)

    if eps_lo <= target_epsilon:
        logger.warning(
            "DP calibration: even sigma=%.3f gives epsilon=%.3f <= target %.3f "
            "over %d steps (q=%.3f); using sigma=%.3f (more private than the "
            "target).", lo, eps_lo, target_epsilon, steps, q, lo)
        return lo, eps_lo
    if eps_hi >= target_epsilon:
        floor = float(np.log(1.0 / delta) / (max(eff_alphas) - 1))
        logger.warning(
            "DP calibration: target epsilon=%.3f is UNREACHABLE over %d steps "
            "(q=%.3f); best is epsilon=%.3f at sigma=%.1f. With the RDP alpha "
            "grid topping out at %g, epsilon cannot fall below ~%.3f at "
            "delta=%.1e regardless of sigma -- extend rdp_alphas to higher "
            "orders to certify a smaller epsilon.",
            target_epsilon, steps, q, eps_hi, hi, max(eff_alphas), floor, delta)
        return hi, eps_hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        eps_mid = epsilon_for(mid)
        if abs(eps_mid - target_epsilon) <= tol:
            return mid, eps_mid
        if eps_mid > target_epsilon:       # too much epsilon -> add noise
            lo = mid
        else:                              # too little epsilon -> remove noise
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, epsilon_for(mid)


# =========================================================================
# Single-seed trainer
# =========================================================================

class Trainer:
    """Trains AEGIS-MG on a single full graph with node masks.

    Transductive setting: the full graph is loaded once; the train/val/
    test masks (covering labeled nodes only — see data.py) select which
    nodes contribute to the loss and to evaluation.
    """

    def __init__(self, model, dp_module, config: TrainConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.dp = dp_module
        self.cfg = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999), eps=1e-8,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(config.epochs - config.warmup_epochs, 1),
            eta_min=config.min_lr,
        )
        self.use_amp = config.mixed_precision and device.type == "cuda"
        self.scaler = _make_grad_scaler() if self.use_amp else None

        self.ckpt_dir = Path(config.checkpoint_dir) / config.run_tag
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_f1 = -1.0
        self.patience_counter = 0
        self.history: List[Dict] = []

    # ---- Main entry point ----

    def fit(
        self,
        data,                  # PyG Data with masks
        view_tensors,          # list of dicts per view (see data.py)
        dataset_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full training loop with adversarial training + DP-SGD."""
        labels = data.y.to(self.device)
        train_mask = data.train_mask.to(self.device)
        val_mask = data.val_mask.to(self.device)

        nf, ei, et, ea = self._views_to_device(data, view_tensors)
        if dataset_size is None:
            dataset_size = int(train_mask.sum().item())

        class_w = compute_class_weights(labels[train_mask]).to(self.device)
        logger.info("Class weights: %s", class_w.cpu().numpy())

        # ---- DP noise calibration (Table 6): choose sigma for the target
        # epsilon over the FULL epoch budget, rather than a fixed sigma that
        # trips the budget partway through.  Full-graph transductive training
        # performs ONE DP step per epoch with sampling rate q = train_nodes /
        # dataset_size (= 1.0 for full-batch; no subsampling amplification).
        # NOTE: if val early-stopping fires before `epochs`, the run spends
        # LESS than the target and the trainer reports the ACTUAL spent
        # epsilon; for points exactly on the target-epsilon grid, set
        # patience >= epochs during the sweep so the full budget is used.
        if (self.dp is not None and self.cfg.enable_dp
                and getattr(self.cfg, "dp_calibrate_noise", True)
                and self.cfg.dp_target_epsilon
                and self.cfg.dp_target_epsilon > 0):
            n_train = int(train_mask.sum().item())
            q = min(1.0, n_train / max(int(dataset_size), 1))
            sigma, eps_hat = calibrate_noise_multiplier(
                self.cfg.dp_target_epsilon,
                steps=self.cfg.epochs, sample_rate=q,
                delta=self.dp.delta, alphas=self.dp.alphas,
            )
            logger.info(
                "DP calibration: sigma %.4f (was %.4f) -> epsilon ~%.3f at "
                "delta=%.1e over %d steps (q=%.3f).",
                sigma, self.dp.sigma, eps_hat, self.dp.delta,
                self.cfg.epochs, q)
            self.dp.sigma = float(sigma)

        t0 = time.time()
        epoch = -1

        for epoch in range(self.cfg.epochs):
            train_loss = self._train_step(
                nf, ei, et, ea, labels, train_mask, class_w,
                epoch, dataset_size,
            )
            val_metrics = self._evaluate(nf, ei, et, ea, labels, val_mask)

            if epoch < self.cfg.warmup_epochs:
                lr_scale = (epoch + 1) / self.cfg.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.cfg.learning_rate * lr_scale
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "train_loss": train_loss, "lr": lr}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if self.dp:
                row["dp_epsilon"] = self.dp.get_epsilon()
            self.history.append(row)

            if epoch % self.cfg.log_every == 0:
                logger.info(
                    "Epoch %3d  loss=%.4f  val_f1=%.4f  val_auc=%.4f  "
                    "lr=%.2e", epoch, train_loss,
                    val_metrics["f1"], val_metrics["auc_roc"], lr,
                )

            vf1 = val_metrics["f1"]
            if vf1 > self.best_val_f1 + self.cfg.min_delta:
                self.best_val_f1 = vf1
                self.patience_counter = 0
                self._save_best()
            else:
                self.patience_counter += 1

            if self.cfg.save_every and (epoch + 1) % self.cfg.save_every == 0:
                torch.save(self.model.state_dict(),
                           self.ckpt_dir / "last.pt")

            if self.patience_counter >= self.cfg.patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

            if self.dp and self.dp.get_epsilon() >= self.cfg.dp_target_epsilon:
                logger.info("DP budget exhausted at epoch %d (eps=%.2f).",
                            epoch, self.dp.get_epsilon())
                break

        if self.cfg.restore_best_weights:
            self.load_best()

        duration = time.time() - t0
        logger.info("Training complete in %.2f hours.", duration / 3600)
        return {
            "best_val_f1": self.best_val_f1,
            "epochs_trained": epoch + 1,
            "duration_hours": duration / 3600,
            "history": self.history,
        }

    # ---- Training step (Algorithm 2) ----

    def _train_step(self, nf, ei, et, ea, labels, mask, class_w,
                    epoch, dataset_size) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        with _autocast(self.use_amp):
            outputs = self.model(nf, ei, et, ea)
            logits = outputs["logits"]

            ce_loss = F.cross_entropy(logits[mask], labels[mask],
                                      weight=class_w)
            disc_loss = outputs.get(
                "discrepancy_loss",
                torch.zeros((), device=self.device))
            total_loss = ce_loss + self.cfg.disc_loss_weight * disc_loss

            if (self.cfg.adversarial_training
                    and epoch >= self.cfg.adv_warmup_epochs):
                # LEAKAGE FIX: attack generation and adversarial loss are
                # restricted to TRAINING nodes via `loss_mask`.
                adv_loss = self.model.compute_adversarial_loss(
                    nf, ei, et, labels, loss_mask=mask,
                    edge_attrs=ea, class_weights=class_w,
                )
                total_loss = total_loss + self.cfg.adv_loss_weight * adv_loss

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.gradient_clip_norm)

        if self.dp and self.cfg.enable_dp:
            # NOTE: if AMP later skips this optimizer step (inf/NaN
            # gradients), the privacy accountant has still charged the
            # step — the reported epsilon is conservative (an upper
            # bound), never optimistic.
            self.dp.clip_and_noise_gradients(
                self.model,
                batch_size=int(mask.sum().item()),
                dataset_size=dataset_size,
            )

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return float(total_loss.detach())

    # ---- Evaluation ----

    @torch.no_grad()
    def _evaluate(self, nf, ei, et, ea, labels, mask) -> Dict[str, float]:
        self.model.eval()
        outputs = self.model(nf, ei, et, ea)
        return self._metrics_from_logits(
            outputs["logits"][mask], labels[mask])

    @staticmethod
    def _metrics_from_logits(logits, y) -> Dict[str, float]:
        if y.numel() == 0:
            logger.warning(
                "Evaluation mask selects ZERO nodes; returning zero "
                "metrics.  Check the split configuration (see data.py "
                "warnings).")
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                    "auc_roc": 0.0, "auc_pr": 0.0}
        y_true = y.cpu().numpy()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]
        two_class = len(np.unique(y_true)) > 1
        return {
            "f1": f1_score(y_true, y_pred, average="binary",
                           zero_division=0),
            "precision": precision_score(y_true, y_pred, average="binary",
                                         zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary",
                                   zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_prob) if two_class else 0.0,
            "auc_pr": (average_precision_score(y_true, y_prob)
                       if two_class else 0.0),
        }

    @torch.no_grad()
    def evaluate_test(self, data, view_tensors) -> Dict[str, Any]:
        """Full test evaluation with confusion matrix."""
        self.model.eval()
        labels = data.y.to(self.device)
        mask = data.test_mask.to(self.device)
        if int(mask.sum()) == 0:
            raise ValueError(
                "Test mask selects zero nodes — evaluation is impossible. "
                "Under the temporal split this means no labeled address "
                "first appears in the test window; check the label time "
                "distribution against the split ratios, or use the "
                "random stratified split (data.py logs the same warning "
                "at preparation time)."
            )
        nf, ei, et, ea = self._views_to_device(data, view_tensors)

        outputs = self.model(nf, ei, et, ea)
        logits = outputs["logits"][mask]
        y_true = labels[mask].cpu().numpy()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        y_pred = probs.argmax(axis=1)

        metrics = self._metrics_from_logits(logits, labels[mask])
        metrics.update({
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1]).tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": probs[:, 1],
        })
        return metrics

    # ---- Certified evaluation (Cohen CERTIFY on TEST nodes) ----

    @torch.no_grad()
    def evaluate_certified(self, data, view_tensors,
                           epsilon: float = 0.20,
                           certify_output: Optional[Dict] = None,
                           ) -> Tuple[Dict[str, Any], Optional[Dict]]:
        """Certified accuracy at `epsilon` on (a sample of) test nodes.

        CERTIFY costs (n0 + N) classifier calls per node; the evaluation
        therefore certifies up to cfg.max_certify_nodes test nodes
        (uniformly sampled with a fixed RNG) and reports that count.
        Pass a previous `certify_output` to re-threshold at a different
        epsilon without re-sampling.

        Returns (result_dict, certify_output) so callers can reuse the
        expensive certification across epsilons.
        """
        self.model.eval()
        if self.model.defense is None:
            return ({"certified_accuracy": None,
                     "abstention_rate": None,
                     "note": "certified defense disabled"}, None)

        labels = data.y.to(self.device)
        test_idx = torch.where(data.test_mask.to(self.device))[0]

        if certify_output is None:
            if len(test_idx) > self.cfg.max_certify_nodes:
                g = torch.Generator(device="cpu").manual_seed(0)
                perm = torch.randperm(len(test_idx), generator=g)
                test_idx = test_idx[
                    perm[: self.cfg.max_certify_nodes].to(test_idx.device)]
                logger.info(
                    "Certifying a fixed random sample of %d test nodes "
                    "(out of %d).", len(test_idx),
                    int(data.test_mask.sum()))
            nf, ei, et, ea = self._views_to_device(data, view_tensors)
            base_fn, fused = self.model.get_base_classifier_fn(nf, ei, et, ea)
            certify_output = self.model.defense.certify(
                fused, base_fn, node_idx=test_idx)

        node_idx = certify_output["node_idx"]
        y = labels[node_idx].cpu()
        preds = certify_output["predicted_class"].cpu()
        radius = certify_output["certified_radius"].cpu()
        certified = certify_output["certified_mask"].cpu()
        n = len(y)

        correct = preds == y
        cert_correct = correct & certified & (radius >= epsilon)

        result = {
            "epsilon": epsilon,
            "n_nodes_certification_run_on": n,
            "certified_accuracy": float(cert_correct.sum()) / max(n, 1),
            "abstention_rate": float((~certified).sum()) / max(n, 1),
            "fraction_certified_above_eps": float(
                (certified & (radius >= epsilon)).sum()) / max(n, 1),
            "mean_certified_radius": (float(radius[certified].mean())
                                      if certified.any() else 0.0),
            "median_certified_radius": (float(radius[certified].median())
                                        if certified.any() else 0.0),
        }
        logger.info(
            "Certified eval (eps=%.2f, n=%d): cert_acc=%.4f, "
            "abstention=%.2f%%, mean_R=%.4f",
            epsilon, n, result["certified_accuracy"],
            result["abstention_rate"] * 100,
            result["mean_certified_radius"],
        )
        return result, certify_output

    # ---- Adversarial evaluation (PGD attack on test nodes) ----

    def evaluate_adversarial(self, data, view_tensors,
                             epsilon: float = 0.20,
                             steps: int = 20,
                             attack: str = "pgd") -> Dict[str, float]:
        """Accuracy under an l_inf adversarial attack on node features.

        attack="pgd": multi-step projected gradient descent (Madry et al.
        2018).  attack="fgsm": single-step (Goodfellow et al. 2015) — the
        steps argument is ignored and one full-epsilon step is taken.

        The attack maximizes the loss of the DETERMINISTIC base path
        (defense noise disabled during gradient computation) to avoid
        gradient masking through smoothing noise; the final evaluation
        uses the model's standard smoothed eval forward.  An EOT attack
        (Athalye et al. 2018) would be stronger still — note this as a
        limitation if reported.  Also returns clean_accuracy (no attack)
        on the same nodes so Table 5's "Clean Accuracy" row is consistent
        with the attacked rows.
        """
        self.model.eval()
        labels = data.y.to(self.device)
        mask = data.test_mask.to(self.device)
        if int(mask.sum()) == 0:
            raise ValueError("Test mask selects zero nodes (see "
                             "evaluate_test for the cause).")
        nf, ei, et, ea = self._views_to_device(data, view_tensors)
        is_fgsm = attack.lower() == "fgsm"
        if is_fgsm:
            steps = 1

        # Clean accuracy on the same nodes (standard smoothed forward).
        with torch.no_grad():
            clean_preds = self.model(nf, ei, et, ea)["logits"][mask].argmax(-1)
            clean_acc = (clean_preds == labels[mask]).float().mean().item()

        # Deterministic surrogate during the attack: zero smoothing noise
        # and a single draw, so gradients flow through h(z) exactly (no
        # gradient masking through stochasticity).  Restored afterwards;
        # the FINAL accuracy is measured with the standard smoothed
        # forward.
        defense = self.model.defense
        saved_samples = saved_sigma = None
        if defense is not None:
            saved_samples = defense.eval_samples
            saved_sigma = defense.sigma_0
            defense.eval_samples = 1
            defense.sigma_0 = 0.0

        perturbed = [f.clone().detach() for f in nf]
        # FGSM takes one full-epsilon step; PGD uses 2*eps/steps (Madry).
        step_size = epsilon if is_fgsm else epsilon / max(steps, 1) * 2

        for _ in range(steps):
            for p in perturbed:
                p.requires_grad_(True)
            out = self.model(perturbed, ei, et, ea)
            loss = F.cross_entropy(out["logits"][mask], labels[mask])
            grads = torch.autograd.grad(loss, perturbed)
            with torch.no_grad():
                for i, (p, g) in enumerate(zip(perturbed, grads)):
                    p += step_size * g.sign()
                    delta = (p - nf[i]).clamp_(-epsilon, epsilon)
                    p.copy_(nf[i] + delta)
            for p in perturbed:
                p.requires_grad_(False)

        if defense is not None:
            defense.eval_samples = saved_samples
            defense.sigma_0 = saved_sigma

        with torch.no_grad():
            out = self.model(perturbed, ei, et, ea)
            preds = out["logits"][mask].argmax(dim=-1)
            acc = (preds == labels[mask]).float().mean().item()

        return {
            "attack": "fgsm" if is_fgsm else "pgd",
            "epsilon": epsilon,
            "steps": steps,
            "clean_accuracy": clean_acc,
            "robust_accuracy": acc,
            # Backward-compatible alias (older callers read pgd_accuracy).
            "pgd_accuracy": acc,
        }

    # ---- Helpers ----

    def _views_to_device(self, data, view_tensors):
        K = len(view_tensors)
        nf = [data.x.to(self.device) for _ in range(K)]
        ei = [v["edge_index"].to(self.device) for v in view_tensors]
        et = [v["edge_timestamp"].to(self.device) for v in view_tensors]
        ea = [v["edge_attr"].to(self.device) for v in view_tensors]
        return nf, ei, et, ea

    def _save_best(self):
        torch.save(self.model.state_dict(),
                   self.ckpt_dir / "best_model.pt")

    def load_best(self):
        path = self.ckpt_dir / "best_model.pt"
        if path.exists():
            self.model.load_state_dict(
                torch.load(path, map_location=self.device,
                           weights_only=True))
            logger.info("Loaded best model from %s.", path)


# =========================================================================
# Multi-seed experiment runner
# =========================================================================

def run_multi_seed_experiment(
    model_factory: Callable[[], Tuple[Any, Any]],
    data_factory: Callable[..., Tuple[Any, List[Dict], Dict]],
    config: TrainConfig,
    device: torch.device,
    eval_epsilons: Optional[List[float]] = None,
    run_security_eval: bool = True,
) -> Dict[str, Any]:
    """Run the full experiment across multiple seeds.

    Per-seed results are collected for downstream statistical testing
    (Wilcoxon over n = num_seeds x num_datasets pairs).

    Seed protocol: each seed controls model initialization and every
    stochastic training component.  If `data_factory` accepts a `seed`
    keyword argument it is called as data_factory(seed=seed), so a
    RANDOM-split pipeline can re-split per seed (DIAM protocol: 5 random
    2:1:1 splits).  Under the TEMPORAL split the data is deterministic
    and a zero-argument factory that returns the same prepared dataset
    is correct — only initialization varies across seeds, and the paper
    must state which protocol was used.
    """
    if eval_epsilons is None:
        eval_epsilons = [0.05, 0.10, 0.20, 0.30, 0.50]

    factory_takes_seed = "seed" in inspect.signature(data_factory).parameters

    all_results = []

    for seed in config.seeds:
        logger.info("===== Seed %d =====", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model, dp_module = model_factory()
        if factory_takes_seed:
            data, views, meta = data_factory(seed=seed)
        else:
            data, views, meta = data_factory()

        seed_cfg = TrainConfig(**{**asdict(config),
                                  "run_tag": f"{config.run_tag}_seed{seed}"})
        trainer = Trainer(model, dp_module, seed_cfg, device)
        train_summary = trainer.fit(data, views)

        test_metrics = trainer.evaluate_test(data, views)

        security = {}
        if run_security_eval:
            cert_out = None
            for eps in eval_epsilons:
                adv = trainer.evaluate_adversarial(data, views, epsilon=eps)
                cert, cert_out = trainer.evaluate_certified(
                    data, views, epsilon=eps, certify_output=cert_out)
                security[f"eps_{eps:.2f}"] = {**adv, **cert}

        dp_report = dp_module.get_privacy_report() if dp_module else {}

        all_results.append({
            "seed": seed,
            "test": {k: v for k, v in test_metrics.items()
                     if k not in ("y_true", "y_pred", "y_prob")},
            "security": security,
            "dp": dp_report,
            "train_summary": {
                "best_val_f1": train_summary["best_val_f1"],
                "epochs": train_summary["epochs_trained"],
                "hours": train_summary["duration_hours"],
            },
        })

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    f1s = [r["test"]["f1"] for r in all_results]
    agg = {
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "per_seed": all_results,
        "num_seeds": len(config.seeds),
    }
    logger.info("Aggregate F1: %.4f ± %.4f (%d seeds).",
                agg["mean_f1"], agg["std_f1"], agg["num_seeds"])
    return agg


# =========================================================================
# Cross-chain transfer with direct-training control (Reviewer 1)
# =========================================================================

def run_cross_chain_transfer(
    model_factory,
    source_data_factory,
    target_data_factory,
    config: TrainConfig,
    device: torch.device,
    fine_tune_lr: float = 1e-4,
    fine_tune_epochs: int = 20,
) -> Dict[str, Any]:
    """Cross-chain transfer: transfer / zero-shot / direct (control).

    NOTE: source and target must yield the same number of graph views and
    the same node/edge feature dimensionality for weights to transfer.
    """
    results: Dict[str, Any] = {}

    src_data, src_views, _ = source_data_factory()
    tgt_data, tgt_views, _ = target_data_factory()
    if len(src_views) != len(tgt_views):
        raise ValueError(
            f"Source has {len(src_views)} graph views but target has "
            f"{len(tgt_views)}; transfer requires matching K.  Configure "
            f"both chains with the same graph_types."
        )

    # --- Setting 1: Transfer (pre-train + fine-tune) ---
    logger.info("Cross-chain: TRANSFER setting.")
    model, dp = model_factory()
    pre_cfg = TrainConfig(**{**asdict(config),
                             "run_tag": f"{config.run_tag}_xfer_pretrain"})
    trainer = Trainer(model, dp, pre_cfg, device)
    trainer.fit(src_data, src_views)

    ft_cfg = TrainConfig(**{
        **asdict(config),
        "epochs": fine_tune_epochs, "learning_rate": fine_tune_lr,
        "patience": 5, "adversarial_training": False, "enable_dp": False,
        "run_tag": f"{config.run_tag}_xfer_finetune",
    })
    ft_trainer = Trainer(model, None, ft_cfg, device)
    ft_trainer.fit(tgt_data, tgt_views)
    results["transfer_f1"] = ft_trainer.evaluate_test(
        tgt_data, tgt_views)["f1"]

    # --- Setting 2: Zero-shot ---
    logger.info("Cross-chain: ZERO-SHOT setting.")
    model_zs, dp_zs = model_factory()
    zs_cfg = TrainConfig(**{**asdict(config),
                            "run_tag": f"{config.run_tag}_xfer_zeroshot"})
    trainer_zs = Trainer(model_zs, dp_zs, zs_cfg, device)
    trainer_zs.fit(src_data, src_views)
    results["zero_shot_f1"] = trainer_zs.evaluate_test(
        tgt_data, tgt_views)["f1"]

    # --- Setting 3: Direct training on target (Reviewer 1 control) ---
    logger.info("Cross-chain: DIRECT training (control).")
    model_d, dp_d = model_factory()
    d_cfg = TrainConfig(**{**asdict(config),
                           "run_tag": f"{config.run_tag}_xfer_direct"})
    trainer_d = Trainer(model_d, dp_d, d_cfg, device)
    trainer_d.fit(tgt_data, tgt_views)
    results["direct_f1"] = trainer_d.evaluate_test(
        tgt_data, tgt_views)["f1"]

    results["abs_degradation"] = results["direct_f1"] - results["transfer_f1"]
    results["rel_degradation_pct"] = (
        results["abs_degradation"] / max(results["direct_f1"], 1e-8) * 100)

    logger.info(
        "Cross-chain: transfer=%.4f, zero_shot=%.4f, direct=%.4f, "
        "degradation=%.2f%%",
        results["transfer_f1"], results["zero_shot_f1"],
        results["direct_f1"], results["rel_degradation_pct"],
    )
    return results


# =========================================================================
# Ablation study with security metrics (Reviewer 4)
# =========================================================================

# Maps config.yaml `disable_module` names to handler keys.  An UNKNOWN
# component name raises immediately — silently running the full model
# under a "w/o X" label would corrupt the ablation table.
ABLATION_HANDLERS = {
    "temporal_encoder", "cross_graph_fusion", "edge_attention",
    "adversarial", "differential_privacy", "certified_defense",
    "discrepancy",
}


def run_ablation(
    model_factory,
    data_factory,
    config: TrainConfig,
    device: torch.device,
    components: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Ablation study disabling one component at a time.

    `components` entries: {"name": <table label>, "disable_module": <key
    in ABLATION_HANDLERS>}.  For each configuration, reports clean AND
    security metrics (Reviewer 4).
    """
    if components is None:
        components = [
            {"name": "continuous_time_encoding",
             "disable_module": "temporal_encoder"},
            {"name": "multi_graph_fusion",
             "disable_module": "cross_graph_fusion"},
            {"name": "enhanced_attention",
             "disable_module": "edge_attention"},
            {"name": "adversarial_training", "disable_module": "adversarial"},
            {"name": "differential_privacy",
             "disable_module": "differential_privacy"},
            {"name": "certified_defense",
             "disable_module": "certified_defense"},
            {"name": "discrepancy_module", "disable_module": "discrepancy"},
        ]

    for comp in components:
        if comp["disable_module"] not in ABLATION_HANDLERS:
            raise ValueError(
                f"Unknown ablation component "
                f"'{comp['disable_module']}'.  Known: "
                f"{sorted(ABLATION_HANDLERS)}.  Refusing to run — an "
                f"unhandled name would silently evaluate the FULL model "
                f"under an ablation label."
            )

    results = []
    logger.info("Ablation: FULL model.")
    full_result = _run_single_ablation(
        model_factory, data_factory, config, device,
        ablation_name="full", disable_module=None,
    )
    results.append(full_result)

    for comp in components:
        logger.info("Ablation: w/o %s.", comp["name"])
        result = _run_single_ablation(
            model_factory, data_factory, config, device,
            ablation_name=f"w/o_{comp['name']}",
            disable_module=comp["disable_module"],
        )
        result["delta_f1"] = result["test_f1"] - full_result["test_f1"]
        results.append(result)
    return results


def _run_single_ablation(
    model_factory, data_factory, config, device,
    ablation_name, disable_module,
) -> Dict[str, Any]:
    """Run one ablation configuration."""
    model, dp = model_factory()
    run_cfg = TrainConfig(**{**asdict(config),
                             "run_tag": f"{config.run_tag}_abl_"
                                        f"{ablation_name}".replace("/", "_")})

    if disable_module == "temporal_encoder":
        model.time_encoder = _IdentityTimeEncoder(
            model.config.temporal_encoding_dim)
    elif disable_module == "cross_graph_fusion":
        model.fusion = _AverageFusion(model.config.hidden_dim,
                                      model.config.num_graph_types)
    elif disable_module == "edge_attention":
        for layer in model.gat_layers:
            layer.use_edge_features = False
    elif disable_module == "adversarial":
        run_cfg = TrainConfig(**{**asdict(run_cfg),
                                 "adversarial_training": False})
    elif disable_module == "differential_privacy":
        dp = None
        run_cfg = TrainConfig(**{**asdict(run_cfg), "enable_dp": False})
    elif disable_module == "certified_defense":
        model.defense = None
    elif disable_module == "discrepancy":
        # "w/o discrepancy module" (Table 9): bypass the discrepancy
        # weighting network (uniform 1/K fusion weights) AND remove the
        # contrastive loss.  Zeroing only the loss would leave the
        # discrepancy network active and mislabel the ablation.
        model.fusion.use_discrepancy_weights = False
        run_cfg = TrainConfig(**{**asdict(run_cfg), "disc_loss_weight": 0.0})

    data, views, _ = data_factory()
    trainer = Trainer(model, dp, run_cfg, device)
    trainer.fit(data, views)

    test = trainer.evaluate_test(data, views)
    cert, _ = trainer.evaluate_certified(data, views, epsilon=0.20)
    adv = trainer.evaluate_adversarial(data, views, epsilon=0.20)

    # Inference latency over 10 smoothed-eval forwards
    nf, ei, et, ea = trainer._views_to_device(data, views)
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(nf, ei, et, ea)
    latency_ms = (time.perf_counter() - t_start) / 10 * 1000

    return {
        "name": ablation_name,
        "test_f1": test["f1"],
        "test_auc_roc": test["auc_roc"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "certified_accuracy_0.20": cert.get("certified_accuracy"),
        "pgd_accuracy_0.20": adv.get("pgd_accuracy"),
        "abstention_rate": cert.get("abstention_rate"),
        "mean_certified_radius": cert.get("mean_certified_radius"),
        "inference_latency_ms": latency_ms,
    }


# ---- Ablation helper modules ----

class _IdentityTimeEncoder(nn.Module):
    """Replaces ContinuousTimeEncoder with zeros for the ablation."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time_deltas):
        return torch.zeros(*time_deltas.shape, self.dim,
                           device=time_deltas.device)


class _AverageFusion(nn.Module):
    """Replaces CrossGraphDiscrepancyFusion with simple averaging."""
    def __init__(self, hidden_dim, num_graphs):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, graph_features, compute_loss=True):
        return torch.stack(graph_features).mean(dim=0), None


# =========================================================================
# Membership Inference Attack (Table 6 MIA column)
# =========================================================================

@torch.no_grad()
def membership_inference_attack(
    trainer: "Trainer",
    data,
    view_tensors,
    num_attack_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Loss/confidence-threshold membership inference (Yeom et al. 2018).

    Members = training nodes; non-members = test nodes (both labeled).
    The attacker observes the model's per-node loss and predicts
    "member" when the loss is below a threshold tuned on a balanced
    held-out half (a standard, code-free-of-shadow-models baseline; a
    shadow-model attack a la Shokri et al. 2017 is strictly stronger but
    requires training auxiliary models).

    Returns
    -------
    attack_success_rate : balanced accuracy of the best-threshold attack
        on the evaluation half (0.5 == random == ideal privacy).
    advantage           : 2 * (attack_success_rate - 0.5), the
        membership-advantage metric (Yeom et al.).
    auc                 : ROC-AUC of the loss score as a membership
        discriminator (threshold-free).

    NOTE: lower is better.  Report alongside the realized DP epsilon, not
    the target (see DifferentialPrivacyModule honesty note).  This is an
    EMPIRICAL privacy probe; it does not certify the DP guarantee.
    """
    model = trainer.model
    model.eval()
    device = trainer.device
    labels = data.y.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)

    nf, ei, et, ea = trainer._views_to_device(data, view_tensors)
    logits = model(nf, ei, et, ea)["logits"]
    per_node_loss = F.cross_entropy(logits, labels.clamp(min=0),
                                    reduction="none")

    member_idx = torch.where(train_mask)[0]
    nonmember_idx = torch.where(test_mask)[0]
    n = min(len(member_idx), len(nonmember_idx),
            num_attack_samples // 2)
    if n < 10:
        return {"attack_success_rate": float("nan"), "advantage": float("nan"),
                "auc": float("nan"), "n_members": int(len(member_idx)),
                "n_nonmembers": int(len(nonmember_idx)),
                "note": "Too few labeled members/non-members for MIA."}

    g = torch.Generator(device="cpu").manual_seed(seed)
    mi = member_idx[torch.randperm(len(member_idx), generator=g)[:n]]
    ni = nonmember_idx[torch.randperm(len(nonmember_idx), generator=g)[:n]]

    member_loss = per_node_loss[mi].cpu().numpy()
    nonmember_loss = per_node_loss[ni].cpu().numpy()
    # Score = -loss (high score => likely member).  ROC-AUC is
    # threshold-free; success rate uses the best threshold on a tuning
    # half, evaluated on the other half (no test-on-train).
    scores = np.concatenate([-member_loss, -nonmember_loss])
    y_member = np.concatenate([np.ones(n), np.zeros(n)])

    auc = roc_auc_score(y_member, scores) if len(np.unique(y_member)) > 1 \
        else 0.5

    rng = np.random.default_rng(seed)
    perm = rng.permutation(2 * n)
    half = n  # tune on first half, eval on second
    tune, eval_ = perm[:half], perm[half:]
    thresholds = np.unique(scores[tune])
    best_t, best_acc = thresholds[0], 0.0
    for t in thresholds:
        pred = (scores[tune] >= t).astype(float)
        acc = ((pred == y_member[tune]).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    eval_pred = (scores[eval_] >= best_t).astype(float)
    success = float((eval_pred == y_member[eval_]).mean())

    logger.info("MIA: success=%.4f (advantage=%.4f), AUC=%.4f "
                "(n=%d members vs %d non-members).",
                success, 2 * (success - 0.5), auc, n, n)
    return {
        "attack_success_rate": success,
        "advantage": 2 * (success - 0.5),
        "auc": float(auc),
        "n_members": int(n),
        "n_nonmembers": int(n),
        "note": ("Yeom et al. (2018) loss-threshold attack; 0.5 == random "
                 "== ideal privacy.  Empirical probe, not a DP certificate."),
    }


# =========================================================================
# Scalability benchmark (Table 7)
# =========================================================================

@torch.no_grad()
def scalability_benchmark(
    model,
    data,
    view_tensors,
    device: torch.device,
    node_subset_sizes: List[int],
    num_timing_runs: int = 20,
    warmup_runs: int = 3,
) -> List[Dict[str, Any]]:
    """Throughput / latency / memory across node-count subsets.

    Each requested size MUST be <= the actual node count of `data`
    (sizes are subsets of a REAL graph — never synthetic padding; the
    config comment enforces this as policy).  An induced subgraph on the
    first `size` nodes is timed with the standard inference forward.

    Latency is measured per forward over the induced subgraph; throughput
    is nodes / mean_latency (the "transactions per second" the paper
    reports — state this methodology in the caption so the headline and
    the table agree, resolving Reviewer 1's 17,847-vs-2,347 concern).

    Returns one row per size with throughput_tps, latency_p50_ms,
    latency_p99_ms, memory_gb (CUDA peak; 0 on CPU), gpu_util (None here
    — fill from an external nvidia-smi sampler if needed).
    """
    model = model.to(device).eval()
    full_nodes = data.x.size(0)
    rows: List[Dict[str, Any]] = []

    for size in node_subset_sizes:
        if size > full_nodes:
            logger.warning(
                "Scalability: requested size %d > available %d nodes; "
                "SKIPPING (do not report sizes exceeding any real "
                "dataset).", size, full_nodes)
            continue

        keep = torch.arange(size, device=device)
        keep_set = torch.zeros(full_nodes, dtype=torch.bool, device=device)
        keep_set[keep] = True

        nf = [data.x[:size].to(device)
              for _ in range(len(view_tensors))]
        sub_views, sub_edges = [], 0
        for v in view_tensors:
            ei = v["edge_index"].to(device)
            emask = keep_set[ei[0]] & keep_set[ei[1]]
            sub_views.append((
                ei[:, emask],
                v["edge_timestamp"].to(device)[emask],
                v["edge_attr"].to(device)[emask],
            ))
            sub_edges += int(emask.sum())
        ei_s = [s[0] for s in sub_views]
        et_s = [s[1] for s in sub_views]
        ea_s = [s[2] for s in sub_views]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        for _ in range(warmup_runs):
            model(nf, ei_s, et_s, ea_s)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times = []
        for _ in range(num_timing_runs):
            t0 = time.perf_counter()
            model(nf, ei_s, et_s, ea_s)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)

        times = np.array(times)
        p50 = float(np.percentile(times, 50))
        p99 = float(np.percentile(times, 99))
        mean_s = float(times.mean()) / 1000.0
        tps = size / mean_s if mean_s > 0 else 0.0
        mem_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                  if device.type == "cuda" else 0.0)

        row = {
            "nodes": int(size),
            "edges": int(sub_edges),
            "throughput_tps": float(tps),
            "latency_p50_ms": p50,
            "latency_p99_ms": p99,
            "memory_gb": float(mem_gb),
            "gpu_util": None,
        }
        rows.append(row)
        logger.info("Scalability: %d nodes / %d edges -> %.0f TPS, "
                    "P50=%.1fms P99=%.1fms, mem=%.2fGB.",
                    size, sub_edges, tps, p50, p99, mem_gb)
    return rows


# =========================================================================
# Utilities
# =========================================================================

def save_experiment_results(results: Dict, path: str):
    """Save results to JSON (numpy-safe)."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info("Results saved to %s.", path)
