"""
AEGIS-MG: Neural Network Architecture for Multi-Graph Fraud Detection
with Certified Defense and Differential Privacy

Architecture modules:
  1. ContinuousTimeEncoder       — Fourier features + learnable decay (§4.2)
  2. EnhancedGraphAttentionLayer — Edge-aware, DIRECTIONAL multi-head GAT (§4.3)
  3. CrossGraphDiscrepancyFusion — Inter-graph attention + contrastive (§4.4)
     (Renamed from MultiGraphDiscrepancyModule to distinguish from
      DIAM's intra-graph MGD; see Ding et al. CIKM 2024.)
  4. CertifiedDefenseLayer       — Randomized smoothing (§4.5)
     Follows Cohen et al. (ICML 2019):
       - Two-stage CERTIFY (selection + estimation)
       - Clopper-Pearson binomial CI for p_A lower bound (vectorized)
       - Abstention when p_A <= 0.5
       - Certified radius R = sigma * Phi^{-1}(p_A)
     SCOPE: certifies ell_2 perturbations of the continuous fused
     representation ONLY.  Does NOT certify structural/topology attacks
     (see Lai et al. IoT-J 2023).
     IMPORTANT: certification uses a FIXED sigma_0 by default.  Cohen's
     guarantee assumes the smoothing measure does not depend on the input
     being certified; an input-dependent sigma(x) changes the smoothing
     distribution under perturbation and voids the certificate.  The
     adaptive variance of Eq. 4 is therefore treated as a TRAINING-TIME
     augmentation only unless `certify_with_adaptive_sigma=True` is set
     explicitly (in which case the caveat must be stated in the paper).
  5. DifferentialPrivacyModule   — DP-SGD-style mechanism (§4.5)
     Follows Abadi et al. (CCS 2016) + Mironov (CSF 2017):
       - Gradient clipping to norm C, Gaussian noise N(0, sigma^2 C^2 / B^2)
         on the MEAN gradient (equivalent to noise N(0, sigma^2 C^2) on the
         SUM, as in Abadi et al.)
       - Rényi DP accounting for the subsampled Gaussian mechanism
     HONESTY NOTE (must be reflected in the paper): this implementation
     clips the AGGREGATED gradient, not per-sample gradients.  The reported
     epsilon is therefore an approximation; for a formally valid guarantee
     use Opacus PrivacyEngine (per-sample clipping) and note that node-level
     DP on GNNs additionally requires bounding each node's influence through
     message passing (see e.g. Daigavane et al. 2022).  The accountant also
     assumes Poisson subsampling at rate q; full-batch training has q = 1
     and receives NO subsampling amplification (a warning is emitted).
  6. AEGISModel                  — Full pipeline (§4.1, §4.7)

Spectral normalization uses Miyato et al. (ICLR 2018), not AdaLipGNN
(Singh et al. TSP 2025 uses a soft log-product penalty — different mechanism).

Author: [Redacted for double-blind review]
Version: 2.3.0
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
from scipy import stats as scipy_stats

logger = logging.getLogger("aegis.models")


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class ModelConfig:
    """Configuration for the AEGIS-MG model."""

    # Core architecture
    input_dim: int = 32          # matches extract_node_features() output
    hidden_dim: int = 256
    output_dim: int = 2          # binary: legitimate vs illicit
    num_layers: int = 6          # GAT layers
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"

    # Temporal encoder (§4.2).  encoding_dim MUST equal
    # 2 * num_fourier_components (sin + cos per component).
    temporal_encoding_dim: int = 64
    num_fourier_components: int = 32
    max_temporal_distance: float = 86400.0  # 24 hours

    # Transaction edge attributes (amounts, fees, ...).  Set to the actual
    # per-edge attribute dimension of the dataset, or 0 to disable.
    edge_attr_dim: int = 0

    # Cross-graph fusion (§4.4)
    num_graph_types: int = 2     # set dynamically from the data
    cross_graph_heads: int = 4
    discrepancy_temperature: float = 0.1
    contrastive_loss_weight: float = 0.1

    # Adversarial training (§4.7, empirical defense)
    enable_adversarial_training: bool = True
    adv_epsilon: float = 0.3     # PGD perturbation budget
    adv_steps: int = 10          # PGD iterations
    adv_step_size: float = 0.01
    adv_norm: str = "linf"       # "linf" or "l2"
    adv_loss_weight: float = 0.5 # lambda_adv in Eq. 7

    # Certified defense via randomized smoothing (§4.5)
    enable_certified_defense: bool = True
    smoothing_sigma: float = 0.10      # base sigma_0
    selection_samples: int = 100       # n0 in CERTIFY
    certification_samples: int = 10000 # N in CERTIFY
    confidence_alpha: float = 0.001    # failure probability
    adaptive_variance: bool = True     # training-time augmentation only
    adaptive_lambda: float = 0.5
    eval_smoothing_samples: int = 50   # MC draws for smoothed eval forward

    # Differential privacy (§4.5)
    enable_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0  # clip norm C
    dp_noise_multiplier: float = 1.1
    dp_rdp_alphas: List[float] = field(
        default_factory=lambda: [2, 5, 10, 25, 50, 100]
    )

    # Engineering
    use_spectral_norm: bool = True   # Miyato et al. (ICLR 2018)
    use_edge_features: bool = True   # edge-feature bias in attention
                                     # (set False for the "w/o enhanced
                                     #  attention" ablation)
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.temporal_encoding_dim != 2 * self.num_fourier_components:
            raise ValueError(
                f"temporal_encoding_dim ({self.temporal_encoding_dim}) must "
                f"equal 2 * num_fourier_components "
                f"({2 * self.num_fourier_components})."
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")


# =========================================================================
# 1. Continuous Temporal Encoder (§4.2)
# =========================================================================

class ContinuousTimeEncoder(nn.Module):
    """Learnable Fourier features with gating and adaptive decay.

    phi(dt) = [sin(2*pi*w_k(dt) + psi_k(dt)), cos(...)]_{k=1..K}
    with learnable frequency/phase maps, a sigmoid gate, and a learnable
    exponential decay.  The decay parameter alpha is stored unconstrained
    and passed through softplus to guarantee positivity (initialized so
    that softplus(raw) ~= 0.1, matching the paper's "alpha init 0.1").
    """

    def __init__(self, encoding_dim: int = 64,
                 max_time: float = 86400.0,
                 num_components: int = 32):
        super().__init__()
        if encoding_dim != 2 * num_components:
            raise ValueError("encoding_dim must equal 2 * num_components.")
        self.encoding_dim = encoding_dim
        self.max_time = max_time

        self.freq_linear = nn.Linear(1, num_components)
        self.phase_linear = nn.Linear(1, num_components)

        self.gate = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.LayerNorm(encoding_dim * 2),
            nn.GELU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.Sigmoid(),
        )

        # softplus^{-1}(0.1) ≈ -2.252 so that the EFFECTIVE decay starts
        # at 0.1 as stated in the paper.
        self.decay_alpha = nn.Parameter(torch.full((encoding_dim,), -2.252))

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_deltas: shape [...], non-negative seconds (each edge's age
                relative to the most recent edge in its graph view).
        Returns:
            Encoded time: shape [..., encoding_dim].

        SCALE (important): the benchmark graphs span MONTHS, while `max_time`
        (config.max_temporal_distance) defaults to one day. Dividing the raw
        delta by a fixed one-day constant would push dt_norm to ~100+, and the
        decay term exp(-softplus(alpha) * dt_norm) would saturate to ~0 for
        essentially every edge older than a few days -- silently destroying the
        temporal signal, with the learnable decay unable to recover because it
        is fighting a runaway scale. We instead normalize by the LARGER of the
        observed maximum delta and `max_time` (the latter a floor so that very
        short graphs are not over-amplified). The normalization constant is
        DETACHED: `decay_alpha` then controls the decay rate over the resulting
        [0, 1] scale -- a well-conditioned range -- rather than the scale itself.
        """
        dt = time_deltas.clamp(min=0).to(torch.float32)
        floor = torch.as_tensor(self.max_time, dtype=dt.dtype, device=dt.device)
        scale = torch.clamp(dt.max().detach(), min=floor)   # graph-relative, floored
        dt_norm = (dt / scale).unsqueeze(-1)

        freq = self.freq_linear(dt_norm)
        phase = self.phase_linear(dt_norm)
        angles = 2 * math.pi * freq
        phi = torch.cat([torch.sin(angles + phase),
                         torch.cos(angles + phase)], dim=-1)

        phi = phi * self.gate(phi)
        decay = torch.exp(-F.softplus(self.decay_alpha) * dt_norm)
        return phi * decay


# =========================================================================
# 2. Enhanced Graph Attention Layer (§4.3)
# =========================================================================

class EnhancedGraphAttentionLayer(nn.Module):
    """Edge-aware multi-head graph attention with DIRECTIONAL aggregation.

    Implements the paper's Eq. 3 and the directional update:
      h_v^{in}  = sum_{u in N_in(v)}  alpha_uv * W_V h_u
      h_v^{out} = sum_{u in N_out(v)} beta_vu  * W_V h_u
      h_v^{l+1} = LayerNorm( W_O [ h_v^{in} || h_v^{out} ] + h_v^{l} )

    Attention is computed sparsely per edge (O(E) memory), with a
    segment-softmax over each node's incoming (resp. outgoing) edges.
    Edge features (temporal encoding [+ optional projected transaction
    attributes]) enter as a per-head additive bias (W_E e_uv).

    `use_edge_features=False` removes the edge bias — this is the
    "w/o enhanced attention" ablation, reducing the layer to a plain
    directional dot-product GAT.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_heads: int = 8, edge_dim: int = 64,
                 dropout: float = 0.1, use_edge_features: bool = True):
        super().__init__()
        assert output_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features

        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        if use_edge_features:
            self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
        # Directional concat -> 2 * output_dim into W_O
        self.W_o = nn.Linear(2 * output_dim, output_dim)

        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)
        self.res_proj = (nn.Linear(input_dim, output_dim)
                         if input_dim != output_dim else nn.Identity())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim]
            edge_index: [2, num_edges]  (src -> dst)
            edge_features: [num_edges, edge_dim] or None
        Returns:
            Updated x: [num_nodes, output_dim]
        """
        N = x.size(0)
        residual = self.res_proj(x)

        if edge_index.numel() == 0:
            # Graph view with no edges: identity update.
            return self.norm(residual)

        src, dst = edge_index

        Q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim)

        # Raw per-edge scores (shared by both directions)
        scores = (Q[dst] * K[src]).sum(dim=-1) / math.sqrt(self.head_dim)
        if self.use_edge_features and edge_features is not None:
            scores = scores + self.W_e(edge_features)  # [E, H]

        # ---- Incoming aggregation (softmax over edges sharing a dst) ----
        attn_in = self.drop(self._segment_softmax(scores, dst, N))
        h_in = torch.zeros(N, self.num_heads, self.head_dim,
                           device=x.device, dtype=x.dtype)
        h_in.index_add_(0, dst, attn_in.unsqueeze(-1) * V[src])

        # ---- Outgoing aggregation (softmax over edges sharing a src) ----
        attn_out = self.drop(self._segment_softmax(scores, src, N))
        h_out = torch.zeros(N, self.num_heads, self.head_dim,
                            device=x.device, dtype=x.dtype)
        h_out.index_add_(0, src, attn_out.unsqueeze(-1) * V[dst])

        h_in = h_in.reshape(N, self.output_dim)
        h_out = h_out.reshape(N, self.output_dim)
        out = self.W_o(torch.cat([h_in, h_out], dim=-1))
        return self.norm(out + residual)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, indices: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """Numerically stable softmax over variable-length segments.

        Computed in float32 regardless of the incoming dtype: under
        autocast the scores arrive in half precision, where exp/sum
        accumulation is unsafe and `index_reduce_` half-precision support
        varies across PyTorch versions.  The result is cast back to the
        caller's dtype.
        """
        in_dtype = scores.dtype
        scores32 = scores.float()
        max_vals = torch.full((num_nodes, scores32.size(1)), float("-inf"),
                              device=scores32.device, dtype=torch.float32)
        max_vals.index_reduce_(0, indices, scores32, "amax",
                               include_self=True)
        exp_scores = (scores32 - max_vals[indices]).exp()
        sum_exp = torch.zeros(num_nodes, scores32.size(1),
                              device=scores32.device, dtype=torch.float32)
        sum_exp.index_add_(0, indices, exp_scores)
        return (exp_scores / (sum_exp[indices] + 1e-10)).to(in_dtype)


# =========================================================================
# 3. Cross-Graph Discrepancy Fusion (§4.4)
# =========================================================================
# RENAMED from MultiGraphDiscrepancyModule.  This module captures
# INTER-graph discrepancy (same node across different graph types);
# DIAM's MGD captures INTRA-graph discrepancy (node vs. its neighbors
# within one directed multigraph).  The two are complementary.

class CrossGraphDiscrepancyFusion(nn.Module):
    """Cross-graph attention and contrastive fusion for K graph views.

    For K == 1 the module degenerates gracefully: the single encoded view
    is gated and returned, and the contrastive loss is skipped.
    """

    def __init__(self, hidden_dim: int, num_graphs: int = 2,
                 num_heads: int = 4, temperature: float = 0.1,
                 contrast_weight: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_graphs = num_graphs
        self.temperature = temperature
        self.contrast_weight = contrast_weight
        # Ablation switch ("w/o discrepancy module", Table 9): when False,
        # the discrepancy network's softmax weights are replaced by
        # uniform 1/K weights (the gates and cross-attention remain).
        # The contrastive loss is controlled separately by the trainer.
        self.use_discrepancy_weights = True

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ) for _ in range(num_graphs)
        ])

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True,
        )

        self.disc_net = nn.Sequential(
            nn.Linear(hidden_dim * num_graphs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_graphs),
        )

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            ) for _ in range(num_graphs)
        ])

    def forward(self, graph_features: List[torch.Tensor],
                compute_loss: bool = True
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            graph_features: list of K tensors, each [N, hidden_dim].
        Returns:
            fused: [N, hidden_dim];  loss: scalar or None.
        """
        assert len(graph_features) == self.num_graphs, (
            f"Expected {self.num_graphs} graph views, "
            f"got {len(graph_features)}."
        )
        N = graph_features[0].size(0)

        encoded = [enc(feat) for enc, feat in
                   zip(self.encoders, graph_features)]

        stacked = torch.stack(encoded, dim=1)               # [N, K, D]
        attended, _ = self.cross_attn(stacked, stacked, stacked)

        disc = self.disc_net(torch.cat(encoded, dim=-1))    # [N, K]
        if self.use_discrepancy_weights:
            weights = F.softmax(disc / self.temperature, dim=-1)
        else:
            # Ablation: uniform fusion weights (discrepancy net bypassed).
            weights = torch.full_like(disc, 1.0 / self.num_graphs)

        fused = torch.zeros(N, self.hidden_dim,
                            device=graph_features[0].device,
                            dtype=graph_features[0].dtype)
        for i in range(self.num_graphs):
            g = self.gates[i](torch.cat([encoded[i], attended[:, i]], dim=-1))
            fused = fused + weights[:, i:i + 1] * g * encoded[i]

        loss = None
        if compute_loss and self.num_graphs > 1:
            loss = self._contrastive_loss(encoded)
        return fused, loss

    def _contrastive_loss(self, encoded: List[torch.Tensor]) -> torch.Tensor:
        """Mean pairwise cosine similarity across views (§4.4).

        Minimizing this term penalizes views that collapse to identical
        representations, preserving cross-view discrepancy.
        """
        total = encoded[0].new_zeros(())
        count = 0
        for i in range(self.num_graphs):
            for j in range(i + 1, self.num_graphs):
                total = total + F.cosine_similarity(
                    encoded[i], encoded[j], dim=-1).mean()
                count += 1
        return self.contrast_weight * total / max(count, 1)


# =========================================================================
# 4. Certified Defense Layer (§4.5)
# =========================================================================

class CertifiedDefenseLayer(nn.Module):
    """Randomized smoothing with Cohen et al. (ICML 2019) CERTIFY.

    The deterministic refinement h(z) = z + 0.1 * denoiser(z) is part of
    the BASE classifier.  Consistency contract (required for the
    certificate to be meaningful):

      * Training:      classifier sees  h(z + eps),  eps ~ N(0, sigma^2 I)
                       (true Gaussian augmentation, Cohen §3.3).
      * Smoothed eval: average of h(z + eps) over MC draws.
      * CERTIFY:       votes of  classifier(h(z + eps))  with the SAME
                       fixed sigma_0.

    Certification uses fixed sigma_0 by default; see module docstring.
    """

    def __init__(self, hidden_dim: int, sigma_0: float = 0.1,
                 adaptive: bool = True, adaptive_lambda: float = 0.5,
                 n0: int = 100, N: int = 10000, alpha: float = 0.001,
                 eval_samples: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma_0 = sigma_0
        self.adaptive = adaptive
        self.adaptive_lambda = adaptive_lambda
        self.n0 = n0
        self.N = N
        self.alpha = alpha
        self.eval_samples = eval_samples

        if adaptive:
            # Eq. 4: sigma^2(x) = sigma_0^2 * (1 + lambda * a(x)),
            # a(x) = sigmoid(MLP(x)) in [0, 1].  TRAINING-TIME ONLY.
            self.variance_adapter = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    # ---- Deterministic refinement (part of the base classifier) ----

    def refine(self, z: torch.Tensor) -> torch.Tensor:
        """h(z) = z + 0.1 * denoiser(z)."""
        return z + 0.1 * self.denoiser(z)

    def get_training_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Per-node sigma used for TRAINING-time Gaussian augmentation."""
        if self.adaptive:
            a = self.variance_adapter(x)
            return self.sigma_0 * torch.sqrt(1.0 + self.adaptive_lambda * a)
        return x.new_full((x.size(0), 1), self.sigma_0)

    def forward(self, x: torch.Tensor, training: bool = True
                ) -> torch.Tensor:
        """Gaussian-augmented refinement (train) / MC smoothing (eval)."""
        if training:
            noise = torch.randn_like(x) * self.get_training_sigma(x)
            return self.refine(x + noise)
        # Smoothed evaluation: MC average of h(x + eps) with FIXED sigma_0
        # (consistent with the certification measure).
        total = torch.zeros_like(x)
        for _ in range(self.eval_samples):
            noise = torch.randn_like(x) * self.sigma_0
            total = total + self.refine(x + noise)
        return total / self.eval_samples

    # ---- CERTIFY (Cohen et al. 2019) ----

    @torch.no_grad()
    def certify(
        self,
        x: torch.Tensor,
        base_classifier_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        node_idx: Optional[torch.Tensor] = None,
        certify_with_adaptive_sigma: bool = False,
        draw_batch: int = 25,
        num_classes: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """Two-stage CERTIFY with Clopper-Pearson CI and abstention.

        Args:
            x: [num_nodes, hidden_dim] fused embeddings (pre-defense).
            base_classifier_fn: callable(z_noisy [n, D], node_idx [n])
                -> predicted classes [n].  It MUST apply self.refine
                internally (use AEGISModel.get_base_classifier_fn).
            node_idx: optional 1-D LongTensor of node indices to certify
                (e.g. test nodes).  Certifying all nodes of a large graph
                with N = 10,000 draws is usually infeasible.
            certify_with_adaptive_sigma: if True, certifies each node with
                its own sigma(x).  DISABLED by default because Cohen's
                guarantee assumes an input-independent smoothing measure;
                enabling this requires an explicit caveat in the paper.
            draw_batch: number of noise draws evaluated per classifier call.
            num_classes: size of the label space (binary fraud detection
                uses 2; kept configurable so the layer is reusable).

        Returns dict with predicted_class (-1 = abstain), certified_radius,
        certified_mask, p_A_lower, abstention_rate, sigma_used.
        """
        if node_idx is None:
            node_idx = torch.arange(x.size(0), device=x.device)
        x_sub = x[node_idx]
        n = x_sub.size(0)

        if certify_with_adaptive_sigma and self.adaptive:
            logger.warning(
                "Certifying with input-dependent sigma(x).  Cohen's "
                "guarantee assumes a fixed smoothing measure; report this "
                "deviation explicitly in the paper."
            )
            sigma = self.get_training_sigma(x_sub)          # [n, 1]
        else:
            sigma = x_sub.new_full((n, 1), self.sigma_0)

        row_idx = torch.arange(n, device=x.device)

        def _vote_counts(num_draws: int, target: Optional[torch.Tensor]
                         ) -> torch.Tensor:
            """Run num_draws noisy classifications.

            If target is None: return [n, num_classes] class counts
            (selection).  Else: return [n] counts of votes equal to
            target (estimation).
            """
            if target is None:
                counts = torch.zeros(n, num_classes, device=x.device)
            else:
                counts = torch.zeros(n, device=x.device)
            remaining = num_draws
            while remaining > 0:
                b = min(draw_batch, remaining)
                # [b, n, D] noisy copies, classified per draw
                noise = torch.randn(b, n, x_sub.size(1),
                                    device=x.device) * sigma
                for k in range(b):
                    preds = base_classifier_fn(x_sub + noise[k], node_idx)
                    if target is None:
                        counts[row_idx, preds] += 1
                    else:
                        counts += (preds == target).float()
                remaining -= b
            return counts

        # Stage 1: selection (n0 draws) — guess the top class.
        counts0 = _vote_counts(self.n0, target=None)
        c_A = counts0.argmax(dim=1)

        # Stage 2: estimation (N fresh draws) — count votes for c_A.
        n_A = _vote_counts(self.N, target=c_A)

        # Clopper-Pearson one-sided lower bound (vectorized):
        # p_A_lower = BetaInvCDF(alpha; n_A, N - n_A + 1);  0 when n_A = 0.
        n_A_np = n_A.cpu().numpy()
        with np.errstate(all="ignore"):
            p_low = scipy_stats.beta.ppf(self.alpha, n_A_np,
                                         self.N - n_A_np + 1)
        p_low = np.where(n_A_np <= 0, 0.0, p_low)
        p_low = np.nan_to_num(p_low, nan=0.0)
        p_A_lower = torch.from_numpy(p_low).to(x.device).float()

        certified_mask = p_A_lower > 0.5
        sigma_flat = sigma.squeeze(-1)
        radius = torch.zeros(n, device=x.device)
        if certified_mask.any():
            phi_inv = scipy_stats.norm.ppf(
                p_A_lower[certified_mask].cpu().numpy())
            radius[certified_mask] = (
                sigma_flat[certified_mask]
                * torch.from_numpy(phi_inv).to(x.device).float()
            )

        predicted_class = c_A.clone()
        predicted_class[~certified_mask] = -1
        abstention_rate = 1.0 - certified_mask.float().mean().item()

        logger.info(
            "CERTIFY: %d/%d nodes certified (%.1f%% abstention), "
            "mean R=%.4f, median R=%.4f (sigma_0=%.3f, n0=%d, N=%d, "
            "alpha=%.4f).",
            int(certified_mask.sum()), n, abstention_rate * 100,
            radius[certified_mask].mean().item() if certified_mask.any() else 0,
            radius[certified_mask].median().item() if certified_mask.any() else 0,
            self.sigma_0, self.n0, self.N, self.alpha,
        )

        return {
            "node_idx": node_idx,
            "predicted_class": predicted_class,
            "certified_radius": radius,
            "certified_mask": certified_mask,
            "p_A_lower": p_A_lower,
            "abstention_rate": abstention_rate,
            "sigma_used": sigma_flat,
        }


# =========================================================================
# 5. Differential Privacy Module (§4.5)
# =========================================================================

class DifferentialPrivacyModule:
    """DP-SGD-style mechanism with RDP accounting.

    See the honesty note in the module docstring: aggregated-gradient
    clipping is an APPROXIMATION of per-sample clipping; for a formally
    valid epsilon use Opacus.  The noise magnitude is fixed by (sigma, C)
    — never learned or data-dependent.
    """

    def __init__(self, max_grad_norm: float = 1.0,
                 noise_multiplier: float = 1.1,
                 target_epsilon: float = 1.0,
                 delta: float = 1e-5,
                 rdp_alphas: Optional[List[float]] = None):
        self.C = max_grad_norm
        self.sigma = noise_multiplier
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.alphas = [a for a in (rdp_alphas or [2, 5, 10, 25, 50, 100])
                       if a > 1]

        self.steps = 0
        self._rdp_spent = np.zeros(len(self.alphas))
        self._warned_full_batch = False

    def clip_and_noise_gradients(
        self, model: nn.Module, batch_size: int, dataset_size: int
    ) -> Dict[str, float]:
        """Clip the TOTAL gradient to norm C, then add Gaussian noise.

        Abadi et al. (2016) add N(0, sigma^2 C^2 I) to the SUM of clipped
        per-sample gradients; on the MEAN gradient (what loss.backward()
        with mean reduction produces) the equivalent noise std is
        sigma * C / B.
        """
        B = max(int(batch_size), 1)
        q = min(1.0, B / max(int(dataset_size), 1))
        if q >= 1.0 and not self._warned_full_batch:
            logger.warning(
                "DP accounting: sampling rate q = 1 (full-batch training). "
                "No subsampling amplification applies; the reported epsilon "
                "uses the unamplified Gaussian-mechanism bound."
            )
            self._warned_full_batch = True

        # Global clip across ALL parameters (one gradient vector of norm <= C)
        total_norm = torch.sqrt(sum(
            (p.grad.detach() ** 2).sum()
            for p in model.parameters() if p.grad is not None
        ))
        clip_coef = float(min(1.0, self.C / (total_norm.item() + 1e-10)))

        noise_std = self.sigma * self.C / B
        for param in model.parameters():
            if param.grad is None:
                continue
            param.grad.mul_(clip_coef)
            param.grad.add_(torch.randn_like(param.grad) * noise_std)

        self._account_step(q)
        self.steps += 1

        eps = self.get_epsilon()
        if self.steps == 1 and eps >= self.target_epsilon:
            logger.warning(
                "DP calibration: the FIRST step already spends epsilon = "
                "%.3f >= target %.3f (noise_multiplier = %.2f, q = %.3f). "
                "Training will stop immediately and any utility measured "
                "at this target is meaningless.  Recalibrate the noise "
                "multiplier per target epsilon (e.g. Opacus "
                "get_noise_multiplier) — REQUIRED for the Table 6 privacy "
                "sweep, especially at small epsilon under full-batch "
                "training.", eps, self.target_epsilon, self.sigma, q,
            )
        return {
            "epsilon_spent": eps,
            "epsilon_target": self.target_epsilon,
            "delta": self.delta,
            "steps": self.steps,
            "sampling_rate": q,
            "budget_exhausted": eps >= self.target_epsilon,
        }

    def _account_step(self, q: float) -> None:
        """Accumulate per-step RDP.

        Subsampled Gaussian lead-term bound (Mironov 2017; Wang et al.
        2019): RDP_alpha ~= alpha * q^2 / (2 sigma^2) for q < 1, valid in
        the small-q / sigma >= 1 regime.  At q = 1 the exact Gaussian
        mechanism bound RDP_alpha = alpha / (2 sigma^2) is used.  This
        implementation is intentionally simple and is NOT guaranteed to
        be conservative in all regimes — for publication-grade accounting,
        cross-check the reported epsilon with Opacus RDPAccountant before
        reporting.
        """
        for i, alpha in enumerate(self.alphas):
            if q >= 1.0:
                rdp_step = alpha / (2 * self.sigma ** 2)
            else:
                rdp_step = alpha * q ** 2 / (2 * self.sigma ** 2)
            self._rdp_spent[i] += rdp_step

    def get_epsilon(self) -> float:
        """eps = min_alpha [ RDP(alpha) + log(1/delta) / (alpha - 1) ]."""
        candidates = [
            self._rdp_spent[i] + math.log(1.0 / self.delta) / (alpha - 1)
            for i, alpha in enumerate(self.alphas)
        ]
        return min(candidates) if candidates else float("inf")

    def get_privacy_report(self) -> Dict[str, float]:
        eps = self.get_epsilon()
        return {
            "epsilon": eps,
            "delta": self.delta,
            "steps": self.steps,
            "noise_multiplier": self.sigma,
            "clip_norm": self.C,
            "budget_remaining": max(0.0, self.target_epsilon - eps),
            "note": ("Approximate accounting; aggregated-gradient clipping. "
                     "Validate with Opacus before reporting."),
        }


# =========================================================================
# 6. AEGIS-MG Full Model (§4.1, §4.7)
# =========================================================================

class AEGISModel(nn.Module):
    """Complete AEGIS-MG architecture.

    Pipeline per graph view k:
      input projection -> temporal (+ edge-attr) edge features
      -> L directional GAT layers -> per-view embedding
    Then: CrossGraphDiscrepancyFusion -> (optional) CertifiedDefenseLayer
    -> classification head over [view_1 || ... || view_K || fused]
    (head input dim = hidden_dim * (K + 1)).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        K = config.num_graph_types

        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ) for _ in range(K)
        ])

        self.time_encoder = ContinuousTimeEncoder(
            config.temporal_encoding_dim,
            config.max_temporal_distance,
            config.num_fourier_components,
        )

        # Optional projection of raw transaction edge attributes into the
        # temporal-encoding space, so that attention genuinely uses
        # transaction attributes (amounts, fees, ...) as stated in §4.3.
        self.edge_attr_proj = None
        if config.edge_attr_dim > 0:
            self.edge_attr_proj = nn.Linear(
                config.edge_attr_dim, config.temporal_encoding_dim
            )

        self.gat_layers = nn.ModuleList()
        for _ in range(config.num_layers):
            layer = EnhancedGraphAttentionLayer(
                config.hidden_dim, config.hidden_dim,
                config.num_heads, config.temporal_encoding_dim,
                config.dropout, use_edge_features=config.use_edge_features,
            )
            if config.use_spectral_norm:
                layer = self._apply_spectral_norm(layer)
            self.gat_layers.append(layer)

        self.fusion = CrossGraphDiscrepancyFusion(
            config.hidden_dim, K, config.cross_graph_heads,
            config.discrepancy_temperature,
            config.contrastive_loss_weight,
        )

        self.defense = None
        if config.enable_certified_defense:
            self.defense = CertifiedDefenseLayer(
                config.hidden_dim, config.smoothing_sigma,
                config.adaptive_variance, config.adaptive_lambda,
                config.selection_samples, config.certification_samples,
                config.confidence_alpha, config.eval_smoothing_samples,
            )

        head_input_dim = config.hidden_dim * (K + 1)
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )

        self._init_weights()

    # ---- Shared encoder path ----

    def encode_views(self, node_features, edge_indices, edge_timestamps,
                     edge_attrs=None) -> List[torch.Tensor]:
        """Compute per-view node embeddings (input proj + GAT stack)."""
        embeds = []
        for k in range(self.config.num_graph_types):
            x = self.input_projs[k](node_features[k])

            ets = edge_timestamps[k]
            if ets.numel() > 0:
                # Per-edge AGE relative to the most recent edge in this view.
                # PRECISION: absolute Unix timestamps (~1.7e9 s) are not exactly
                # representable in float32 (spacing ~128 s), so the delta is
                # taken in the tensor's native dtype (the data pipeline supplies
                # float64) before the small-magnitude result is cast down.
                # SCALE: the encoder normalizes these deltas by the graph's own
                # time span (floored by max_temporal_distance), so month-long
                # graphs do not saturate the decay -- see
                # ContinuousTimeEncoder.forward.
                time_deltas = (ets.max() - ets).to(torch.float32)
                e_feat = self.time_encoder(time_deltas)
                if (self.edge_attr_proj is not None and edge_attrs is not None
                        and edge_attrs[k] is not None):
                    e_feat = e_feat + self.edge_attr_proj(edge_attrs[k])
            else:
                e_feat = None

            for gat in self.gat_layers:
                if self.config.gradient_checkpointing and self.training:
                    x = checkpoint(gat, x, edge_indices[k], e_feat,
                                   use_reentrant=False)
                else:
                    x = gat(x, edge_indices[k], e_feat)
            embeds.append(x)
        return embeds

    def forward(
        self,
        node_features: List[torch.Tensor],
        edge_indices: List[torch.Tensor],
        edge_timestamps: List[torch.Tensor],
        edge_attrs: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: K tensors, each [num_nodes, input_dim].
            edge_indices: K tensors, each [2, num_edges_k].
            edge_timestamps: K tensors, each [num_edges_k].
            edge_attrs: optional K tensors, each [num_edges_k, edge_attr_dim].
        Returns:
            dict with "logits", "predictions", optionally "discrepancy_loss".
        """
        outputs: Dict[str, torch.Tensor] = {}

        graph_embeds = self.encode_views(
            node_features, edge_indices, edge_timestamps, edge_attrs)

        fused, disc_loss = self.fusion(
            graph_embeds, compute_loss=self.training)
        if disc_loss is not None:
            outputs["discrepancy_loss"] = disc_loss

        if self.defense is not None:
            fused = self.defense(fused, training=self.training)

        combined = torch.cat(graph_embeds + [fused], dim=-1)
        logits = self.classifier(combined)

        outputs["logits"] = logits
        outputs["predictions"] = F.softmax(logits, dim=-1)
        return outputs

    # ---- Base classifier for CERTIFY ----

    @torch.no_grad()
    def get_base_classifier_fn(self, node_features, edge_indices,
                               edge_timestamps, edge_attrs=None):
        """Return (classify_fn, fused) for CERTIFY.

        classify_fn(z_noisy [n, D], node_idx [n]) -> class indices [n].
        It applies the deterministic refinement h(.) of the defense layer
        — the SAME function the classifier was trained on — so the
        certified function matches the trained base classifier.

        The per-view embeddings are deterministic w.r.t. the smoothing
        noise (noise is applied to the fused representation only, which is
        exactly what Theorem 3.1 certifies — see the scope statement).
        """
        was_training = self.training
        self.eval()
        graph_embeds = self.encode_views(
            node_features, edge_indices, edge_timestamps, edge_attrs)
        fused, _ = self.fusion(graph_embeds, compute_loss=False)
        if was_training:
            self.train()

        defense = self.defense
        classifier = self.classifier

        def classify_fn(z_noisy: torch.Tensor,
                        node_idx: torch.Tensor) -> torch.Tensor:
            # The certified base classifier must be deterministic given
            # the noise draw: force eval mode (no dropout) for the
            # refinement and classification head even if the surrounding
            # model has been switched back to train mode.
            was_training = classifier.training
            if was_training:
                classifier.eval()
                if defense is not None:
                    defense.eval()
            try:
                z = defense.refine(z_noisy) if defense is not None else z_noisy
                views = [g[node_idx] for g in graph_embeds]
                combined = torch.cat(views + [z], dim=-1)
                return classifier(combined).argmax(dim=-1)
            finally:
                if was_training:
                    classifier.train()
                    if defense is not None:
                        defense.train()

        return classify_fn, fused

    # ---- Adversarial training (Algorithm 2, lines 9-12) ----

    def compute_adversarial_loss(
        self,
        node_features: List[torch.Tensor],
        edge_indices: List[torch.Tensor],
        edge_timestamps: List[torch.Tensor],
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        edge_attrs: Optional[List[Optional[torch.Tensor]]] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PGD adversarial training loss.

        `loss_mask` restricts BOTH attack generation and the adversarial
        loss to training nodes.  Without it, gradients of validation/test
        labels would steer the perturbation — label leakage.
        """
        eps = self.config.adv_epsilon
        steps = self.config.adv_steps
        alpha = self.config.adv_step_size
        use_linf = self.config.adv_norm == "linf"

        # Algorithm 2, line 9: random start delta^(0) ~ Uniform(-eps, eps)
        # (l_inf) or a uniformly scaled random direction inside the l2
        # ball.  Starting from the clean point weakens PGD (Madry et al.
        # 2018) and would not match the algorithm stated in the paper.
        perturbed = []
        with torch.no_grad():
            for f in node_features:
                if use_linf:
                    delta0 = torch.empty_like(f).uniform_(-eps, eps)
                else:
                    delta0 = torch.randn_like(f)
                    d_norm = delta0.norm(dim=-1, keepdim=True) + 1e-10
                    scale = torch.rand(f.size(0), 1, device=f.device) * eps
                    delta0 = delta0 / d_norm * scale
                perturbed.append((f + delta0).detach())
        for p in perturbed:
            p.requires_grad_(True)

        for _ in range(steps):
            out = self.forward(perturbed, edge_indices, edge_timestamps,
                               edge_attrs)
            loss = F.cross_entropy(out["logits"][loss_mask],
                                   labels[loss_mask],
                                   weight=class_weights)
            grads = torch.autograd.grad(loss, perturbed)
            with torch.no_grad():
                for i, (feat, grad) in enumerate(zip(perturbed, grads)):
                    if use_linf:
                        feat += alpha * grad.sign()
                        delta = (feat - node_features[i]).clamp_(-eps, eps)
                    else:
                        g_norm = grad.norm(dim=-1, keepdim=True) + 1e-10
                        feat += alpha * grad / g_norm
                        delta = feat - node_features[i]
                        d_norm = delta.norm(dim=-1, keepdim=True)
                        delta = delta / torch.clamp(d_norm / eps, min=1.0)
                    feat.copy_(node_features[i] + delta)
            for feat in perturbed:
                feat.requires_grad_(True)

        adv_inputs = [p.detach() for p in perturbed]
        adv_out = self.forward(adv_inputs, edge_indices, edge_timestamps,
                               edge_attrs)
        return F.cross_entropy(adv_out["logits"][loss_mask],
                               labels[loss_mask],
                               weight=class_weights)

    # ---- Utilities ----

    def _apply_spectral_norm(self, module: nn.Module) -> nn.Module:
        """Spectral normalization (Miyato et al. 2018) on Linear layers."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, spectral_norm(child))
            elif len(list(child.children())) > 0:
                self._apply_spectral_norm(child)
        return module

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_model_stats(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "model_size_mb": total * 4 / (1024 ** 2),
        }


# =========================================================================
# Factory
# =========================================================================

def create_model(config_dict: Optional[Dict] = None) -> AEGISModel:
    """Create AEGIS-MG model from a config dict."""
    config = ModelConfig(**(config_dict or {}))
    model = AEGISModel(config)
    stats = model.get_model_stats()
    logger.info("Model: %s params (%.1f MB), K=%d graph views.",
                f"{stats['total_parameters']:,}",
                stats["model_size_mb"], config.num_graph_types)
    return model
