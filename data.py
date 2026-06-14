"""
AEGIS-MG -- Data Pipeline for Multi-Blockchain Transaction-Graph Construction
Engineering Applications of Artificial Intelligence (EAAI), submission EAAI-25-25248.

This module builds the node-classification datasets used in the paper. It is
written to be *reproducible and provenance-faithful*: the four cryptocurrency
benchmarks are the datasets released with DIAM (Ding et al., CIKM 2024), and
this loader reads them in their published `*_graph_dict.npz` form and validates
the loaded graph against DIAM's Table 1 statistics. Binance Smart Chain (BSC)
has no public benchmark and is built from raw BSCScan exports + externally
sourced labels.

Provenance (labels MUST come from verified external sources -- never heuristics):
  - Ethereum-S : Yuan et al. (2020), Etherscan/XBlock phishing labels.        [npz]
  - Ethereum-P : Chen et al. (2020), Etherscan phishing (illicit) + licit
                 wallet/finance-service labels (normal).                       [npz]
  - Bitcoin-M  : Wu et al. (2021), WalletExplorer; gambling/mixing = illicit.  [npz]
  - Bitcoin-L  : Ding et al. (2024, DIAM), WalletExplorer labels.              [npz]
  - BSC-Cross  : BSCScan API exports; community-reported cross-chain fraud.    [parquet]

Label semantics (DIAM-compatible, identical across all sources):
  y =  1  illicit          y =  0  legitimate          y = -1  unknown/unlabeled
  Training / validation / testing use ONLY labeled nodes (y >= 0). Treating
  unlabeled addresses as legitimate would inject massive label noise and silently
  change the task; this pipeline never does that. Evaluating on the labeled
  subset (where the illicit share is 18-49%, not the whole-graph <1%) is what
  makes precision/F1 meaningful -- see the response to Reviewer 4 on the
  base-rate point.

Graph model (matches DIAM exactly):
  - Bitcoin: ADDRESS graph. A transaction with |S| senders and |R| receivers
    expands to |S| x |R| directed edges (UTXO fan-out). This is the rule the
    earlier draft violated; it is implemented here with a vectorized explode.
  - Ethereum / BSC: ACCOUNT graph; one directed edge per transaction.

Node features (32, our own -- NOT Elliptic's 166, which require Elliptic Inc.'s
proprietary entity-intelligence engine and cannot be recomputed from public
chain data; see Weber et al. 2019). DIAM itself uses Edge2Seq over edge
sequences and needs no node features; we compute 32 uniform features so that a
single fixed input dimension is shared across all chains (required for the
cross-chain transfer experiment). The DIAM-provided X can optionally be used
instead via `use_provided_node_features=True` (disabled by default because the
provided dimensions differ per dataset: 48/48/69/89).

Edge features: the DIAM edge-attribute schema MINUS the absolute timestamp, plus
log1p(amount); zero-padded to a common width so weights transfer across chains.
The absolute timestamp is deliberately EXCLUDED from edge attributes (the
temporal encoder consumes time *deltas* separately) so that, under a temporal
split, the model cannot use "later timestamp => test split" as a shortcut.

Leakage prevention (Reviewer 4):
  - The RobustScaler for node and edge features is fitted on TRAINING data only
    (train nodes / pre-cutoff edges) and then applied to the whole graph.
  - PROTOCOL SCOPE (state in the paper, Section 5.1): the setting is
    TRANSDUCTIVE. Node features and message passing use the full graph; under
    the temporal split only the LABELS are partitioned by time. A strictly
    inductive temporal protocol would also restrict feature computation to
    pre-cutoff edges. Metadata records this as feature_scope.

Multi-graph views (K is data-dependent, NEVER faked):
  - "tx":       value-transfer edges (always present).
  - "contract": edges touching contract addresses, built ONLY when genuine
    contract data is supplied. The public DIAM graphs contain no contract data,
    so they run with K = 1; the pipeline never duplicates the tx view to inflate
    the multi-graph claim. A real second view for Ethereum can be sourced from
    XBlock-ETH contract-call data (Zheng et al. 2020) -- see the project notes.

A note on imports: torch / torch_geometric are imported lazily inside the
tensorization step, so the pure graph-construction logic in this file can be
exercised and unit-tested with numpy/pandas alone.

Author: [Redacted for double-blind review]
Version: 2.3.0 (camera-ready)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("aegis.data")

N_NODE_FEATURES = 32          # fixed; matches model.input_dim and is uniform across chains
DEFAULT_MAX_EDGE_FEATURES = 8 # common edge-attr width (Bitcoin-L has 7 after dropping ts)
SECONDS_PER_DAY = 86_400


# ===========================================================================
# Dataset registry  (single source of truth for provenance + schema)
# ===========================================================================

@dataclass(frozen=True)
class RefStats:
    """Published statistics used to validate a loaded graph (DIAM Table 1)."""
    nodes: int
    edges: int
    illicit: int
    normal: int


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to load and interpret one dataset."""
    family: str                       # "ethereum" | "bitcoin" | "bsc"
    source: str                       # "npz" | "parquet"
    label_source: str                 # human-readable provenance (goes into metadata)
    lowercase_addresses: bool         # ETH/BSC hex => True; Bitcoin base58 => False
    # npz path -----------------------------------------------------------------
    npz_file: Optional[str] = None    # e.g. "EthereumS_graph_dict.npz"
    n_edge_attr: Optional[int] = None # expected #columns of edge_attr in the npz
    amount_col: Optional[int] = None  # column of edge_attr holding the transfer amount
    fee_col: Optional[int] = None     # column holding the fee, or None if absent
    ts_col: int = -1                  # column holding the (absolute) timestamp
    expected: Optional[RefStats] = None
    # parquet path -------------------------------------------------------------
    parquet_subdir: Optional[str] = None
    labels_file: Optional[str] = None


# DIAM Table 1 statistics are exact; a mismatch after loading means the wrong /
# corrupted file was downloaded, so we validate against these.
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "ethereum_s": DatasetSpec(
        family="ethereum", source="npz", lowercase_addresses=True,
        label_source="Etherscan/XBlock phishing labels (Yuan et al., 2020)",
        npz_file="EthereumS_graph_dict.npz", n_edge_attr=2,
        amount_col=0, fee_col=None, ts_col=-1,
        expected=RefStats(nodes=1_329_729, edges=6_794_521, illicit=1_660, normal=1_700),
    ),
    "ethereum_p": DatasetSpec(
        family="ethereum", source="npz", lowercase_addresses=True,
        label_source="Etherscan phishing (illicit) + licit services (Chen et al., 2020)",
        npz_file="EthereumP_graph_dict.npz", n_edge_attr=2,
        amount_col=0, fee_col=None, ts_col=-1,
        expected=RefStats(nodes=2_973_489, edges=13_551_303, illicit=1_165, normal=3_418),
    ),
    "bitcoin_m": DatasetSpec(
        family="bitcoin", source="npz", lowercase_addresses=False,
        label_source="WalletExplorer; gambling/mixing = illicit (Wu et al., 2021)",
        npz_file="BitcoinM_graph_dict.npz", n_edge_attr=5,
        amount_col=1, fee_col=None, ts_col=-1,   # [in_amt, out_amt, n_in, n_out, ts]
        expected=RefStats(nodes=2_505_841, edges=14_181_316, illicit=46_930, normal=213_026),
    ),
    "bitcoin_l": DatasetSpec(
        family="bitcoin", source="npz", lowercase_addresses=False,
        label_source="WalletExplorer labels (Ding et al., 2024, DIAM)",
        npz_file="BitcoinL_graph_dict.npz", n_edge_attr=8,
        # [in_amt, out_amt, n_in, n_out, fee, total_in, total_out, ts]
        amount_col=1, fee_col=4, ts_col=-1,
        expected=RefStats(nodes=20_085_231, edges=203_419_765, illicit=362_391, normal=1_271_556),
    ),
    "bsc": DatasetSpec(
        family="bsc", source="parquet", lowercase_addresses=True,
        label_source="BSCScan API exports; community-reported cross-chain fraud",
        parquet_subdir="bsc", labels_file="labels/bsc_community_reports.csv",
        expected=None,  # no public ground truth to validate against
    ),
}

# Bare family names are ambiguous (two Ethereum and two Bitcoin datasets), so we
# refuse them and ask for an explicit dataset id -- the paper reports five
# distinct datasets and must not silently collapse them.
_FAMILY_ALIASES = {"ethereum", "bitcoin"}


def resolve_dataset(name: str) -> Tuple[str, DatasetSpec]:
    """Map a dataset identifier to its spec, with a helpful error otherwise."""
    key = name.strip().lower().replace("-", "_")
    if key in DATASET_REGISTRY:
        return key, DATASET_REGISTRY[key]
    if key in _FAMILY_ALIASES:
        raise ValueError(
            f"'{name}' is ambiguous -- there are two {key} datasets. "
            f"Use an explicit id: "
            f"{[k for k, s in DATASET_REGISTRY.items() if s.family == key]}."
        )
    raise ValueError(
        f"Unknown dataset '{name}'. Valid ids: {sorted(DATASET_REGISTRY)}."
    )


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class DataConfig:
    """Configuration for the data pipeline (aligned with config.yaml)."""

    data_root: str = "./data"

    # Subdirectories holding raw parquet exports (parquet path only).
    bitcoin_subdir: str = "bitcoin"
    ethereum_subdir: str = "ethereum"
    bsc_subdir: str = "bsc"

    # External ground-truth label files (parquet path only; npz carries y).
    # CSV columns: address,label[,category]  (label: 0=legit, 1=illicit).
    bitcoin_labels_file: str = "labels/bitcoin_walletexplorer.csv"
    ethereum_labels_file: str = "labels/ethereum_etherscan_phishing.csv"
    bsc_labels_file: str = "labels/bsc_community_reports.csv"

    # Optional per-dataset fraud-category files (CSV: node_id|address,category)
    # required ONLY for the zero-day holdout experiment.
    categories_files: Dict[str, str] = field(default_factory=dict)

    # Feature dimensions.
    max_node_features: int = N_NODE_FEATURES
    max_edge_features: int = DEFAULT_MAX_EDGE_FEATURES
    use_provided_node_features: bool = False  # use DIAM's X (per-dataset dim) instead of our 32

    # Graph views to attempt; "contract" is realized only if real contract data exists.
    graph_types: List[str] = field(default_factory=lambda: ["tx", "contract"])

    # Splitting.
    temporal_split: bool = True
    train_ratio: float = 0.50
    val_ratio: float = 0.25
    test_ratio: float = 0.25

    # Which datasets to operate on (use registry ids, NOT bare families).
    datasets: List[str] = field(default_factory=lambda: [
        "ethereum_s", "ethereum_p", "bitcoin_m", "bitcoin_l", "bsc",
    ])

    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    chunk_size: int = 100_000
    random_seed: int = 42

    validate_reference: bool = True  # check loaded npz against DIAM Table 1

    # Optional npz file-name overrides keyed by dataset id.
    npz_files: Dict[str, str] = field(default_factory=dict)

    # ---- path helpers --------------------------------------------------------
    def npz_path(self, dataset: str, spec: DatasetSpec) -> Path:
        fname = self.npz_files.get(dataset, spec.npz_file)
        return Path(self.data_root) / fname

    def parquet_dir(self, spec: DatasetSpec) -> Path:
        sub = spec.parquet_subdir or getattr(self, f"{spec.family}_subdir")
        return Path(self.data_root) / sub

    def label_path(self, spec: DatasetSpec) -> Path:
        rel = spec.labels_file or getattr(self, f"{spec.family}_labels_file")
        return Path(self.data_root) / rel


# ===========================================================================
# Small helpers
# ===========================================================================

_BAD_ADDR = {"", "nan", "none", "null", "<na>"}


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _resolve_timestamp_column(edge_attr: np.ndarray, declared_idx: int) -> int:
    """Return the edge_attr column that actually holds Unix timestamps.

    DIAM stores the timestamp as the last edge-attribute column, but rather than
    trust that blindly we verify the declared column has values in a plausible
    Unix-seconds range (the four DIAM datasets are from 2015-2019, ~1.4e9-1.6e9).
    If it does not, we look for a column that does and warn. This guards against
    a silently mis-ordered file feeding garbage to the temporal split/encoder.
    """
    n_cols = edge_attr.shape[1]
    declared = declared_idx % n_cols

    def plausible(col: int) -> bool:
        finite = edge_attr[np.isfinite(edge_attr[:, col]), col]
        if finite.size == 0:
            return False
        return 5.0e8 <= float(np.median(finite)) <= 2.2e9

    if plausible(declared):
        return declared
    for c in range(n_cols):
        if plausible(c):
            logger.warning(
                "Declared timestamp column %d is not in Unix-seconds range; "
                "using detected column %d instead.", declared, c)
            return c
    logger.warning(
        "No edge-attribute column looks like a Unix timestamp (declared %d). "
        "Temporal splitting and the temporal encoder may be unreliable -- "
        "verify the edge_attr schema for this dataset.", declared)
    return declared


def _validate_reference(dataset: str, spec: DatasetSpec,
                        num_nodes: int, num_edges: int,
                        labels: np.ndarray) -> None:
    """Warn loudly if a loaded graph does not match its published statistics."""
    if spec.expected is None:
        return
    exp = spec.expected
    got = {
        "nodes": int(num_nodes), "edges": int(num_edges),
        "illicit": int((labels == 1).sum()), "normal": int((labels == 0).sum()),
    }
    want = {"nodes": exp.nodes, "edges": exp.edges,
            "illicit": exp.illicit, "normal": exp.normal}
    mismatches = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if mismatches:
        details = ", ".join(f"{k}: got {g}, expected {w} (Δ{g - w:+d})"
                            for k, (g, w) in mismatches.items())
        logger.warning(
            "[%s] loaded graph does NOT match DIAM Table 1 (%s). Verify you "
            "downloaded the official graph_dict; reported numbers must match "
            "the dataset you actually evaluate on.", dataset, details)
    else:
        logger.info("[%s] graph statistics match DIAM Table 1 exactly.", dataset)


# ===========================================================================
# Ground-truth label loading (parquet path; npz carries y directly)
# ===========================================================================

def load_ground_truth_labels(
    label_path: Path,
    address_to_id: Dict[str, int],
    num_nodes: int,
    lowercase: bool = True,
) -> Tuple[np.ndarray, Optional[Dict[int, str]]]:
    """Load externally verified labels; every unlisted address is UNKNOWN (-1).

    `lowercase` MUST mirror the normalization used to build `address_to_id`:
    True for Ethereum/BSC (hex; EIP-55 casing is a checksum, not identity),
    False for Bitcoin (base58 is case-sensitive -- lowercasing merges addresses).
    """
    labels = np.full(num_nodes, -1, dtype=np.int64)
    categories: Optional[Dict[int, str]] = None

    if not label_path.exists():
        raise FileNotFoundError(
            f"Ground-truth label file not found: {label_path}. Labels MUST come "
            f"from verified external sources (WalletExplorer for Bitcoin, "
            f"Etherscan/XBlock for Ethereum, community reports for BSC). Refusing "
            f"to proceed without real labels -- heuristic labels (e.g. top-volume "
            f"'whale' addresses) measure a different task and are not comparable.")

    df = pd.read_csv(label_path, dtype={"address": str})
    df["address"] = df["address"].str.strip()
    if lowercase:
        df["address"] = df["address"].str.lower()
    df["label"] = df["label"].astype(int)

    has_cat = "category" in df.columns
    if has_cat:
        categories = {}

    df["node_id"] = df["address"].map(address_to_id)
    matched = df.dropna(subset=["node_id"])
    node_ids = matched["node_id"].astype(int).to_numpy()
    labels[node_ids] = matched["label"].to_numpy()
    if has_cat:
        for nid, cat in zip(node_ids, matched["category"].astype(str)):
            categories[int(nid)] = cat

    n_match, n_total = len(matched), len(df)
    logger.info("Labels from %s: matched %d/%d addresses (%d illicit, %d legit, "
                "%d unknown of %d nodes).", label_path.name, n_match, n_total,
                int((labels == 1).sum()), int((labels == 0).sum()),
                int((labels == -1).sum()), num_nodes)
    if n_match < 0.5 * max(n_total, 1):
        logger.warning("Fewer than half of label-file addresses matched the "
                       "graph (%d/%d) -- check address normalization (case, "
                       "prefixes) and the transaction time window.", n_match, n_total)
    return labels, categories


# ===========================================================================
# Node feature extraction (32 features; fully vectorized, path-agnostic)
# ===========================================================================

def extract_node_features(
    num_nodes: int,
    edge_index: np.ndarray,    # [2, E] int
    amount: np.ndarray,        # [E] float  (per-edge transfer value)
    fee: np.ndarray,           # [E] float  (per-edge fee; zeros if unavailable)
    timestamp: np.ndarray,     # [E] float  (per-edge absolute Unix seconds)
    name: str = "",
) -> np.ndarray:
    """Compute 32 node features in O(E log E) with no per-node Python loop.

      [0] in-degree            [1] out-degree           [2] total degree
      [3] total received       [4] total sent
      [5] mean received        [6] mean sent
      [7] std received         [8] std sent
      [9] total fee paid       [10] mean fee paid
     [11] activity span (s)    [12] mean inter-tx time  [13] std inter-tx time
     [14] unique in-counterparties   [15] unique out-counterparties
     [16] in/out degree ratio  [17] sent/(sent+received) value ratio
     [18-31] 14-bin temporal activity histogram

    These are OUR features computed from public edge attributes -- NOT Elliptic's
    166 proprietary features. Returns UNNORMALIZED float64; normalization is
    applied later with a scaler fitted on training nodes only (leakage-safe).
    """
    f = np.zeros((num_nodes, N_NODE_FEATURES), dtype=np.float64)
    if edge_index.size == 0:
        logger.info("Extracted %d node features for %d nodes (%s): graph has no "
                    "edges.", N_NODE_FEATURES, num_nodes, name)
        return f

    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    amount = np.nan_to_num(np.asarray(amount, dtype=np.float64))
    fee = np.nan_to_num(np.asarray(fee, dtype=np.float64))
    timestamp = np.nan_to_num(np.asarray(timestamp, dtype=np.float64))

    # Degrees
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float64)
    f[:, 0], f[:, 1], f[:, 2] = in_deg, out_deg, in_deg + out_deg

    # Amount sums / means / stds via E[X^2] - E[X]^2
    in_sum = np.bincount(dst, weights=amount, minlength=num_nodes)
    out_sum = np.bincount(src, weights=amount, minlength=num_nodes)
    in_sq = np.bincount(dst, weights=amount ** 2, minlength=num_nodes)
    out_sq = np.bincount(src, weights=amount ** 2, minlength=num_nodes)
    f[:, 3], f[:, 4] = in_sum, out_sum
    with np.errstate(invalid="ignore", divide="ignore"):
        in_mean = np.where(in_deg > 0, in_sum / np.maximum(in_deg, 1), 0.0)
        out_mean = np.where(out_deg > 0, out_sum / np.maximum(out_deg, 1), 0.0)
        in_var = np.where(in_deg > 1, in_sq / np.maximum(in_deg, 1) - in_mean ** 2, 0.0)
        out_var = np.where(out_deg > 1, out_sq / np.maximum(out_deg, 1) - out_mean ** 2, 0.0)
    f[:, 5], f[:, 6] = in_mean, out_mean
    f[:, 7] = np.sqrt(np.clip(in_var, 0, None))
    f[:, 8] = np.sqrt(np.clip(out_var, 0, None))

    # Fees (paid by the sender)
    fee_sum = np.bincount(src, weights=fee, minlength=num_nodes)
    f[:, 9] = fee_sum
    f[:, 10] = np.where(out_deg > 0, fee_sum / np.maximum(out_deg, 1), 0.0)

    # Activity span + inter-transaction-time statistics over each node's incident
    # edges (both directions), via sorted segments.
    inc_nodes = np.concatenate([src, dst])
    inc_ts = np.concatenate([timestamp, timestamp])
    order = np.lexsort((inc_ts, inc_nodes))
    s_nodes, s_ts = inc_nodes[order], inc_ts[order]
    seg_start = np.r_[True, s_nodes[1:] != s_nodes[:-1]]
    seg_nodes = s_nodes[seg_start]
    first_ts = s_ts[seg_start]
    last_ts = s_ts[np.r_[seg_start[1:], True]]
    f[seg_nodes, 11] = last_ts - first_ts

    diffs = np.diff(s_ts)
    same = s_nodes[1:] == s_nodes[:-1]
    d_nodes = s_nodes[1:][same]
    d_vals = diffs[same]
    d_sum = np.bincount(d_nodes, weights=d_vals, minlength=num_nodes)
    d_sq = np.bincount(d_nodes, weights=d_vals ** 2, minlength=num_nodes)
    d_cnt = np.bincount(d_nodes, minlength=num_nodes).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_mean = np.where(d_cnt > 0, d_sum / np.maximum(d_cnt, 1), 0.0)
        d_var = np.where(d_cnt > 1, d_sq / np.maximum(d_cnt, 1) - d_mean ** 2, 0.0)
    f[:, 12] = d_mean
    f[:, 13] = np.sqrt(np.clip(d_var, 0, None))

    # Unique counterparties (drop duplicate (node, counterparty) pairs)
    in_pairs = np.unique(np.stack([dst, src], axis=1), axis=0)
    out_pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    f[:, 14] = np.bincount(in_pairs[:, 0], minlength=num_nodes)
    f[:, 15] = np.bincount(out_pairs[:, 0], minlength=num_nodes)

    # Ratios
    f[:, 16] = in_deg / np.maximum(in_deg + out_deg, 1)
    f[:, 17] = out_sum / np.maximum(in_sum + out_sum, 1e-10)

    # 14-bin temporal histogram over the global time range
    t_min, t_max = float(timestamp.min()), float(timestamp.max())
    t_range = max(t_max - t_min, 1.0)
    bins = np.minimum(((inc_ts - t_min) / t_range * 14).astype(np.int64), 13)
    flat = inc_nodes * 14 + bins
    hist = np.bincount(flat, minlength=num_nodes * 14)
    f[:, 18:32] = hist.reshape(num_nodes, 14)

    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info("Extracted %d node features for %d nodes (%s), vectorized.",
                N_NODE_FEATURES, num_nodes, name)
    return f


# ===========================================================================
# Leakage-safe normalization
# ===========================================================================

def normalize_train_fitted(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """RobustScaler fitted on rows where fit_mask is True, applied to all rows.

    Fitting on the full array would leak validation/test statistics into
    preprocessing (Reviewer 4). RobustScaler maps zero-IQR columns to scale 1.0,
    so there is no division by zero; nan_to_num is a final guard.
    """
    scaler = RobustScaler()
    fit_rows = values[fit_mask] if np.asarray(fit_mask).any() else values
    scaler.fit(fit_rows)
    out = scaler.transform(values)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ===========================================================================
# DIAM .npz loader  (primary path for the four cryptocurrency benchmarks)
# ===========================================================================

def load_diam_npz(config: DataConfig, dataset: str, spec: DatasetSpec) -> Dict[str, Any]:
    """Load a DIAM `*_graph_dict.npz` and prepare a chain bundle.

    The npz stores: edge_index [2, E], edge_attr [E, D], X [N, F], y [N] in
    {-1, 0, 1}. We keep the published edge_index and y verbatim (y *is* the
    external WalletExplorer/Etherscan ground truth), split the timestamp out of
    edge_attr, build the model edge attributes (schema minus timestamp, plus
    log1p(amount)), and recompute our 32 uniform node features.
    """
    path = config.npz_path(dataset, spec)
    if not path.exists():
        raise FileNotFoundError(
            f"DIAM graph file not found: {path}. Download the official datasets "
            f"from https://huggingface.co/datasets/Tommy-DING/"
            f"crypto-illicit-account-detection-multigraphs (file '{spec.npz_file}') "
            f"or the authors' SharePoint, and place them under '{config.data_root}'.")

    z = np.load(path, allow_pickle=False)
    for key in ("edge_index", "edge_attr", "y"):
        if key not in z:
            raise KeyError(f"{path.name} is missing required array '{key}'.")

    edge_index = np.ascontiguousarray(z["edge_index"]).astype(np.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    edge_attr = np.asarray(z["edge_attr"], dtype=np.float64)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr[:, None]
    y = np.asarray(z["y"]).astype(np.int64).reshape(-1)

    num_nodes = int(max(int(edge_index.max()) + 1 if edge_index.size else 0, y.shape[0]))
    num_edges = int(edge_index.shape[1])

    if spec.n_edge_attr is not None and edge_attr.shape[1] != spec.n_edge_attr:
        logger.warning("[%s] edge_attr has %d columns but the DIAM schema expects "
                       "%d; column-to-semantic mapping may be off.",
                       dataset, edge_attr.shape[1], spec.n_edge_attr)

    # Identify and split off the absolute timestamp.
    ts_col = _resolve_timestamp_column(edge_attr, spec.ts_col)
    timestamps = edge_attr[:, ts_col].astype(np.float64)
    keep_cols = [c for c in range(edge_attr.shape[1]) if c != ts_col]
    edge_attr_no_ts = edge_attr[:, keep_cols]

    # Map raw amount / fee for feature extraction (account for the dropped ts col).
    def _shift(idx: Optional[int]) -> Optional[int]:
        if idx is None:
            return None
        idx = idx % edge_attr.shape[1]
        return idx if idx < ts_col else idx - 1
    amount_idx, fee_idx = _shift(spec.amount_col), _shift(spec.fee_col)
    amount = (edge_attr_no_ts[:, amount_idx] if amount_idx is not None
              and amount_idx < edge_attr_no_ts.shape[1]
              else edge_attr_no_ts[:, 0] if edge_attr_no_ts.shape[1] else np.zeros(num_edges))
    fee = (edge_attr_no_ts[:, fee_idx] if fee_idx is not None
           and fee_idx < edge_attr_no_ts.shape[1] else np.zeros(num_edges))

    # Model edge attributes: schema (minus ts) + log1p(amount).
    log_amt = np.log1p(np.clip(amount, 0, None))[:, None]
    edge_attrs_model = np.concatenate([edge_attr_no_ts, log_amt], axis=1)

    _validate_reference(dataset, spec, num_nodes, num_edges, y)

    if config.use_provided_node_features and "X" in z:
        node_features = np.asarray(z["X"], dtype=np.float64)
        logger.warning("[%s] using DIAM-provided node features X with dim %d "
                       "(cross-chain transfer requires a uniform dim and will "
                       "not work unless every dataset matches).",
                       dataset, node_features.shape[1])
    else:
        node_features = extract_node_features(
            num_nodes, edge_index, amount, fee, timestamps, name=dataset)

    # Optional fraud-category mapping (node_id-keyed) for the zero-day experiment.
    categories = _load_categories(config, dataset, num_nodes)

    views = {"tx": {"edge_index": edge_index,
                    "edge_attrs": edge_attrs_model,
                    "timestamps": timestamps}}
    return {
        "node_features": node_features, "labels": y, "categories": categories,
        "views": views, "num_nodes": num_nodes, "num_edges": num_edges,
        "contract_mask": None,
    }


def _load_categories(config: DataConfig, dataset: str,
                     num_nodes: int) -> Optional[Dict[int, str]]:
    """Load a node_id->category map from an optional CSV (node_id,category)."""
    rel = config.categories_files.get(dataset)
    if not rel:
        return None
    path = Path(config.data_root) / rel
    if not path.exists():
        logger.warning("[%s] categories file '%s' not found; zero-day holdout "
                       "will be unavailable for this dataset.", dataset, path)
        return None
    df = pd.read_csv(path)
    id_col = "node_id" if "node_id" in df.columns else df.columns[0]
    cats = {int(n): str(c) for n, c in zip(df[id_col], df["category"])
            if 0 <= int(n) < num_nodes}
    logger.info("[%s] loaded %d fraud-category annotations.", dataset, len(cats))
    return cats


# ===========================================================================
# Raw parquet builders  (BSC, and any chain rebuilt from transactions)
# ===========================================================================

def _read_parquet(data_dir: Path) -> pd.DataFrame:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Place raw "
                                f"transaction parquet files there.")
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}.")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    logger.info("Loaded %d transactions from %d parquet files (%s).",
                len(df), len(files), data_dir)
    return df


def construct_bitcoin_graph(transactions: pd.DataFrame, label_path: Path) -> Dict[str, Any]:
    """Directed Bitcoin ADDRESS graph (DIAM protocol): each transaction with |S|
    senders and |R| receivers expands to |S| x |R| directed edges. Vectorized via
    pandas `explode` (no per-row iterrows). Edge attributes (7, DIAM schema minus
    absolute timestamp): [amount, fee, n_inputs, n_outputs, total_input,
    total_output, log1p(amount)].
    """
    tx = transactions.copy()
    has_multi = "senders" in tx.columns

    for col, default in [("fee", 0.0), ("n_inputs", 1), ("n_outputs", 1)]:
        if col not in tx.columns:
            tx[col] = default
    if "total_input" not in tx.columns:
        tx["total_input"] = tx.get("amount", 0.0)
    if "total_output" not in tx.columns:
        tx["total_output"] = tx.get("amount", 0.0)

    if has_multi:
        # base58 is CASE-SENSITIVE: strip whitespace only, never lowercase.
        tx["senders"] = tx["senders"].astype(str).str.strip().str.split(";")
        tx["receivers"] = tx["receivers"].astype(str).str.strip().str.split(";")
        if "amounts" in tx.columns:
            amounts_series = tx["amounts"].astype(str).str.strip().str.split(";")
        else:
            amounts_series = pd.Series([None] * len(tx), index=tx.index)

        def _pair(receivers, amts, total_out):
            n = len(receivers)
            if amts is None or len(amts) != n:
                per = float(_to_float(total_out)) / max(n, 1)
                vals = [per] * n
            else:
                vals = [_to_float(a) for a in amts]
            return list(zip(receivers, vals))

        tx["_recv"] = [_pair(r, a, t) for r, a, t
                       in zip(tx["receivers"], amounts_series, tx["total_output"])]
        cols = ["senders", "_recv", "fee", "timestamp",
                "n_inputs", "n_outputs", "total_input", "total_output"]
        e = (tx[cols].explode("senders").explode("_recv")
             .dropna(subset=["senders", "_recv"]))
        e["sender"] = e["senders"].astype(str).str.strip()
        e["receiver"] = e["_recv"].map(lambda p: str(p[0]).strip())
        e["amount"] = e["_recv"].map(lambda p: float(p[1]))
        edges = e.drop(columns=["senders", "_recv"])
    else:
        edges = tx.copy()
        edges["sender"] = edges["sender"].astype(str).str.strip()
        edges["receiver"] = edges["receiver"].astype(str).str.strip()
        if "amount" not in edges.columns:
            edges["amount"] = 0.0

    edges = edges[~edges["sender"].str.lower().isin(_BAD_ADDR)
                  & ~edges["receiver"].str.lower().isin(_BAD_ADDR)]
    logger.info("Bitcoin graph: %d transactions -> %d directed edges.",
                len(transactions), len(edges))
    return _finalize_graph(
        edges, label_path,
        edge_attr_cols=["amount", "fee", "n_inputs", "n_outputs",
                        "total_input", "total_output"],
        contract_mask=None, lowercase_labels=False)


def construct_ethereum_graph(transactions: pd.DataFrame, label_path: Path,
                             include_contracts: bool = True) -> Dict[str, Any]:
    """Directed account graph for Ethereum / BSC (one edge per transaction). Edge
    attributes (6, schema minus timestamp): [amount, fee, gas_used, gas_price,
    status, log1p(amount)]. Hex addresses are lowercased (EIP-55 is a checksum).
    """
    tx = transactions.copy()
    tx["sender"] = tx["sender"].astype(str).str.strip().str.lower()
    tx["receiver"] = tx["receiver"].astype(str).str.strip().str.lower()
    if "amount" not in tx.columns:
        tx["amount"] = 0.0
    for col, default in [("fee", 0.0), ("gas_used", 0.0),
                         ("gas_price", 0.0), ("status", 1.0)]:
        if col not in tx.columns:
            tx[col] = default

    tx = tx[~tx["sender"].isin(_BAD_ADDR) & ~tx["receiver"].isin(_BAD_ADDR)]

    contract_addrs: Optional[set] = None
    if include_contracts and "contract_address" in tx.columns:
        contract_addrs = {str(c).strip().lower()
                          for c in tx["contract_address"].dropna().unique()}

    return _finalize_graph(
        tx, label_path,
        edge_attr_cols=["amount", "fee", "gas_used", "gas_price", "status"],
        contract_mask=contract_addrs, lowercase_labels=True)


def _finalize_graph(edges: pd.DataFrame, label_path: Path,
                    edge_attr_cols: List[str], contract_mask: Optional[set],
                    lowercase_labels: bool) -> Dict[str, Any]:
    """Shared finalization for the parquet path: node mapping, arrays, labels."""
    all_addresses = pd.unique(pd.concat([edges["sender"], edges["receiver"]]))
    address_to_id = {a: i for i, a in enumerate(all_addresses)}
    num_nodes = len(all_addresses)

    src = edges["sender"].map(address_to_id).to_numpy(dtype=np.int64)
    dst = edges["receiver"].map(address_to_id).to_numpy(dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)

    amount = edges["amount"].astype(float).to_numpy()
    fee = (edges["fee"].astype(float).to_numpy() if "fee" in edges.columns
           else np.zeros(len(edges)))
    attr_arrays = [edges[c].astype(float).to_numpy() for c in edge_attr_cols]
    attr_arrays.append(np.log1p(np.clip(amount, 0, None)))
    edge_attrs = np.column_stack(attr_arrays)
    timestamps = edges["timestamp"].astype(float).to_numpy()

    labels, categories = load_ground_truth_labels(
        label_path, address_to_id, num_nodes, lowercase=lowercase_labels)

    cmask = None
    if contract_mask:
        cmask = np.zeros(num_nodes, dtype=bool)
        for ca in contract_mask:
            if ca in address_to_id:
                cmask[address_to_id[ca]] = True

    return {
        "edge_index": edge_index, "edge_attrs": edge_attrs, "timestamps": timestamps,
        "amount_raw": amount, "fee_raw": fee, "labels": labels,
        "categories": categories, "num_nodes": num_nodes, "contract_mask": cmask,
    }


def load_parquet_chain(config: DataConfig, dataset: str, spec: DatasetSpec) -> Dict[str, Any]:
    """Build a chain bundle from raw parquet transactions + external labels."""
    tx_df = _read_parquet(config.parquet_dir(spec))
    label_path = config.label_path(spec)
    if spec.family == "bitcoin":
        g = construct_bitcoin_graph(tx_df, label_path)
    else:  # ethereum / bsc account graph
        g = construct_ethereum_graph(tx_df, label_path, include_contracts=True)

    node_features = extract_node_features(
        g["num_nodes"], g["edge_index"], g["amount_raw"], g["fee_raw"],
        g["timestamps"], name=dataset)
    views = {"tx": {"edge_index": g["edge_index"],
                    "edge_attrs": g["edge_attrs"],
                    "timestamps": g["timestamps"]}}
    bundle = {
        "node_features": node_features, "labels": g["labels"],
        "categories": g["categories"], "views": views,
        "num_nodes": g["num_nodes"], "num_edges": int(g["edge_index"].shape[1]),
        "contract_mask": g["contract_mask"],
    }
    _add_contract_view(bundle, config)
    return bundle


def _add_contract_view(bundle: Dict[str, Any], config: DataConfig) -> None:
    """Realize a 'contract' view ONLY if genuine contract data exists. The
    pipeline never duplicates the tx view -- a duplicated view would inflate the
    multi-graph claim without adding information.
    """
    if "contract" not in config.graph_types:
        return
    cm = bundle.get("contract_mask")
    tx = bundle["views"]["tx"]
    if cm is not None and cm.any():
        src, dst = tx["edge_index"]
        mask = cm[src] | cm[dst]
        if mask.sum() > 0:
            bundle["views"]["contract"] = {
                "edge_index": tx["edge_index"][:, mask],
                "edge_attrs": tx["edge_attrs"][mask],
                "timestamps": tx["timestamps"][mask],
            }
            logger.info("Contract view: %d/%d edges.", int(mask.sum()), mask.size)
            return
    logger.warning("No genuine contract data; running with K=1 (tx view only). "
                   "The model's num_graph_types must match the available views.")


# ===========================================================================
# Splitting  (labeled nodes only)
# ===========================================================================

def temporal_node_split(edge_index: np.ndarray, timestamps: np.ndarray,
                        labels: np.ndarray, train_ratio: float = 0.50,
                        val_ratio: float = 0.25
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Assign each LABELED node to a split by its earliest incident-edge time.

    ETGraph-style: first `train_ratio` of the time range trains, next `val_ratio`
    validates, the rest tests. CAVEAT (state in the paper): assignment by
    earliest activity skews val/test toward addresses that first appear late;
    prepare_dataset warns if any split is empty or has no positives.
    """
    num_nodes = len(labels)
    node_min_ts = np.full(num_nodes, np.inf)
    np.minimum.at(node_min_ts, edge_index[0], timestamps)
    np.minimum.at(node_min_ts, edge_index[1], timestamps)
    finite = timestamps[np.isfinite(timestamps)]
    t_max = float(finite.max()) if finite.size else 0.0
    node_min_ts[np.isinf(node_min_ts)] = t_max + 1.0

    t_min = float(finite.min()) if finite.size else 0.0
    t_range = max(t_max - t_min, 1.0)
    train_cut = t_min + t_range * train_ratio
    val_cut = t_min + t_range * (train_ratio + val_ratio)

    labeled = labels >= 0
    train_idx = np.where(labeled & (node_min_ts <= train_cut))[0]
    val_idx = np.where(labeled & (node_min_ts > train_cut) & (node_min_ts <= val_cut))[0]
    test_idx = np.where(labeled & (node_min_ts > val_cut))[0]
    return train_idx, val_idx, test_idx, train_cut


def random_node_split(labels: np.ndarray, train_ratio: float = 0.50,
                      val_ratio: float = 0.25, seed: int = 42
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DIAM-style random 2:1:1 split over LABELED nodes, stratified by class so
    every split contains positives."""
    labeled_idx = np.where(labels >= 0)[0]
    strat = labels[labeled_idx]
    test_ratio = max(1.0 - train_ratio - val_ratio, 1e-6)
    train_idx, temp_idx = train_test_split(
        labeled_idx, test_size=val_ratio + test_ratio,
        random_state=seed, shuffle=True, stratify=strat)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed, shuffle=True, stratify=labels[temp_idx])
    return train_idx, val_idx, test_idx


# ===========================================================================
# Dataset assembly  (split + leakage-safe normalization + tensorization)
# ===========================================================================

def prepare_dataset(config: DataConfig, dataset: str,
                    holdout_category: Optional[str] = None):
    """Load, split, normalize (train-fitted), and tensorize one dataset.

    Returns
    -------
    data        : PyG Data with x, y, train/val/test masks (labeled nodes only).
    view_tensors: list of dicts per view, keys "edge_index", "edge_timestamp",
                  "edge_attr" (order == metadata["view_names"]).
    metadata    : statistics, split info, provenance.
    """
    import torch                                  # lazy: keep construction torch-free
    from torch_geometric.data import Data

    dataset_id, spec = resolve_dataset(dataset)
    if config.max_node_features != N_NODE_FEATURES and not config.use_provided_node_features:
        logger.warning("config.max_node_features=%d but the extractor produces %d; "
                       "the model input_dim must equal the feature count.",
                       config.max_node_features, N_NODE_FEATURES)

    if spec.source == "npz":
        bundle = load_diam_npz(config, dataset_id, spec)
    else:
        bundle = load_parquet_chain(config, dataset_id, spec)

    labels = np.asarray(bundle["labels"], dtype=np.int64).copy()
    categories = bundle["categories"]
    views = bundle["views"]
    num_nodes = bundle["num_nodes"]
    tx_view = views["tx"]
    view_names = list(views.keys())

    # ---- Zero-day holdout (real; never a placeholder) ----
    holdout_nodes = None
    labels_for_split = labels
    if holdout_category is not None:
        if not categories:
            raise ValueError(
                f"Zero-day holdout for '{holdout_category}' requires a fraud "
                f"category per node (config.categories_files['{dataset_id}']). "
                f"Without it the experiment cannot be run honestly -- refusing to "
                f"emit placeholder numbers.")
        holdout_nodes = np.array(
            [nid for nid, cat in categories.items()
             if cat == holdout_category and 0 <= nid < num_nodes and labels[nid] == 1],
            dtype=np.int64)
        if holdout_nodes.size == 0:
            raise ValueError(f"No illicit nodes with category '{holdout_category}'.")
        labels_for_split = labels.copy()
        labels_for_split[holdout_nodes] = -1   # hide from the ordinary split

    # ---- Split over labeled nodes ----
    if config.temporal_split:
        train_idx, val_idx, test_idx, train_cut = temporal_node_split(
            tx_view["edge_index"], tx_view["timestamps"], labels_for_split,
            config.train_ratio, config.val_ratio)
        tx_fit = tx_view["timestamps"] <= train_cut
        split_method = "temporal"
    else:
        train_cut = None
        train_idx, val_idx, test_idx = random_node_split(
            labels_for_split, config.train_ratio, config.val_ratio, config.random_seed)
        train_set = np.zeros(num_nodes, dtype=bool)
        train_set[train_idx] = True
        s, d = tx_view["edge_index"]
        tx_fit = train_set[s] & train_set[d]
        if not tx_fit.any():
            tx_fit = train_set[s] | train_set[d]
        split_method = "random_stratified"

    if holdout_nodes is not None:
        # Zero-day test set: held-out illicit nodes + ordinary legitimate test
        # nodes (so precision remains measurable).
        legit_test = test_idx[labels[test_idx] == 0] if test_idx.size else test_idx
        test_idx = np.unique(np.concatenate([holdout_nodes, legit_test]))

    # ---- Leakage-safe normalization (fit on training data only) ----
    node_fit = np.zeros(num_nodes, dtype=bool)
    node_fit[train_idx] = True
    x_norm = normalize_train_fitted(bundle["node_features"], node_fit)

    view_tensors: List[Dict[str, "torch.Tensor"]] = []
    for name in view_names:
        v = views[name]
        if name == "tx":
            v_fit = tx_fit
        elif config.temporal_split:
            v_fit = v["timestamps"] <= train_cut
        else:
            vs, vd = v["edge_index"]
            v_fit = node_fit[vs] & node_fit[vd]
        if not np.asarray(v_fit).any():
            logger.warning("View '%s': no edges inside the training window/"
                           "subgraph; fitting its scaler on ALL of this view's "
                           "edges (minor preprocessing leakage, this view only).",
                           name)
            v_fit = np.ones(len(v["timestamps"]), dtype=bool)

        e_norm = normalize_train_fitted(v["edge_attrs"], v_fit)
        # Pad to a common edge-attr width so weights transfer across chains.
        width = config.max_edge_features
        if e_norm.shape[1] < width:
            pad = np.zeros((e_norm.shape[0], width - e_norm.shape[1]), dtype=e_norm.dtype)
            e_norm = np.concatenate([e_norm, pad], axis=1)
        elif e_norm.shape[1] > width:
            raise ValueError(
                f"View '{name}' has {e_norm.shape[1]} edge attributes > "
                f"max_edge_features={width}. Raise the config ceiling.")
        view_tensors.append({
            "edge_index": torch.from_numpy(np.ascontiguousarray(v["edge_index"])).long(),
            # float64: absolute Unix seconds lose ~128 s of precision in float32;
            # the model casts the (small) per-edge deltas down itself.
            "edge_timestamp": torch.from_numpy(v["timestamps"].astype(np.float64)),
            "edge_attr": torch.from_numpy(e_norm),
        })

    # ---- Masks (labeled nodes only) ----
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=torch.from_numpy(x_norm),
                y=torch.from_numpy(labels.astype(np.int64)), num_nodes=num_nodes)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    labeled = int((labels >= 0).sum())
    metadata = {
        "dataset": dataset_id, "family": spec.family, "source": spec.source,
        "label_source": spec.label_source,
        "num_nodes": int(num_nodes), "num_edges": int(bundle["num_edges"]),
        "num_views": len(view_names), "view_names": view_names,
        "illicit_count": int((labels == 1).sum()),
        "legitimate_count": int((labels == 0).sum()),
        "unlabeled_count": int((labels == -1).sum()),
        "labeled_illicit_ratio": float((labels == 1).sum() / max(labeled, 1)),
        "node_feature_dim": int(x_norm.shape[1]),
        "edge_attr_dim": int(view_tensors[0]["edge_attr"].shape[1]),
        "split": {
            "method": split_method,
            "train_nodes": int(train_mask.sum()), "val_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "train_illicit": int((data.y[train_mask] == 1).sum()),
            "val_illicit": int((data.y[val_mask] == 1).sum()),
            "test_illicit": int((data.y[test_mask] == 1).sum()),
            "holdout_category": holdout_category,
            "normalization": "RobustScaler fitted on training data only",
        },
        # TRANSDUCTIVE: features and message passing use the full graph; under the
        # temporal split only the LABELS are split by time. Scalers are
        # train-fitted to avoid preprocessing leakage; a strictly inductive
        # protocol would additionally restrict feature computation to pre-cutoff
        # edges. State this in Section 5.1.
        "feature_scope": "full_graph_transductive",
    }

    logger.info("[%s] split=%s train=%d val=%d test=%d (illicit %d/%d/%d).",
                dataset_id, split_method, metadata["split"]["train_nodes"],
                metadata["split"]["val_nodes"], metadata["split"]["test_nodes"],
                metadata["split"]["train_illicit"], metadata["split"]["val_illicit"],
                metadata["split"]["test_illicit"])
    for nm in ("train", "val", "test"):
        if metadata["split"][f"{nm}_nodes"] == 0:
            logger.warning("[%s] %s split has ZERO nodes -- training/evaluation on "
                           "it is impossible; adjust ratios or use the random "
                           "stratified split.", dataset_id, nm)
        elif metadata["split"][f"{nm}_illicit"] == 0:
            logger.warning("[%s] %s split has ZERO illicit nodes -- F1/recall is "
                           "undefined; check the temporal split boundaries.",
                           dataset_id, nm)
    return data, view_tensors, metadata


# Backward-compatible alias (aegis-mg.py imports prepare_chain_dataset).
def prepare_chain_dataset(config: DataConfig, chain: str,
                          holdout_category: Optional[str] = None):
    """Alias for prepare_dataset; `chain` accepts a registry dataset id."""
    return prepare_dataset(config, chain, holdout_category=holdout_category)


# ===========================================================================
# Convenience summary / smoke entry point
# ===========================================================================

def summarize_dataset(config: DataConfig, dataset: str) -> Dict[str, Any]:
    """Load a dataset and return its metadata (no model construction)."""
    _, _, meta = prepare_dataset(config, dataset)
    return meta


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Inspect an AEGIS-MG dataset.")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--dataset", default="ethereum_s",
                    help=f"one of {sorted(DATASET_REGISTRY)}")
    ap.add_argument("--random-split", action="store_true",
                    help="use DIAM-style random 2:1:1 instead of temporal")
    args = ap.parse_args()
    cfg = DataConfig(data_root=args.data_root, temporal_split=not args.random_split)
    import json
    print(json.dumps(summarize_dataset(cfg, args.dataset), indent=2))
