# AEGIS-MG — Adversarially Robust Multi-Graph Neural Networks for Cryptocurrency Fraud Detection

Reference implementation for the manuscript *Adversarially Robust Multi-Graph
Neural Networks for Cryptocurrency Fraud Detection*, currently under
double-blind review. This repository reproduces every table and figure in the
paper from the released code and the publicly documented datasets.

> **Anonymity.** This repository is anonymized for review. It contains no
> author names, affiliations, acknowledgements, or institution-specific paths.
> Please do not open issues or pull requests that could de-anonymize the
> authors.

---

## 1. What this is (and what it is not)

AEGIS-MG is a graph neural network for account-level fraud detection on
cryptocurrency transaction graphs. Its design goal is **robustness under
attack**, not a new state of the art on clean detection. The honest summary of
the results, reproduced by this code, is:

- On clean data, AEGIS-MG is **competitive** with strong baselines and ahead of
  conventional graph models, but it **does not surpass** the accuracy-leading
  baseline (DIAM).
- Under a projected-gradient-descent attack (`l2`, `eps = 0.20`), AEGIS-MG
  **retains 83–88%** of its clean accuracy, where an undefended graph model
  retains **under one half**. This is the contribution.
- The certified guarantee is a **conservative `l2` floor** (about half of nodes
  certified at `eps = 0.20`, with abstentions reported), strong differential
  privacy is **costly** under full-batch training (`eps <= 1` drives utility
  toward the no-skill floor), and cross-chain transfer leaves **~10%
  degradation** after fine-tuning.

The code reports these costs rather than hiding them. There is no "Combined"
super-dataset, no 100M-node throughput claim, and no 166-dimensional Elliptic
feature set in this repository.

---

## 2. Repository layout

| File | Purpose |
|------|---------|
| `aegis-mg.py` | Experiment runner / CLI entry point (`--experiment ...`). |
| `models.py`   | Model definitions: `ContinuousTimeEncoder`, `EnhancedGraphAttentionLayer`, `CrossGraphDiscrepancyFusion`, `CertifiedDefenseLayer`, `DifferentialPrivacyModule`, `AEGISModel`. |
| `train.py`    | `Trainer` and `TrainConfig`: training loop, early stopping, checkpointing. |
| `data.py`     | Dataset registry, loading, DIAM-Table-1 validation, 32-feature pipeline, temporal splits. |
| `analysis.py` | Statistical tests (Wilcoxon, effect size), certification analysis, false-alert accounting, reporting. |
| `config.yaml` | All hyperparameters, dataset ids, and evaluation protocols. |

Graph operators are implemented directly in `models.py`; **no `torch-geometric`
dependency is required**.

---

## 3. Requirements

- Python >= 3.9
- `torch >= 2.1`
- `numpy`, `pandas`, `pyyaml`, `scipy`, `scikit-learn`

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.1" numpy pandas pyyaml scipy scikit-learn
```

A CUDA GPU is recommended. The paper's runs use four NVIDIA RTX 3090 (24 GB);
single-GPU operation is supported up to the Bitcoin-L scale (peak ~22 GB).

---

## 4. Data setup

The five datasets are resolved through `data.DATASET_REGISTRY`. The four
cryptocurrency benchmarks load directly from the published **DIAM
`*_graph_dict.npz`** files (`edge_index`, `edge_attr`, `y`); the loader
**validates each graph against DIAM Table 1** and warns on any mismatch. Binance
Smart Chain is loaded from BSCScan exports with community-reported labels and is
treated as **contingent**.

| id | Chain | Source of labels | Status |
|----|-------|------------------|--------|
| `ethereum_s` | Ethereum | Etherscan / XBlock phishing | validated |
| `ethereum_p` | Ethereum | Etherscan phishing + licit | validated |
| `bitcoin_m`  | Bitcoin  | WalletExplorer | validated |
| `bitcoin_l`  | Bitcoin  | WalletExplorer (DIAM) | validated |
| `bsc`        | BSC      | BSCScan community reports | **contingent** |

Expected directory tree under `./data` (see `config.yaml -> data`):

```
data/
├── ethereum/   EthereumS_graph_dict.npz, EthereumP_graph_dict.npz
├── bitcoin/    BitcoinM_graph_dict.npz, BitcoinL_graph_dict.npz
├── bsc/        *.parquet
└── labels/
    ├── ethereum_etherscan_phishing.csv
    ├── bitcoin_walletexplorer.csv
    └── bsc_community_reports.csv   (address,label[,category])
```

Notes that the pipeline enforces:
- **32 node features** are computed by the pipeline (not Elliptic's 166), so a
  single input dimension transfers across chains.
- Absolute timestamps are **excluded** from edge attributes (the temporal
  encoder consumes time deltas) to prevent a temporal-split shortcut.
- Nodes with `y = -1` (npz) or unlabeled addresses (parquet) are **excluded**
  from train/val/test, following the DIAM convention.

Inspect a dataset without training:

```bash
python data.py --data-root ./data --dataset ethereum_s
```

---

## 5. Running the experiments

The runner is `aegis-mg.py`. The `--experiment` flag selects what to run; each
experiment is repeated over **five seeds (42–46)** for valid statistics.

```bash
# Detection on one dataset
python aegis-mg.py --experiment standard   --dataset bitcoin_l

# Empirical robustness (FGSM/PGD/CW) and randomized-smoothing certification
python aegis-mg.py --experiment adversarial --dataset ethereum_s

# Differential-privacy sweep over epsilon
python aegis-mg.py --experiment privacy     --dataset bitcoin_m

# Cross-chain transfer (direct / fine-tuned / zero-shot)
python aegis-mg.py --experiment cross_chain --source ethereum_s --target bitcoin_m

# Ablation, scalability, zero-day holdout
python aegis-mg.py --experiment ablation    --dataset bitcoin_l
python aegis-mg.py --experiment scalability --dataset bitcoin_l
python aegis-mg.py --experiment zero_day    --dataset bitcoin_l

# Everything (all datasets / all transfer pairs)
python aegis-mg.py --experiment all
```

Full CLI: `--config` (default `config.yaml`), `--dataset/--chain`, `--source`,
`--target`, `--device`, `--output-dir`. Results are written under
`paths.results_dir`; `analysis.py` aggregates them into the reported tables.

### Experiment → paper map

| `--experiment` | Produces | Paper |
|----------------|----------|-------|
| `standard`    | Detection F1 / precision / recall / AUC | Tables 2–3, Fig. (detection) |
| `adversarial` | PGD-0.20 retention; certified accuracy + abstention + radius | Tables 4 & 6, Figs. (robustness), (security A) |
| `privacy`     | F1 vs `epsilon`, no-skill floor | Table 5, Fig. (security B) |
| `cross_chain` | Direct / fine-tuned / zero-shot, degradation | Table 7, Fig. (transfer) |
| `scalability` | Training time / peak memory vs graph size | Table (scalability), Fig. (results D) |
| `ablation`    | Component contributions (security; see caveat) | Table (ablation) |
| `zero_day`    | Held-out-category generalization | Limitations |

---

## 6. Reproducibility

- Random seed **42** for all stochastic components; statistics use seeds
  **42–46** (`system.num_seeds: 5`).
- Significance tests use **5 seeds × 5 datasets = 25 paired observations**
  (`analysis.py`); the older "n = 6, p < 0.01" claim was not attainable and has
  been corrected.
- Deterministic mode (`system.deterministic: true`) additionally requires
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment for full GPU determinism.
- Certification follows Cohen et al. (2019) with **fixed `sigma_0 = 0.10`** and
  `N = 10000` samples; the adaptive variance is a training-time augmentation
  only and is **not** used at certification time.

---

## 7. Honest scope and caveats

These are stated in the paper and enforced or flagged in the code:

1. **Multi-view fusion is inactive on these benchmarks.** Every public dataset
   provides a single relation view (`K = 1`), so the cross-graph attention and
   contrastive term contribute nothing here; their ablation on this data yields
   no change, and they are **not credited** with detection gains. A genuinely
   multi-view dataset is needed to evaluate them.
2. **Certification is conservative and scoped.** It certifies `l2` perturbations
   of the continuous fused representation only — **not** structural (edge/node)
   attacks. `N = 10000` (Cohen uses 100000) is a deliberate cost trade-off; the
   maximum certifiable radius at `alpha = 1e-3` is `~ sigma_0 * 2.86`.
3. **The privacy accounting is approximate.** The implementation clips the
   aggregated gradient as a stand-in for strict per-sample clipping; the
   reported `epsilon` should be **validated with a per-sample accountant
   (Opacus)** before any strong privacy claim.
4. **BSC is contingent.** Community-sourced labels, reported separately and not
   aggregated with the four validated benchmarks.
5. **The `zero_day` experiment requires per-node category labels**
   (`data.categories_files`); without them it **raises** rather than emitting
   placeholder numbers.
6. **Training is costly.** A Bitcoin-L epoch is roughly 4× the cost of DIAM
   (multi-head attention + the PGD inner loop).

---

## 8. Expected results (sanity ranges)

| Quantity | Expected |
|----------|----------|
| Detection F1 (validated four) | ~88–95% (DIAM leads at ~94.7% mean) |
| PGD-0.20 retention, AEGIS-MG | 83–88% (undefended < 50%) |
| Certified-correct at `eps = 0.20` | ~50% mean, with 11–21% abstention, radius ~0.30 |
| Privacy at `eps = 1` (full-batch) | F1 near the no-skill floor |
| Cross-chain degradation (fine-tuned) | ~10% on average |
| Bitcoin-L training | ~1320 s / epoch, ~22 GB peak |

If your numbers depart materially from these ranges, check the DIAM-Table-1
validation warnings emitted by `data.py` first.

---

## 9. License & citation

Released for peer review. A license and citation block will be added upon
de-anonymization. Please cite the accompanying manuscript (details withheld for
double-blind review).
