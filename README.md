# LOX and LOX-R: Local Outlier eXplanations

Code for reproducing experiments in the CIKM 2026 submission.

## Structure

```
├── src/pyxod/            # Core library
│   ├── explainers.py     # LOX (Algorithm 1) and LOX-R (Algorithm 2)
│   ├── base.py           # BaseLOXExplainer
│   ├── metrics.py        # Evaluation metrics
│   └── utils.py          # Utilities
├── run_experiments.py    # Full experiment runner
├── pyproject.toml        # Dependencies
└── README.md
```

## Installation

```bash
pip install -e ".[explain]"
```

## Reproducing Results

```bash
python run_experiments.py --datasets cancer adult credit heart bank_marketing \
    covertype mammography shuttle satimage \
    synthetic_context synthetic_guaranteed fred_md
```

Results are written to `results/` by default. Key outputs:
- `all_results.csv` — per-instance evaluation (all datasets × detectors × methods × k)
- `summary.csv` — aggregated metrics
- `detector_auc.csv` — detector performance

## Algorithms

| Method | Description |
|--------|-------------|
| **LOX** | Perturbation-based rank-shift explainer. Measures expected rank change when each feature is replaced by random reference values. |
| **LOX-R** | Rank-aware conditional explainer. Combines normalized rank shift (via kNN conditional replacement) with residual-to-manifold correction. |

## Baselines (included in runner)

- SHAP-Permutation
- LIME-ScoreReg (local Ridge on anomaly scores)
- PFI-Rank (local permutation feature importance)
- LOFO-Rank (leave-one-feature-out rank shift)

## Detectors

PyOD: LODA, Isolation Forest, LOF, DeepSVDD.

## Evaluation Metrics

- Fidelity (Jaccard@k vs white-box reference)
- Kendall-tau@k
- Delta-rank (rank shift after top-k neutralization)
- Infidelity
- Context-counterfactual validity (CCV)
