# Phase 5 — Forecasting

## Phase objective

Build and validate a model that predicts topic emergence 3 years ahead of
conventional detection, with all evaluation metrics pre-specified on OSF
before training. The primary architecture is a Heterogeneous Graph
Transformer (HGT) or Temporal Graph Network (TGN) implemented in PyTorch
Geometric over heterogeneous Paper/Author/Topic nodes. Mandatory baselines:
a naive last-3-years moving average, ARIMA per topic, a simple MLP on
topic-level time-series features, and a without-graph ablation. Temporal
cross-validation strictly partitions training (1995–2017), validation
(2018–2020), and test (2021–2025). Success criterion: the GNN beats the
best baseline by >5 percentage points on emergence AUC at a 3-year horizon
and the improvement is statistically significant by Wilcoxon signed-rank
across topics. A null result is documented honestly and the paper pivots
toward F1 and F2.
