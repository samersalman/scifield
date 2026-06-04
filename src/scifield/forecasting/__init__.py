"""V1-S12 forecasting — leakage-safe topic-level time-series features plus
classical forecasting baselines. :mod:`scifield.forecasting.data` materializes
one row per ``(topic_id, origin_year)`` with strictly-trailing features, the
multiplicative-share-growth emergence label, and the temporal train/val/test
split (the test set is sealed; never materialized in S12). The
:mod:`scifield.forecasting.baselines` package supplies the four floors the
future GNN must beat — naive moving-average, per-topic ARIMA, a torch-CPU MLP,
and the ``no_graph`` GNN-minus-edges ablation — scored on the validation set by
emergence AUC and share MAPE.
"""
