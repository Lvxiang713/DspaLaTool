---
id: metrics
title: Metrics and fidelity
keywords: acc mcc f1 precision recall agreement fidelity rmse r2 score delta metric evaluation holdout
---

# Metrics and fidelity

Classification reports ACC, MCC, F1 (macro), Precision (macro) and
Recall (macro). The comparison tables place the original model and each
extracted expression side by side with per-metric deltas, so a negative delta
shows how much performance the expression loses relative to the model.

Fidelity measures how closely an expression reproduces the original model on
the same evaluation split: prediction agreement, plus score RMSE and score R2
for score closeness. Agreement near 1 with small score error supports using
the expression for interpretation; large gaps mean the approximation should
not be read as the model itself.

The holdout split is observational. An optional external CSV is evaluated
after training and never influences training or parameter selection.
