---
id: result_interpretation
title: Interpreting results
keywords: interpret read results report agreement fidelity ranking shap good performance delta
---

# Interpreting results

1. Read the original-model metrics first, then the expression metrics and
   their deltas. A small negative delta means the expression keeps most of the
   model's behavior.
2. Check fidelity: prediction agreement close to 1 with small score error
   supports reading the expression as a surrogate for the model on that split.
3. Compare coefficient ranking with SHAP. Overlap between the two gives
   stronger support than either alone.
4. Expand a feature to drill into its individual terms when the aggregate
   bar hides opposing interaction contributions.
5. Keep the interpretation at the level of the trained model. An expression
   that reproduces model decisions is evidence about the model, not a causal
   statement about the domain.
