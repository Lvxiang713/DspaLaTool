---
id: reanalysis
title: Reanalyzing a saved model
keywords: reanalyze reanalysis reuse saved model without training another expression compare
---

# Reanalyzing a saved model

Choose a completed model in "Reanalyze a saved model", select the expression
methods to add, and run analysis only. The original encoder, split and
training settings are reused; frozen artifact hashes are verified first, and
results are written to a new run folder that never changes the parent.

Use this to compare geometric and Taylor 1/2/3 extractions from the same
trained model, or to add SHAP later. Reanalysis evaluates the original model
from its audited frozen kernel export; it does not load or retrain a network
checkpoint.
