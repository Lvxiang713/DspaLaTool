---
id: shap_and_coefficients
title: SHAP and coefficient ranking
keywords: shap coefficient ranking importance contribution drilldown interaction feature
---

# SHAP and coefficient ranking

Coefficient ranking is a descriptive norm of the standardized expression
coefficients (and Hessian/cubic terms for Taylor). It describes the extracted
expression itself.

Equation SHAP uses the training-only background to quantify each feature's
contribution to the expression score. Geometric and Taylor orders 1 and 2 use
exact grouped interventional polynomial SHAP; Taylor order 3 uses a sampled
grouped permutation estimate. The optional SHAP switch also estimates
original-model SHAP with the bundled sampled permutation method, which is
approximate rather than exact.

In charts and predictors, a feature name can be expanded to drill down into
every nonzero term containing that feature, ranked by absolute coefficient
for the expression or by allocated signed contribution for one sample.
Displayed sample contributions split interaction terms equally among their
coordinates; they are not SHAP values.
