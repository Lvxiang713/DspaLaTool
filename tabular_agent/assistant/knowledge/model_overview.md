---
id: model_overview
title: Model overview
keywords: model dspala dspalarefiner landmark kernel vote classifier architecture mixture exponential distance
---

# Model overview

The classifier is the native DSpaLaRefiner architecture. The following
complete-input formulas describe its frozen inference. The interface supports
binary and multiclass classification.

1. Standardize the input.

    z_j = (x_j - mu_j) / sigma_j

x_j is input feature j. mu_j and sigma_j are fitted on source-training rows
and remain frozen at evaluation. Categorical inputs first undergo the saved
training-fitted encoding; j indexes encoded coordinates.

2. Compute the distance to each learned landmark.

    d_cl(z) = sum_j b_clj (z_j - t_clj)^2
    b_clj = |gamma_c| W_clj^2 >= 0

Class c has L_c learned landmarks. t_clj is coordinate j of landmark l
for that class. W_clj controls its sensitivity to feature j. Squaring W
ensures nonnegative distance weights; gamma sets the distance scale.
Landmark positions, feature weights and vote weights are learned.

3. Convert distance into a positive kernel response and combine votes.

    k_cl(z) = exp[-d_cl(z)]
    a_cl = vote_scale * |v_cl| / L_c
    F_c(z) = sum_(l=1 to L_c) a_cl k_cl(z)

v_cl is the learned vote parameter; the voting layer has no bias.
Close matches give a larger kernel response. Each class score F_c combines
all its landmark responses rather than selecting only the nearest landmark.
These are raw class scores, not calibrated class probabilities.

4. Make the prediction.

    predicted_class = argmax_c F_c(z)
    Q_native(z) = F_1(z) - F_0(z)

For binary classification, Q_native > 0 predicts class 1; Q_native <= 0
predicts class 0. Class indices follow the labels saved with the model.

5. Learn class-discriminating geometry with the training objective.

    L_total = L_CE + lambda * L_tether
    L_CE = -sum_n sum_c y_tilde_nc log softmax(F(z_n))_c
    y_tilde_nc = (1[y_n=c] + 0.05) / (1 + 0.05*C)

Here C is the number of classes and lambda is the tether weight. The code
uses label-smoothed cross-entropy on the raw scores. For a training sample n,
let h_nc = 1 when y_n=c and 0 otherwise. The own/other tether objective is:

    L_tether =
      -sum_c (1/L_c) sum_l sum_n
       log[h_nc*k_cl(z_n) + (1-h_nc)*(1-k_cl(z_n)) + epsilon]
    epsilon = 1e-5

For a same-class sample, this favors a larger kernel response, or closer
weighted distance. For another-class sample, it favors a smaller response.
Tether therefore constrains the learned geometry using labeled source data;
it does not freeze landmark positions or force them to equal source samples.
The formula describes the complete-input own/other mode used by this workflow.

6. Distinguish native inference from an extracted equation.

The full exported kernel mixture F_c reproduces frozen native inference
within the audit tolerance (rtol 2e-5, atol 1e-6). The geometric additive
and Taylor expressions are approximations derived from that frozen model.
Their fidelity is evaluated separately; they are not algebraically identical
to the full mixture. In particular, geometric extraction approximates
log F_c with an additive quadratic S_c, and its binary score is S_1-S_0.
The native score difference and the extracted quadratic score are different
numerical quantities even when they produce the same class prediction.
