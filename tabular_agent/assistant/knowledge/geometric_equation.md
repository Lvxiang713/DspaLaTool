---
id: geometric_equation
title: Geometric additive equation
keywords: geometric additive equation quadratic expression extraction default global interpretable
---

# Geometric additive equation

This is the default extraction method. The following formulas describe
complete inputs. z_j = (x_j-mu_j)/sigma_j uses frozen training statistics.

1. Start with the frozen model's exact class score.

    F_c(z) = sum_l a_cl exp[-d_cl(z)]
    d_cl(z) = sum_j b_clj (z_j-t_clj)^2
    a_cl = vote_scale*|v_cl|/L_c
    b_clj = |gamma_c| W_clj^2

Define total vote mass M_c = sum_l a_cl and normalized prior weights
p_cl = a_cl/M_c. Then:

    F_c(z) = M_c * sum_l p_cl exp[-d_cl(z)]
    log F_c(z) = log M_c + log E_p[exp(-d_cl(z))]

2. Make the geometric approximation.

Replace the weighted arithmetic mean of kernel responses by their weighted
geometric mean:

    G_c(z) = M_c * product_l exp[-d_cl(z)]^(p_cl)
           = M_c * exp[-sum_l p_cl d_cl(z)]
    S_c(z) = log G_c(z) = log M_c - sum_l p_cl d_cl(z)

This replacement is the approximation. It is not a Taylor expansion.
After this step, all remaining expansion is exact algebra.
Jensen's inequality gives S_c(z) <= log F_c(z), but different class gaps
mean this inequality does not guarantee preservation of the predicted class.

3. Expand each squared distance.

    (z_j-t_clj)^2 = z_j^2 - 2*t_clj*z_j + t_clj^2

Substitution and collection of equal powers give:

    S_c(z) = beta_c + sum_j [A_cj*z_j^2 + B_cj*z_j]
    beta_c = log M_c - sum_j sum_l p_cl*b_clj*t_clj^2
    A_cj = -sum_l p_cl*b_clj
    B_cj = 2*sum_l p_cl*b_clj*t_clj

Every coefficient comes directly from frozen model parameters. No labels
are fitted to obtain these coefficients. No cross-products z_i*z_j appear
because the distance is diagonal in feature coordinates and its weighted
average preserves that additive form.

4. Construct the binary decision equation.

    Q_geo(z) = S_1(z)-S_0(z)
             = beta_0 + sum_j [A_j*z_j^2+B_j*z_j]
    beta_0 = beta_1(class)-beta_0(class)
    A_j = A_1j-A_0j
    B_j = B_1j-B_0j

In this last expression beta_0 denotes the binary intercept, not the
class-0 intercept. Predict class 1 if Q_geo>0 and class 0 otherwise.
Multiclass prediction chooses the largest S_c.

Although each class-specific A_cj is nonpositive, their binary difference
A_j can have either sign. Thus a binary feature curve can open upward or
downward.

5. Interpret the approximation correctly.

The native mixture can have cross-feature interactions even though its
individual distances are diagonal: exponentiation and mixture aggregation
create joint responses. The geometric log-score approximation removes those
explicit cross-feature terms. Its prediction agreement and classification
metrics are evaluated against the original model on the same split.
