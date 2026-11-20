---
id: taylor_equations
title: Taylor approximations
keywords: taylor hessian cubic first second third order interaction local approximation reference expansion
---

# Taylor approximations

The implemented Taylor methods expand f_c(z)=log F_c(z), the logarithm of
the frozen native class vote. They do not expand F_c itself and do not fit
polynomial coefficients to labels. The formulas below cover complete inputs.

1. Define the function and reference.

    h_cl(z) = log a_cl - sum_j b_clj*(z_j-t_clj)^2
    f_c(z) = log sum_l exp[h_cl(z)]
    Delta = z-r

r is the reference vector. The interface uses r=0 in source-standardized
coordinates, corresponding to the source-training mean for numeric features.
For encoded categorical coordinates zero is a mathematical reference and
need not represent a valid category. Derivatives are evaluated at r, not
along a training or gradient-descent trajectory.

2. First-order (linear) approximation.

    S_c^(1)(z) = f_c(r) + g_c^T*Delta
    g_ci = partial f_c / partial z_i, evaluated at r

This retains the value and slope of the log class score at the reference.

3. Second-order (Hessian) approximation.

    S_c^(2)(z) = f_c(r) + g_c^T*Delta + (1/2)*Delta^T*H_c*Delta
    H_cij = partial^2 f_c / (partial z_i partial z_j), evaluated at r

Hessian is the matrix of second derivatives, not a separate alternative to
Taylor. It is precisely the second-order term of a Taylor expansion.
At r=0, the binary score Q^(2)=S_1^(2)-S_0^(2) becomes:

    Q^(2)(z) = beta_0 + sum_j [A_j*z_j^2+B_j*z_j]
                        + sum_(i<j) C_ij*z_i*z_j
    beta_0 = f_1(0)-f_0(0)
    B_j = g_1j-g_0j
    A_j = (H_1jj-H_0jj)/2
    C_ij = H_1ij-H_0ij

There is no extra factor 1/2 on C_ij when only i<j is summed: the symmetric
matrix product contains both (i,j) and (j,i).

4. Third-order (cubic) approximation.

    S_c^(3)(z) = S_c^(2)(z)
                + (1/6)*sum_i sum_j sum_k T_cijk*Delta_i*Delta_j*Delta_k
    T_cijk = partial^3 f_c / (partial z_i partial z_j partial z_k), at r

T is symmetric. After combining identical monomials, a class-specific
z_i^3 term has coefficient T_ciii/6; z_i^2*z_j (i != j) has coefficient
T_ciij/2; z_i*z_j*z_k with i<j<k has coefficient T_cijk.
For binary prediction, subtract the corresponding class-0 coefficients from
class-1 coefficients. These multiplicities explain why a raw derivative
tensor entry is not always the coefficient of a collected polynomial term.

5. Obtain derivatives directly from the model.

At r, define normalized response weights:

    q_cl = exp[h_cl(r)] / sum_s exp[h_cs(r)]
    u_cli = partial h_cl / partial z_i at r
           = -2*b_cli*(r_i-t_cli)
    V_clij = partial^2 h_cl / (partial z_i partial z_j)
            = -2*b_cli*1[i=j]

Then expectations and covariances below use q_cl as weights over landmarks:

    g_ci = E_q[u_i]
    H_cij = E_q[V_ij] + Cov_q(u_i,u_j)
    T_cijk = E_q[(u_i-g_ci)*(u_j-g_cj)*(u_k-g_ck)]
             + Cov_q(V_ij,u_k)
             + Cov_q(V_ik,u_j)
             + Cov_q(V_jk,u_i)

Each h_cl is quadratic, so its own third derivatives are zero.
The log-mixture nevertheless has nonzero third derivatives through the
response-weighted covariance and third central moment above.
At r=0, u_cli=2*b_cli*t_cli, and q_cl is proportional to
a_cl*exp[-sum_j b_clj*t_clj^2].

All these quantities use the learned landmark positions, distance weights
and vote coefficients of the frozen model. Off-diagonal Hessian entries
come from covariance of landmark slopes; this explains how interactions can
exist in the native mixture despite diagonal per-landmark distances.

6. Understand the approximation order.

For a smooth f_c, an order-p expansion has a local remainder
O(||z-r||^(p+1)) as z approaches r. Its retained value and derivatives at r
are exact; its predictions away from r are approximate.
Third order does not guarantee better global accuracy than second order.
All methods are compared on the same evaluation split.

Taylor orders 1/2 support up to 256 encoded features; order 3 supports up
to 40 because the derivative tensor grows cubically. One-hot coordinates
count toward these limits. Multiclass prediction uses argmax_c S_c^(p);
binary prediction uses S_1^(p)-S_0^(p)>0, with ties assigned to class 0.
