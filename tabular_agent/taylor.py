"""Analytic Taylor derivatives of frozen log kernel mixtures."""
import copy
import numpy as np


def extract_taylor(geometric, order=2):
    if isinstance(order, bool) or order not in (1, 2, 3):
        raise ValueError("Taylor order must be 1, 2 or 3.")
    artifact = copy.deepcopy(geometric)
    p = len(artifact["mean"])
    equations = []
    for kernel in artifact["kernels"]:
        b, t = np.array(kernel["B"]), np.array(kernel["center"])
        missing, coef = np.array(kernel["missing"]), np.array(kernel["coef"])
        keep = coef > 0
        b, t, missing, coef = b[keep], t[keep], missing[keep], coef[keep]
        terms = np.log(coef) - (b*t*t).sum(1)
        maximum = terms.max()
        posterior = np.exp(terms-maximum)
        mass = posterior.sum()
        posterior /= mass
        value = float(maximum + np.log(mass))
        slopes = np.concatenate([2*b*t, b*t*t-missing], axis=1)
        gradient = posterior @ slopes
        hessian = np.zeros((2*p, 2*p))
        third = None
        if order >= 2:
            centered = slopes-gradient
            hessian = centered.T @ (posterior[:,None]*centered)
            ix = np.arange(p)
            hessian[ix,ix] -= 2*(posterior@b)
            cross = -2*(posterior@(b*t))
            hessian[ix,p+ix] += cross
            hessian[p+ix,ix] += cross
        if order == 3:
            # d^3 logsumexp = E[d^3 logkernel] + 3 sym Cov(d^2 logkernel,d logkernel)
            #                 + third centered moment of logkernel slopes.
            third = np.einsum("li,lj,lk,l->ijk",centered,centered,centered,posterior,optimize=True)
            for j in range(p):
                cov_zz = (-2*b[:,j]*posterior) @ centered
                cov_zm = (-2*b[:,j]*t[:,j]*posterior) @ centered
                for a,c,cov in ((j,j,cov_zz),(j,p+j,cov_zm),(p+j,j,cov_zm)):
                    third[a,c,:] += cov
                    third[a,:,c] += cov
                    third[:,a,c] += cov
                value_zzm = float(posterior@(2*b[:,j]))
                third[j,j,p+j] += value_zzm
                third[j,p+j,j] += value_zzm
                third[p+j,j,j] += value_zzm
        equations.append({
            "intercept":value, "linear":gradient.tolist(), "H":hessian.tolist(),
            "A":(.5*np.diag(hessian)[:p]).tolist(), "B":gradient[:p].tolist(),
            "D":(gradient[p:]+.5*np.diag(hessian)[p:]).tolist()})
        if third is not None:
            equations[-1]["T"] = third.tolist()
    artifact.update(classes=equations,kind="native_taylor",approximation_order=order,
                    reference={"standardized_observed": [0.]*p, "missing_indicators":[0.]*p},
                    derivation=f"Order-{order} Taylor expansion of frozen log class votes at z=0, m=0. "
                    "u=((1-m)z,m). Joint derivatives include observed and missing coordinates. "
                    "Order 3 adds the symmetric third-derivative tensor. This is a local approximation.")
    return artifact


def coordinates(artifact, x):
    z = (x-np.array(artifact["mean"]))/np.array(artifact["std"])
    m = ~np.isfinite(z)
    return np.concatenate([np.where(m,0,z),m.astype(float)],axis=1)


def grouped_shap(artifact, background, samples, groups, class_index=-1, contrast=False):
    """Exact group-interventional SHAP for a degree<=2 polynomial.

    Missing indicators move with their observed coordinate. Encoded category
    columns move together. Background covariance is retained for absent groups.
    """
    u, bg = coordinates(artifact,samples), coordinates(artifact,background)
    p = len(groups)
    eq = artifact["classes"][class_index]
    linear, h = np.array(eq["linear"]), np.array(eq["H"])
    if contrast:
        linear = linear-np.array(artifact["classes"][0]["linear"])
        h = h-np.array(artifact["classes"][0]["H"])
    mean = bg.mean(0)
    result = np.zeros((len(samples),p))
    for group in dict.fromkeys(groups):
        cols = [j for j,g in enumerate(groups) if g==group]
        ix = np.array(cols+[j+p for j in cols])
        other = np.setdiff1d(np.arange(2*p),ix)
        a, ab = u[:,ix], bg[:,ix]
        own = h[np.ix_(ix,ix)]
        val = (a-mean[ix])@linear[ix]
        val += .5*(np.einsum("ni,ij,nj->n",a,own,a) -
                    np.einsum("ni,ij,nj->n",ab,own,ab).mean())
        cross = h[np.ix_(ix,other)]
        if len(other):
            val += .5*(np.einsum("ni,ij,nj->n",a,cross,u[:,other]) -
                        np.einsum("ni,ij,nj->n",ab,cross,bg[:,other]).mean() +
                        a@cross@mean[other] - u[:,other]@cross.T@mean[ix])
        result[:,cols[0]] = val
    return result
