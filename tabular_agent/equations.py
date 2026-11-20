"""NumPy-only equation evaluation; no training library or checkpoint required."""
import numpy as np


def scores(artifact, x):
    if artifact["kind"] == "native_taylor":
        from .taylor import coordinates
        u = coordinates(artifact, x)
        p = x.shape[1]
        terms = []
        for e in artifact["classes"]:
            # Half of each cross product is allocated to each coordinate.
            allocated = u*np.array(e["linear"]) + .5*u*(u@np.array(e["H"]))
            if "T" in e:
                allocated += u*np.einsum("ijk,nj,nk->ni",np.array(e["T"]),u,u,optimize=True)/6
            terms.append(allocated[:,:p]+allocated[:,p:])
        contributions = np.stack(terms,axis=1)
        return contributions.sum(2)+[e["intercept"] for e in artifact["classes"]],contributions
    z = (x - artifact["mean"]) / artifact["std"]
    m = ~np.isfinite(z)
    z = np.nan_to_num(z)
    contributions = np.stack([np.where(m, e["D"], z * z * e["A"] + z * e["B"])
                              for e in artifact["classes"]], axis=1)
    return contributions.sum(2) + [e["intercept"] for e in artifact["classes"]], contributions


def kernel_votes(artifact, x):
    z = np.nan_to_num((x - artifact["mean"]) / artifact["std"])
    m = ~np.isfinite(x)
    votes = []
    for k in artifact["kernels"]:
        d = np.where(m[:, None, :], np.array(k["missing"])[None, :, :],
                     np.array(k["B"])[None, :, :] * (z[:, None, :] - k["center"]) ** 2).sum(2)
        votes.append(np.exp(-d) @ k["coef"])
    return np.stack(votes, axis=1)


def kernel_log_scores(artifact, x):
    """Stable log mixture scores, including very small kernel responses."""
    z = np.nan_to_num((x - artifact["mean"]) / artifact["std"])
    m = ~np.isfinite(x)
    result = []
    for kernel in artifact["kernels"]:
        coef = np.asarray(kernel["coef"])
        positive = coef > 0
        distance = np.where(m[:,None,:],np.array(kernel["missing"])[None,:,:],
                            np.array(kernel["B"])[None,:,:]*(z[:,None,:]-kernel["center"])**2).sum(2)
        terms = np.log(coef[positive])[None,:]-distance[:,positive]
        maximum = terms.max(1)
        result.append(maximum+np.log(np.exp(terms-maximum[:,None]).sum(1)))
    return np.stack(result,axis=1)
