"""Grouped, sampled permutation SHAP for an arbitrary deterministic score."""
import numpy as np


def permutation_shap(predict_score, explain_x, background, groups, permutations=4, seed=0):
    """Sample feature orders and average over a fixed source-only background.

    Categorical indicator columns move together as one original feature.
    Each sampled path telescopes exactly; finite-order SHAP estimates remain
    approximations. Missingness moves together with the feature value.
    """
    explain_x, background = np.asarray(explain_x), np.asarray(background)
    if not len(explain_x) or not len(background) or permutations < 1:
        raise ValueError("SHAP needs non-empty explanation/background rows and positive permutations.")
    names = list(dict.fromkeys(groups))
    columns = [np.flatnonzero(np.asarray(groups) == name) for name in names]
    rng = np.random.default_rng(seed)
    baseline = np.asarray(predict_score(background),dtype=float).mean()
    endpoint = np.asarray(predict_score(explain_x),dtype=float)
    phi = np.zeros((len(explain_x),len(names)))
    evaluations = len(background)+len(explain_x)
    for i, x in enumerate(explain_x):
        orders = []
        for draw in range((permutations+1)//2):
            order = rng.permutation(len(names))
            orders.append(order)
            if len(orders)<permutations:
                orders.append(order[::-1])
        for order in orders:
            paths = []
            for bg in background:
                current = bg.copy()
                paths.append(current.copy())
                for group in order:
                    current[columns[group]] = x[columns[group]]
                    paths.append(current.copy())
            path_scores = np.asarray(predict_score(np.asarray(paths)),dtype=float).reshape(len(background),len(names)+1)
            increments = np.diff(path_scores,axis=1).mean(0)
            phi[i,order] += increments/permutations
            evaluations += len(paths)
    residual = phi.sum(1)+baseline-endpoint
    return {"values":phi,"feature_names":names,"base_value":float(baseline),
            "max_additivity_error":float(np.max(np.abs(residual))),
            "score_evaluations":evaluations,"permutations":permutations,
            "background_rows":len(background),"explained_rows":len(explain_x)}
