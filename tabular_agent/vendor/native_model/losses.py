# -*- coding: utf-8 -*-
"""Native tether loss and summed cross-entropy with 0.05 label smoothing.

Total loss is classification loss plus tether_rate times tether loss.
The native filtering order and streaming-statistic formulas are preserved.
"""
import os

import torch
from torch import nn

# 'own_other' (default, = original study) : own class attracted AND
#                                          other class repelled
# 'own_only'  (experimental)             : ONLY the own-class
#                                          attraction; the inter-class
#                                          repulsion term is dropped.
# 'repel_only'(experimental)             : ONLY the inter-class
#                                          repulsion; the own-class
#                                          attraction term is dropped.
TETHER_MODE = os.environ.get('TETHER_MODE', 'own_other')

INF_FILL = 1e5        # original: WDD.infFillNum
DIVIDE_BIAS = 1e-30   # original: WDD.divide_bias


class TetherLoss(nn.Module):
    """Tether loss over landmark distances (see module docstring).

    This module is parameter-free; it is instantiated inside each
    DSpaLaRefiner_Layer and called with the distances produced by
    ``DSpaLaRefiner_Layer.weighted_distance``.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dist, mean_byf, miss_mask, yb, gamma, pos_position,
                num_t, filter_nan=True):
        """Exact port of the original ``WDD.cmpDi`` formula.

        dist      : [nt, n]   weighted squared landmark<->sample distances
        mean_byf  : [nt, f]   per-feature mean squared distances
                              (per-batch in training, running buffer in eval)
        miss_mask : [n, f]    True == missing feature
        yb        : [n, 2]    one-hot labels
        gamma     : scalar parameter (shared across layers)
        pos_position : class index whose column of yb marks "own class"
        num_t     : number of landmarks of this layer
        """
        if filter_nan:
            nan_mask = torch.isnan(dist)
            dist = dist.masked_fill(nan_mask, INF_FILL)
            nan_mask = torch.isinf(dist)
            dist = dist.masked_fill(nan_mask, INF_FILL)
            nan_mask = dist > INF_FILL
            dist = dist.masked_fill(nan_mask, INF_FILL)

        miss_mask_unsq = miss_mask.unsqueeze(0).repeat(mean_byf.shape[0], 1, 1)
        dist_punish = mean_byf.unsqueeze(1).repeat(1, dist.shape[1], 1) \
            .masked_fill(~miss_mask_unsq, 0).sum(-1)      # nt*n
        pr_ij = 1 - torch.exp(-1 * torch.abs(gamma) * (dist + dist_punish))

        if filter_nan:
            # NOTE: the original fills `dist` (not pr_ij) here — kept for
            # bit-exact parity even though it is almost certainly a bug.
            nan_mask = torch.isnan(pr_ij)
            dist = dist.masked_fill(nan_mask, 1e-5)
            nan_mask = torch.isinf(pr_ij)
            dist = dist.masked_fill(nan_mask, 1e-5)

        pos_mask = yb[:, pos_position].clone().flatten().view([1, -1])  # 1*n
        if TETHER_MODE == 'repel_only':
            # ONLY inter-class repulsion: maximise log(q) on the samples of the
            # OTHER class; own-class samples contribute exactly 0.
            pr = pr_ij * (1.0 - pos_mask) + pos_mask
            out = (torch.log(pr + 1e-5) * (1.0 - pos_mask)).sum()
        elif TETHER_MODE == 'own_only':
            # ONLY same-class attraction: maximise log(1 - q) on the samples of
            # this layer's own class; samples of the other class contribute
            # exactly 0 (pr==1 -> log 1 == 0, then masked out).
            pr = (1.0 - pr_ij) * pos_mask + (1.0 - pos_mask)
            out = (torch.log(pr + 1e-5) * pos_mask).sum()
        else:
            pos_sign = pos_mask * -2 + 1
            pr = pr_ij * pos_sign + pos_mask
            out = torch.log(pr + 1e-5).sum()
        if torch.isnan(out):
            raise FloatingPointError('TetherLoss became NaN')
        return out / num_t


class SmoothedCrossEntropy(nn.Module):
    """Classification loss (original ``computeLoss``): smooth the one-hot
    targets with +0.05, renormalize, then CrossEntropyLoss(sum) on the
    raw 2-logits."""

    def __init__(self, smooth=0.05):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, logits, y_onehot):
        y = y_onehot + self.smooth
        y = y / y.sum(dim=-1).view([-1, 1])
        return self.ce(logits, y)
