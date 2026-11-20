# -*- coding: utf-8 -*-
"""Learned landmark positions, feature weights and kernel voting.

Native distance calls accumulate streaming statistics even during evaluation.
Reset them at each training epoch; use the frozen adapter for stable inference.
Preserve construction order to retain the native RNG sequence.
"""
import torch
from torch import nn

from .losses import TetherLoss, INF_FILL, DIVIDE_BIAS

MISS_HEAD_DIM = 512   # original: self.miss_head


class DSpaLaRefiner_Layer(nn.Module):
    """Landmark refiner layer for ONE class (formerly ``WDD``).

    Parameters
    ----------
    input_features : int
        Feature dimension f.
    alpha, gamma, delta : float
        Initial values of the (later shared) scalar parameters.
    num_t : int
        Number of landmarks (original ``numT``).
    compute_mode : str
        'Di' (per-feature distances, used throughout the study),
        'Bf' or 'Normal' (kept for API parity).
    dropout_rate : float
        0 disables dropout (study setting).
    """

    def __init__(self, input_features, alpha, gamma, delta, device='cuda:0',
                 dtype=torch.float32, early_stop_thres=100, compute_mode='Di',
                 num_t=4, dropout_rate=0.15):
        super().__init__()
        # --- construction order reproduces the original RNG consumption ---
        self.missing_head = nn.Linear(MISS_HEAD_DIM, 1, bias=False)
        torch.nn.init.zeros_(self.missing_head.weight)
        self.input_features = input_features
        self.feature_weights = nn.Parameter(
            2 * torch.rand([num_t, input_features], dtype=dtype, device=device) - 1,
            requires_grad=True)
        self.missing_feature_weights_pu = nn.Parameter(
            2 * torch.rand([num_t, input_features], dtype=dtype, device=device) - 1,
            requires_grad=True)
        self.missing_weight = nn.Parameter(
            2 * torch.rand([1, input_features], dtype=dtype, device=device) - 1,
            requires_grad=True)
        self.random_feature_anchor = nn.Parameter(
            2 * torch.rand([1, input_features], dtype=dtype, device=device) - 1,
            requires_grad=True)
        self.landmark_positions = None  # created by generate_landmarks()
        self.num_t = num_t
        self.relu = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=dtype).to(device),
                                  requires_grad=True)
        self.device = device
        self.compute_mode = compute_mode
        self.dtype = dtype
        self.initialized = False
        self.dropout_rate = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('feature_mean', torch.zeros(input_features))
        self.register_buffer('feature_std', torch.zeros(input_features))
        self.register_buffer('stream_dist_sum',
                             torch.zeros([num_t, input_features], dtype=dtype))
        self.register_buffer('stream_obs_count',
                             torch.zeros([input_features], dtype=dtype))
        self.register_buffer('class_feature_mean', torch.zeros(input_features))
        self.register_buffer('class_feature_std', torch.zeros(input_features))
        self.tether_loss = TetherLoss()
        self.generate_landmarks()

    # ------------------------------------------------ landmark handling
    def freeze_landmarks(self):
        self.landmark_positions.requires_grad_(False)
        self.feature_weights.requires_grad_(False)

    def thaw_landmarks(self):
        self.landmark_positions.requires_grad_(True)
        self.feature_weights.requires_grad_(True)

    def generate_landmarks(self):
        """(Re)sample landmark positions from the per-class feature
        statistics (originally ``generateT``; consumes the global RNG —
        39 ``torch.normal`` calls in feature order)."""
        random_arrays = [
            torch.normal(mean=self.class_feature_mean[i],
                         std=self.class_feature_std[i],
                         size=(self.num_t,), dtype=self.dtype, device=self.device)
            for i in range(len(self.class_feature_mean))]
        self.landmark_positions = nn.Parameter(
            torch.stack(random_arrays, dim=1), requires_grad=True)

    def initialize_landmarks(self, data_loader):
        """NaN-guarded landmark initialisation (originally
        ``sampleInitPos``): re-generates landmarks until the tether term
        over the training data is finite."""
        if self.initialized:
            return
        retry = 0
        with torch.no_grad():
            while True:
                retry += 1
                needs_reinit = False
                for xb, miss_mask, yb in data_loader:
                    xb = xb.type(self.dtype).to(self.device)
                    miss_mask = miss_mask.to(self.device)
                    yb = yb.type(self.dtype).to(self.device)
                    xb = self.norm(xb)
                    xb.masked_fill_(miss_mask, 0)
                    tmp_loss = self.tether_term(xb, miss_mask, yb,
                                                filter_nan=False)
                    if torch.isnan(tmp_loss):
                        needs_reinit = True
                        self.generate_landmarks()
                        break
                if not needs_reinit:
                    break
                assert retry < 1000
        self.initialized = True

    # ------------------------------------------------ statistics
    def norm(self, xb):
        return (xb - self.feature_mean) / self.feature_std

    def reset_streaming_stats(self):
        """Zero the streaming buffers (original epoch start:
        ``sum_byf *= 0; sum_mask *= 0``)."""
        self.stream_dist_sum *= 0
        self.stream_obs_count *= 0

    # ------------------------------------------------ distances
    def weighted_distance(self, xb, mask=None, w=None):
        """Weighted squared distances landmarks <-> samples
        (originally ``distDi``).

        Returns (dist [nt,n], mean_byf [nt,f], obs_count_byf [f],
                 dist_per_feature [nt,n,f]).
        SIDE EFFECT (deliberate, see module docstring): accumulates
        ``stream_dist_sum`` / ``stream_obs_count`` on every call.
        """
        if w is not None:
            tmp_w = w.unsqueeze(1)                      # nt * 1 * f
        else:
            tmp_w = self.feature_weights.unsqueeze(1)   # nt * 1 * f
        diff = self.landmark_positions.unsqueeze(1) - xb.unsqueeze(0)  # nt*n*f
        if mask is not None:
            diff = diff.masked_fill(mask.unsqueeze(0), 0)
        tmp_diff = (tmp_w * diff) ** 2                  # nt * n * f
        obs_count_byf = torch.logical_not(mask).sum(dim=0)
        tmp_diff_sum = tmp_diff.sum(1)
        mean_byf = tmp_diff_sum / (obs_count_byf + DIVIDE_BIAS)
        self.stream_dist_sum += tmp_diff_sum.detach()
        self.stream_obs_count += obs_count_byf.detach()
        return tmp_diff.sum(axis=-1), mean_byf, obs_count_byf, tmp_diff

    def weighted_distance_bf(self, xb, mask=None):
        tmp_w = self.feature_weights
        return tmp_w * torch.abs(self.landmark_positions - xb)

    # ------------------------------------------------ tether term
    def forward(self, xb, miss_mask, yb, filter_nan=True, pos_position=0):
        return self.tether_term(xb, miss_mask, yb, filter_nan=filter_nan,
                                pos_position=pos_position)

    def tether_term(self, xb, miss_mask, yb, filter_nan=True, pos_position=0):
        if self.compute_mode == 'Di':
            return self._tether_term_di(xb, miss_mask, yb, filter_nan,
                                        pos_position)
        elif self.compute_mode == 'Bf':
            return self._tether_term_bf(xb, miss_mask, yb, filter_nan)
        else:
            return self._tether_term_normal(xb, filter_nan, yb)

    def _tether_term_di(self, xb, miss_mask, yb, filter_nan=True,
                        pos_position=0):
        """Tether loss for compute_mode 'Di' (originally ``cmpDi``).

        Training mode uses per-batch statistics; eval mode uses the
        RUNNING streaming buffers (see module docstring). The formula
        itself lives in losses.TetherLoss.
        """
        if self.training:
            dist, mean_byf, _, _ = self.weighted_distance(xb, mask=miss_mask)
        else:
            dist, _, _, _ = self.weighted_distance(xb, mask=miss_mask)
            mean_byf = self.stream_dist_sum / self.stream_obs_count.unsqueeze(0)
        return self.tether_loss(dist, mean_byf, miss_mask, yb, self.gamma,
                                pos_position, self.num_t, filter_nan)

    def _tether_term_bf(self, xb, miss_mask, yb, filter_nan=True):
        dist_ij = self.weighted_distance_bf(xb)
        dist = torch.exp(-1 * self.gamma * dist_ij) ** self.delta
        dist = dist.masked_fill(miss_mask, 1e-5)
        pr_ij = dist.sum(dim=1)
        pos_mask = yb[:, 1].clone().flatten()
        pos_sign = pos_mask * -2 + 1
        pr = torch.exp(-1 * pr_ij) * pos_sign + pos_mask
        if filter_nan:
            nan_mask = torch.isnan(pr)
            pr = pr.masked_fill(nan_mask, 1e-5)
            nan_mask = torch.isinf(pr)
            pr = pr.masked_fill(nan_mask, 1e-5)
        return torch.log(pr + 1e-5).sum()

    def _tether_term_normal(self, xb, filter_nan=True, yb=None):
        # NOTE: the original cmpNormal unpacked 3 values from a 4-value
        # return (broken). 'Normal' compute mode is unused in the study;
        # this is the corrected minimal version.
        dist, mean_byf, obs_count, _ = self.weighted_distance(xb)
        if filter_nan:
            nan_mask = torch.isnan(dist)
            dist = dist.masked_fill(nan_mask, INF_FILL)
            nan_mask = torch.isinf(dist)
            dist = dist.masked_fill(nan_mask, INF_FILL)
            nan_mask = dist > INF_FILL
            dist = dist.masked_fill(nan_mask, INF_FILL)
        pr_ij = 1 - torch.exp(-1 * self.gamma * dist)
        if filter_nan:
            nan_mask = torch.isnan(pr_ij)
            dist = dist.masked_fill(nan_mask, 1e-5)
            nan_mask = torch.isinf(pr_ij)
            dist = dist.masked_fill(nan_mask, 1e-5)
        pos_mask = yb[:, 1].clone().flatten()
        pos_sign = pos_mask * -2 + 1
        pr = pr_ij * pos_sign + pos_mask
        return torch.log(pr + 1e-5).sum()


class AbsVotingLayer(nn.Module):
    """Landmark voting layer (originally ``AbsLinear``): votes over the
    per-landmark probabilities with |weight|; the absolute weight of a
    landmark is its influence on the class logit."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = bias
        torch.nn.init.normal_(self.linear.weight, mean=0, std=1)

    def forward(self, x):
        weights = torch.abs(self.linear.weight)
        weighted_sum_abs = torch.matmul(x, weights.T)
        if self.bias:
            return torch.sum(weighted_sum_abs, dim=1, keepdim=True) + self.linear.bias
        return torch.sum(weighted_sum_abs, dim=1, keepdim=True)
