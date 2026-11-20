# -*- coding: utf-8 -*-
"""Class-specific landmark kernels with nonnegative voting.

Raw class votes feed classification loss without an extra softmax.
Construction order and streaming-statistic behavior match the native model.
"""
import numpy as np
import os

import torch
from torch import nn

from .layers import DSpaLaRefiner_Layer, AbsVotingLayer
from .losses import SmoothedCrossEntropy


class DSpaLaRefiner(nn.Module):
    """Distance-Space Landmark Refiner."""

    def __init__(self, input_features, num_class, gamma=2. ** 0, alpha=2. ** 0,
                 delta=2. ** 0, device='cuda:0', dtype=torch.float32,
                 early_stop_thres=100, compute_mode='Di', num_t=32,
                 batch_size=128, dropout_rate=0.15, dist_scale=None,
                 mini_batch=False, use_scheduler=False, set_global_w=False,
                 last_act=nn.Softmax(-1), record_epoch_targets=False,
                 perform_refiner_opt=True, record_epoch_step_size=10,
                 dist_loss_top_k=None, out_save_path='out', tether_rate=0.0,
                 record_best_epoch=False, record_thr_mcc=None,
                 val_mcc_stop_thre=None):
        super().__init__()
        self.out_save_path = out_save_path
        self.input_features = input_features
        self.num_class = num_class
        self.tether_rate = tether_rate
        assert isinstance(num_t, (int, list))
        if isinstance(num_t, int):
            num_t = [num_t] * num_class
        else:
            assert len(num_t) == num_class

        # --- construction order reproduces the original RNG consumption ---
        self.refiner_layers = nn.ModuleList([])
        for i in range(num_class):
            self.refiner_layers.append(DSpaLaRefiner_Layer(
                input_features, alpha, gamma, delta, device=device,
                dtype=dtype, early_stop_thres=early_stop_thres,
                num_t=num_t[i], dropout_rate=dropout_rate,
                compute_mode=compute_mode))

        global_gamma = nn.Parameter(torch.tensor(gamma, dtype=dtype).to(device),
                                    requires_grad=True)
        global_delta = nn.Parameter(torch.tensor(delta, dtype=dtype).to(device),
                                    requires_grad=True)
        global_alpha = nn.Parameter(torch.tensor(alpha, dtype=dtype).to(device),
                                    requires_grad=True)
        for layer in self.refiner_layers:
            layer.gamma = global_gamma
            layer.delta = global_delta
            layer.alpha = global_alpha
        if set_global_w:
            self.global_w = nn.Parameter(
                torch.full((1, input_features), 1 / input_features)
                .type(dtype).to(device), requires_grad=True)
            self.global_miss_w = nn.Parameter(
                torch.full((1, input_features), 1 / input_features)
                .type(dtype).to(device), requires_grad=True)
            for layer in self.refiner_layers:
                layer.feature_weights = self.global_w
                layer.missing_weight = self.global_miss_w

        self.early_stop_thres = early_stop_thres
        self.compute_mode = compute_mode
        self.record_thr_mcc = record_thr_mcc
        if val_mcc_stop_thre:
            self.val_mcc_stop_thre = val_mcc_stop_thre
        self.dropout_rate = dropout_rate
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        self.num_t = num_t
        self.voting_layers = nn.ModuleList([])
        for i in range(num_class):
            self.voting_layers.append(AbsVotingLayer(num_t[i], 1, bias=False))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.last_act = last_act
        self.device = device
        self.dtype = dtype
        self.relu = nn.ReLU()
        self.cls_loss = SmoothedCrossEntropy()
        self.vote_scale = 1.
        self.dist_loss_p = 2
        self.dist_scale = dist_scale
        self.total_t_num = int(np.sum(num_t))
        self.mini_dist_eps = 1e-10
        self.dist_loss_top_k = dist_loss_top_k
        self.batch_size = batch_size      # only used by compute_dist_loss
        self.mini_batch = mini_batch
        self.use_scheduler = use_scheduler
        self.perform_refiner_opt = perform_refiner_opt
        self.record_epoch_targets = record_epoch_targets
        self.record_best_epoch = record_best_epoch
        self.record_epoch_step_size = record_epoch_step_size
        self.register_buffer('feature_mean', torch.zeros(input_features))
        self.register_buffer('feature_std', torch.zeros(input_features))
        self.register_buffer('class_feature_mean',
                             torch.zeros([num_class, input_features]))
        self.register_buffer('class_feature_std',
                             torch.zeros([num_class, input_features]))

    # ------------------------------------------------ statistics
    def norm(self, xb):
        return (xb - self.feature_mean) / self.feature_std

    def compute_feature_statistics(self, train_loader):
        """Masked feature statistics from the TRAINING data only
        (originally ``processStatistic``). Registers feature_mean/std on
        the model and every layer, plus per-class normed statistics, then
        re-generates each layer's landmarks (consumes the global RNG)."""
        tmp_data, tmp_mask, tmp_label = [], [], []
        for xb, miss_mask, yb in train_loader:
            tmp_data.append(xb)
            tmp_mask.append(miss_mask)
            tmp_label.append(yb)
        tmp_arr = torch.cat(tmp_data, axis=0)
        if self.compute_mode == 'Normal':
            avg = tmp_arr.sum(axis=0) / tmp_arr.size(0)
            diff = (tmp_arr - avg) ** 2
            std = (diff.sum(axis=0) / tmp_arr.size(0)) ** 0.5
            raise NotImplementedError(
                "compute_mode='Normal' has no per-class statistics in the "
                "original implementation either")
        tmp_mask_t = torch.cat(tmp_mask, axis=0)
        tmp_label_t = torch.cat(tmp_label, axis=0)
        avg = tmp_arr.sum(axis=0) / (tmp_arr.size(0) - tmp_mask_t.sum(axis=0))
        diff = (tmp_arr - avg) ** 2
        diff.masked_fill_(tmp_mask_t, 0)
        std = (diff.sum(axis=0)
               / (tmp_arr.size(0) - tmp_mask_t.sum(axis=0))) ** 0.5
        n_class = yb.shape[-1]
        avg_by_label = [
            tmp_arr[tmp_label_t[:, i] == 1].sum(axis=0)
            / (tmp_arr[tmp_label_t[:, i] == 1].size(0)
               - tmp_mask_t[tmp_label_t[:, i] == 1].sum(axis=0))
            for i in range(n_class)]
        diff_by_label = [
            (tmp_arr[tmp_label_t[:, i] == 1] - avg_by_label[i]) ** 2
            for i in range(n_class)]
        diff_by_label = [
            diff_by_label[i].masked_fill_(tmp_mask_t[tmp_label_t[:, i] == 1], 0)
            for i in range(n_class)]
        std_by_label = [
            (diff_by_label[i].sum(axis=0)
             / (tmp_arr[tmp_label_t[:, i] == 1].size(0)
                - tmp_mask_t[tmp_label_t[:, i] == 1].sum(axis=0))) ** 0.5
            for i in range(n_class)]

        avg = avg.type(self.dtype).to(self.device)
        std = std.type(self.dtype).to(self.device)
        avg_by_label_t = torch.concat([i.unsqueeze(0) for i in avg_by_label],
                                      axis=0).type(self.dtype).to(self.device)
        std_by_label_t = torch.concat([i.unsqueeze(0) for i in std_by_label],
                                      axis=0).type(self.dtype).to(self.device)

        self.register_buffer('feature_mean', avg)
        self.register_buffer('feature_std', std)
        for layer in self.refiner_layers:
            layer.register_buffer('feature_mean', avg)
            layer.register_buffer('feature_std', std)

        avg_by_label_norm = self.norm(avg_by_label_t)
        std_by_label_norm = std_by_label_t / std
        self.register_buffer('class_feature_mean', avg_by_label_norm)
        self.register_buffer('class_feature_std', std_by_label_norm)
        for idx, layer in enumerate(self.refiner_layers):
            layer.register_buffer('class_feature_mean', avg_by_label_norm[idx, :])
            layer.register_buffer('class_feature_std', std_by_label_norm[idx, :])
            layer.generate_landmarks()

    def mask_with_rand(self, x, mask, scale=1e-5):
        """Zero the missing entries; at scale=0 this is a pure masked_fill
        but still consumes the global RNG (torch.rand_like) — kept for
        bit-exact reproduction of the original training stream."""
        return x.masked_fill(mask, 0) + (
            2 * torch.rand_like(mask.float(), dtype=self.dtype,
                                device=self.device) - 1
        ).masked_fill(~mask, 0) * scale

    # ------------------------------------------------ landmark spread loss
    def compute_dist_loss(self):
        """Landmark spread regularizer (original ``computeDistLoss``).
        Unused in the study (dist_scale=None) — kept for API parity.
        NOTE: calls backward() internally, like the original."""
        t_points = torch.cat([layer.landmark_positions
                              for layer in self.refiner_layers], dim=0)
        dist_loss = None
        for i in range(int(np.ceil(t_points.size(0) / self.batch_size))):
            s_pos = i * self.batch_size
            e_pos = (i + 1) * self.batch_size
            tmp_points = t_points[s_pos:e_pos, :].clone()
            sq = ((torch.abs(tmp_points.unsqueeze(1) - t_points.unsqueeze(0))
                   ** self.dist_loss_p) * self.dist_scale).sum(dim=-1)
            eye = torch.zeros_like(sq, dtype=bool, device=self.device)
            eye[torch.arange(tmp_points.size(0)),
                torch.arange(s_pos, s_pos + tmp_points.size(0))] = 1
            sq = sq.masked_fill(eye, 1e10)
            if self.dist_loss_top_k is not None:
                topk = torch.topk(sq.detach(), self.dist_loss_top_k, dim=1,
                                  largest=False).values[:, -1] \
                    .view([sq.size(0), 1])
                sq = sq.masked_fill(sq > topk, 1e10)
            zero_mask = sq < self.mini_dist_eps
            sq = sq.masked_fill(zero_mask, self.mini_dist_eps)
            curr = (sq ** -1).sum()
            curr.backward()
            dist_loss = curr.item() if dist_loss is None \
                else dist_loss + curr.item()
        if self.dist_loss_top_k:
            return dist_loss / self.total_t_num / self.dist_loss_top_k
        return dist_loss / self.total_t_num ** 2

    # ------------------------------------------------ training step
    def train_step(self, xb, yb, miss_mask):
        """One training forward+backward on a batch (original
        ``processOneBatch('N', ...)`` training path).

        Returns dict(tether_weighted, cls, ori, pred) of python floats /
        label lists. Performs total.backward() + grad clip internally.
        """
        xb = self.norm(xb)
        if self.compute_mode != 'Normal':
            xb = self.mask_with_rand(xb, miss_mask, scale=0)
        tether = 0
        if self.perform_refiner_opt:
            for i, layer in enumerate(self.refiner_layers):
                tether = tether - layer(xb, miss_mask, yb, pos_position=i)

        probs = []
        for i in range(self.num_class):
            layer = self.refiner_layers[i]
            vote = self.voting_layers[i]
            miss_mask_unsq = miss_mask.unsqueeze(0).repeat(
                layer.feature_weights.shape[0], 1, 1)
            dist, mean_byf, _, _ = layer.weighted_distance(xb, miss_mask)
            dist_punish = mean_byf.unsqueeze(1).repeat(1, dist.shape[1], 1) \
                .masked_fill(~miss_mask_unsq, 0).sum(-1)
            d = torch.exp(-1 * torch.abs(layer.gamma) * (dist + dist_punish))
            if self.training and self.dropout_rate:
                d = self.dropout(d)
            probs.append(vote(d.transpose(1, 0)) / layer.num_t * self.vote_scale)
        cls_label = torch.cat(probs, dim=1)          # n * 2 (raw logits)
        cls = self.cls_loss(cls_label, yb)
        # CE_MODE=off : drop the cross-entropy term entirely (tether only).
        _ce_off = os.environ.get('CE_MODE', 'on') == 'off'
        total = (self.tether_rate * tether) if _ce_off \
            else (self.tether_rate * tether + cls)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        tether_weighted = (self.tether_rate * tether.item()
                           if self.perform_refiner_opt else 0.0)
        return {'tether_weighted': tether_weighted,
                'cls': cls.item(),
                'ori': list(torch.argmax(yb.detach().cpu(), dim=1).numpy()),
                'pred': list(torch.argmax(cls_label.detach().cpu(), dim=1).numpy())}

    # ------------------------------------------------ eval step
    def eval_step(self, xb, yb, miss_mask, return_pred_prob=False):
        """One eval forward on a batch (original ``processOneBatch('N',
        ...)`` eval path). Uses the RUNNING streaming buffers — see
        layers.py docstring (stateful predict)."""
        xb = self.norm(xb)
        if self.compute_mode != 'Normal':
            xb = self.mask_with_rand(xb, miss_mask, scale=0)
        tether = 0
        if self.perform_refiner_opt:
            for i, layer in enumerate(self.refiner_layers):
                tether = tether - layer(xb, miss_mask, yb, pos_position=i)

        probs, dist_probs, dist_f_probs = [], [], []
        for i in range(self.num_class):
            layer = self.refiner_layers[i]
            vote = self.voting_layers[i]
            miss_mask_unsq = miss_mask.unsqueeze(0).repeat(
                layer.feature_weights.shape[0], 1, 1)
            dist, _, _, dist_f = layer.weighted_distance(xb, miss_mask)
            mean_byf = layer.stream_dist_sum / layer.stream_obs_count.unsqueeze(0)
            dist_punish = mean_byf.unsqueeze(1).repeat(1, dist.shape[1], 1) \
                .masked_fill(~miss_mask_unsq, 0).sum(-1)
            d = torch.exp(-1 * torch.abs(layer.gamma) * (dist + dist_punish))
            if return_pred_prob:
                dist_probs.append(d.detach().cpu())
                dist_f_probs.append(dist_f.detach().cpu())
            probs.append(vote(d.transpose(1, 0)) / layer.num_t * self.vote_scale)
        cls_label = torch.cat(probs, dim=1)
        cls = self.cls_loss(cls_label, yb)
        tether_weighted = (self.tether_rate * tether.item()
                           if self.perform_refiner_opt else 0.0)
        out = {'tether_weighted': tether_weighted, 'cls': cls.item(),
               'ori': list(torch.argmax(yb.detach().cpu(), dim=1).numpy()),
               'pred': list(torch.argmax(cls_label.detach().cpu(), dim=1).numpy())}
        if return_pred_prob:
            out['cls_label'] = cls_label.detach().cpu()
            out['dist_probs'] = dist_probs
            out['dist_f_probs'] = dist_f_probs
        return out

    # ------------------------------------------------ predict
    def predict(self, test_loader, return_dataset=False):
        """Full test-set evaluation (original ``predict``).

        STATEFUL (streaming buffers, see layers.py): the archived metrics
        correspond to exactly ONE predict as the first forward after
        loading a checkpoint, eval batch size 512.
        """
        self.eval()
        if return_dataset:
            collect = {'x': None, 'y': None, 'mask': None, 'y_pred': None,
                       'pred_probability': None, 'dist_prob': None}
        with torch.no_grad():
            t_loss, c_loss = None, None
            ori_labels, pred_labels = [], []
            for xb, miss_mask, yb in test_loader:
                xb = xb.type(self.dtype).to(self.device)
                miss_mask = miss_mask.to(self.device)
                yb = yb.type(self.dtype).to(self.device)
                r = self.eval_step(xb, yb, miss_mask, return_pred_prob=True)
                t_loss = r['tether_weighted'] if t_loss is None \
                    else t_loss + r['tether_weighted']
                c_loss = r['cls'] if c_loss is None else c_loss + r['cls']
                ori_labels += r['ori']
                pred_labels += r['pred']
                if return_dataset:
                    for key, val in (('x', xb.cpu()), ('y', yb.cpu()),
                                     ('mask', miss_mask.cpu()),
                                     ('pred_probability', r['cls_label'])):
                        collect[key] = val if collect[key] is None \
                            else torch.cat([collect[key], val], dim=0)
                    if collect['dist_prob'] is None:
                        collect['dist_prob'] = {
                            'class%s' % i: r['dist_probs'][i]
                            for i in range(len(r['dist_probs']))}
                    else:
                        for i in range(len(collect['dist_prob'])):
                            collect['dist_prob']['class%s' % i] = torch.cat(
                                [collect['dist_prob']['class%s' % i],
                                 r['dist_probs'][i]], dim=1)
            c_loss = c_loss / len(pred_labels)
            t_loss = t_loss / len(pred_labels)
            loss = c_loss + t_loss
            if return_dataset:
                collect['x'] = collect['x'].cpu().detach().numpy()
                collect['y'] = torch.argmax(collect['y'].detach().cpu(),
                                            dim=1).numpy()
                collect['mask'] = collect['mask'].cpu().detach().numpy()
                collect['y_pred'] = np.array(pred_labels)
                collect['pred_probability'] = np.array(
                    collect['pred_probability'].cpu())
                return loss, ori_labels, pred_labels, collect
        return loss, ori_labels, pred_labels
