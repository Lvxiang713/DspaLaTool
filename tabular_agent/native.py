"""Unmodified native architecture with a safe source-statistics adapter."""
import hashlib
import os
import random
import types
from contextlib import contextmanager

import numpy as np
import torch
from .vendor.native_model.network import DSpaLaRefiner
from .equations import scores, kernel_votes


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def frozen(model):
    """Native evaluation cannot change training state or RNG streams."""
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    nrng, prng, training = np.random.get_state(), random.getstate(), model.training
    originals = []
    try:
        model.eval()
        for layer in model.refiner_layers:
            original = layer.weighted_distance
            sd, oc = layer.stream_dist_sum.clone(), layer.stream_obs_count.clone()
            def call(self, *args, _original=original, _sd=sd, _oc=oc, **kwargs):
                result = _original(*args, **kwargs)
                self.stream_dist_sum.copy_(_sd)
                self.stream_obs_count.copy_(_oc)
                return result
            originals.append((layer, original))
            layer.weighted_distance = types.MethodType(call, layer)
        with torch.no_grad():
            yield
    finally:
        for layer, original in originals:
            layer.weighted_distance = original
        model.load_state_dict(state)
        model.train(training)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(nrng)
        random.setstate(prng)


def tensors(x, y, classes, device):
    mask = ~np.isfinite(x)
    return (torch.tensor(np.nan_to_num(x), dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.bool, device=device),
            torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long, device=device), classes).float())


def train(x, y, config, callback=None):
    if os.environ.get("CE_MODE", "on") != "on" or os.environ.get("TETHER_MODE", "own_other") != "own_other":
        raise ValueError("Unset CE_MODE/TETHER_MODE overrides. This workflow requires CE on and own_other tether.")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = config["device"]
    nclass = int(np.max(y)) + 1
    model = DSpaLaRefiner(x.shape[1], nclass, device=device, num_t=config["landmarks"],
                         dropout_rate=config["dropout"], tether_rate=config["tether"]).to(device)
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    std[std < 1e-8] = 1
    # Class-empty observed coordinates fall back to source statistics.
    cm, cs = [], []
    for c in range(nclass):
        subset = x[y == c]
        count = np.isfinite(subset).sum(0)
        m = np.divide(np.nansum(subset, 0), count, out=mean.copy(), where=count > 0)
        s = np.sqrt(np.divide(np.nansum((subset - m) ** 2, 0), count,
                              out=std.copy() ** 2, where=count > 0))
        cm.append((m - mean) / std)
        cs.append(s / std)
    with torch.no_grad():
        model.feature_mean.copy_(torch.tensor(mean, device=device))
        model.feature_std.copy_(torch.tensor(std, device=device))
        model.class_feature_mean.copy_(torch.tensor(np.array(cm), device=device))
        model.class_feature_std.copy_(torch.tensor(np.array(cs), device=device))
        for c, layer in enumerate(model.refiner_layers):
            layer.feature_mean.copy_(model.feature_mean)
            layer.feature_std.copy_(model.feature_std)
            layer.class_feature_mean.copy_(model.class_feature_mean[c])
            layer.class_feature_std.copy_(model.class_feature_std[c])
            layer.generate_landmarks()
    xb, mask, yb = tensors(x, y, nclass, device)
    optimizer = torch.optim.Rprop(model.parameters(), lr=config["learning_rate"])
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        for layer in model.refiner_layers:
            layer.reset_streaming_stats()
        losses, tether_losses = [], []
        for start in range(0, len(x), config["batch_size"]):
            end = start + config["batch_size"]
            result = model.train_step(xb[start:end], yb[start:end], mask[start:end])
            losses.append(result["cls"])
            tether_losses.append(result["tether_weighted"])
        optimizer.step()
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError(f"Non-finite model state at epoch {epoch + 1}. Inspect scaling and configuration.")
        row = {"epoch": epoch + 1, "classification_loss_sum": float(sum(losses)),
               "tether_loss_weighted_sum": float(sum(tether_losses)),
               "total_loss_sum": float(sum(losses) + sum(tether_losses)),
               "training_n": len(x),
               "total_loss_mean": float(sum(losses) + sum(tether_losses))/len(x)}
        history.append(row)
        if callback is not None:
            before = state_hash(model)
            callback(model, row)
            if before != state_hash(model):
                raise AssertionError("Monitoring changed model state.")
    return model, history


def predict_native(model, x, batch_size=256):
    xb, mask, yb = tensors(x, np.zeros(len(x), dtype=int), model.num_class, model.device)
    result = []
    with frozen(model):
        for start in range(0, len(x), batch_size):
            out = model.eval_step(xb[start:start + batch_size], yb[start:start + batch_size],
                                  mask[start:start + batch_size], return_pred_prob=True)
            result.append(out["cls_label"].cpu().numpy())
    return np.concatenate(result)


def extract(model):
    kernels, equations = [], []
    for layer, vote in zip(model.refiner_layers, model.voting_layers):
        coef = (vote.linear.weight.detach().abs().flatten() / layer.num_t * model.vote_scale).cpu().double().numpy()
        weights = layer.feature_weights.detach().cpu().double().numpy() ** 2
        gamma = float(layer.gamma.detach().abs())
        center = layer.landmark_positions.detach().cpu().double().numpy()
        penalty = (layer.stream_dist_sum / layer.stream_obs_count[None, :]).detach().cpu().double().numpy() * gamma
        b = gamma * weights
        if coef.sum() <= 0 or not np.isfinite(penalty).all():
            raise ValueError("Cannot extract a class with zero vote mass or undefined source missing penalties.")
        p = coef / coef.sum()
        bbar = p @ b
        bt = p @ (b * center)
        bt2 = p @ (b * center ** 2)
        equations.append({"intercept": float(np.log(coef.sum()) - bt2.sum()),
                          "A": (-bbar).tolist(), "B": (2 * bt).tolist(),
                          "D": (bt2 - p @ penalty).tolist()})
        kernels.append({"coef": coef.tolist(), "B": b.tolist(), "center": center.tolist(),
                        "missing": penalty.tolist()})
    return {"mean": model.feature_mean.cpu().tolist(), "std": model.feature_std.cpu().tolist(),
            "classes": equations, "kernels": kernels, "kind": "native_geometric_quadratic",
            "derivation": "Coefficient-weighted mean distance, followed by exact algebraic expansion."}


