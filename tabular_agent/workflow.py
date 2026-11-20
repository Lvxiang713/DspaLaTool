"""Auditable local workflow. Holdout data never select parameters."""
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, matthews_corrcoef, f1_score,
                             precision_score, recall_score, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import torch

from .data import Encoder, read_table, category_values
from .native import train, extract, scores, kernel_votes, predict_native, state_hash
from .report import write_report, write_predictor
from .equations import kernel_log_scores
from .explain import permutation_shap
from .taylor import extract_taylor, grouped_shap


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, pred, task):
    if task == "regression":
        return {"RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "R2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None}
    return {"ACC": float(accuracy_score(y, pred)), "MCC": float(matthews_corrcoef(y, pred)),
            "F1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "Precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
            "Recall_macro": float(recall_score(y, pred, average="macro", zero_division=0))}


def regression_adapter(x, y, config):
    """Separate fitted surrogate, never mislabeled algebraic native extraction."""
    mean, std = np.nanmean(x, 0), np.nanstd(x, 0)
    std[std < 1e-8] = 1
    z = np.nan_to_num((x - mean) / std)
    mask = ~np.isfinite(x)
    design = np.column_stack([z*z, z, mask.astype(float)])
    original = RandomForestRegressor(n_estimators=100, random_state=config["seed"], n_jobs=1)
    original.fit(np.column_stack([z, mask.astype(float)]), y)
    surrogate = Ridge(alpha=1).fit(design, original.predict(np.column_stack([z, mask.astype(float)])))
    n = x.shape[1]
    eq = {"intercept": float(surrogate.intercept_), "A": surrogate.coef_[:n].tolist(),
          "B": surrogate.coef_[n:2*n].tolist(), "D": surrogate.coef_[2*n:].tolist()}
    artifact = {"kind": "regression_fitted_quadratic", "mean": mean.tolist(), "std": std.tolist(),
                "classes": [eq], "derivation": "Ridge-fitted additive surrogate of a random forest. Not algebraic DSpaLa extraction."}
    def predict(v):
        return original.predict(np.column_stack([np.nan_to_num((v-mean)/std), (~np.isfinite(v)).astype(float)]))
    return original, artifact, predict


def run(csv, target, output, config, external=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    dump(output / "status.json", {"stage": "data", "status": "running"})
    try:
        reused = None
        if config.get("reuse_analysis"):
            parent = Path(config["reuse_analysis"]).resolve()
            manifest = json.loads((parent/"ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
            for name in ("encoder.json","equation.json","summary.json"):
                if hashlib.sha256((parent/name).read_bytes()).hexdigest() != manifest.get(name):
                    raise ValueError(f"Saved model integrity check failed: {name}")
            reused = json.loads((parent/"summary.json").read_text(encoding="utf-8"))
            if hashlib.sha256(Path(csv).read_bytes()).hexdigest() != reused["input_sha256"]:
                raise ValueError("Reanalysis requires the original source CSV and split.")
            if reused.get("external_sha256") != (hashlib.sha256(Path(external).read_bytes()).hexdigest() if external else None):
                raise ValueError("Reanalysis requires the original external evaluation CSV.")
            overrides = {k:config[k] for k in ("approximation","approximation_order","no_shap",
                         "original_shap","reuse_analysis","requested_methods") if k in config}
            config = dict(reused["configuration"], **overrides)
            if config["task"] != "classification":
                raise ValueError("Saved-model reanalysis currently supports classification.")
        approximation = config.get("approximation","geometric")
        order = config.get("approximation_order",2)
        if approximation not in ("geometric","taylor") or isinstance(order,bool) or order not in (1,2,3):
            raise ValueError("Choose geometric or Taylor approximation with order 1, 2 or 3.")
        if config["task"] != "classification" and approximation == "taylor":
            raise ValueError("Native Taylor extraction requires classification.")
        if config["task"] not in ("classification","regression"):
            raise ValueError("Task must be classification or regression.")
        for name in ("epochs","landmarks","batch_size"):
            if isinstance(config[name],bool) or not isinstance(config[name],int) or config[name]<1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(config["tether"]) or config["tether"]<0:
            raise ValueError("tether must be finite and non-negative.")
        if not np.isfinite(config["learning_rate"]) or config["learning_rate"]<=0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 <= config["dropout"] < 1:
            raise ValueError("dropout must be in [0,1).")
        frame = read_table(csv)
        if target not in frame:
            raise ValueError(f"Target column {target!r} is absent.")
        if frame[target].isna().any():
            raise ValueError("Target values must be present in every training row.")
        if not 0 < config["test_fraction"] < 1:
            raise ValueError("test_fraction must be between zero and one.")
        target_labels = category_values(frame[target])
        if target_labels.isna().any():
            raise ValueError("Target contains missing or non-finite labels.")
        labels = sorted(target_labels.unique().tolist()) if config["task"] == "classification" else []
        if reused and labels != reused["classes"]:
            raise ValueError("Saved classes do not match the original target.")
        if config["task"] == "classification" and len(labels) < 2:
            raise ValueError("Classification needs at least two target classes.")
        y = np.array([labels.index(v) for v in target_labels]) if labels else pd.to_numeric(frame[target]).to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError("Target contains non-finite values.")
        if labels and np.bincount(y).min()<2:
            raise ValueError("Every class needs at least two source rows for a stratified training/holdout split.")
        indices = np.arange(len(frame))
        tr, te = train_test_split(indices, test_size=config["test_fraction"], random_state=config["seed"],
                                  stratify=y if labels else None)
        encoder = (Encoder.load(json.loads((parent/"encoder.json").read_text(encoding="utf-8")))
                   if reused else Encoder().fit(frame.drop(columns=target).iloc[tr],config.get("feature_schema")))
        x = encoder.transform(frame.drop(columns=target))
        trainx, trainy = x[tr], y[tr]
        requested = config.get("requested_methods",[])
        if (approximation == "taylor" or any(m.startswith("taylor") for m in requested)) and len(encoder.names)>256:
            raise ValueError("Taylor extraction currently supports at most 256 encoded features.")
        if ((approximation == "taylor" and order==3) or "taylor3" in requested) and len(encoder.names)>40:
            raise ValueError("Third-order extraction currently supports at most 40 encoded features.")
        datasets = {"source_holdout":(x[te],y[te])}
        if external:
            ef = read_table(external)
            ex = encoder.transform(ef)
            ey = None
            if target in ef:
                if labels:
                    external_labels = category_values(ef[target])
                    if not external_labels.isin(labels).all():
                        raise ValueError("External target includes unknown/missing classes.")
                    ey = np.array([labels.index(v) for v in external_labels])
                else:
                    ey = pd.to_numeric(ef[target]).to_numpy(float)
                    if not np.isfinite(ey).all():
                        raise ValueError("External regression target must contain finite values.")
            datasets["external"] = ex,ey
        dump(output / "config.json", config)
        dump(output / "encoder.json", encoder.export())
        history = []
        dump(output / "status.json", {"stage": "training", "status": "running"})
        if reused:
            artifact = json.loads((parent/"equation.json").read_text(encoding="utf-8"))
            # Reconstruct the geometric expression directly from frozen kernels.
            equations = []
            for kernel in artifact["kernels"]:
                coef,b,t,u = [np.array(kernel[k]) for k in ("coef","B","center","missing")]
                p = coef/coef.sum()
                bt2 = p@(b*t*t)
                equations.append({"intercept":float(np.log(coef.sum())-bt2.sum()),
                                  "A":(-p@b).tolist(),"B":(2*(p@(b*t))).tolist(),
                                  "D":(bt2-p@u).tolist()})
            artifact = {k:artifact[k] for k in ("mean","std","kernels")}
            artifact.update(classes=equations,kind="native_geometric_quadratic",
                            derivation="Coefficient-weighted mean distance and algebraic expansion.")
            native = lambda v: kernel_votes(artifact,v)
            fidelity_audit = {"reused_frozen_export":True,"parent":str(parent),
                              "parent_native_audit":reused["audit"],
                              "training_performed":False}
            if approximation == "taylor":
                artifact = extract_taylor(artifact,order)
        elif labels:
            def monitor(model, row):
                # Observational only: fixed epochs, no best-holdout checkpoint selection.
                with (output / "monitor.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if config.get("monitor_sink"):
                    with Path(config["monitor_sink"]).open("a",encoding="utf-8") as f:
                        f.write(json.dumps(row)+"\n")
            model, history = train(trainx, trainy, config, monitor)
            torch.save(model.state_dict(), output / "native_state.pt")
            before = state_hash(model)
            artifact = extract(model)
            if approximation == "taylor":
                artifact = extract_taylor(artifact,order)
            native = lambda v: predict_native(model, v)
            votes = native(x[te])
            equivalent = kernel_votes(artifact, x[te])
            np.testing.assert_allclose(equivalent, votes, rtol=2e-5, atol=1e-6)
            assert before == state_hash(model), "Inference mutated the model."
            fidelity_audit = {"exact_kernel_native_allclose": True, "rtol": 2e-5, "atol": 1e-6,
                              "max_abs_error": float(np.max(np.abs(votes-equivalent))), "state_unchanged": True}
        else:
            model, artifact, native = regression_adapter(trainx, trainy, config)
            import joblib
            joblib.dump(model, output / "regression_model.joblib")
            fidelity_audit = {"exact_kernel_native_allclose": None, "reason": "Separate regression fitted-surrogate adapter."}
        dump(output / "equation.json", artifact)
        coefficients = []
        for c, eq in enumerate(artifact["classes"]):
            for j, feature in enumerate(encoder.names):
                coefficients.append({"class": labels[c] if labels else "regression",
                                     "feature": feature, "original_feature": encoder.groups[j],
                                     "A": eq["A"][j], "B": eq["B"][j], "D": eq["D"][j],
                                     "source_mean": artifact["mean"][j], "source_std": artifact["std"][j],
                                     "class_intercept": eq["intercept"]})
        pd.DataFrame(coefficients).to_csv(output / "coefficients.csv", index=False)
        if approximation == "taylor":
            pair_rows = []
            p = len(encoder.names)
            names = [f"observed:{n}" for n in encoder.names]+[f"missing:{n}" for n in encoder.names]
            for c,eq in enumerate(artifact["classes"]):
                h = np.array(eq["H"])
                for i in range(2*p):
                    for j in range(i+1,2*p):
                        if h[i,j] != 0:
                            pair_rows.append({"class":labels[c],"coordinate_i":names[i],
                                              "coordinate_j":names[j],"coefficient":h[i,j]})
            pd.DataFrame(pair_rows,columns=["class","coordinate_i","coordinate_j","coefficient"]).to_csv(
                output/"interaction_coefficients.csv",index=False)
            if order==3:
                import itertools
                cubic_rows = []
                for c,eq in enumerate(artifact["classes"]):
                    tensor = np.array(eq["T"])
                    for i,j,k in itertools.combinations_with_replacement(range(2*p),3):
                        multiplicity = 1 if i==k else 3 if i==j or j==k else 6
                        value = tensor[i,j,k]*multiplicity/6
                        if value != 0:
                            cubic_rows.append({"class":labels[c],"coordinate_i":names[i],"coordinate_j":names[j],
                                               "coordinate_k":names[k],"coefficient":value})
                pd.DataFrame(cubic_rows).to_csv(output/"cubic_coefficients.csv",index=False)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        dump(output / "status.json", {"stage": "analysis", "status": "running"})
        results = {}
        for name, (xx, yy) in datasets.items():
            q, _ = scores(artifact, xx)
            ori = native(xx)
            original_pred = ori.argmax(1) if labels else ori
            quadratic_pred = q.argmax(1) if labels else q[:, 0]
            row = {"n": len(xx), "original": metrics(yy, original_pred, config["task"]) if yy is not None else None,
                   "quadratic": metrics(yy, quadratic_pred, config["task"]) if yy is not None else None}
            if labels:
                logvote = np.log(np.maximum(ori, np.finfo(float).tiny))
                # Contrast to class 0 removes a common score offset.
                a, b = (q[:, 1:]-q[:, :1]).ravel(), (logvote[:, 1:]-logvote[:, :1]).ravel()
                row["prediction_agreement"] = float(np.mean(original_pred == quadratic_pred))
            else:
                a, b = quadratic_pred, original_pred
            results[name] = row
        # Exact additive interventional SHAP for quadratic score contrast.
        shap_enabled = not config.get("no_shap", False)
        closure = None
        cubic_shap = None
        if shap_enabled:
            _, background = scores(artifact, trainx)
            _, held = scores(artifact, x[te])
            ref = background[:, 1]-background[:, 0] if len(labels)==2 else background[:, -1]
            obs = held[:, 1]-held[:, 0] if len(labels)==2 else held[:, -1]
            phi = obs-ref.mean(0)
            if approximation == "taylor":
                if order<3:
                    phi = grouped_shap(artifact,trainx,x[te],encoder.groups,
                                        class_index=1 if len(labels)==2 else -1,contrast=len(labels)==2)
                else:
                    rng = np.random.default_rng(config["seed"]+103)
                    samples = x[te][rng.choice(len(te),min(config.get("shap_rows",16),len(te)),replace=False)]
                    bg = trainx[rng.choice(len(trainx),min(config.get("shap_background",8),len(trainx)),replace=False)]
                    def polynomial_score(rows):
                        value,_ = scores(artifact,rows)
                        return value[:,1]-value[:,0] if len(labels)==2 else value[:,-1]
                    cubic_shap = permutation_shap(polynomial_score,samples,bg,encoder.groups,
                                                  config.get("shap_permutations",4),config["seed"]+104)
                    phi = np.zeros((len(samples),len(encoder.names)))
                    for j,g in enumerate(cubic_shap["feature_names"]):
                        phi[:,encoder.groups.index(g)] = cubic_shap["values"][:,j]
        grouped = {}
        for j, group in enumerate(encoder.groups):
            grouped.setdefault(group, []).append(j)
        importance = []
        eqs = artifact["classes"]
        aa = np.array(eqs[1]["A"])-eqs[0]["A"] if len(labels)==2 else np.array(eqs[-1]["A"])
        bb = np.array(eqs[1]["B"])-eqs[0]["B"] if len(labels)==2 else np.array(eqs[-1]["B"])
        for group, cols in grouped.items():
            entry = {"feature": group, "coefficient_norm": float(np.linalg.norm(np.r_[aa[cols], bb[cols]]))}
            if approximation == "taylor":
                h = np.array(eqs[1]["H"])-eqs[0]["H"] if len(labels)==2 else np.array(eqs[-1]["H"])
                linear = np.array(eqs[1]["linear"])-eqs[0]["linear"] if len(labels)==2 else np.array(eqs[-1]["linear"])
                ix = cols+[j+len(encoder.names) for j in cols]
                entry["coefficient_norm"] = float(np.sqrt(np.sum(linear[ix]**2)+.5*np.sum(h[ix,:]**2)))
                if order==3:
                    tensor = np.array(eqs[1]["T"])-eqs[0]["T"] if len(labels)==2 else np.array(eqs[-1]["T"])
                    entry["coefficient_norm"] = float(np.sqrt(entry["coefficient_norm"]**2+
                                                              np.sum(tensor[ix,:,:]**2)/6))
            if shap_enabled:
                entry["mean_abs_SHAP"] = float(np.abs(phi[:, cols].sum(1)).mean())
            importance.append(entry)
        importance = pd.DataFrame(importance).sort_values("mean_abs_SHAP" if shap_enabled else "coefficient_norm", ascending=False)
        importance.to_csv(output / "feature_importance.csv", index=False)
        base_intercept = eqs[1]["intercept"]-eqs[0]["intercept"] if len(labels)==2 else eqs[-1]["intercept"]
        if shap_enabled:
            closure = (cubic_shap["max_additivity_error"] if cubic_shap is not None else
                       np.max(np.abs(phi.sum(1) + ref.mean(0).sum() + base_intercept - (obs.sum(1)+base_intercept))))
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        axes = axs.ravel()
        top = importance.head(12).iloc[::-1]
        if shap_enabled:
            axes[0].barh(top.feature, top.mean_abs_SHAP, color="#009e78")
            axes[0].set_title("Taylor expression SHAP value" if approximation=="taylor" else "Quadratic SHAP value")
            axes[0].set_xlabel("Mean absolute score contribution")
        else:
            axes[0].axis("off")
            axes[0].text(.5,.5,"SHAP analysis is off",ha="center",va="center")
        if labels:
            axes[1].bar(["Original", "Taylor" if approximation=="taylor" else "Quadratic"], [results["source_holdout"][k]["ACC"] for k in ["original", "quadratic"]], color=["#798ca0", "#9a63bf"])
            axes[1].set_ylim(0, 1)
            axes[1].set_title("Source holdout accuracy")
        else:
            axes[1].bar(["Original", "Quadratic"], [results["source_holdout"][k]["RMSE"] for k in ["original", "quadratic"]])
            axes[1].set_title("Source holdout RMSE")
        topcoef = importance.sort_values("coefficient_norm",ascending=False).head(12).iloc[::-1]
        axes[2].barh(topcoef.feature,topcoef.coefficient_norm,color="#087ba4")
        axes[2].set_title("Standardized Taylor coefficients" if approximation=="taylor" else "Standardized quadratic coefficients")
        axes[2].set_xlabel("Linear / Hessian coefficient norm" if approximation=="taylor" else "Norm of A and B")
        hq,_ = scores(artifact,x[te])
        ho = native(x[te])
        xx = np.log(np.maximum(ho[:,1],np.finfo(float).tiny))-np.log(np.maximum(ho[:,0],np.finfo(float).tiny)) if len(labels)==2 else np.log(np.maximum(ho[:,-1],np.finfo(float).tiny)) if labels else ho
        yy = hq[:,1]-hq[:,0] if len(labels)==2 else hq[:,-1]
        axes[3].scatter(xx,yy,s=14,alpha=.6,color="#9a63bf")
        lower,upper=min(xx.min(),yy.min()),max(xx.max(),yy.max())
        axes[3].plot([lower,upper],[lower,upper],color="black",lw=1)
        axes[3].set_xlabel("Original model score")
        axes[3].set_ylabel("Taylor score" if approximation=="taylor" else "Quadratic score")
        axes[3].set_title("Score fidelity on source holdout")
        fig.tight_layout()
        fig.savefig(output / "analysis.png", dpi=180)
        plt.close(fig)
        original_shap = None
        if shap_enabled and config.get("original_shap",False):
            dump(output / "status.json",{"stage":"original_model_SHAP","status":"running"})
            rng = np.random.default_rng(config["seed"]+101)
            explanation = x[te][rng.choice(len(te),min(config.get("shap_rows",16),len(te)),replace=False)]
            bg = trainx[rng.choice(len(trainx),min(config.get("shap_background",8),len(trainx)),replace=False)]
            def original_score(rows):
                if labels:
                    logs = kernel_log_scores(artifact,rows)
                    return logs[:,1]-logs[:,0] if len(labels)==2 else logs[:,-1]
                return native(rows)
            original_shap = permutation_shap(original_score,explanation,bg,encoder.groups,
                                             config.get("shap_permutations",4),config["seed"]+102)
            ranking = pd.DataFrame({"feature":original_shap["feature_names"],
                                    "mean_abs_SHAP":np.abs(original_shap["values"]).mean(0)})
            ranking = ranking.sort_values("mean_abs_SHAP",ascending=False)
            ranking.to_csv(output / "original_model_SHAP.csv",index=False)
            fig,ax = plt.subplots(figsize=(7,5))
            top_original = ranking.head(12).iloc[::-1]
            ax.barh(top_original.feature,top_original.mean_abs_SHAP,color="#9a63bf")
            ax.set_title("Original model SHAP value")
            ax.set_xlabel("Mean absolute sampled permutation contribution")
            fig.tight_layout()
            fig.savefig(output / "original_model_SHAP.png",dpi=180)
            plt.close(fig)
            original_shap = {k:v for k,v in original_shap.items() if k not in ("values","feature_names")}
            original_shap.update({"method":"Sampled grouped interventional permutation SHAP; antithetic feature orders",
                                  "object":"stable log frozen-kernel class1-minus-class0 score" if len(labels)==2 else "stable log frozen-kernel last-class score" if labels else "original regression prediction",
                                  "scope":"Fixed source-training background and sampled source-holdout rows; finite permutations are approximate"})
        summary = {"task": config["task"], "classes": labels, "training_n": len(tr), "holdout_n": len(te),
                   "encoded_features": len(encoder.names), "dropped_features": encoder.dropped,
                   "input_sha256": hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
                   "external_sha256":hashlib.sha256(Path(external).read_bytes()).hexdigest() if external else None,
                   "feature_schema":config.get("feature_schema",{}),
                   "configuration": config, "results": results, "audit": fidelity_audit,
                   "SHAP": {"object": ("Taylor score contrast class 1 minus class 0" if len(labels)==2 else "last class Taylor score")
                            if approximation=="taylor" else
                            "quadratic score contrast class 1 minus class 0" if len(labels)==2 else "last class quadratic score" if labels else "quadratic regression score",
                            "method": ("Sampled grouped permutation SHAP for cubic expression; source-training background"
                                       if cubic_shap is not None else
                                       "Exact grouped quadratic interventional SHAP; joint training-background moments"
                                       if approximation=="taylor" else "Exact additive interventional SHAP; training-partition background"),
                            "grouping": "Encoded columns summed by original feature",
                            "max_additivity_error": float(closure)} if shap_enabled else None,
                   "original_model_SHAP":original_shap,
                   "elapsed_seconds": time.time()-started,
                   "protocol": "Training-only encoder/statistics; fixed epochs; source holdout is observational and never selects the model.",
                   "monitor": "Loss and epoch monitoring always enabled; no per-epoch predictive metrics or projections."
                              if config["task"] == "classification" else
                              "Not applicable: the regression adapter has no native epoch monitor.",
                   "equation_kind": artifact["kind"]}
        summary["training_performed"] = reused is None
        dump(output / "summary.json", summary)
        write_report(output, summary)
        write_predictor(output, artifact, encoder.export(), labels)
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                    if p.is_file() and p.name not in ("status.json","ARTIFACT_MANIFEST.json")}
        dump(output / "ARTIFACT_MANIFEST.json",manifest)
        dump(output / "status.json", {"stage": "complete", "status": "complete"})
        return summary
    except Exception as e:
        dump(output / "status.json", {"stage": "failed", "status": "error", "type": type(e).__name__, "message": str(e)})
        raise
