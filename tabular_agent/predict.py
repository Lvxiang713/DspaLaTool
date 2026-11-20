"""Batch prediction from saved JSON equations without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Encoder, read_table
from .equations import scores, kernel_votes


def load_artifact(directory):
    root = Path(directory)
    manifest = json.loads((root / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    required = ("encoder.json", "equation.json", "summary.json")
    for name in required:
        expected = manifest.get(name)
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise ValueError(f"Artifact integrity check failed: {name}")
    encoder = Encoder.load(json.loads((root / "encoder.json").read_text(encoding="utf-8")))
    equation = json.loads((root / "equation.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return encoder, equation, summary


def predict_frame(directory, frame, original_kernel=False):
    encoder, artifact, summary = load_artifact(directory)
    x = encoder.transform(frame)
    if not len(x):
        raise ValueError("Prediction CSV has no rows.")
    quadratic, contributions = scores(artifact, x)
    labels = summary["classes"]
    output = pd.DataFrame(index=np.arange(len(x)))
    if labels:
        pred = quadratic.argmax(1)
        output["prediction"] = [labels[i] for i in pred]
        if len(labels) == 2:
            q = quadratic[:,1] - quadratic[:,0]
            terms = contributions[:,1] - contributions[:,0]
            output["equation_score"] = q
            # Stable sigmoid without overflow.
            normalized = np.empty_like(q)
            positive = q >= 0
            normalized[positive] = 1/(1+np.exp(-q[positive]))
            expq = np.exp(q[~positive])
            normalized[~positive] = expq/(1+expq)
            output["normalized_score"] = normalized
        else:
            output["equation_score"] = quadratic[np.arange(len(x)),pred]
            terms = contributions[np.arange(len(x)),pred]
            output["explained_class"] = output["prediction"]
        if original_kernel:
            votes = kernel_votes(artifact,x)
            output["original_kernel_prediction"] = [labels[i] for i in votes.argmax(1)]
    else:
        if original_kernel:
            raise ValueError("Original-kernel export is available only for native classification.")
        output["prediction"] = quadratic[:,0]
        output["equation_score"] = quadratic[:,0]
        terms = contributions[:,0]
    # Group observed contributions only, matching the exported web predictor.
    for group in dict.fromkeys(encoder.groups):
        columns = [j for j,g in enumerate(encoder.groups) if g == group]
        observed = np.isfinite(x[:,columns]).any(1)
        value = terms[:,columns].sum(1)
        output[f"contribution:{group}"] = np.where(observed,value,np.nan)
    return output


def main():
    parser = argparse.ArgumentParser(description="Predict a CSV from an existing saved quadratic analysis")
    parser.add_argument("run", help="Completed analysis directory")
    parser.add_argument("csv", help="New rows with the original feature columns")
    parser.add_argument("--output",required=True,help="New prediction CSV, never overwritten")
    parser.add_argument("--original-kernel",action="store_true",help="Also evaluate the exact exported native kernel mixture")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists. Choose a new CSV path.")
    result = predict_frame(args.run,read_table(args.csv),args.original_kernel)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create also protects against a second writer after the check.
    with output.open("x",encoding="utf-8",newline="") as stream:
        result.to_csv(stream,index=False)
    print(f"Predicted {len(result)} rows. Saved {output}.")


if __name__ == "__main__":
    main()
