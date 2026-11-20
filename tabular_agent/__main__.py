import argparse
import json
from pathlib import Path
from .workflow import run


def main():
    parser = argparse.ArgumentParser(description="Train a tabular model and export an explicit quadratic predictor.")
    parser.add_argument("csv", help="Training CSV")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, help="A NEW output directory")
    parser.add_argument("--external", help="Optional external CSV. Never used for training or selection.")
    parser.add_argument("--schema",help="JSON mapping feature names to numeric, categorical or ignore")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--landmarks", type=int, default=128)
    parser.add_argument("--tether", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--test-fraction", type=float, default=.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--approximation", choices=["geometric","taylor"], default="geometric")
    parser.add_argument("--approximation-order", type=int, choices=[1,2,3], default=2)
    parser.add_argument("--monitor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--original-shap",action="store_true",help="Also estimate grouped permutation SHAP for the original frozen model")
    parser.add_argument("--no-shap",action="store_true",help="Disable all SHAP computation and output")
    parser.add_argument("--shap-rows",type=int,default=16)
    parser.add_argument("--shap-background",type=int,default=8)
    parser.add_argument("--shap-permutations",type=int,default=4)
    args = vars(parser.parse_args())
    csv, target, output, external = [args.pop(k) for k in ("csv", "target", "output", "external")]
    schema = args.pop("schema")
    if schema:
        args["feature_schema"] = json.loads(Path(schema).read_text(encoding="utf-8-sig"))
    if args["epochs"] < 1 or args["landmarks"] < 1 or args["batch_size"] < 1:
        parser.error("epochs, landmarks and batch-size must be positive")
    if any(args[k]<1 for k in ("shap_rows","shap_background","shap_permutations")):
        parser.error("SHAP sample sizes and permutations must be positive")
    result = run(csv, target, output, args, external)
    print(json.dumps({"output": output, "results": result["results"]}, indent=2))


if __name__ == "__main__":
    main()
