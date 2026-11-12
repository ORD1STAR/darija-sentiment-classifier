"""Command line entry point: `python -m darija_sentiment <command>`."""

import argparse
import json
from pathlib import Path

from darija_sentiment.data import DEFAULT_DATA

DEFAULT_MODEL = Path("artifacts") / "baseline.joblib"
DEFAULT_SPLITS = Path("data") / "splits"
DEFAULT_REPORTS = Path("reports")
DEFAULT_PARAMS = Path("artifacts") / "best_params.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darija-sentiment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the TF-IDF baseline")
    p_train.add_argument("--data", default=str(DEFAULT_DATA))
    p_train.add_argument("--model", default=str(DEFAULT_MODEL))
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-char", action="store_true", help="drop the char n-gram block")
    p_train.add_argument("--no-word", action="store_true", help="drop the word n-gram block")
    p_train.add_argument("--params", help="JSON file written by `tune`")

    p_tune = sub.add_parser("tune", help="cross-validate C and class weights on train")
    p_tune.add_argument("--data", default=str(DEFAULT_DATA))
    p_tune.add_argument("--out", default=str(DEFAULT_PARAMS))
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.add_argument("--folds", type=int, default=5)

    p_split = sub.add_parser("split", help="write a stratified 70/15/15 split")
    p_split.add_argument("--data", default=str(DEFAULT_DATA))
    p_split.add_argument("--out", default=str(DEFAULT_SPLITS))
    p_split.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("evaluate", help="evaluate a saved model on a split")
    p_eval.add_argument("--data", default=str(DEFAULT_DATA))
    p_eval.add_argument("--model", default=str(DEFAULT_MODEL))
    p_eval.add_argument("--split", choices=("train", "val", "test"), default="test")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--out", default=str(DEFAULT_REPORTS))

    p_val = sub.add_parser("validate", help="check a labeled CSV against the schema")
    p_val.add_argument("path")
    p_val.add_argument("--allow-unlabeled", action="store_true")

    p_agree = sub.add_parser("agreement", help="Cohen's kappa between two label files")
    p_agree.add_argument("file_a")
    p_agree.add_argument("file_b")

    p_pred = sub.add_parser("predict", help="classify one comment")
    p_pred.add_argument("text")
    p_pred.add_argument("--model", default=str(DEFAULT_MODEL))

    args = parser.parse_args(argv)

    if args.command == "train":
        from darija_sentiment.baseline import train

        params = {}
        if args.params:
            from darija_sentiment.tune import load_params

            params = load_params(args.params)
        metrics = train(
            args.data,
            args.model,
            seed=args.seed,
            char_ngrams=not args.no_char,
            word_ngrams=not args.no_word,
            **params,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "split":
        from darija_sentiment.data import write_splits

        print(json.dumps(write_splits(args.data, args.out, seed=args.seed)))
    elif args.command == "tune":
        from darija_sentiment.tune import tune

        print(json.dumps(tune(args.data, args.out, seed=args.seed, folds=args.folds), indent=2))
    elif args.command == "evaluate":
        from darija_sentiment.evaluate import evaluate

        metrics = evaluate(args.model, args.data, args.split, seed=args.seed, out_dir=args.out)
        f1, acc = metrics["macro_f1"], metrics["accuracy"]
        print(f"macro F1 ({args.split}): {f1:.3f}  accuracy: {acc:.3f}")
        print(f"report written to {args.out}/metrics.json")
    elif args.command == "validate":
        from darija_sentiment.validate import validate_file

        problems = validate_file(args.path, allow_unlabeled=args.allow_unlabeled)
        for problem in problems:
            print(f"- {problem}")
        print("valid" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    elif args.command == "agreement":
        from darija_sentiment.agreement import GOOD_KAPPA, agreement_files

        result = agreement_files(args.file_a, args.file_b)
        print(f"shared ids: {result['n_shared']}")
        print(f"raw agreement: {result['raw_agreement']:.3f}")
        print(f"Cohen's kappa: {result['kappa']:.3f}")
        print(result["table"].to_string())
        if result["kappa"] < GOOD_KAPPA:
            print(f"warning: kappa below {GOOD_KAPPA}, refine the guide and relabel unclear cases")
    elif args.command == "predict":
        from darija_sentiment.predict import Predictor

        result = Predictor(args.model).predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
