"""Command line entry point: `python -m darija_sentiment <command>`."""

import argparse
import json
from pathlib import Path

from darija_sentiment.data import DEFAULT_DATA

DEFAULT_MODEL = Path("artifacts") / "baseline.joblib"
DEFAULT_SPLITS = Path("data") / "splits"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darija-sentiment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the TF-IDF baseline")
    p_train.add_argument("--data", default=str(DEFAULT_DATA))
    p_train.add_argument("--model", default=str(DEFAULT_MODEL))
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-char", action="store_true", help="drop the char n-gram block")
    p_train.add_argument("--no-word", action="store_true", help="drop the word n-gram block")

    p_split = sub.add_parser("split", help="write a stratified 70/15/15 split")
    p_split.add_argument("--data", default=str(DEFAULT_DATA))
    p_split.add_argument("--out", default=str(DEFAULT_SPLITS))
    p_split.add_argument("--seed", type=int, default=42)

    p_pred = sub.add_parser("predict", help="classify one comment")
    p_pred.add_argument("text")
    p_pred.add_argument("--model", default=str(DEFAULT_MODEL))

    args = parser.parse_args(argv)

    if args.command == "train":
        from darija_sentiment.baseline import train

        metrics = train(
            args.data,
            args.model,
            seed=args.seed,
            char_ngrams=not args.no_char,
            word_ngrams=not args.no_word,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "split":
        from darija_sentiment.data import write_splits

        print(json.dumps(write_splits(args.data, args.out, seed=args.seed)))
    elif args.command == "predict":
        from darija_sentiment.predict import Predictor

        result = Predictor(args.model).predict(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
