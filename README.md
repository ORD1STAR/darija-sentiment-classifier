# Darija Sentiment Classifier

Sentiment classification (positive / negative / neutral) for Algerian Darija
comments written in Arabic script, Latin script (Arabizi) or mixed with French.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m darija_sentiment train                  # trains the TF-IDF baseline
python -m darija_sentiment predict "sa7a ftorkom, video zina"
python -m pytest -q
python -m ruff check .
```

`data/processed/sample.csv` is a small hand-written seed set (96 comments) used
to develop the pipeline. The real dataset is collected and labeled separately.

## Layout

```
darija_sentiment/   package: cleaning, data loading, baseline, inference, CLI
data/processed/     labeled CSV files (text, label, script, source, topic)
tests/              pytest suite
```
