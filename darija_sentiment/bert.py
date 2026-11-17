"""Fine-tune DziriBERT (or any BERT-style model) for 3-class sentiment.

Default settings follow the project plan: lr 2e-5, batch 16, 4 epochs,
max length 128, weight decay 0.01, 10% warmup. The best checkpoint is picked by
validation macro F1. Heavy imports (torch, transformers) stay inside functions
so the rest of the package works without them.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from darija_sentiment import LABELS
from darija_sentiment.data import load_dataset, split_three

DEFAULT_MODEL = "alger-ia/dziribert"
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


@dataclass
class BertConfig:
    model_name: str = DEFAULT_MODEL
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 4
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


class TextDataset:
    """Minimal torch-style dataset of tokenized texts and integer labels."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        import torch

        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def encode(tokenizer, df, max_length: int) -> TextDataset:
    enc = tokenizer(list(df["clean"]), truncation=True, max_length=max_length, padding="max_length")
    return TextDataset(dict(enc), [LABEL2ID[label] for label in df["label"]])


def train_bert(
    data_path: str | Path,
    out_dir: str | Path,
    config: BertConfig | None = None,
    tokenizer=None,
    model=None,
) -> dict:
    """Fine-tune and save the best checkpoint to `out_dir/best`. Returns validation metrics.

    `tokenizer` and `model` can be passed in (used by tests with a tiny offline model);
    otherwise they are downloaded from `config.model_name`.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = config or BertConfig()
    set_seed(config.seed)
    out_dir = Path(out_dir)
    splits = split_three(load_dataset(data_path), seed=config.seed)

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model_name)
    model = model or AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    train_ds = encode(tokenizer, splits["train"], config.max_length)
    val_ds = encode(tokenizer, splits["val"], config.max_length)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config.seed,
        report_to=[],
        logging_strategy="epoch",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    result = {
        "val_macro_f1": float(metrics["eval_macro_f1"]),
        "val_accuracy": float(metrics["eval_accuracy"]),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "config": asdict(config),
    }
    best = out_dir / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class BertPredictor:
    """Same interface as `Predictor`, backed by a fine-tuned transformer."""

    def __init__(self, model_dir: str | Path, max_length: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.max_length = max_length

    def predict(self, text: str) -> dict:
        import torch

        from darija_sentiment.clean import clean_text

        enc = self.tokenizer(
            clean_text(text), truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0].tolist()
        scores = {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}
        return {"label": max(scores, key=scores.get), "scores": scores}
