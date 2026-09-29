"""Phase 8 (SHOULD, CUDA only): M4 = distilbert-base-uncased fine-tuned on the 77 intents.

Standalone on purpose: it only needs data/processed/{train,val}.csv + labels.json and writes
models/distilbert/ (+ results/train_meta_m4_distilbert.json), so it can also run on Google Colab
(copy data/processed and this file, run it, copy models/distilbert back).

Settings (spec section 6): max_length 64, lr 5e-5, batch 32, weight decay 0.01, warmup ratio 0.1,
fp16 on CUDA, evaluate every epoch on validation macro-F1, keep the best epoch.
Epochs: up to 20 with early stopping (stop after 3 epochs without a validation macro-F1 gain).
The first run used a fixed 3 epochs and was still improving (archived as results/m4_3epoch_*).

Run: python -m src.train_distilbert [--force]     (or: python src/train_distilbert.py)
"""
import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "distilbert"
CKPT = ROOT / "models" / "distilbert_checkpoints"   # temporary, deleted after training
RESULTS = ROOT / "results"

MODEL_NAME = "distilbert-base-uncased"
SEED, MAX_LEN, LR, BATCH, WEIGHT_DECAY, WARMUP_RATIO = 42, 64, 5e-5, 32, 0.01, 0.1
MAX_EPOCHS, PATIENCE = 20, 3   # early stopping on validation macro-F1


class IntentDataset(torch.utils.data.Dataset):
    """Tokenised queries + integer labels; padding is done per batch by the collator."""
    def __init__(self, texts, label_ids, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = list(label_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def compute_metrics(pred):
    logits, y = pred
    return {"macro_f1": f1_score(y, logits.argmax(-1), average="macro"),
            "accuracy": float((logits.argmax(-1) == y).mean())}


def main(force=False):
    if (OUT / "config.json").exists() and not force:
        print("m4_distilbert: model exists; skipping (use --force)")
        return
    if not torch.cuda.is_available():
        print("DistilBERT not run (no GPU)")
        return
    set_seed(SEED)
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    label2id = {l: i for i, l in enumerate(labels)}
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)
    train_ds = IntentDataset(train["text"], train["label"].map(label2id), tokenizer)
    val_ds = IntentDataset(val["text"], val["label"].map(label2id), tokenizer)

    # warmup: transformers 5 takes a float ratio in warmup_steps; 4.x has warmup_ratio
    params = inspect.signature(TrainingArguments.__init__).parameters
    warmup = {"warmup_ratio": WARMUP_RATIO} if "warmup_ratio" in params else {"warmup_steps": WARMUP_RATIO}
    args = TrainingArguments(
        output_dir=str(CKPT), num_train_epochs=MAX_EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=128,
        weight_decay=WEIGHT_DECAY, fp16=True, seed=SEED, data_seed=SEED,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,   # best + latest are kept
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_strategy="epoch", report_to="none", **warmup)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])

    device = torch.cuda.get_device_name(0)
    print(f"training on {device} (model device: {next(model.parameters()).device})")
    start = time.perf_counter()
    trainer.train()
    secs = time.perf_counter() - start

    # Per-epoch validation history -> best epoch (the checkpoint the Trainer reloaded)
    history = [{"epoch": int(round(h["epoch"])), "val_macro_f1": round(h["eval_macro_f1"], 4),
                "val_accuracy": round(h["eval_accuracy"], 4)}
               for h in trainer.state.log_history if "eval_macro_f1" in h]
    best = max(history, key=lambda h: h["val_macro_f1"])
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    shutil.rmtree(CKPT, ignore_errors=True)

    epochs_run = len(history)
    meta = {"model": "m4_distilbert", "base_model": MODEL_NAME, "chosen_epoch": best["epoch"],
            "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE, "epochs_run": epochs_run,
            "stopped_early": epochs_run < MAX_EPOCHS,
            "val_macro_f1": best["val_macro_f1"], "train_seconds": round(secs, 2),
            "train_device": f"GPU ({device})", "epoch_history": history,
            "settings": {"max_length": MAX_LEN, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR, "batch": BATCH,
                         "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO, "fp16": True,
                         "seed": SEED, "warmup_arg_used": list(warmup)[0],
                         "transformers_version": __import__("transformers").__version__},
            "train_note": f"fine-tuning on {device}; latency is still measured on CPU"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "train_meta_m4_distilbert.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    try:   # log to the run log when run inside the project
        from src.utils import log_run
        log_run("python -m src.train_distilbert", secs,
                f"trained on {device}; {epochs_run} epochs run (max {MAX_EPOCHS}, patience {PATIENCE}); "
                f"best epoch {best['epoch']} val macro-F1 {best['val_macro_f1']}; "
                f"per-epoch val macro-F1 {[h['val_macro_f1'] for h in history]}; "
                f"saved models/distilbert/")
    except ImportError:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
