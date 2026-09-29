"""Light text cleaning for the TF-IDF models only (embedding/transformer models get raw text).

Steps: lowercase -> currency symbols to <cur> -> numbers to <num> -> collapse whitespace.
No stop-word removal and no stemming: words like "not", "why", "still" carry intent.
Run:   python -m src.preprocess   (saves 10 before/after examples)
"""
import re

import pandas as pd

import config

CURRENCY_RE = re.compile(r"[£$€₹]")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = str(text).lower().strip()
    text = CURRENCY_RE.sub(" <cur> ", text)
    text = NUMBER_RE.sub(" <num> ", text)
    return SPACE_RE.sub(" ", text).strip()


def save_examples(n=10):
    """Save before/after pairs, preferring queries that actually change (numbers, currency)."""
    train = pd.read_csv(config.DATA_PROC / "train.csv")
    has_special = train[train["text"].str.contains(r"[\d£$€₹]", regex=True)]
    picks = pd.concat([has_special.sample(7, random_state=config.SEED),
                       train.drop(has_special.index).sample(n - 7, random_state=config.SEED)])
    out = pd.DataFrame({"before": picks["text"], "after": picks["text"].map(clean_text),
                        "label": picks["label"]})
    out.to_csv(config.RESULTS / "preprocessing_examples.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    save_examples()
