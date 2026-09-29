"""Phase 3: M1 TF-IDF + Logistic Regression and M2 TF-IDF + LinearSVC.

Features (shared by M1 and M2): word 1-2 grams + char_wb 2-5 grams, sublinear TF,
on text cleaned by src.preprocess.clean_text. C is chosen from a 2-value grid by
validation macro-F1; the model trained on the train split with that C is kept
(it is NOT refit on train+val, so validation stays clean for threshold tuning).

Run: python -m src.train_tfidf [--force]
"""
import argparse
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import config
from src.preprocess import clean_text
from src.utils import Timer, log_run, save_json, set_seed


def make_features():
    """Word + character TF-IDF, concatenated side by side."""
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean_text, analyzer="word",
                                 ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                                 ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


def make_classifier(key, C):
    if key == "m1_tfidf_lr":
        return LogisticRegression(C=C, max_iter=config.LR_MAX_ITER, random_state=config.SEED)
    return LinearSVC(C=C, random_state=config.SEED)


def train_one(key, grid, train, val, force=False):
    path = config.MODELS / f"{key}.joblib"
    if path.exists() and not force:
        print(f"{key}: model exists; skipping (use --force)")
        return
    set_seed()
    trials, best = [], None
    for C in grid:
        with Timer() as t:
            pipe = Pipeline([("tfidf", make_features()), ("clf", make_classifier(key, C))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pipe.fit(train["text"], train["label"])
        val_f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
        print(f"{key}  C={C:<5} val macro-F1={val_f1:.4f}  ({t.seconds:.1f}s)")
        trials.append({"C": C, "val_macro_f1": round(val_f1, 4), "train_seconds": round(t.seconds, 2)})
        if best is None or val_f1 > best[1]:
            best = (pipe, val_f1, C, t.seconds)

    pipe, val_f1, C, secs = best
    joblib.dump(pipe, path)
    meta = {"model": key, "chosen_C": C, "val_macro_f1": round(val_f1, 4),
            "train_seconds": round(secs, 2), "grid": trials,
            "n_features": int(pipe.named_steps["tfidf"].transform(["x"]).shape[1])}
    save_json(meta, config.RESULTS / f"train_meta_{key}.json")
    log_run(f"python -m src.train_tfidf ({key})", sum(tr["train_seconds"] for tr in trials),
            f"C grid {grid} -> chose C={C} (val macro-F1 {val_f1:.4f}); saved {path.name}")


def main(force=False):
    train = pd.read_csv(config.DATA_PROC / "train.csv")
    val = pd.read_csv(config.DATA_PROC / "val.csv")
    train_one("m1_tfidf_lr", config.M1_C_GRID, train, val, force)
    train_one("m2_tfidf_svm", config.M2_C_GRID, train, val, force)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
