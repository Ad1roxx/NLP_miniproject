"""Phase 4: M3 = frozen all-MiniLM-L6-v2 sentence embeddings + Logistic Regression.

Raw text is encoded once (384-dim, L2-normalised) and cached to models/embeddings/<split>.npy.
Only the Logistic Regression is trained; C is chosen from {1, 10} by validation macro-F1.
The encoder is saved to models/sbert_encoder/ so the demo works offline.

Run: python -m src.train_sbert [--force]
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import config
from src.utils import Timer, log_run, save_json, set_seed

SPLITS = ["train", "val", "test", "oos_val", "oos_test"]
ENCODER_DIR = config.MODELS / "sbert_encoder"


def load_encoder():
    """Load the saved local copy if present, otherwise download from Hugging Face."""
    from sentence_transformers import SentenceTransformer
    src = str(ENCODER_DIR) if ENCODER_DIR.exists() else config.SBERT_NAME
    return SentenceTransformer(src)


def get_embeddings(split, encoder=None, force=False):
    """Return cached embeddings for a split, encoding them first if needed."""
    path = config.EMB_CACHE / f"{split}.npy"
    if path.exists() and not force:
        return np.load(path)
    texts = pd.read_csv(config.DATA_PROC / f"{split}.csv")["text"].tolist()
    emb = encoder.encode(texts, batch_size=config.SBERT_BATCH, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True)
    np.save(path, emb)
    return emb


def main(force=False):
    set_seed()
    model_path = config.MODELS / "m3_sbert_lr.joblib"
    if model_path.exists() and not force:
        print("m3_sbert_lr: model exists; skipping (use --force)")
        return

    # 1) Encode every split once (the encoder stays frozen)
    with Timer() as t_enc:
        encoder = load_encoder()
        if not ENCODER_DIR.exists():
            encoder.save(str(ENCODER_DIR))
        embs = {s: get_embeddings(s, encoder, force) for s in SPLITS}
    device = str(encoder.device)
    print(f"embeddings ready on {device}: " + ", ".join(f"{s}={e.shape}" for s, e in embs.items())
          + f"  ({t_enc.seconds:.1f}s)")

    # 2) Train the classifier; choose C on validation
    y_train = pd.read_csv(config.DATA_PROC / "train.csv")["label"]
    y_val = pd.read_csv(config.DATA_PROC / "val.csv")["label"]
    trials, best = [], None
    for C in config.M3_C_GRID:
        with Timer() as t:
            clf = LogisticRegression(C=C, max_iter=config.LR_MAX_ITER, random_state=config.SEED)
            clf.fit(embs["train"], y_train)
        val_f1 = f1_score(y_val, clf.predict(embs["val"]), average="macro")
        print(f"m3_sbert_lr  C={C:<3} val macro-F1={val_f1:.4f}  ({t.seconds:.1f}s)")
        trials.append({"C": C, "val_macro_f1": round(val_f1, 4), "train_seconds": round(t.seconds, 2)})
        if best is None or val_f1 > best[1]:
            best = (clf, val_f1, C, t.seconds)

    clf, val_f1, C, secs = best
    joblib.dump(clf, model_path)
    save_json({"model": "m3_sbert_lr", "chosen_C": C, "val_macro_f1": round(val_f1, 4),
               # training cost = encoding the training split + fitting the classifier
               "train_seconds": round(secs, 2), "encode_all_splits_seconds": round(t_enc.seconds, 2),
               "encode_device": device, "grid": trials, "embedding_dim": int(embs["train"].shape[1])},
              config.RESULTS / "train_meta_m3_sbert_lr.json")
    log_run("python -m src.train_sbert", t_enc.seconds + sum(tr["train_seconds"] for tr in trials),
            f"encoded {len(SPLITS)} splits on {device} in {t_enc.seconds:.1f}s; "
            f"C grid {config.M3_C_GRID} -> C={C} (val macro-F1 {val_f1:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
