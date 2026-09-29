"""Phase 5a: in-scope evaluation on the official BANKING77 test set + cost table.

For every trained model:
  - caches score matrices for val / test / oos_val / oos_test in models/scores/ (used by oos.py)
  - accuracy, macro precision/recall/F1 (weighted-F1 only if the test set is not balanced)
  - per-class F1, predictions file with top-3 labels
  - cost: median CPU latency (batch 1, 200 test queries), size on disk, training time
Outputs: results/results_table.{csv,json} (T1), results/cost_table.csv (T4),
         results/per_class_f1_<model>.csv, results/predictions_<model>.csv
Run: python -m src.evaluate [--force]
"""
import argparse
import statistics
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import config
from src import predict
from src.utils import Timer, cuda_available, hardware, load_json, log_run, save_json, set_seed

SPLITS = ["val", "test", "oos_val", "oos_test"]


def dir_size_mb(path):
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def model_size_mb(key):
    return round(sum(dir_size_mb(f.parent if f.name == "config.json" else f)
                     for f in predict.model_files(key)), 1)


def cache_scores(key, force=False):
    """Score every evaluation split once and cache the matrices."""
    paths = {s: config.SCORES / f"{key}_{s}.npy" for s in SPLITS}
    if all(p.exists() for p in paths.values()) and not force:
        return {s: np.load(p) for s, p in paths.items()}
    device = "cuda" if (key == "m4_distilbert" and cuda_available()) else "cpu"
    m = predict.load_model(key, device=device)
    scores = {}
    for s in SPLITS:
        texts = pd.read_csv(config.DATA_PROC / f"{s}.csv")["text"]
        emb = np.load(config.EMB_CACHE / f"{s}.npy") if key == "m3_sbert_lr" else None
        scores[s] = score_texts_checked(m, texts, emb)
        np.save(paths[s], scores[s])
    return scores


def score_texts_checked(m, texts, emb):
    sc = predict.score_texts(m, texts, embeddings=emb)
    assert sc.shape == (len(texts), len(predict.LABELS))
    return sc


def cached_latency(key, texts, force=False):
    """Latency is measured once per model and cached, so reruns leave existing rows unchanged."""
    path = config.RESULTS / f"latency_{key}.json"
    if path.exists() and not force:
        return load_json(path)["median_latency_ms_cpu_batch1"]
    ms = measure_latency(key, texts)
    save_json({"model": key, "median_latency_ms_cpu_batch1": ms, "n_queries": len(texts),
               "batch_size": 1, "device": "cpu", "hardware": hardware()}, path)
    return ms


def measure_latency(key, texts):
    """Median milliseconds per query, batch size 1, on CPU (includes encoding/tokenising)."""
    m = predict.load_model(key, device="cpu")
    for t in texts[:10]:                      # warm-up
        predict.score_texts(m, [t])
    times = []
    for t in texts:
        start = time.perf_counter()
        predict.score_texts(m, [t])
        times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 2)


def main(force=False):
    set_seed()
    labels = np.array(predict.LABELS)
    test = pd.read_csv(config.DATA_PROC / "test.csv")
    balanced = load_json(config.RESULTS / "dataset_stats.json")["banking77"]["test_exactly_balanced"]
    rows, cost_rows = [], []

    with Timer() as t_all:
        for key in predict.available_models():
            scores = cache_scores(key, force)
            sc = scores["test"]
            pred = labels[sc.argmax(1)]
            y = test["label"].values
            has_proba = key in config.PROB_MODELS

            p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
            meta = load_json(config.RESULTS / f"train_meta_{key}.json")
            row = {"model": key, "name": config.MODEL_NAMES[key],
                   "accuracy": round(accuracy_score(y, pred), 4),
                   "macro_precision": round(p, 4), "macro_recall": round(r, 4), "macro_f1": round(f, 4),
                   "val_macro_f1": meta["val_macro_f1"],
                   "chosen_hyperparameter": meta.get("chosen_C", meta.get("chosen_epoch")),
                   "oos_gate": "yes" if has_proba else "no (no probabilities)"}
            if not balanced:
                row["weighted_f1"] = round(f1_score(y, pred, average="weighted"), 4)
            rows.append(row)
            print(f"{key:15s} test acc={row['accuracy']:.4f}  macro-F1={row['macro_f1']:.4f}")

            # Per-class precision / recall / F1
            pc = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
            pd.DataFrame({"label": labels, "precision": pc[0].round(4), "recall": pc[1].round(4),
                          "f1": pc[2].round(4), "support": pc[3]}) \
              .sort_values("f1").to_csv(config.RESULTS / f"per_class_f1_{key}.csv", index=False)

            # Predictions file with top-3
            top3 = np.argsort(-sc, axis=1)[:, :3]
            pr = pd.DataFrame({"text": test["text"], "true": y, "pred": pred,
                               "correct": y == pred,
                               "confidence": sc.max(1).round(4) if has_proba else np.nan})
            for k in range(3):
                pr[f"top{k + 1}"] = labels[top3[:, k]]
                pr[f"top{k + 1}_prob"] = sc[np.arange(len(sc)), top3[:, k]].round(4) if has_proba else np.nan
            pr.to_csv(config.RESULTS / f"predictions_{key}.csv", index=False)

            # Cost
            lat_texts = test["text"].sample(config.LATENCY_N, random_state=config.SEED).tolist()
            cost_rows.append({"model": key, "name": config.MODEL_NAMES[key],
                              "median_latency_ms_cpu_batch1": cached_latency(key, lat_texts, force),
                              "size_on_disk_mb": model_size_mb(key),
                              "train_seconds": meta["train_seconds"],
                              "train_device": meta.get("train_device", "CPU"),
                              "note": (f"classifier fit only; frozen MiniLM encoding of all splits took "
                                       f"{meta['encode_all_splits_seconds']}s on {meta['encode_device']}"
                                       if "encode_all_splits_seconds" in meta else
                                       meta.get("train_note", ""))})

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "results_table.csv", index=False)
    save_json({"test_set": "official BANKING77 test (3,080 queries, 77 intents)",
               "weighted_f1_reported": not balanced,
               "weighted_f1_note": "test set is exactly balanced (40 per intent), so weighted-F1 equals macro-F1 and is omitted" if balanced else "",
               "rows": rows}, config.RESULTS / "results_table.json")
    # The three one-change comparisons from the spec (each isolates one design choice)
    f1 = dict(zip(table["model"], table["macro_f1"]))
    pairs = [("m2_tfidf_svm", "m1_tfidf_lr", "classifier (TF-IDF fixed): LinearSVC vs LogReg"),
             ("m3_sbert_lr", "m1_tfidf_lr", "representation (LogReg fixed): MiniLM vs TF-IDF"),
             ("m4_distilbert", "m3_sbert_lr", "fine-tuning: DistilBERT vs frozen MiniLM + LogReg")]
    save_json([{"comparison": d, "model": a, "baseline": b, "macro_f1_model": f1[a], "macro_f1_baseline": f1[b],
                "macro_f1_difference": round(f1[a] - f1[b], 4)} for a, b, d in pairs if a in f1 and b in f1],
              config.RESULTS / "model_comparisons.json")
    cost = pd.DataFrame(cost_rows)
    cost["latency_hardware"] = hardware().split(";")[0]
    cost.to_csv(config.RESULTS / "cost_table.csv", index=False)
    print(table.to_string(index=False))
    print(cost.to_string(index=False))
    log_run("python -m src.evaluate" + (" --force" if force else ""), t_all.seconds,
            "; ".join(f"{r['model']} test macro-F1 {r['macro_f1']}" for r in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
