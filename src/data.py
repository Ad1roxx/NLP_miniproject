"""Phase 1: download (cached), load, split and save BANKING77 + the CLINC150 OOS queries.

Outputs: data/processed/{train,val,test,oos_val,oos_test}.csv, labels.json,
         results/dataset_stats.json
Run:     python -m src.data [--force]
"""
import argparse
import json

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

import config
from src.utils import Timer, load_json, log_run, save_json, set_seed


def download(url, dest, force=False):
    """Download once; later runs reuse the cached file in data/raw/."""
    if dest.exists() and not force:
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"downloaded {url} -> {dest.name}")


def load_banking(path):
    """Read a BANKING77 CSV and standardise columns to (text, label)."""
    df = pd.read_csv(path)
    print(f"{path.name}: columns = {list(df.columns)}")
    # The official files use 'text' and 'category'; map whatever the intent column is to 'label'
    label_col = [c for c in df.columns if c != "text"][0]
    return df.rename(columns={label_col: "label"})[["text", "label"]]


def norm(s):
    """Normalisation used only for the duplicate check (lowercase + strip)."""
    return s.str.lower().str.strip()


def main(force=False):
    set_seed()
    out = config.DATA_PROC
    if (out / "oos_test.csv").exists() and not force:
        print("processed data exists; skipping (use --force to rebuild)")
        return

    with Timer() as t:
        raw_train = config.DATA_RAW / "banking_train.csv"
        raw_test = config.DATA_RAW / "banking_test.csv"
        raw_clinc = config.DATA_RAW / "clinc_data_full.json"
        download(config.BANKING_TRAIN_URL, raw_train, force)
        download(config.BANKING_TEST_URL, raw_test, force)
        download(config.CLINC_URL, raw_clinc, force)

        full_train = load_banking(raw_train)
        test = load_banking(raw_test)
        labels = sorted(full_train["label"].unique())

        # Stop if the data is not what the spec expects
        counts_ok = (len(full_train) == config.EXPECTED_TRAIN and len(test) == config.EXPECTED_TEST
                     and len(labels) == config.EXPECTED_LABELS
                     and set(test["label"]) == set(labels))
        if not counts_ok:
            raise SystemExit(f"Unexpected counts: train={len(full_train)} test={len(test)} "
                             f"labels={len(labels)} -- stopping as instructed.")

        # Stratified 10% validation split; the official test set is never touched
        train, val = train_test_split(full_train, test_size=config.VAL_FRACTION,
                                      stratify=full_train["label"], random_state=config.SEED)

        # CLINC150: keep only the out-of-scope queries (in-scope CLINC intents are ignored)
        clinc = load_json(raw_clinc)
        print(f"CLINC keys: {list(clinc.keys())}")
        oos = {k: pd.DataFrame(clinc[k], columns=["text", "label"])
               for k in ("oos_train", "oos_val", "oos_test")}
        for k, df in oos.items():
            assert (df["label"] == config.OOS_LABEL).all(), k

        # Save processed splits (oos_train is deliberately unused: models train on in-scope data only)
        train.to_csv(out / "train.csv", index=False)
        val.to_csv(out / "val.csv", index=False)
        test.to_csv(out / "test.csv", index=False)
        oos["oos_val"].to_csv(out / "oos_val.csv", index=False)
        oos["oos_test"].to_csv(out / "oos_test.csv", index=False)
        save_json(labels, out / "labels.json")

        # Duplicates between train/val and test: logged, NOT removed
        test_norm = set(norm(test["text"]))
        dup_train = int(norm(train["text"]).isin(test_norm).sum())
        dup_val = int(norm(val["text"]).isin(test_norm).sum())
        dup_train_val = int(norm(train["text"]).isin(set(norm(val["text"]))).sum())
        dup_examples = sorted(set(norm(full_train["text"])) & test_norm)[:10]

        train_counts = full_train["label"].value_counts()
        test_counts = test["label"].value_counts()
        val_counts = val["label"].value_counts()
        stats = {
            "banking77": {
                "official_train_rows": len(full_train), "test_rows": len(test),
                "n_labels": len(labels),
                "train_split_rows": len(train), "val_split_rows": len(val),
                "train_per_class_min": int(train_counts.min()),
                "train_per_class_max": int(train_counts.max()),
                "train_per_class_mean": round(float(train_counts.mean()), 2),
                "train_imbalance_ratio_max_over_min": round(float(train_counts.max() / train_counts.min()), 2),
                "val_per_class_min": int(val_counts.min()), "val_per_class_max": int(val_counts.max()),
                "test_per_class_min": int(test_counts.min()), "test_per_class_max": int(test_counts.max()),
                "test_exactly_balanced": bool(test_counts.nunique() == 1),
                "train_per_class_counts": {k: int(v) for k, v in train_counts.sort_index().items()},
                "test_per_class_counts": {k: int(v) for k, v in test_counts.sort_index().items()},
            },
            "clinc_oos": {k: len(v) for k, v in oos.items()},
            "clinc_oos_note": "oos_val tunes the threshold; oos_test is final evaluation; oos_train is unused.",
            "duplicates_with_test_after_lowercase_strip": {
                "train_split_rows_also_in_test": dup_train,
                "val_split_rows_also_in_test": dup_val,
                "train_split_rows_also_in_val": dup_train_val,
                "examples": dup_examples,
                "action": "logged only, not removed",
            },
            "split": {"method": "stratified train_test_split", "val_fraction": config.VAL_FRACTION,
                      "random_state": config.SEED},
        }
        save_json(stats, config.RESULTS / "dataset_stats.json")

    print(json.dumps({k: v for k, v in stats["banking77"].items() if "counts" not in k}, indent=1))
    print("OOS:", stats["clinc_oos"])
    print("duplicates:", {k: v for k, v in stats["duplicates_with_test_after_lowercase_strip"].items()
                          if k != "examples"})
    log_run("python -m src.data" + (" --force" if force else ""), t.seconds,
            f"train {len(train)} / val {len(val)} / test {len(test)}, {len(labels)} labels; "
            f"OOS {stats['clinc_oos']}; test balanced={stats['banking77']['test_exactly_balanced']}; "
            f"train/val rows duplicated in test: {dup_train}/{dup_val}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
