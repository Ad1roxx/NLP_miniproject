"""Phase 2: EDA figures F1 (class distribution) and F2 (query length); adds length stats.

Run: python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.utils import Timer, load_json, log_run, save_json

# One figure style for the whole project: white background, readable fonts, quiet axes
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.edgecolor": config.INK_MUTED, "axes.labelcolor": config.INK,
    "xtick.color": config.INK_MUTED, "ytick.color": config.INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": config.GRID, "axes.axisbelow": True,
}
plt.rcParams.update(STYLE)


def main():
    with Timer() as t:
        train = pd.read_csv(config.DATA_PROC / "train.csv")
        val = pd.read_csv(config.DATA_PROC / "val.csv")
        test = pd.read_csv(config.DATA_PROC / "test.csv")
        full_train = pd.concat([train, val])  # the official training file (train + our val split)

        # F1: examples per intent in the official training file, sorted
        counts = full_train["label"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.bar(range(len(counts)), counts.values, color=config.MODEL_COLORS["m1_tfidf_lr"], width=0.8)
        ax.axhline(counts.mean(), color=config.INK_MUTED, ls="--", lw=1.2)
        ax.text(1, counts.mean() + 3, f"mean = {counts.mean():.0f}", color=config.INK_MUTED)
        ax.set_xticks([])
        ax.set_xlabel(f"77 intents, sorted by training examples (fewest: {counts.index[0]}, "
                      f"most: {counts.index[-1]})")
        ax.set_ylabel("Training examples")
        ax.set_title(f"F1. BANKING77 training set is imbalanced: {counts.min()} to {counts.max()} "
                     "examples per intent (test set: 40 each)")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(config.FIGURES / "f1_class_distribution.png", dpi=200)
        plt.close(fig)

        # F2: query length in whitespace tokens
        lengths = full_train["text"].str.split().str.len()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(lengths, bins=range(0, int(lengths.max()) + 2), color=config.MODEL_COLORS["m1_tfidf_lr"],
                edgecolor="white", linewidth=1)
        ax.axvline(lengths.mean(), color=config.INK, ls="--", lw=1.2)
        ax.text(lengths.mean() + 0.8, ax.get_ylim()[1] * 0.9, f"mean = {lengths.mean():.1f} tokens",
                color=config.INK)
        ax.set_xlabel("Query length (whitespace tokens)")
        ax.set_ylabel("Number of queries")
        ax.set_title("F2. Banking queries are short (official training set)")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(config.FIGURES / "f2_query_length.png", dpi=200)
        plt.close(fig)

        # Add length statistics to dataset_stats.json
        stats = load_json(config.RESULTS / "dataset_stats.json")
        test_len = test["text"].str.split().str.len()
        stats["query_length_tokens"] = {
            "train_mean": round(float(lengths.mean()), 2), "train_median": float(np.median(lengths)),
            "train_max": int(lengths.max()), "train_p99": float(np.percentile(lengths, 99)),
            "test_mean": round(float(test_len.mean()), 2), "test_max": int(test_len.max()),
            "tokeniser": "whitespace split of raw text",
        }
        save_json(stats, config.RESULTS / "dataset_stats.json")
    print(stats["query_length_tokens"])
    log_run("python -m src.eda", t.seconds, "F1, F2 saved; length stats added to dataset_stats.json")


if __name__ == "__main__":
    main()
