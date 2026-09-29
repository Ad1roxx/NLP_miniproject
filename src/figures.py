"""Result figures F5-F8, all read from files in results/ (and cached test scores for F7/F8b).

F5 model comparison, F6 per-class F1 (best model), F7 zoomed confusion heatmap,
F8a threshold sweep on validation, F8b confidence histograms on test (demo model).
Run: python -m src.figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
from src import predict
from src.eda import STYLE  # same style as the EDA figures
from src.error_analysis import best_model
from src.utils import Timer, load_json, log_run

plt.rcParams.update(STYLE)
OUT = config.FIGURES


def f5_model_comparison():
    t = pd.read_csv(config.RESULTS / "results_table.csv")
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(t)
    width = 0.8 / n
    for i, r in enumerate(t.itertuples()):
        xs = np.arange(len(metrics)) + (i - (n - 1) / 2) * width
        vals = [getattr(r, m) for m, _ in metrics]
        bars = ax.bar(xs, vals, width * 0.92, color=config.MODEL_COLORS[r.model], label=r.name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, color=config.INK)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    lo = t[["accuracy", "macro_f1"]].min().min()
    ax.set_ylim(max(0, np.floor(lo * 20) / 20 - 0.05), 1.0)   # zoomed axis: stated in the label
    ax.set_ylabel("Score on BANKING77 test (axis starts above 0)")
    ax.set_title("F5. In-scope test performance per model (3,080 queries, 77 intents)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "f5_model_comparison.png", dpi=200)
    plt.close(fig)


def f6_per_class_f1(key):
    pc = pd.read_csv(config.RESULTS / f"per_class_f1_{key}.csv").sort_values("f1")
    sel = pd.concat([pc.head(15), pc.tail(5)])
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.r_[np.arange(15), np.arange(15, 20) + 0.8]   # small gap between the two groups
    colors = [config.MODEL_COLORS[key] if i >= 15 else config.INK_MUTED for i in range(len(sel))]
    ax.barh(y, sel["f1"], color=colors, height=0.75)
    for yi, v in zip(y, sel["f1"]):
        ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=10, color=config.INK)
    ax.set_yticks(y, sel["label"].str.replace("_", " "), fontsize=10)
    ax.axhline(15.4, color=config.INK_MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Test F1")
    ax.set_title(f"F6. Test F1 of the hardest 15 (grey) and easiest 5 (colour) intents\n"
                 f"{config.MODEL_NAMES[key]}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "f6_per_class_f1.png", dpi=200)
    plt.close(fig)


def f7_confusion_zoom(key, n_intents=15):
    pr = pd.read_csv(config.RESULTS / f"predictions_{key}.csv")
    cp = pd.read_csv(config.RESULTS / "confused_pairs.csv").drop_duplicates("rank")
    # intents that appear most often in the top confused pairs
    order = []
    for r in cp.itertuples():
        for x in (r.intent_a, r.intent_b):
            if x not in order:
                order.append(x)
    order = order[:n_intents]
    sub = pr[pr["true"].isin(order)]
    cm = confusion_matrix(sub["true"], sub["pred"], labels=order)
    off = cm.astype(float)
    np.fill_diagonal(off, np.nan)   # diagonal (correct) would swamp the colour scale
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(off, cmap="Blues", vmin=0)
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm[i, j]
            if i == j:
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color=config.INK_MUTED)
            elif v:
                ax.text(j, i, v, ha="center", va="center", fontsize=10,
                        color="white" if v > np.nanmax(off) * 0.6 else config.INK)
    names = [o.replace("_", " ") for o in order]
    ax.set_xticks(range(len(order)), names, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)), names, fontsize=9)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.grid(False)
    ax.set_title(f"F7. Where the {n_intents} most-confused intents go wrong\n"
                 f"{config.MODEL_NAMES[key]}, test set; grey diagonal = correct count")
    fig.colorbar(im, ax=ax, fraction=0.04, label="Misclassified test queries")
    fig.tight_layout()
    fig.savefig(OUT / "f7_confusion_zoom.png", dpi=200)
    plt.close(fig)


def f8a_threshold_sweep():
    th = load_json(config.RESULTS / "thresholds.json")["models"]
    keys = list(th)
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        sw = pd.read_csv(config.RESULTS / f"threshold_sweep_{key}.csv")
        c = config.MODEL_COLORS[key]
        ax.plot(sw["tau"], sw["in_scope_acc_with_rejection"], color=c, lw=2, label="In-scope acc. with rejection")
        ax.plot(sw["tau"], sw["oos_recall"], color=c, lw=2, ls="--", label="OOS recall")
        tau = th[key]["tau"]
        ax.axvline(tau, color=config.INK, lw=1)
        right = tau > 0.75   # keep the label inside the panel
        ax.text(tau - 0.02 if right else tau + 0.02, 0.05, f"chosen tau = {tau:.2f}", color=config.INK,
                ha="right" if right else "left")
        ax.set_title(config.MODEL_NAMES[key])
        ax.set_xlabel("Threshold tau on max probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("Validation score")
    from matplotlib.lines import Line2D   # legend explains line style, so draw it in neutral ink
    handles = [Line2D([], [], color=config.INK, lw=2), Line2D([], [], color=config.INK, lw=2, ls="--")]
    lbls = ["In-scope acc. with rejection", "OOS recall"]
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle("F8a. Threshold sweep on validation (BANKING77 val + 100 CLINC OOS): "
                 "raising tau rejects more OOS but also more banking queries", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "f8a_threshold_sweep.png", dpi=200)
    plt.close(fig)


def f8b_confidence_hist(key):
    tau = load_json(config.RESULTS / "thresholds.json")["models"][key]["tau"]
    conf_in = np.load(config.SCORES / f"{key}_test.npy").max(1)
    conf_oos = np.load(config.SCORES / f"{key}_oos_test.npy").max(1)
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(conf_in, bins=bins, color=config.MODEL_COLORS[key], alpha=0.85,
            label=f"BANKING77 test (in-scope, n={len(conf_in)})", edgecolor="white", linewidth=0.8)
    ax.hist(conf_oos, bins=bins, color=config.INK_MUTED, alpha=0.75,
            label=f"CLINC oos_test (out-of-scope, n={len(conf_oos)})", edgecolor="white", linewidth=0.8)
    ax.axvline(tau, color=config.INK, lw=1.5)
    ax.text(tau - 0.01, ax.get_ylim()[1] * 0.92, "< human agent", ha="right", color=config.INK)
    ax.text(tau + 0.01, ax.get_ylim()[1] * 0.92, f"accept (tau = {tau:.2f}) >", ha="left", color=config.INK)
    ax.set_xlabel("Max predicted probability (confidence)")
    ax.set_ylabel("Number of test queries")
    ax.set_title(f"F8b. Confidence separates in-scope from OOS queries - {config.MODEL_NAMES[key]}")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "f8b_confidence_hist.png", dpi=200)
    plt.close(fig)


def f9_m4_validation_curves():
    """Per-epoch validation macro-F1 of both DistilBERT runs (initial 3-epoch run vs early stopping)."""
    final = load_json(config.RESULTS / "train_meta_m4_distilbert.json")
    initial_path = config.RESULTS / "m4_3epoch_train_meta.json"
    fig, ax = plt.subplots(figsize=(9, 5))
    c = config.MODEL_COLORS["m4_distilbert"]
    if initial_path.exists():
        h0 = load_json(initial_path)["epoch_history"]
        ax.plot([h["epoch"] for h in h0], [h["val_macro_f1"] for h in h0], color=config.INK_MUTED,
                lw=2, ls="--", marker="o", ms=8, label="Initial run: fixed 3 epochs")
    h1 = final["epoch_history"]
    ax.plot([h["epoch"] for h in h1], [h["val_macro_f1"] for h in h1], color=c, lw=2, marker="o", ms=8,
            label=f"Final run: early stopping, {final['epochs_run']} epochs run")
    ax.axvline(final["chosen_epoch"], color=config.INK, lw=1)
    ax.text(final["chosen_epoch"] + 0.15, 0.72, f"best epoch {final['chosen_epoch']}\n"
            f"val macro-F1 {final['val_macro_f1']:.4f}", color=config.INK)
    ax.set_xticks(range(1, final["epochs_run"] + 1))
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("F9. DistilBERT validation curve: the 3-epoch run stopped while still improving\n"
                 f"(final run: early stopping, patience {final['early_stopping_patience']})")
    ax.legend(loc="center", bbox_to_anchor=(0.4, 0.3), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "f9_m4_validation_curves.png", dpi=200)
    plt.close(fig)


def demo_model():
    """Same default as the Streamlit app: best validation balanced score among gated models."""
    return predict.default_model()


def main():
    with Timer() as t:
        best = best_model()
        best = best if best in config.PROB_MODELS else best_model(prob_only=True)
        f5_model_comparison()
        f6_per_class_f1(best)
        f7_confusion_zoom(best)
        f8a_threshold_sweep()
        f8b_confidence_hist(demo_model())
        if (config.RESULTS / "train_meta_m4_distilbert.json").exists():
            f9_m4_validation_curves()
    print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
    log_run("python -m src.figures", t.seconds, f"F5-F9 written (best model {best})")


if __name__ == "__main__":
    main()
