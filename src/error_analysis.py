"""Phase 5b: error analysis on the best model (highest test macro-F1).

Outputs:
  results/confused_pairs.csv     (T3) 15 most-confused intent pairs (both directions), 3 real examples each,
                                 plus an empty "type (to label manually)" column
  results/high_conf_errors.csv   10 most confident wrong predictions
  results/error_summary.json     share of errors the OOS gate would catch (confidence < tau)
  results/example_predictions.csv (T5) 5 correct, 5 wrong, 5 OOS test queries with outputs
Run: python -m src.error_analysis
"""
from collections import Counter

import numpy as np
import pandas as pd

import config
from src import predict
from src.utils import Timer, load_json, log_run, save_json, set_seed


def best_model(prob_only=False):
    """Model with the highest TEST macro-F1 (used for error analysis, as the spec asks)."""
    t = pd.read_csv(config.RESULTS / "results_table.csv")
    if prob_only:
        t = t[t["model"].isin(config.PROB_MODELS)]
    return t.sort_values("macro_f1", ascending=False).iloc[0]["model"]


def main():
    set_seed()
    key = best_model()
    if key not in config.PROB_MODELS:     # confidences are needed below
        print(f"best model {key} has no probabilities; using best probabilistic model instead")
        key = best_model(prob_only=True)
    pr = pd.read_csv(config.RESULTS / f"predictions_{key}.csv")
    thresholds = load_json(config.RESULTS / "thresholds.json")["models"]
    tau = thresholds[key]["tau"]
    errors = pr[~pr["correct"]]
    print(f"best model: {key}  errors: {len(errors)} / {len(pr)}  tau={tau}")

    with Timer() as t:
        # --- T3: most-confused unordered pairs, counting both directions ---
        pair_counts = Counter(tuple(sorted((r.true, r.pred))) for r in errors.itertuples())
        rows = []
        for rank, ((a, b), n) in enumerate(pair_counts.most_common(15), start=1):
            ex = errors[((errors["true"] == a) & (errors["pred"] == b)) |
                        ((errors["true"] == b) & (errors["pred"] == a))] \
                .sort_values("confidence", ascending=False)
            for e in ex.head(3).itertuples():
                rows.append({"rank": rank, "intent_a": a, "intent_b": b, "pair_errors_total": n,
                             "a_predicted_as_b": int(((errors["true"] == a) & (errors["pred"] == b)).sum()),
                             "b_predicted_as_a": int(((errors["true"] == b) & (errors["pred"] == a)).sum()),
                             "text": e.text, "true": e.true, "pred": e.pred, "confidence": e.confidence,
                             # left empty on purpose: lexical / semantic / contextual-ambiguous,
                             # to be decided by reading the examples
                             "type (to label manually)": ""})
        cp = pd.DataFrame(rows)
        cp.insert(0, "model", key)
        cp.to_csv(config.RESULTS / "confused_pairs.csv", index=False)

        # --- 10 most confident wrong predictions ---
        hce = errors.sort_values("confidence", ascending=False)
        n_over_09 = int((hce["confidence"] > 0.9).sum())
        hce = (hce[hce["confidence"] > 0.9] if n_over_09 >= 10 else hce).head(10)
        hce.insert(0, "model", key)
        hce[["model", "text", "true", "pred", "confidence", "top2", "top2_prob"]] \
            .to_csv(config.RESULTS / "high_conf_errors.csv", index=False)

        # --- How many test errors would the OOS gate catch? (for every gated model) ---
        summary = {"best_model": key, "per_model": {}}
        for k, th in thresholds.items():
            p = pd.read_csv(config.RESULTS / f"predictions_{k}.csv")
            err = p[~p["correct"]]
            summary["per_model"][k] = {
                "tau": th["tau"], "n_test_errors": len(err),
                "errors_below_tau": int((err["confidence"] < th["tau"]).sum()),
                "share_of_errors_caught_by_gate": round(float((err["confidence"] < th["tau"]).mean()), 4),
                "errors_with_conf_over_0.9": int((err["confidence"] > 0.9).sum()),
                "median_conf_of_errors": round(float(err["confidence"].median()), 4),
                "median_conf_of_correct": round(float(p[p["correct"]]["confidence"].median()), 4),
                "correct_rejected_by_gate": int((p[p["correct"]]["confidence"] < th["tau"]).sum()),
            }
        save_json(summary, config.RESULTS / "error_summary.json")

        # --- T5: example predictions (5 correct, 5 wrong, 5 OOS) with the gate applied ---
        oos_test = pd.read_csv(config.DATA_PROC / "oos_test.csv")
        s_oos = np.load(config.SCORES / f"{key}_oos_test.npy")
        labels = np.array(predict.LABELS)
        oos_df = pd.DataFrame({"text": oos_test["text"], "true": config.OOS_LABEL,
                               "pred": labels[s_oos.argmax(1)], "confidence": s_oos.max(1).round(4)})
        ex = pd.concat([
            pr[pr["correct"]].sample(5, random_state=config.SEED).assign(case="correct"),
            errors.sample(5, random_state=config.SEED).assign(case="wrong intent"),
            oos_df.sample(5, random_state=config.SEED).assign(case="out-of-scope"),
        ])[["case", "text", "true", "pred", "confidence"]]
        ex["tau"] = tau
        ex["gate_decision"] = np.where(ex["confidence"] >= tau, "accepted -> " + ex["pred"], "rejected -> human agent")
        ex.insert(0, "model", key)
        ex.to_csv(config.RESULTS / "example_predictions.csv", index=False)

    print(cp.drop_duplicates("rank")[["rank", "intent_a", "intent_b", "pair_errors_total"]]
          .to_string(index=False))
    print({k: v for k, v in summary["per_model"].items()})
    log_run("python -m src.error_analysis", t.seconds,
            f"best={key}; {len(errors)} test errors; gate (tau={tau}) would catch "
            f"{summary['per_model'][key]['share_of_errors_caught_by_gate']:.1%} of them")


if __name__ == "__main__":
    main()
