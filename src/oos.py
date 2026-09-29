"""Phase 6: out-of-scope (OOS) gate by maximum-probability threshold.

1. Validation pool = BANKING77 val (in-scope) + CLINC oos_val.
2. Sweep tau = 0.00 .. 0.99.
3. At each tau: in-scope accuracy with rejection (correct AND conf >= tau) and OOS recall (conf < tau).
4. Pick tau maximising the mean of the two; freeze it in results/thresholds.json.
5. Only then score the test pool once: BANKING77 test + CLINC oos_test (78 classes: 77 intents + OOS).
Only models with probabilities are gated (M1, M3, M4); M2 LinearSVC is not.

Outputs: results/thresholds.json, results/threshold_sweep_<model>.csv, results/oos_results.csv (T2),
         results/oos_false_accepts.csv, results/oos_false_rejects.csv
Run: python -m src.oos
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from src import predict
from src.utils import Timer, log_run, save_json, set_seed


def load_scores(key, split):
    return np.load(config.SCORES / f"{key}_{split}.npy")


def gate_metrics(conf, correct, n_in):
    """conf/correct cover in-scope rows first, then OOS rows. Returns the two sweep metrics."""
    return lambda tau: (
        float(np.mean(correct[:n_in] & (conf[:n_in] >= tau))),  # in-scope acc with rejection
        float(np.mean(conf[n_in:] < tau)),                        # OOS recall
    )


def sweep(key):
    """Threshold sweep on VALIDATION data only."""
    labels = np.array(predict.LABELS)
    y_val = pd.read_csv(config.DATA_PROC / "val.csv")["label"].values
    s_in, s_oos = load_scores(key, "val"), load_scores(key, "oos_val")
    conf = np.concatenate([s_in.max(1), s_oos.max(1)])
    correct = np.concatenate([labels[s_in.argmax(1)] == y_val, np.zeros(len(s_oos), bool)])
    f = gate_metrics(conf, correct, len(s_in))
    rows = []
    for tau in config.TAU_GRID:
        acc, rec = f(tau)
        rows.append({"tau": tau, "in_scope_acc_with_rejection": round(acc, 4),
                     "oos_recall": round(rec, 4), "balanced_score": round((acc + rec) / 2, 4)})
    sw = pd.DataFrame(rows)
    sw.to_csv(config.RESULTS / f"threshold_sweep_{key}.csv", index=False)
    best = sw.loc[sw["balanced_score"].idxmax()]   # first (lowest) tau on ties
    return float(best["tau"]), best


def test_once(key, tau):
    """Final evaluation at the frozen tau on the untouched test pool."""
    labels = np.array(predict.LABELS)
    test = pd.read_csv(config.DATA_PROC / "test.csv")
    oos_test = pd.read_csv(config.DATA_PROC / "oos_test.csv")
    s_in, s_oos = load_scores(key, "test"), load_scores(key, "oos_test")
    conf_in, conf_oos = s_in.max(1), s_oos.max(1)
    pred_in, pred_oos = labels[s_in.argmax(1)], labels[s_oos.argmax(1)]

    rej_in, rej_oos = conf_in < tau, conf_oos < tau
    tp = int(rej_oos.sum())                # OOS correctly sent to a human
    fp = int(rej_in.sum())                 # in-scope wrongly sent to a human
    fn = int((~rej_oos).sum())             # OOS routed to an intent
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    # 78-class accuracy over the whole pool (OOS counts as correct when rejected)
    final_in = np.where(rej_in, config.OOS_LABEL, pred_in)
    acc78 = (np.sum(final_in == test["label"].values) + tp) / (len(test) + len(oos_test))
    y_bin = np.r_[np.zeros(len(conf_in)), np.ones(len(conf_oos))]
    wrong_in = pred_in != test["label"].values
    acc_no_gate = float(np.mean(~wrong_in))
    acc_rej = float(np.mean(~wrong_in & ~rej_in))
    row = {
        "model": key, "name": config.MODEL_NAMES[key], "tau": tau,
        "oos_recall": round(rec, 4), "oos_precision": round(prec, 4),
        "oos_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "in_scope_acc_with_rejection": round(acc_rej, 4),
        "in_scope_acc_no_gate": round(acc_no_gate, 4),
        # Gate cost = accuracy lost by gating = correct answers that were sent to a human
        "gate_cost_acc_points": round(100 * (acc_no_gate - acc_rej), 1),
        "false_rejection_rate": round(float(rej_in.mean()), 4),
        # false_rejection_rate splits into: wrong answers the gate caught + correct answers it rejected
        "share_in_scope_errors_caught": round(float(np.mean(wrong_in & rej_in)), 4),
        "share_in_scope_correct_rejected": round(float(np.mean(~wrong_in & rej_in)), 4),
        "accuracy_78_class": round(float(acc78), 4),
        "auroc": round(float(roc_auc_score(y_bin, 1 - np.r_[conf_in, conf_oos])), 4),
        "n_in_scope": len(test), "n_oos": len(oos_test),
        "oos_precision_note": f"OOS precision depends on the test mix ({len(oos_test)} OOS : "
                              f"{len(test)} in-scope); a different mix gives a different precision",
    }
    fa = pd.DataFrame({"model": key, "text": oos_test["text"], "routed_to_intent": pred_oos,
                       "confidence": conf_oos.round(4)})[~rej_oos] \
        .sort_values("confidence", ascending=False).head(10)
    fr = pd.DataFrame({"model": key, "text": test["text"], "true": test["label"], "pred": pred_in,
                       "pred_correct": pred_in == test["label"].values,
                       "confidence": conf_in.round(4)})[rej_in] \
        .sort_values("confidence").head(10)
    return row, fa, fr


def main():
    set_seed()
    gated = [k for k in predict.available_models() if k in config.PROB_MODELS]
    with Timer() as t:
        # Step 1-4: choose and freeze every tau BEFORE any test scoring
        thresholds = {}
        for key in gated:
            tau, best = sweep(key)
            thresholds[key] = {"tau": tau, "val_in_scope_acc_with_rejection": best["in_scope_acc_with_rejection"],
                               "val_oos_recall": best["oos_recall"], "val_balanced_score": best["balanced_score"]}
            print(f"{key}: tau={tau:.2f} (val acc-with-rejection {best['in_scope_acc_with_rejection']:.4f}, "
                  f"val OOS recall {best['oos_recall']:.4f})")
        save_json({"selection": "argmax over tau of mean(in-scope acc with rejection, OOS recall) "
                                "on BANKING77 val + CLINC oos_val; test never used",
                   "models": thresholds}, config.RESULTS / "thresholds.json")

        # Step 5: test once at the frozen tau
        rows, fas, frs = [], [], []
        for key in gated:
            row, fa, fr = test_once(key, thresholds[key]["tau"])
            rows.append(row); fas.append(fa); frs.append(fr)
        res = pd.DataFrame(rows)
        res.to_csv(config.RESULTS / "oos_results.csv", index=False)
        pd.concat(fas).to_csv(config.RESULTS / "oos_false_accepts.csv", index=False)
        pd.concat(frs).to_csv(config.RESULTS / "oos_false_rejects.csv", index=False)
    print(res.to_string(index=False))
    log_run("python -m src.oos", t.seconds,
            "; ".join(f"{r['model']} tau={r['tau']} OOS recall {r['oos_recall']} "
                      f"acc-w-rej {r['in_scope_acc_with_rejection']} AUROC {r['auroc']}" for r in rows))


if __name__ == "__main__":
    main()
