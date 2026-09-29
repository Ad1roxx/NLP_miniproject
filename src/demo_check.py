"""Check the Streamlit demo examples against every gated model at its tuned tau.

The OOS examples must be rejected by the demo's default model; this script only reports what
happens (tau and models are never adjusted to make an example pass).
behaves_as_intended = accepted for banking examples / rejected for OOS examples;
intent_correct = whether the top intent matches the expected intent (banking examples only).
Output: results/demo_examples_check.csv
Run: python -m src.demo_check
"""
import pandas as pd

import config
from src import predict
from src.utils import log_run


def examples():
    cp = pd.read_csv(config.RESULTS / "confused_pairs.csv").sort_values(["rank", "confidence"],
                                                                         ascending=[True, False])
    # (type, text, expected intent); the tricky example's expected intent is its test-set label
    return [("supported", "I am still waiting on my card", "card_arrival"),
            ("supported", "How do I change my PIN?", "change_pin"),
            ("supported", "my new card still hasn't shown up", "card_arrival"),
            ("tricky banking (from confused_pairs.csv)", cp.iloc[0]["text"], cp.iloc[0]["true"]),
            ("out of scope", "Book me a flight to Delhi", config.OOS_LABEL),
            ("out of scope", "What's the weather tomorrow?", config.OOS_LABEL)]


def main():
    th = predict.load_thresholds()
    default = predict.default_model()
    rows = []
    for key in th:
        m = predict.load_model(key)
        for kind, text, expected in examples():
            r = predict.classify(m, text, th[key]["tau"])
            ok = r["accepted"] if kind != "out of scope" else not r["accepted"]
            rows.append({"model": key, "is_demo_default": key == default, "tau": th[key]["tau"],
                         "example_type": kind, "text": text, "top_intent": r["intent"],
                         "confidence": round(r["confidence"], 4), "expected_intent": expected,
                         "intent_correct": r["intent"] == expected if expected != config.OOS_LABEL else None,
                         "decision": "accepted" if r["accepted"] else "rejected -> human agent",
                         "behaves_as_intended": ok})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "demo_examples_check.csv", index=False)
    print(df.drop(columns="text").to_string(index=False))
    bad = df[df["is_demo_default"] & ~df["behaves_as_intended"]]
    log_run("python -m src.demo_check", 0,
            f"default {default}: {'all ' + str(len(examples())) + ' examples behave as intended' if bad.empty else 'FAILING: ' + '; '.join(bad['text'])}; "
            f"other models failing: {'; '.join(df[~df['is_demo_default'] & ~df['behaves_as_intended']].apply(lambda r: r['model'] + ': ' + r['text'], axis=1)) or 'none'}")


if __name__ == "__main__":
    main()
