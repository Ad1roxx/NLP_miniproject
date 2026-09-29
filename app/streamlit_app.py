"""BankBot Router demo: one page, loads saved models, trains nothing, runs on CPU and offline.

Run: streamlit run app/streamlit_app.py
Optional URL parameters (used for screenshots): ?q=<query> pre-fills and analyses a query,
?model=<model key> preselects a model (e.g. m4_distilbert).

Page order follows the routing story: message → routing decision → top predictions / model controls
→ technical details. All predictions come from src.predict (unchanged); app/ui.py only formats them.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make config / src importable
import config
from src import predict
from src.utils import load_json
import ui

st.set_page_config(page_title="BankBot Router", layout="wide")
st.html(ui.CSS)


@st.cache_resource
def get_model(key):
    return predict.load_model(key, device="cpu")


def tricky_example():
    """A real misclassified test query from the top confused pair (results/confused_pairs.csv)."""
    path = config.RESULTS / "confused_pairs.csv"
    if path.exists():
        return pd.read_csv(path).sort_values(["rank", "confidence"], ascending=[True, False]).iloc[0]["text"]
    return "My transfer is pending."


# (button label, exact text sent to the model). The texts are identical to src/demo_check.py.
EXAMPLES = [
    ("Waiting on my card", "I am still waiting on my card"),
    ("Change my PIN", "How do I change my PIN?"),
    ("New card not here", "my new card still hasn't shown up"),
    ("Card delivery time", tricky_example()),
    ("Flight to Delhi", "Book me a flight to Delhi"),
    ("Weather tomorrow", "What's the weather tomorrow?"),
]


def short_name(key):
    """'M3 MiniLM + LogReg' -> 'M3 · MiniLM + LogReg'"""
    return config.MODEL_NAMES[key].replace(" ", " · ", 1)


@st.cache_data
def read_csv(name):
    path = config.RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def model_summary(key):
    """One line of test-set facts for the selected model, read from results/ (nothing computed here)."""
    parts = []
    res, oos, cost = read_csv("results_table.csv"), read_csv("oos_results.csv"), read_csv("cost_table.csv")
    if res is not None and key in set(res["model"]):
        parts.append(f"Test macro-F1 <b>{res.set_index('model').loc[key, 'macro_f1']:.4f}</b>")
    if oos is not None and key in set(oos["model"]):
        parts.append(f"OOS recall <b>{oos.set_index('model').loc[key, 'oos_recall']:.3f}</b>")
    if cost is not None and key in set(cost["model"]):
        parts.append(f"<b>{cost.set_index('model').loc[key, 'median_latency_ms_cpu_batch1']:.2f} ms</b>/query on CPU")
    return " · ".join(parts)


@st.dialog("Supported intents", width="large")
def intents_dialog(key, predicted):
    """All 77 intents with training-set size and the selected model's per-intent test scores."""
    stats = load_json(config.RESULTS / "dataset_stats.json") if (config.RESULTS / "dataset_stats.json").exists() else {}
    train_counts = stats.get("banking77", {}).get("train_per_class_counts", {})
    pc = read_csv(f"per_class_f1_{key}.csv")
    table = pd.DataFrame({"label": predict.LABELS})
    table["Intent"] = table["label"].map(predict.readable)
    table["Training examples"] = table["label"].map(train_counts)
    table = table.sort_values("Intent", key=lambda s: s.str.lower())
    if pc is not None:
        table = table.merge(pc[["label", "f1", "precision", "recall"]], on="label", how="left") \
                     .rename(columns={"f1": "Test F1", "precision": "Precision", "recall": "Recall"})
    st.html(f'<div class="bb-small">The router can send a message to any of these <b>{len(table)}</b> BANKING77 '
            f'intents; anything else should go to a human agent. Scores are for <b>{short_name(key)}</b> on the '
            'official test set (40 queries per intent); training counts are from the official training file.</div>')
    if predicted:
        row = table[table["label"] == predicted]
        if len(row) and "Test F1" in row:
            st.html(f'<div class="bb-small" style="margin-top:6px">Current prediction: <b>{predict.readable(predicted)}</b>'
                    f' · test F1 {row["Test F1"].iloc[0]:.2f} for this model.</div>')
    search = st.text_input("Search intents", placeholder="e.g. card, transfer, top up", label_visibility="collapsed")
    if search:
        table = table[table["label"].str.contains(search.strip().replace(" ", "_"), case=False)
                      | table["Intent"].str.contains(search.strip(), case=False)]
    cols = [c for c in ["Intent", "Training examples", "Test F1", "Precision", "Recall"] if c in table]
    st.dataframe(table[cols], hide_index=True, use_container_width=True, height=420,
                 column_config={"Test F1": st.column_config.ProgressColumn("Test F1", min_value=0.0, max_value=1.0,
                                                                            format="%.2f"),
                                "Precision": st.column_config.NumberColumn(format="%.2f"),
                                "Recall": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"{len(table)} intents shown. Click a column header to sort, e.g. by Test F1 to see the hardest intents.")


# ---------------- state: the message currently being analysed ----------------
def analyse(text):
    st.session_state["query"] = text
    st.session_state["analyzed"] = text


def analyse_typed():
    st.session_state["analyzed"] = st.session_state.get("query", "")


thresholds = predict.load_thresholds()
models = predict.available_models()
default = predict.default_model()
if "query" not in st.session_state:          # first load: optional URL parameters
    q = st.query_params.get("q", "")
    st.session_state["query"] = q
    st.session_state["analyzed"] = q
    m = st.query_params.get("model")
    st.session_state["model_key"] = m if m in models else default

# ---------------- page skeleton (filled below, so controls can sit under the result) ----------------
h_left, h_right = st.columns([3, 1.3], vertical_alignment="bottom")
h_left.html(ui.header_title())
status_slot = h_right.empty()
with h_right.container(key="intents"):
    open_intents = st.button(f"View all {len(predict.LABELS)} supported intents", use_container_width=True)
st.html(ui.RULE)

st.html('<div class="bb-eyebrow">Analyze customer message</div>')
c_in, c_btn = st.columns([6, 1], vertical_alignment="bottom")
c_in.text_input("Customer message", key="query", on_change=analyse_typed, label_visibility="collapsed",
                placeholder="e.g. I was charged twice for the same payment")
with c_btn.container(key="analyze"):
    st.button("Analyze →", on_click=analyse_typed, use_container_width=True)

with st.container(key="examples"):
    cols = st.columns(len(EXAMPLES), gap="small")
    for col, (label, text) in zip(cols, EXAMPLES):
        col.button(label, key=f"ex_{label}", on_click=analyse, args=(text,), help=text,
                   use_container_width=True)

result_slot = st.container()
c_top, c_ctrl = st.columns([3, 2], gap="medium")

# ---------------- model controls (secondary, beside the ranked list) ----------------
with c_ctrl:
    st.html('<div class="bb-eyebrow">Model controls</div>')
    with st.container(key="controls"):
        key = st.selectbox("Model", models, key="model_key",
                           format_func=lambda k: short_name(k) + ("  (default)" if k == default else ""))
        model = get_model(key)
        if model["has_proba"]:
            tuned = thresholds[key]["tau"]
            tau = st.slider("Confidence threshold τ", 0.0, 0.99, float(tuned), 0.01, key=f"tau_{key}")
            st.html(f'<div class="bb-small">Tuned on validation data: <b>τ = {tuned:.2f}</b>. '
                    'Messages below τ go to a human agent.</div>')
        else:
            tau = None
            st.html('<div class="bb-small">LinearSVC gives no probabilities, so this model has '
                    '<b>no confidence gate</b>: every message is routed to its top intent.</div>')
        summary = model_summary(key)
        if summary:
            st.html(f'<div class="bb-small" style="margin-top:6px">{summary}</div>')

status_slot.html(ui.header_status(short_name(key)))
if open_intents:   # opened here because it needs the selected model
    analysed = st.session_state.get("analyzed", "").strip()
    intents_dialog(key, predict.classify(model, analysed, 0.0)["intent"] if analysed else None)

# ---------------- routing decision + top predictions ----------------
query = st.session_state.get("analyzed", "").strip()
with result_slot:
    if not query:
        st.html(ui.empty_result())
    else:
        r = predict.classify(model, query, tau if tau is not None else 0.0)
        st.html(ui.result_panel(query, r, tau, predict.readable))
if query:
    with c_top:
        st.html(ui.top_predictions(r["top3"], model["has_proba"], predict.readable))

# ---------------- technical details (tertiary) ----------------
st.html('<div class="bb-section"></div>')
with st.expander("How the system works"):
    st.html(ui.how_it_works())
    pipeline_png = config.DIAGRAMS / "pipeline.png"
    if pipeline_png.exists():
        st.image(str(pipeline_png), caption="Experiment pipeline (F3)", width=420)

with st.expander("Model results"):
    res = config.RESULTS / "results_table.csv"
    if res.exists():
        st.caption("In-scope results on the BANKING77 test set (3,080 queries)")
        st.dataframe(pd.read_csv(res)[["name", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                                       "val_macro_f1", "oos_gate"]], hide_index=True)
    oos = config.RESULTS / "oos_results.csv"
    if oos.exists():
        st.caption("Out-of-scope gate on test (3,080 in-scope + 1,000 CLINC OOS queries). "
                   "OOS precision depends on this 1,000 : 3,080 mix.")
        st.dataframe(pd.read_csv(oos)[["name", "tau", "oos_recall", "oos_precision", "in_scope_acc_with_rejection",
                                       "gate_cost_acc_points", "false_rejection_rate", "auroc"]], hide_index=True)
