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
header_slot = st.empty()

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

header_slot.html(ui.header(short_name(key)))

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
