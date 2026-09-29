"""BankBot Router demo: one page, loads saved models, trains nothing, runs on CPU.

Run: streamlit run app/streamlit_app.py
Optional URL parameter ?q=<query> pre-fills and classifies a query (used for screenshots).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make config / src importable
import config
from src import predict

st.set_page_config(page_title="BankBot Router", page_icon="🏦", layout="centered")


@st.cache_resource
def get_model(key):
    return predict.load_model(key, device="cpu")


def tricky_example():
    """A real misclassified test query from the top confused pair (results/confused_pairs.csv)."""
    path = config.RESULTS / "confused_pairs.csv"
    if path.exists():
        return pd.read_csv(path).sort_values(["rank", "confidence"], ascending=[True, False]).iloc[0]["text"]
    return "My transfer is pending."


EXAMPLES = [
    ("Supported", "I am still waiting on my card"),
    ("Supported", "How do I change my PIN?"),
    ("Supported", "my new card still hasn't shown up"),
    ("Tricky banking", tricky_example()),
    ("Out of scope", "Book me a flight to Delhi"),
    ("Out of scope", "What's the weather tomorrow?"),
]

# ---------------- Sidebar: model + threshold ----------------
thresholds = predict.load_thresholds()
models = predict.available_models()
default = predict.default_model()
st.sidebar.header("Settings")
key = st.sidebar.selectbox("Model", models, index=models.index(default),
                           format_func=lambda k: config.MODEL_NAMES[k] + ("  (default)" if k == default else ""))
model = get_model(key)
if model["has_proba"]:
    tuned = thresholds[key]["tau"]
    tau = st.sidebar.slider("Confidence threshold (tau)", 0.0, 0.99, float(tuned), 0.01, key=f"tau_{key}")
    st.sidebar.caption(f"Tuned tau for this model: **{tuned:.2f}** (chosen on validation data only). "
                       "Queries below tau are sent to a human agent.")
else:
    tau = None
    st.sidebar.info("LinearSVC gives no probabilities, so this model has no out-of-scope gate: "
                    "every query is routed to an intent.")
st.sidebar.caption("Default model = gated model with the best validation score "
                   "(mean of in-scope accuracy with rejection and OOS recall).")

# ---------------- Main page ----------------
st.title("BankBot Router — Banking Support Query Classifier")
st.caption("Routes a customer message to one of 77 banking intents, or hands it to a human agent "
           "when the model is not confident enough.")


def use_example(text):
    st.session_state["query"] = text
    st.session_state["run"] = True


if "query" not in st.session_state:
    q = st.query_params.get("q")
    st.session_state["query"] = q or ""
    st.session_state["run"] = bool(q)

st.write("**Try an example:**")
for row in (EXAMPLES[:3], EXAMPLES[3:]):          # two rows so the full text stays readable
    for col, (kind, text) in zip(st.columns(3), row):
        col.button(text, key=f"ex_{text}", on_click=use_example, args=(text,), help=kind,
                   use_container_width=True)

query = st.text_input("Customer message", key="query", placeholder="Type a banking question...")
if st.button("Classify", type="primary") or st.session_state.pop("run", False):
    if not query.strip():
        st.warning("Please type a message first.")
    else:
        r = predict.classify(model, query, tau if tau is not None else 0.0)
        with st.container(border=True):
            if r["accepted"]:
                st.success("✅ Supported banking query")
            else:
                st.warning("⚠️ Out of scope — route to human agent")
            c1, c2, c3 = st.columns([2, 1, 2])
            conf_txt = f"{r['confidence']:.1%}" if r["confidence"] is not None else "n/a (SVM)"
            for col, label, value in ((c1, "Predicted intent", r["intent_readable"]),
                                      (c2, "Confidence", conf_txt), (c3, "Routed to", r["routed_to"])):
                col.caption(label)
                col.markdown(f"#### {value}")
            if tau is not None:
                st.caption(f"Accepted if confidence ≥ tau = {tau:.2f}")

            st.write("**Top-3 intents**")
            for label, score in r["top3"]:
                if model["has_proba"]:
                    st.progress(min(max(score, 0.0), 1.0), text=f"{predict.readable(label)} — {score:.1%}")
                else:
                    st.write(f"- {predict.readable(label)} (SVM score {score:.2f})")

# ---------------- Expanders ----------------
with st.expander("How it works"):
    pipeline_png = config.DIAGRAMS / "pipeline.png"
    arch_png = config.DIAGRAMS / "architecture.png"
    if arch_png.exists():
        st.image(str(arch_png), caption="System architecture")
    if pipeline_png.exists():
        st.image(str(pipeline_png), caption="Experiment pipeline")
    st.write("The query is classified by the selected model. If its highest probability is below "
             "tau, the query is treated as out-of-scope and handed to a human agent; otherwise it "
             "is routed to the predicted intent's workflow.")

with st.expander("Model results"):
    res = config.RESULTS / "results_table.csv"
    if res.exists():
        st.write("In-scope results on the BANKING77 test set (3,080 queries):")
        st.dataframe(pd.read_csv(res)[["name", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                                       "val_macro_f1", "oos_gate"]], hide_index=True)
    oos = config.RESULTS / "oos_results.csv"
    if oos.exists():
        st.write("Out-of-scope gate on test (3,080 in-scope + 1,000 CLINC OOS queries):")
        st.dataframe(pd.read_csv(oos)[["name", "tau", "oos_recall", "oos_precision", "in_scope_acc_with_rejection",
                                       "gate_cost_acc_points", "false_rejection_rate", "auroc"]], hide_index=True)
        st.caption("OOS precision depends on the 1,000 OOS : 3,080 in-scope test mix.")
