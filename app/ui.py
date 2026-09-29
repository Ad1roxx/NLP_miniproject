"""Presentation layer for the BankBot Router demo: design tokens, CSS and small HTML components.

Only formatting lives here. Every number shown comes from src.predict.classify(), which is unchanged.
All styling targets our own `bb-*` classes, plus containers created with st.container(key=...),
which Streamlit exposes as `.st-key-<key>`. A minimal global reset is the only exception.
No external fonts or CDNs: system font stack only, so the app works offline.
"""
from html import escape

# ---------------------------------------------------------------- design tokens
CSS = """
<style>
:root {
  --bb-bg: #f6f5f1;          /* warm off-white page */
  --bb-surface: #ffffff;
  --bb-border: #e1ded6;
  --bb-border-strong: #cfcbc1;
  --bb-text: #172131;        /* navy-charcoal */
  --bb-muted: #5d6573;       /* slate */
  --bb-accent: #1f3a5f;      /* deep navy, used sparingly */
  --bb-ok: #1d7250;          /* muted emerald */
  --bb-ok-bg: #edf5f0;
  --bb-warn: #8f5a00;        /* muted amber (text-safe) */
  --bb-warn-bg: #fbf3e4;
  --bb-track: #ebe8e1;
  --bb-radius: 6px;
  --bb-font: Inter, "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --bb-mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
}

/* ---- minimal global reset: font, page width, hide Streamlit chrome ---- */
html, body, .stApp { font-family: var(--bb-font); }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 1040px; padding: 1.1rem 1.5rem 2rem; }

/* ---- type ---- */
.bb-eyebrow { font-size: 11.5px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
              color: var(--bb-muted); margin: 0 0 6px; }
.bb-section { margin-top: 18px; }

/* ---- header ---- */
.bb-header { display: flex; justify-content: space-between; align-items: flex-end;
             padding-bottom: 10px; border-bottom: 1px solid var(--bb-border); margin-bottom: 0; }
.bb-brand { font-size: 13px; font-weight: 700; letter-spacing: .14em; color: var(--bb-accent); }
.bb-title { font-size: 21px; font-weight: 600; color: var(--bb-text); margin-top: 2px; }
.bb-subtitle { font-size: 13.5px; color: var(--bb-muted); margin-top: 2px; }
.bb-ready { font-size: 13px; color: var(--bb-muted); white-space: nowrap; }
.bb-ready b { color: var(--bb-text); font-weight: 600; }
.bb-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--bb-ok);
          margin-right: 6px; vertical-align: 1px; }

/* ---- result panel ---- */
.bb-result { background: var(--bb-surface); border: 1px solid var(--bb-border);
             border-radius: var(--bb-radius); margin-top: 14px; }
.bb-result.ok   { border-left: 4px solid var(--bb-ok); }
.bb-result.warn { border-left: 4px solid var(--bb-warn); }
.bb-result.plain{ border-left: 4px solid var(--bb-muted); }
.bb-result-head { display: flex; justify-content: space-between; align-items: center;
                  padding: 12px 20px; border-bottom: 1px solid var(--bb-border); }
.bb-status { font-size: 13px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
             padding: 4px 10px; border-radius: 4px; }
.bb-status.ok   { color: var(--bb-ok);   background: var(--bb-ok-bg); }
.bb-status.warn { color: var(--bb-warn); background: var(--bb-warn-bg); }
.bb-status.plain{ color: var(--bb-muted); background: var(--bb-track); }
.bb-query { font-size: 14px; color: var(--bb-muted); max-width: 60%; text-align: right;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bb-query q { color: var(--bb-text); }

.bb-flow { display: grid; grid-template-columns: 1.25fr 28px 1fr 28px 1.15fr; align-items: stretch;
           padding: 16px 20px 18px; }
.bb-step .bb-value { font-size: 22px; font-weight: 600; color: var(--bb-text); line-height: 1.25; }
.bb-step .bb-note  { font-size: 13.5px; color: var(--bb-muted); margin-top: 4px; }
.bb-step.final .bb-value.ok   { color: var(--bb-ok); }
.bb-step.final .bb-value.warn { color: var(--bb-warn); }
.bb-arrow { color: var(--bb-border-strong); font-size: 20px; display: flex; align-items: center;
            justify-content: center; }
.bb-conf { font-family: var(--bb-mono); font-size: 22px; font-weight: 600; color: var(--bb-text); }

/* confidence meter: thin track, fill = confidence, tick = tau */
.bb-meter { position: relative; height: 6px; background: var(--bb-track); border-radius: 3px; margin: 10px 0 6px; }
.bb-meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--bb-accent); }
.bb-meter .fill.ok { background: var(--bb-ok); }
.bb-meter .fill.warn { background: var(--bb-warn); }
.bb-meter .tick { position: absolute; top: -4px; width: 2px; height: 14px; background: var(--bb-text); }

/* ---- ranked list ---- */
.bb-rank { background: var(--bb-surface); border: 1px solid var(--bb-border); border-radius: var(--bb-radius);
           padding: 4px 16px; }
.bb-row { display: grid; grid-template-columns: 30px 1fr 64px; column-gap: 10px; align-items: center;
          padding: 9px 0; border-bottom: 1px solid var(--bb-border); }
.bb-row:last-child { border-bottom: none; }
.bb-row .n { font-family: var(--bb-mono); font-size: 12.5px; color: var(--bb-muted); }
.bb-row .name { font-size: 15px; color: var(--bb-text); }
.bb-row .pct { font-family: var(--bb-mono); font-size: 14px; color: var(--bb-text); text-align: right; }
.bb-row .bar { grid-column: 2 / 4; height: 3px; background: var(--bb-track); border-radius: 2px; margin-top: 6px; }
.bb-row .bar span { display: block; height: 100%; background: var(--bb-accent); border-radius: 2px; }
.bb-row.first .name { font-weight: 600; }

.bb-empty { border: 1px dashed var(--bb-border-strong); border-radius: var(--bb-radius); padding: 22px 20px;
            color: var(--bb-muted); font-size: 14px; margin-top: 14px; background: var(--bb-surface); }
.bb-small { font-size: 13px; color: var(--bb-muted); line-height: 1.45; }
.bb-small b { color: var(--bb-text); }

/* ---- "how it works" diagram ---- */
.bb-diagram { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 13.5px; margin: 6px 0 4px; }
.bb-node { border: 1px solid var(--bb-border-strong); border-radius: 4px; padding: 6px 10px; background: var(--bb-surface);
           color: var(--bb-text); }
.bb-node.gate { border-color: var(--bb-accent); color: var(--bb-accent); font-weight: 600; }
.bb-to { color: var(--bb-muted); }

/* ---- our keyed containers (st.container(key=...)) ---- */
.st-key-examples button { border-radius: 4px; border: 1px solid var(--bb-border-strong); background: var(--bb-surface);
                          color: var(--bb-text); font-size: 13.5px; padding: 4px 6px; min-height: 34px; }
.st-key-examples button:hover { border-color: var(--bb-accent); color: var(--bb-accent); }
.st-key-examples button p { font-size: 13.5px; }
.st-key-analyze button { border-radius: 4px; background: var(--bb-accent); border: 1px solid var(--bb-accent);
                         color: #fff; font-weight: 600; min-height: 42px; }
.st-key-analyze button:hover { background: #172d4a; color: #fff; }
.st-key-intents button { border-radius: 4px; border: 1px solid var(--bb-border-strong); background: var(--bb-surface);
                         color: var(--bb-accent); font-weight: 600; min-height: 32px; padding: 2px 10px; }
.st-key-intents button p { font-size: 13px; }
.st-key-intents button:hover { border-color: var(--bb-accent); }
.st-key-controls { background: var(--bb-surface); border: 1px solid var(--bb-border); border-radius: var(--bb-radius);
                   padding: 12px 16px 6px; }
</style>
"""


def pct(x):
    return f"{x * 100:.1f}%"


def header_title():
    return """
<div>
  <div class="bb-brand">BANKBOT ROUTER</div>
  <div class="bb-title">Banking support query routing</div>
  <div class="bb-subtitle">77 banking intents · low-confidence messages go to a human agent</div>
</div>"""


def header_status(model_name):
    return f'<div class="bb-ready" style="text-align:right"><span class="bb-dot"></span>Model ready · <b>{escape(model_name)}</b></div>'


RULE = '<div style="border-bottom:1px solid var(--bb-border); margin: 2px 0 4px"></div>'


def empty_result():
    return ('<div class="bb-empty">Enter a customer message or pick an example. The routing decision, '
            'model prediction and confidence check will appear here.</div>')


def result_panel(query, r, tau, readable):
    """Routing decision: model prediction → confidence check → final routing.

    r is the dict returned by src.predict.classify(); tau is None for models without probabilities.
    """
    intent = escape(r["intent_readable"])
    q = escape(query)
    if tau is None:   # LinearSVC: no probabilities, no gate
        state, status = "plain", "Routed without gate"
        pred_label, conf_html = "Model prediction", (
            '<div class="bb-eyebrow">Confidence check</div>'
            '<div class="bb-value">Not available</div>'
            '<div class="bb-note">LinearSVC gives no probabilities, so no threshold is applied.</div>')
        route_label, route_value, route_cls = "Routed to", f"Intent queue: {intent}", ""
        pred_note = "Highest SVM decision score"
    else:
        conf = r["confidence"]
        accepted = r["accepted"]
        state = "ok" if accepted else "warn"
        status = "✓ Supported banking query" if accepted else "! Out of scope · human review"
        pred_label = "Model prediction" if accepted else "Closest predicted intent"
        pred_note = "Top intent from the classifier" if accepted else "Not used: confidence is below the threshold"
        side = "Above" if accepted else "Below"
        conf_html = (
            '<div class="bb-eyebrow">Confidence check</div>'
            f'<div class="bb-conf">{pct(conf)}</div>'
            f'<div class="bb-meter"><div class="fill {state}" style="width:{conf * 100:.1f}%"></div>'
            f'<div class="tick" style="left:calc({tau * 100:.1f}% - 1px)"></div></div>'
            f'<div class="bb-note">{side} threshold (τ = {tau:.2f})</div>')
        route_label = "Final routing"
        route_value = f"Intent queue: {intent}" if accepted else "Human agent"
        route_cls = state
    return f"""
<div class="bb-result {state}">
  <div class="bb-result-head">
    <span class="bb-status {state}">{status}</span>
    <span class="bb-query">Message: <q>{q}</q></span>
  </div>
  <div class="bb-flow">
    <div class="bb-step">
      <div class="bb-eyebrow">{pred_label}</div>
      <div class="bb-value">{intent}</div>
      <div class="bb-note">{pred_note}</div>
    </div>
    <div class="bb-arrow">→</div>
    <div class="bb-step">{conf_html}</div>
    <div class="bb-arrow">→</div>
    <div class="bb-step final">
      <div class="bb-eyebrow">{route_label}</div>
      <div class="bb-value {route_cls}">{route_value}</div>
    </div>
  </div>
</div>"""


def top_predictions(top3, has_proba, readable):
    rows = []
    for i, (label, score) in enumerate(top3, start=1):
        if has_proba:
            value, bar = pct(score), f'<div class="bar"><span style="width:{score * 100:.1f}%"></span></div>'
        else:
            value, bar = f"{score:.2f}", ""
        rows.append(f'<div class="bb-row{" first" if i == 1 else ""}"><span class="n">{i:02d}</span>'
                    f'<span class="name">{escape(readable(label))}</span><span class="pct">{value}</span>{bar}</div>')
    note = "" if has_proba else '<div class="bb-small" style="padding:6px 0 8px">SVM decision scores, not probabilities.</div>'
    return f'<div class="bb-eyebrow">Top predictions</div><div class="bb-rank">{"".join(rows)}{note}</div>'


def how_it_works():
    steps = ["Customer message", "Streamlit interface", "Saved NLP model", "77-intent classifier"]
    chain = "".join(f'<span class="bb-node">{s}</span><span class="bb-to">→</span>' for s in steps)
    return f"""
<div class="bb-diagram">{chain}<span class="bb-node gate">Confidence ≥ τ ?</span></div>
<div class="bb-diagram"><span class="bb-to" style="margin-left:4px">yes →</span><span class="bb-node">Intent queue</span>
<span class="bb-to" style="margin-left:14px">no →</span><span class="bb-node">Human agent</span></div>
<div class="bb-small" style="margin-top:8px">The classifier always picks its closest intent. The routing layer accepts that
intent only when the model's highest probability reaches the threshold τ, which was tuned on validation data only
(mean of in-scope accuracy with rejection and OOS recall). Otherwise the message is handed to a human agent.</div>"""
