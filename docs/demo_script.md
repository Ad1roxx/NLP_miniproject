# Demo script

## Start
```
.venv\Scripts\activate
streamlit run app/streamlit_app.py
```
Open http://localhost:8501. The app loads saved models only (CPU, no training). The first classification with M3 takes a few seconds while the encoder loads.

## What should happen (default model M3 MiniLM + LogReg, tuned τ = 0.58)
Checked by `python -m src.demo_check` → `results/demo_examples_check.csv`:

| Click | Expected result | Confidence |
|---|---|---|
| "I am still waiting on my card" | ✅ Supported → Card arrival | 0.6937 |
| "How do I change my PIN?" | ✅ Supported → Change pin | 0.9774 |
| "my new card still hasn't shown up" | ⚠️ Out of scope → Human agent. **Right intent (Card arrival), but confidence below τ**, so it is a false reject | 0.4006 |
| "How long does a card delivery take?" (tricky; from `results/confused_pairs.csv`) | ✅ Supported → Card delivery estimate. **This is a misroute**: the test label is card_arrival | 0.8942 |
| "Book me a flight to Delhi" | ⚠️ Out of scope → Human agent | 0.2607 |
| "What's the weather tomorrow?" | ⚠️ Out of scope → Human agent | 0.1605 |

## Suggested live sequence (about 3 minutes)
1. "How do I change my PIN?" Accepted with high confidence; the top-3 bars show one clear winner.
2. "Book me a flight to Delhi" Rejected. Point at the top-3 bars: no intent is clearly ahead, so the gate hands off to a human.
3. "How long does a card delivery take?" Accepted but wrong. The model is confident (0.8942), so the gate cannot catch it. This is the card_arrival ↔ card_delivery_estimate confusion from slide 11.
4. "my new card still hasn't shown up": right intent, but rejected (0.4006 < 0.58). This is the price of the gate: a real customer gets a human instead of the card-arrival workflow.
5. Optional: drag the τ slider up to 0.90 and re-run example 3. It is now sent to a human, which shows the trade-off: a higher τ catches more errors but rejects more good queries.

## Honest notes for the presenter
- **"my new card still hasn't shown up" is rejected by the default model.** M3 picks the right intent (card_arrival) but only at 0.4006 confidence, below τ = 0.58, so the customer is sent to a human. M1 (0.9665) and M4 (0.9987) accept it with the correct intent. It was kept as a demo example on purpose: it shows the gate's cost, since some real banking queries get handed off.
- **M4 (DistilBERT, final early-stopping run) at its τ = 0.97** handles all OOS examples correctly, but look at its confidences: "Book me a flight to Delhi" gets 0.9478 and is rejected only because τ is so high. The initial 3-epoch M4 (τ 0.39) had accepted "What's the weather tomorrow?"; that run is archived (`results/m4_3epoch_demo_examples_check.csv`).
- All numbers here: `results/demo_examples_check.csv` (all six examples × every gated model). The examples and τ were **not** changed to make any example pass.
- "I am still waiting on my card" is almost verbatim a training query ("I am still waiting on my card?" is in `data/processed/train.csv`), so it is an easy case.
- M2 (LinearSVC) has no probabilities. With M2 selected, the sidebar shows that there is no gate and every query is routed.

## Backup if the live demo fails
Screenshots (taken headlessly from the running app with M3 at τ = 0.58):
- `docs/screenshots/demo_supported.png`: "How do I change my PIN?"
- `docs/screenshots/demo_tricky.png`: "How long does a card delivery take?"
- `docs/screenshots/demo_oos.png`: "Book me a flight to Delhi"
- `docs/screenshots/demo_new_card.png`: "my new card still hasn't shown up" (false reject)

The app also accepts a URL parameter to pre-fill a query: `http://localhost:8501/?q=Book%20me%20a%20flight%20to%20Delhi`

[TODO: student] Record a short screen capture of the live sequence above as a second backup.
