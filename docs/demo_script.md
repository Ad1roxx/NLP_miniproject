# Demo script

## Start
```
.venv\Scripts\activate
streamlit run app/streamlit_app.py
```
Open http://localhost:8501. The app loads saved models only (CPU, no training). The first classification with M3 takes a few seconds while the encoder loads.

## Layout (redesigned UI)
One page, top to bottom: message box + **Analyze →** (Enter also works) → six example buttons → **routing decision panel** (model prediction → confidence check against τ → final routing) → top-3 predictions beside the **model controls** (model selector, τ slider) → "How the system works" and "Model results" expanders. The result stays on screen when you switch model or move τ, so you can show the routing change directly.

**View all 77 supported intents** (top right) opens a searchable table of every intent the router can route to, with its number of training examples and the selected model's test F1 / precision / recall (from `results/per_class_f1_<model>.csv`); the current prediction is highlighted above the table. Useful when someone asks "what can it actually handle?". The model controls also show the selected model's test macro-F1, OOS recall and CPU latency (from `results/`).

Example buttons show short labels; the text sent to the model is exactly the original example text (hover a button to see it).

## What should happen (default model M3 MiniLM + LogReg, tuned τ = 0.58)
Checked by `python -m src.demo_check` → `results/demo_examples_check.csv`:

| Click | Expected result | Confidence |
|---|---|---|
| "Waiting on my card" → "I am still waiting on my card" | Supported banking query → Intent queue: Card arrival | 0.6937 |
| "Change my PIN" → "How do I change my PIN?" | Supported banking query → Intent queue: Change pin | 0.9774 |
| "New card not here" → "my new card still hasn't shown up" | Out of scope → Human agent. **Closest intent is right (Card arrival), but confidence is below τ**, so it is a false reject | 0.4006 |
| "Card delivery time" → "How long does a card delivery take?" (tricky; from `results/confused_pairs.csv`) | Supported → Intent queue: Card delivery estimate. **This is a misroute**: the test label is card_arrival | 0.8942 |
| "Flight to Delhi" → "Book me a flight to Delhi" | Out of scope → Human agent (closest intent: Card delivery estimate) | 0.2607 |
| "Weather tomorrow" → "What's the weather tomorrow?" | Out of scope → Human agent (closest intent: Card delivery estimate) | 0.1605 |

## Suggested live sequence (about 3 minutes)
1. "Waiting on my card": supported; walk left to right through the panel: model prediction → confidence 69.4% above τ = 0.58 → Intent queue: Card arrival.
2. "Flight to Delhi": out of scope. The model still names a *closest* intent (Card delivery estimate), but at 26.1% it is below τ, so the final routing is Human agent. This separation of model prediction and routing decision is the point of the project.
3. Switch the model to **M4 · DistilBERT** with the same message: the closest intent now gets 94.8% confidence, yet it is still rejected because M4's τ is 0.97. Good moment to explain over-confidence.
4. "Card delivery time" (M3): accepted but wrong. The model is confident (0.8942), so the gate cannot catch it. This is the card_arrival ↔ card_delivery_estimate confusion from slide 11.
5. "New card not here": right intent, but rejected (0.4006 < 0.58). This is the price of the gate: a real customer gets a human instead of the card-arrival workflow.
6. Optional: with "Waiting on my card" on screen, drag τ to 0.70. The result updates immediately: confidence stays 69.4%, routing flips to Human agent. The message is now sent to a human, which shows the trade-off: a higher τ catches more errors but rejects more good queries.

## Honest notes for the presenter
- **"my new card still hasn't shown up" is rejected by the default model.** M3 picks the right intent (card_arrival) but only at 0.4006 confidence, below τ = 0.58, so the customer is sent to a human. M1 (0.9665) and M4 (0.9987) accept it with the correct intent. It was kept as a demo example on purpose: it shows the gate's cost, since some real banking queries get handed off.
- **M4 (DistilBERT, final early-stopping run) at its τ = 0.97** handles all OOS examples correctly, but look at its confidences: "Book me a flight to Delhi" gets 0.9478 and is rejected only because τ is so high. The initial 3-epoch M4 (τ 0.39) had accepted "What's the weather tomorrow?"; that run is archived (`results/m4_3epoch_demo_examples_check.csv`).
- All numbers here: `results/demo_examples_check.csv` (all six examples × every gated model). The examples and τ were **not** changed to make any example pass.
- "I am still waiting on my card" is almost verbatim a training query ("I am still waiting on my card?" is in `data/processed/train.csv`), so it is an easy case.
- M2 (LinearSVC) has no probabilities. With M2 selected, the sidebar shows that there is no gate and every query is routed.

## Backup if the live demo fails
Screenshots (redesigned UI, 1366×768, taken headlessly from the running app; M3 at τ = 0.58 unless stated):
- `docs/screenshots/demo_supported.png`: "I am still waiting on my card" (supported)
- `docs/screenshots/demo_tricky.png`: "How long does a card delivery take?"
- `docs/screenshots/demo_oos.png`: "Book me a flight to Delhi" (out of scope)
- `docs/screenshots/demo_intents.png`: the supported-intents table (M3)
- `docs/screenshots/demo_m4.png`: "Book me a flight to Delhi" with M4 · DistilBERT selected (94.8% but below τ = 0.97)
- `docs/screenshots/demo_new_card.png`: "my new card still hasn't shown up" (false reject)

The app also accepts URL parameters: `?q=` pre-fills and analyses a query, `?model=` preselects a model, e.g. `http://localhost:8501/?model=m4_distilbert&q=Book%20me%20a%20flight%20to%20Delhi`

[TODO: student] Record a short screen capture of the live sequence above as a second backup.
