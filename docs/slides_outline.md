# Slide outline (15 slides)

Every number is from `results/`. Figures are 200-dpi PNGs, ready to paste.

---

### 1. Title
- BankBot Router: Fine-Grained Banking Intent Detection with Out-of-Scope Rejection
- Name: [TODO: student] · Roll no: [TODO: student] · Class/Div: [TODO: student]
- NLP mini-project (IA-II)
- **Say:** "I built a router that sends each bank customer message to the right workflow, or to a human when it isn't sure."

### 2. Contents
- Problem → Dataset → Pipeline & architecture → Methods → Setup → Results → Error analysis → OOS detection → Demo → Inferences → References
- **Say:** "The slides follow the pipeline from data to the demo."

### 3. Case study and problem
- A bank's support chatbot must pick 1 of 77 intents for every message
- Unsupported messages must go to a human agent, not be misrouted
- A plain classifier always picks *some* intent, even for "Book me a flight to Delhi"
- Figure: example queries (use `results/example_predictions.csv`)
- **Say:** "A wrong route is worse than a hand-off."

### 4. Dataset
- BANKING77: 10,003 train / 3,080 test queries, 77 intents; our split 9,002 train / 1,001 validation
- Train imbalanced (35–187 per intent), test balanced (40 each), queries short (mean 11.95 tokens)
- CLINC150 out-of-scope: 100 validation / 1,000 test
- Figures: `figures/f1_class_distribution.png`, `figures/f2_query_length.png`; table: `results/dataset_stats.json`
- **Say:** "77 fine-grained intents, many of them near-duplicates of each other."

### 5. NLP pipeline
- One split, three representation families, shared evaluation and OOS gate
- Each branch isolates one design choice
- Figure: `diagrams/pipeline.png` (source `diagrams/pipeline.mmd`)
- **Say:** "Each branch changes one thing, so every gap between models has one cause."

### 6. System architecture
- Query → Streamlit UI → predict.py → classifier → max-probability gate
- ≥ τ: intent workflow; < τ: human agent
- Figure: `diagrams/architecture.png` (source `diagrams/architecture.mmd`)
- **Say:** "The gate is what makes it safe to deploy."

### 7. Preprocessing and features
- TF-IDF models: lowercase, numbers → `<num>`, currency → `<cur>`; stop words kept, no stemming
- TF-IDF on word 1–2 grams + character 2–5 grams (robust to typos)
- Embedding/transformer models get raw text
- Before/after examples: `results/preprocessing_examples.csv` (e.g. "Why was I charged an additional $1?" → "why was i charged an additional <cur> <num> ?")
- **Say:** "Stop words stay in because 'not', 'why' and 'still' change the intent."

### 8. Models
| | Representation | Classifier |
|---|---|---|
| M1 | TF-IDF | Logistic Regression |
| M2 | TF-IDF | LinearSVC (no probabilities → no gate) |
| M3 | MiniLM-L6-v2 sentence embeddings (frozen) | Logistic Regression |
| M4 | DistilBERT | fine-tuned (GPU), early stopping |
- Source: `config.py`, `results/train_meta_*.json`
- **Say:** "Linear models on two representations, then full fine-tuning."

### 9. Experimental setup
- Seed 42; official test used once; C, the DistilBERT stopping epoch and τ all chosen on validation
- C: M1 = 10, M2 = 1, M3 = 10
- DistilBERT, initial run (fixed 3 epochs): val macro-F1 0.715 → 0.8724 → 0.9079, still rising, so undertrained
- DistilBERT, final run (max 20 epochs, early stopping, patience 3): 12 epochs run, best epoch 9 (val 0.9232). Val per epoch: 0.3957, 0.8091, 0.9114, 0.9123, 0.9091, 0.9218, 0.9173, 0.9189, 0.9232, 0.9232, 0.9232, 0.9208
- Figure: `figures/f9_m4_validation_curves.png`; source `results/m4_3epoch_train_meta.json`, `results/train_meta_m4_distilbert.json`
- Laptop: 16-thread AMD CPU + RTX 4060 Laptop GPU; latency timed on CPU, batch 1
- **Say:** "The first DistilBERT run was still improving at epoch 3, so I retrained it with early stopping. The decision was based on the validation curve."

### 10. Results
- Macro-F1: M3 0.9267 ≈ M4 0.9266 > M1 0.9133 > M2 0.9122 (initial 3-epoch M4: 0.9029)
- Same features, different classifier (M1 vs M2): −0.0011; same classifier, embeddings instead of TF-IDF (M1 → M3): +0.0134; fine-tuning vs frozen embeddings (M3 → M4): −0.0001
- Cost: M1 1.41 ms / 22.9 MB; M3 9.33 ms / 91.7 MB; M4 16.93 ms / 268.8 MB and 187.32 s GPU training
- Figure: `figures/f5_model_comparison.png`; tables: `results/results_table.csv` (T1), `results/cost_table.csv` (T4), `results/model_comparisons.json`
- **Say:** "Changing the representation helped; changing the classifier didn't. Fine-tuning DistilBERT only tied the frozen encoder, at higher cost."

### 11. Error analysis
- Top confused pair: card_arrival ↔ card_delivery_estimate (7 errors)
- Transfer-status intents are the hardest: pending_transfer 0.7826, balance_not_updated_after_bank_transfer 0.7838, transfer_not_received_by_recipient 0.7952
- Read aloud: "How long does a card delivery take?" was labelled card_arrival and predicted card_delivery_estimate (0.8942)
- Figures: `figures/f7_confusion_zoom.png`, `figures/f6_per_class_f1.png`; table: `results/confused_pairs.csv` (T3)
- **Say:** "The errors are between intents that describe states of the same process."

### 12. OOS detection
- τ chosen on validation: M1 0.49, M3 0.58, M4 0.97
- M3 on test: OOS recall 0.925; gate cost 6.5 accuracy points (0.9269 → 0.8617)
- Of the 11.30% of banking queries M3 rejected, 4.77% would have been wrong anyway and 6.53% were correct
- M3's gate catches 65.3% of its own errors (median confidence of errors 0.4728 vs 0.9424)
- M4's confidence is saturated (median confidence of errors 0.9174; 117 errors above 0.9), so its τ had to be 0.97
- Footnote: OOS precision (0.7266 for M3) depends on the 1,000 OOS : 3,080 in-scope test mix
- Figures: `figures/f8a_threshold_sweep.png`, `figures/f8b_confidence_hist.png`; table: `results/oos_results.csv` (T2)
- **Say:** "τ was chosen on validation, never on test. The fine-tuned model is confident even when it's wrong."

### 13. Live demo
- `streamlit run app/streamlit_app.py`, default model M3 (best validation balanced score 0.9101), τ = 0.58
- Run: "How do I change my PIN?" (accepted, change_pin), "How long does a card delivery take?" (confident misroute), "Book me a flight to Delhi" (rejected, 0.2607), "my new card still hasn't shown up" (right intent, card_arrival, but confidence 0.4006 < τ, so sent to a human: a false reject)
- Backup screenshots: `docs/screenshots/demo_supported.png`, `demo_tricky.png`, `demo_oos.png`, `demo_new_card.png`, `demo_m4.png` (M4 selected); steps in `docs/demo_script.md`; checks in `results/demo_examples_check.csv`
- **Say:** "Watch the top-3 bars. When no intent is clearly ahead, the query goes to a human, even when the guess was right."

### 14. Inferences and limitations
- Representation > classifier: +0.0134 macro-F1 (M1 → M3) vs −0.0011 (M1 → M2)
- Fine-tuning (M4, early stopping) tied frozen embeddings (0.9266 vs 0.9267) at higher cost, with over-confident probabilities
- For M3 the gate doubles as an error filter (65.3% of errors caught); the price is 6.5 accuracy points
- OOS differences between models (recall 0.925–0.955, AUROC 0.9692–0.9742) are single-run observations, not a ranking
- Limitations: OOS set is mostly out-of-domain; some CLINC "OOS" queries are real banking requests ("i need to update my address" → edit_personal_details, 0.9962 for M1); 6 train/test duplicates kept; single run; uncalibrated confidence; English only
- Source: `docs/report_draft.md` §8–9
- **Say:** "Every claim here is one number from the results folder."

### 15. Conclusion and references
- Frozen MiniLM + LogReg is the chosen router: macro-F1 0.9267 (tied with fine-tuned DistilBERT at 0.9266, but cheaper), OOS recall 0.925 at τ = 0.58
- The confidence gate makes the router deployable, at a measured cost
- Key references: Casanueva et al. 2020 (BANKING77); Larson et al. 2019 (CLINC OOS); Hendrycks & Gimpel 2017 (max-softmax baseline); Reimers & Gurevych 2019; Sanh et al. 2019; Jurafsky & Martin SLP3
- Full list: `docs/references.md` [TODO: student: convert to the teacher's syntax]
- **Say:** "Small, cheap models plus an honest confidence gate get most of the way there."
