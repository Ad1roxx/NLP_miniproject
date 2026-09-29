# BankBot Router: Fine-Grained Banking Intent Detection with Out-of-Scope Rejection

**Name:** [TODO: student] · **Roll no:** [TODO: student] · **Class/Div:** [TODO: student] · **Course:** NLP mini-project (IA-II)

> Draft generated from the files in `results/`. Every number below is copied from the file named next to it. Sections marked [TODO: student] need your input. Nothing here was estimated.

---

## Abstract

A bank's customer-support chatbot must route each message to the right workflow and hand messages it cannot handle to a human agent instead of misrouting them. We treat routing as 77-way intent classification on the public BANKING77 dataset (10,003 training and 3,080 test queries). We compare four models: TF-IDF features with Logistic Regression (M1) and a linear SVM (M2), frozen all-MiniLM-L6-v2 sentence embeddings with Logistic Regression (M3), and DistilBERT fine-tuned with early stopping (M4). A confidence gate rejects a query when the model's maximum class probability falls below a threshold τ, tuned on validation data only and tested on 1,000 out-of-scope (OOS) queries from CLINC150. On the in-scope test set, M3 (macro-F1 0.9267) and M4 (0.9266) were effectively tied, ahead of M1 (0.9133) and M2 (0.9122). M4 was slower on CPU (16.93 vs 9.33 ms per query) and larger (268.8 vs 91.7 MB). At its tuned τ = 0.58, M3's gate sent 92.5% of OOS test queries to a human, at a cost of 6.5 accuracy points on banking queries. M3's errors are mostly low-confidence (median 0.4728 vs 0.9424 for correct answers), so the gate also caught 65.3% of its misroutes. Fine-tuned DistilBERT was far more confident even when wrong (median confidence of errors 0.9174), so its gate needed τ = 0.97.
*(sources: results/results_table.csv, results/cost_table.csv, results/oos_results.csv, results/error_summary.json)*

---

## 1 Introduction

**Case study.** Online banks answer a large volume of short customer messages: "my card hasn't arrived", "why was I charged?", "I can't verify my identity". A support chatbot must first decide *what the customer wants* (the intent) so the right workflow or FAQ can be triggered. A wrong route is costly: the customer gets an irrelevant answer and loses trust. Some messages are not about banking at all, or ask for something the bot does not support. A standard classifier always picks one of its known intents, so without extra logic these messages are routed confidently to the wrong place.

**Problem statement.** Given a single English customer message, (a) assign one of 77 fine-grained banking intents, and (b) decide whether the prediction is trustworthy enough to act on, or whether the message should go to a human agent.

**Objectives.**
1. Build and compare NLP intent classifiers of increasing complexity: sparse lexical (TF-IDF), dense semantic (sentence embeddings) and fine-tuned transformer (DistilBERT).
2. Add an out-of-scope (OOS) gate based on classifier confidence, with the threshold chosen without touching test data.
3. Analyse which intents are confused and why, and whether the gate catches those errors.
4. Demonstrate the router in an interactive Streamlit app.

## 2 Background

**Intent detection and BANKING77.** Casanueva et al. (2020) introduced BANKING77, a single-domain dataset of 13,083 customer-service queries labelled with 77 fine-grained intents. They chose one domain on purpose, so that many intents overlap and must be told apart by small differences in meaning. They also showed that intent classifiers built on fixed, pretrained sentence encoders can be competitive with full BERT fine-tuning while being much cheaper to train, especially when little training data is available. This motivates our M3 design (frozen encoder + linear classifier).

**Out-of-scope prediction.** Larson et al. (2019) released CLINC150, an intent dataset with 150 in-scope intents across 10 domains plus a set of out-of-scope queries that match none of them. They showed that classifiers with high in-scope accuracy can still perform poorly at recognising out-of-scope queries, and they evaluated threshold-based rejection among other approaches. We use only their out-of-scope queries.

**Harder, in-domain OOS.** Zhang et al. (2022) argue that OOS evaluation should also include *in-domain* out-of-scope queries, which are close in meaning to supported intents but not supported. They found pretrained transformers much less robust on these than on out-of-domain queries. This is why our results on CLINC OOS (mostly unrelated to banking) should be read as the easier case (Section 9).

**Maximum softmax probability baseline.** Hendrycks and Gimpel (2017) showed that the maximum softmax probability of a classifier tends to be higher for correctly classified, in-distribution inputs than for misclassified or out-of-distribution inputs. This makes it a simple baseline for detecting both. They evaluate this separation with threshold-free metrics such as AUROC. Our gate and our AUROC metric follow this baseline directly.

**Sentence embeddings.** Reimers and Gurevych (2019) proposed Sentence-BERT, which fine-tunes BERT-style networks in a siamese/triplet set-up. The resulting fixed-size sentence embeddings can be compared with cosine similarity, and semantically similar sentences lie close together. We use `all-MiniLM-L6-v2`, a sentence-transformers model that maps a sentence to a 384-dimensional vector (model card).

**Transformers and fine-tuning.** Devlin et al. (2019) introduced BERT, a bidirectional transformer pretrained with masked language modelling. It can be fine-tuned for classification by adding a single output layer. Sanh et al. (2019) used knowledge distillation to create DistilBERT. Its authors report that it is 40% smaller and 60% faster than BERT while keeping 97% of its language-understanding performance. That makes it the transformer that fits our hardware and time budget.

**Tools.** Classical models use scikit-learn (Pedregosa et al., 2011). Background on TF-IDF, logistic regression, attention and fine-tuning follows Jurafsky and Martin (SLP, 3rd ed. draft).

## 3 Dataset

| | Value | Source |
|---|---|---|
| BANKING77 official train / test | 10,003 / 3,080 queries, 77 intents | results/dataset_stats.json |
| Our split of the official train file | 9,002 train / 1,001 validation (stratified, seed 42) | results/dataset_stats.json |
| Train examples per intent | 35 to 187 (mean 129.91; max/min ratio 5.34) | results/dataset_stats.json |
| Test examples per intent | exactly 40 for every intent (balanced) | results/dataset_stats.json |
| Query length (official train, whitespace tokens) | mean 11.95, median 10, 99th percentile 43, max 79 | results/dataset_stats.json |
| CLINC150 OOS queries | 100 train (unused) / 100 validation / 1,000 test | results/dataset_stats.json |
| Train rows that also appear in test (lowercased) | 6 (logged, not removed) | results/dataset_stats.json |

- **Figure F1** (`figures/f1_class_distribution.png`): the training set is imbalanced (35 to 187 per intent), but the test set is perfectly balanced. We therefore use macro-F1, which weights every intent equally. Weighted-F1 would equal macro-F1 on this test set and is not reported.
- **Figure F2** (`figures/f2_query_length.png`): queries are short, so a maximum length of 64 subword tokens for DistilBERT loses almost nothing.
- **Validation split.** The official test set is never used for any choice. All choices (the C value, the DistilBERT stopping point and epoch, the threshold τ) are made on the 1,001-query validation split plus the 100 CLINC `oos_val` queries.
- **OOS data.** CLINC's `oos_train` queries are not used. All models are trained on in-scope banking data only, so the gate never sees an OOS example during training.

## 4 Methodology

**Pipeline (Figure F3, `diagrams/pipeline.png`).** One shared split feeds three representation families. All models go through the same evaluation and the same OOS gate, so the comparison is fair.

**Architecture (Figure F4, `diagrams/architecture.png`).** Customer query → Streamlit UI → `src/predict.py` → classifier (77 probabilities) → gate: if max probability ≥ τ, route to that intent's workflow; otherwise hand off to a human agent.

**Preprocessing (TF-IDF models only).** Lowercase; replace numbers with `<num>` and currency symbols (£ $ € ₹) with `<cur>`; collapse whitespace. Stop words are *kept* and nothing is stemmed, because words such as "not", "why" and "still" carry intent ("still waiting on my card" vs "waiting on my card"). Ten before/after examples are in `results/preprocessing_examples.csv`. Embedding and transformer models get raw text, because their own tokenisers expect it.

**Models.** Each comparison changes one thing:
| Model | Representation | Classifier | Isolates |
|---|---|---|---|
| M1 | TF-IDF word 1–2 grams + char_wb 2–5 grams, sublinear TF | Logistic Regression | baseline |
| M2 | same features as M1 | LinearSVC | the classifier (M1 vs M2) |
| M3 | all-MiniLM-L6-v2 sentence embeddings (frozen, 384-d, normalised) | Logistic Regression | the representation (M1 vs M3) |
| M4 | DistilBERT (distilbert-base-uncased) | fine-tuned end to end, 77-way head | fine-tuning (M3 vs M4) |

- Character n-grams make TF-IDF robust to typos and word forms ("transfer"/"transferred").
- Logistic Regression is linear, fast on sparse features, and outputs probabilities, which the gate needs.
- LinearSVC outputs no probabilities, so **M2 has no OOS gate** and is compared on in-scope metrics only. No calibration was added.

**OOS gate.** Confidence is the maximum predicted probability (softmax for M4), the baseline of Hendrycks and Gimpel (2017). For each gated model:
1. The validation pool is BANKING77 validation (1,001) plus CLINC `oos_val` (100).
2. τ is swept from 0.00 to 0.99 in steps of 0.01.
3. At each τ two numbers are computed: *in-scope accuracy with rejection* (correct intent **and** confidence ≥ τ) and *OOS recall* (OOS query with confidence < τ).
4. The τ with the highest mean of the two is chosen. The mean balances them, because the pool is about 10:1 in-scope. τ is frozen in `results/thresholds.json`.
5. The test pool (BANKING77 test 3,080 + CLINC `oos_test` 1,000) is scored once.

## 5 Experimental setup

- **Seed** 42 everywhere (Python, NumPy, PyTorch, scikit-learn `random_state`, Trainer seed).
- **Hyperparameters.** Only the tiny validation choices from the spec were made (`results/train_meta_*.json`):
  - M1: C chosen from {1, 10}; val macro-F1 0.9038 vs 0.9216, so C = 10.
  - M2: C chosen from {0.1, 1}; 0.9066 vs 0.9187, so C = 1.
  - M3: C chosen from {1, 10}; 0.9205 vs 0.9325, so C = 10.
  - M4: max_length 64, lr 5e-5, batch 32, weight decay 0.01, warmup ratio 0.1, fp16, best epoch chosen by validation macro-F1. The number of epochs is covered in the next paragraph.
- **DistilBERT: initial run and final run.** M4 was trained twice. Only validation data informed the change.

  | Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Initial run (fixed 3 epochs) val macro-F1 | 0.715 | 0.8724 | 0.9079 | – | – | – | – | – | – | – | – | – |
  | Final run (max 20 epochs, early stopping, patience 3) val macro-F1 | 0.3957 | 0.8091 | 0.9114 | 0.9123 | 0.9091 | 0.9218 | 0.9173 | 0.9189 | **0.9232** | 0.9232 | 0.9232 | 0.9208 |

  *(results/m4_3epoch_train_meta.json, results/train_meta_m4_distilbert.json; Figure F9 `figures/f9_m4_validation_curves.png`)*

  - **Initial run.** The spec's fixed 3 epochs. Validation macro-F1 rose at every epoch (0.715 → 0.8724 → 0.9079), so the model was clearly still improving when training stopped: it was undertrained.
  - **Final run.** Because of that validation curve, M4 was retrained with up to 20 epochs and early stopping: training stops once validation macro-F1 has not improved for 3 consecutive epochs, and the best checkpoint is kept. Training ran 12 epochs. The best was epoch 9 (0.9232); epochs 10–12 did not beat it, so training stopped. All other settings were unchanged. The decision to retrain was based on the validation curve. For transparency: the initial run had already been scored on the test set (macro-F1 0.9029) before retraining, but the criterion for retraining and the stopping rule used validation data only.
  - **Warmup.** Warmup is 10% of the *planned* schedule. In the final run that is 10% of 20 epochs, so the learning rate was still warming up during epoch 1. This is why its epoch-1 score (0.3957) is lower than the initial run's (0.715).
  - **Archive.** The initial run's results are kept in `results/m4_3epoch_*` for reference. All tables below use the final run.
- **Hardware.** Laptop with an AMD CPU (16 threads) and an NVIDIA GeForce RTX 4060 Laptop GPU. M1–M3 were trained on the CPU. MiniLM encoding and DistilBERT fine-tuning ran on the GPU. All latencies were measured on the CPU at batch size 1. See `results/run_log.md`.
- **Metrics and why.**
  - Accuracy, as the intuitive headline.
  - Macro precision/recall/F1 as the primary metric, because every intent matters equally.
  - Per-class F1 and confusion pairs, to show where errors fall.
  - For the gate: OOS recall, OOS precision, in-scope accuracy with rejection, false-rejection rate and AUROC (OOS as the positive class, score = 1 − max probability).
  - Cost: median latency, size on disk and training time.
- **Library versions.** See `requirements.txt` (scikit-learn 1.9.1, sentence-transformers 6.1.0, transformers 5.17.0, torch 2.14.0).

## 6 Results

### 6.1 In-scope classification (T1, `results/results_table.csv`; Figure F5)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Val macro-F1 |
|---|---|---|---|---|---|
| M1 TF-IDF + LogReg | 0.9130 | 0.9164 | 0.9130 | 0.9133 | 0.9216 |
| M2 TF-IDF + LinearSVC | 0.9120 | 0.9147 | 0.9120 | 0.9122 | 0.9187 |
| **M3 MiniLM + LogReg** | **0.9269** | **0.9303** | **0.9269** | **0.9267** | **0.9325** |
| M4 DistilBERT (early stopping, best epoch 9) | 0.9266 | 0.9292 | 0.9266 | 0.9266 | 0.9232 |

All four models are far above chance (1/77). On test, M3 and M4 differ by 0.0001 macro-F1, which is effectively a tie (`results/model_comparisons.json`). On validation, M3 is ahead (0.9325 vs 0.9232). The initial 3-epoch M4 reached 0.9029 test macro-F1 (`results/m4_3epoch_summary.json`).

### 6.2 Cost (T4, `results/cost_table.csv`)

| Model | Median latency (ms/query, CPU, batch 1) | Size on disk (MB) | Training time (s) |
|---|---|---|---|
| M1 | 1.41 | 22.9 | 16.65 (CPU) |
| M2 | 1.41 | 22.9 | 4.55 (CPU) |
| M3 | 9.33 | 91.7 | 0.96 (classifier fit, CPU) + 12.84 (MiniLM encoding of all splits, GPU) |
| M4 | 16.93 | 268.8 | 187.32 (fine-tuning, 12 epochs, GPU) |

Training times for different devices are not directly comparable. M4 would take much longer on a CPU. That configuration was not run.

### 6.3 Out-of-scope gate (T2, `results/oos_results.csv`; Figures F8a, F8b)

τ was chosen on validation (`results/thresholds.json`). The test pool is 3,080 in-scope + 1,000 OOS queries.

| Model | τ | OOS recall | OOS precision* | OOS F1* | In-scope acc. (no gate) | In-scope acc. with rejection | Gate cost (acc. points) | False-rejection rate | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 0.49 | 0.943 | 0.7276 | 0.8214 | 0.9130 | 0.8494 | 6.4 | 0.1146 | 0.9742 |
| M3 | 0.58 | 0.925 | 0.7266 | 0.8139 | 0.9269 | 0.8617 | 6.5 | 0.1130 | 0.9692 |
| M4 | 0.97 | 0.955 | 0.7449 | 0.8370 | 0.9266 | 0.8662 | 6.0 | 0.1062 | 0.9705 |

\* **OOS precision (and therefore OOS F1) depends on the test mix of 1,000 OOS : 3,080 in-scope queries.** With a different proportion of unsupported messages, precision would change even if the model were identical. OOS recall, false-rejection rate and AUROC do not depend on the mix.

**Where the false rejections come from.** The gate cost is accuracy without the gate minus in-scope accuracy with rejection. That equals the share of in-scope test queries that were *correct* but got rejected. The false-rejection rate splits into two parts:

| Model | False-rejection rate | = wrong answers caught by the gate | + correct answers rejected (= gate cost) |
|---|---|---|---|
| M1 | 0.1146 | 0.0510 | 0.0636 |
| M3 | 0.1130 | 0.0477 | 0.0653 |
| M4 | 0.1062 | 0.0458 | 0.0604 |

(shares of the 3,080 in-scope test queries; `results/oos_results.csv`)

**M4's threshold sits at the top of the scale.** The fine-tuned model's probabilities are highly peaked: its tuned τ is 0.97. On validation, in-scope accuracy with rejection falls from 0.8781 at τ = 0.95 to 0.8651 at 0.97 and 0.8412 at 0.99, while OOS recall moves from 0.93 to 0.95 to 0.96 (`results/threshold_sweep_m4_distilbert.csv`). A gate this close to 1.0 is sensitive to small changes in the model's confidence.

Without a gate, every OOS query is routed to some intent (OOS recall 0.0 at τ = 0.00 on validation, first row of `results/threshold_sweep_*.csv`).

## 7 Error analysis

Error analysis uses the model with the best test macro-F1, **M3** (0.9267 vs M4's 0.9266; 225 errors out of 3,080; `results/error_summary.json`).

**Hardest intents (Figure F6, `results/per_class_f1_m3_sbert_lr.csv`).** The three lowest-F1 intents are all transfer-status intents: `pending_transfer` 0.7826, `balance_not_updated_after_bank_transfer` 0.7838 and `transfer_not_received_by_recipient` 0.7952. Four intents reach F1 = 1.0.

**Most-confused pairs (T3, `results/confused_pairs.csv`; Figure F7).** Confusions are counted in both directions:

| Rank | Pair | Errors (a→b / b→a) | Real example (true → predicted, confidence) |
|---|---|---|---|
| 1 | card_arrival ↔ card_delivery_estimate | 7 (4 / 3) | "How long does a card delivery take?" (card_arrival → card_delivery_estimate, 0.8942) |
| 2 | balance_not_updated_after_bank_transfer ↔ transfer_not_received_by_recipient | 6 (5 / 1) | "I transferred some money but it is yet to arrive." (balance_not_updated… → transfer_not_received…, 0.5902) |
| 3 | exchange_via_app ↔ fiat_currency_support | 5 (2 / 3) | "Can I exchange currencies?" (fiat_currency_support → exchange_via_app, 0.7530) |
| 4 | card_payment_not_recognised ↔ compromised_card | 5 (5 / 0) | "Somebody used my card to make a purchase" (card_payment_not_recognised → compromised_card, 0.7219) |

Pairs 5–15 are in the CSV. The "type (to label manually)" column is left empty on purpose. [TODO: student] Read the three examples per pair and label each pair *lexical* (shared keywords), *semantic* (same meaning, different business action) or *contextual/ambiguous* (the query alone cannot decide). For example, "How long does a card delivery take?" is labelled `card_arrival`, but the wording asks for a delivery estimate. The label depends on context the message does not contain.

**High-confidence errors (`results/high_conf_errors.csv`).** 12 of M3's errors have confidence above 0.9 (`results/error_summary.json`). The ten most confident include "how many transactions can i make with a disposable card" (get_disposable_virtual_card → disposable_card_limits, 0.9904) and "My transfer is pending." (balance_not_updated_after_bank_transfer → pending_transfer, 0.9702). These are the dangerous cases for a bank: the gate cannot catch them. Several look like label ambiguity rather than model failure. [TODO: student] Verify by reading.

**Would the gate catch the errors?** (`results/error_summary.json`)

| Model | τ | Test errors | Errors below τ (caught) | Share caught | Errors with confidence > 0.9 | Median confidence of errors / of correct |
|---|---|---|---|---|---|---|
| M1 | 0.49 | 268 | 157 | 0.5858 | 9 | 0.432 / 0.9365 |
| M3 | 0.58 | 225 | 147 | 0.6533 | 12 | 0.4728 / 0.9424 |
| M4 | 0.97 | 226 | 141 | 0.6239 | 117 | 0.9174 / 0.9988 |

**OOS false accepts (`results/oos_false_accepts.csv`).** Several of the most confidently accepted "OOS" queries are genuine banking requests. "i need to update my address" was routed to `edit_personal_details` with confidence 0.9962 (M1), 0.9855 (M3) and 0.9992 (M4). "is there a way to change your houses address" behaved the same way. By the dataset's labels these count as errors, but a bank would want them routed. Other false accepts are true OOS queries that share banking words, e.g. "add mary to my phone plan, please" → `lost_or_stolen_phone` (M1, 0.9717).

## 8 Inferences

Each statement cites its number and file. All results come from a single run with one seed and one split, and no significance tests were run.

1. **The representation mattered more than the classifier.** With TF-IDF features fixed, swapping Logistic Regression (M1) for LinearSVC (M2) changed macro-F1 by 0.0011 (0.9133 vs 0.9122). Swapping TF-IDF for MiniLM sentence embeddings (M1 → M3) with the same classifier raised it by 0.0134 (0.9133 → 0.9267). *(results/model_comparisons.json, results/results_table.csv)*
2. **With enough epochs, fine-tuning matched frozen embeddings but did not beat them, and it cost more.**
   - M4 with early stopping reached test macro-F1 0.9266 vs M3's 0.9267 (difference −0.0001), and was lower on validation (0.9232 vs 0.9325).
   - It needed 187.32 s of GPU training, answers in 16.93 vs 9.33 ms per query on CPU, and takes 268.8 vs 91.7 MB.
   - The initial 3-epoch run (test 0.9029) was undertrained, as its rising validation curve showed.
   - For this dataset and budget, fine-tuning bought no in-scope accuracy over a frozen encoder with a linear classifier.

   *(results/model_comparisons.json, results/cost_table.csv, results/train_meta_m4_distilbert.json, results/m4_3epoch_summary.json)*
3. **Errors cluster among intents that describe states of the same process.** The three lowest per-class F1 scores are the transfer-status intents (0.7826, 0.7838, 0.7952), and the top confused pair is card_arrival ↔ card_delivery_estimate (7 errors). These intents differ by a business detail that a short message often does not state. *(results/per_class_f1_m3_sbert_lr.csv, results/confused_pairs.csv)*
4. **For M3 the confidence gate also works as an error filter; the fine-tuned model is far more confident when wrong.** M3's wrong answers have median confidence 0.4728 against 0.9424 for correct ones, and at τ = 0.58 its gate rejects 147 of 225 test errors (65.3%). M4's wrong answers have median confidence 0.9174, and 117 of its 226 errors are above 0.9 (M3: 12). Its gate only works because τ was pushed to 0.97. *(results/error_summary.json, results/thresholds.json)*
5. **The gate has a clear, measurable price.** For M3 it sends 92.5% of OOS test queries to a human and costs 6.5 points of in-scope accuracy (0.9269 → 0.8617). That is the share of banking queries answered correctly but still handed to an agent. A further 4.77% of banking queries were rejected but would have been misrouted anyway. For M1 the figures are 94.3%, 6.4 points and 5.10%; for M4, 95.5%, 6.0 points and 4.58%. *(results/oos_results.csv)*
6. **The three gated models separated OOS from in-scope queries about equally well in this run.** OOS recall was 0.955 (M4), 0.943 (M1) and 0.925 (M3); AUROC was 0.9742 (M1), 0.9705 (M4) and 0.9692 (M3). These are single-run observations with one seed and one OOS set of mostly out-of-domain queries. The gaps are small, and we do not claim any model is better at OOS detection. *(results/oos_results.csv)*

## 9 Limitations and future work

**Limitations**
- **The OOS test set is mostly out-of-domain.** Most CLINC OOS queries have nothing to do with banking, which is the easier kind of OOS. Unsupported *banking* questions (in-domain OOS; Zhang et al., 2022) were not evaluated. The optional held-out-intent experiment (E8) was not run.
- **Some CLINC "OOS" queries are genuine banking requests.** "i need to update my address" was routed to `edit_personal_details` at 0.9962 by M1 (0.9855 M3, 0.9992 M4) and counts as a false accept. OOS precision and recall therefore slightly understate how well the gate behaves from a bank's point of view. *(results/oos_false_accepts.csv)*
- **Train/test duplicates.** 6 queries in our training split also appear (after lowercasing) in the official test set. They were logged and deliberately not removed, to keep the official test set intact. They can inflate test scores very slightly. *(results/dataset_stats.json)*
- **Uncalibrated confidence.** Maximum probability is not the probability of being correct. The fine-tuned M4 is strongly over-confident: 117 errors above 0.9 and τ = 0.97. Calibration (e.g. temperature scaling) would likely be needed before relying on its confidence.
- **OOS precision depends on the test mix** (1,000 OOS : 3,080 in-scope) and would differ at a different rate of unsupported messages.
- **Single run.** One seed and one split. The M3–M4 difference (0.0001), the M1–M2 difference, and all OOS differences between models may not be stable.
- **M4 was trained twice.** The retrain was motivated by the validation curve, but it happened after the initial run's test scores were known, so the final M4 is not a strictly single-shot test evaluation. [TODO: student] mention this if asked about leakage.
- **M4 training schedule.** Early stopping used patience 3 and a 20-epoch maximum, with warmup defined over that maximum. Other schedules were not tried, by design.
- **Scope.** English only, one dataset, single-intent messages only, no conversation context, no user study. Latency was measured on one laptop CPU.

**Future work.** In-domain OOS evaluation by holding out intents; confidence calibration, especially for the fine-tuned model; data augmentation for the confused transfer and card-delivery intents; multi-intent messages and conversation context; Hinglish and other languages; a comparison with an LLM router on cost, latency and privacy.

## 10 Conclusion

We built and evaluated a banking intent router on BANKING77 with four models and a validation-tuned confidence gate. Frozen MiniLM sentence embeddings with Logistic Regression and DistilBERT fine-tuned with early stopping were effectively tied as in-scope classifiers (macro-F1 0.9267 vs 0.9266), both above the TF-IDF baselines. The frozen-embedding model was cheaper and gave more useful confidence scores. The maximum-probability gate made the router safe to deploy for out-of-domain messages: with M3 at τ = 0.58 it sent 92.5% of OOS test queries to a human. The cost was 6.5 accuracy points on banking queries, and in exchange it also caught 65.3% of the classifier's own mistakes. The errors that remain are concentrated in a few related intent groups and in a small number of confident mistakes, which are the natural targets for further work.

## References

See `docs/references.md` (to be converted to the required syntax). Works cited in this report: Casanueva et al. (2020); Larson et al. (2019); Zhang et al. (2022); Hendrycks and Gimpel (2017); Reimers and Gurevych (2019); Devlin et al. (2019); Sanh et al. (2019); Pedregosa et al. (2011); Jurafsky and Martin, SLP 3rd ed. draft; all-MiniLM-L6-v2 model card; BANKING77 and CLINC150 repositories.

## Appendix

- **A. Demo screenshots:** `docs/screenshots/demo_supported.png`, `docs/screenshots/demo_tricky.png`, `docs/screenshots/demo_oos.png`, `docs/screenshots/demo_new_card.png`, `docs/screenshots/demo_m4.png` (M4 selected).
- **B. Example predictions (T5):** `results/example_predictions.csv` (5 correct, 5 wrong, 5 OOS test queries with M3's output and gate decision).
- **C. Demo example check:** `results/demo_examples_check.csv` (all six demo examples, every gated model, at each model's tuned τ).
- **D. Initial 3-epoch DistilBERT run:** `results/m4_3epoch_summary.json` and other `results/m4_3epoch_*` files.
- **E. Preprocessing examples:** `results/preprocessing_examples.csv`.
- **F. Run log:** `results/run_log.md`.
