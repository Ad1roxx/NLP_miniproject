# Viva notes

Short answers, with numbers from `results/`. All results come from a single run (seed 42).

**Why BANKING77?**
It is public (CC-BY-4.0) and made of real banking-style customer queries. Its 77 fine-grained intents overlap in wording, so it tests meaning rather than keyword matching, and it is a published benchmark with a citable paper (Casanueva et al., 2020). It has 10,003 train and 3,080 test queries, and the test set has exactly 40 per intent.

**What does TF-IDF calculate?**
Term frequency in the query × inverse document frequency. IDF is the log of (number of documents / documents containing the term), smoothed in scikit-learn. A word that is frequent in this query but rare overall gets a high weight. We use sublinear TF (1 + log tf), and each vector is L2-normalised.

**Why character n-grams too?**
Character 2–5 grams survive typos and word forms ("transfer"/"transferred"/"transfered"), which are common in chat text.

**Why Logistic Regression?**
It is linear and fast on sparse, high-dimensional text, and it outputs class probabilities, which the OOS gate needs. C was chosen on validation: M1 C = 10 (val macro-F1 0.9216 vs 0.9038 at C = 1).

**Why Linear SVM?**
It is a max-margin classifier that is strong on sparse text. With the features fixed, it isolates the effect of the classifier. It changed macro-F1 by only −0.0011 vs M1 (0.9122 vs 0.9133). It gives no probabilities, so it has no OOS gate. We did not add calibration.

**Why not Naive Bayes?**
It assumes features are independent, which overlapping word and character n-grams violate, and its probabilities tend to be poorly calibrated, which matters for thresholding. It would be a reasonable extra baseline, but we did not test it, so we make no accuracy claim about it.

**What are sentence embeddings?**
One fixed-length vector per sentence (384 dimensions for MiniLM-L6), from a transformer trained so that paraphrases land close together in cosine space (Sentence-BERT idea, Reimers & Gurevych 2019).

**Why freeze the encoder?**
Encoding once and training only a classifier is cheap: 12.84 s to encode all splits (GPU) plus 0.96 s to fit the classifier. It also isolates the value of the representation. It gave our best macro-F1, 0.9267.

**Why DistilBERT?**
It is a distilled BERT that its authors report is 40% smaller and 60% faster while keeping 97% of BERT's language-understanding performance (Sanh et al., 2019). It fits a laptop GPU: the final fine-tuning run (12 epochs) took 187.32 s on the RTX 4060.

**Why was DistilBERT trained twice?**
The initial run used a fixed 3 epochs, and its validation macro-F1 was still rising at every epoch (0.715 → 0.8724 → 0.9079), so it was undertrained. Based on that validation curve, I retrained it with up to 20 epochs and early stopping: stop after 3 epochs without a validation macro-F1 gain, and keep the best checkpoint. It ran 12 epochs, and the best was epoch 9 (val 0.9232). To be transparent: the initial run's test score (0.9029) was already computed before the retrain, but the stopping rule used only validation data. The initial run is archived in `results/m4_3epoch_*`.

**Is fine-tuning worth it here?**
Not in this run. The final M4 reached test macro-F1 0.9266 vs 0.9267 for frozen MiniLM + LogReg (difference −0.0001), and it was lower on validation (0.9232 vs 0.9325). It costs more: 16.93 vs 9.33 ms per query on CPU, 268.8 vs 91.7 MB, and 187.32 s of GPU training. Its probabilities are also much more peaked (median confidence of its errors 0.9174).

**What is attention?**
Each token's new representation is a weighted sum of all tokens' value vectors, with weights = softmax(QKᵀ / √d). It lets "card" mean different things in "card hasn't arrived" and "card was declined".

**What is fine-tuning?**
Start from pretrained weights, add a classification head (here 77 outputs), and train the whole network on labelled data with a small learning rate for a few epochs. We kept the best epoch by validation macro-F1 (epoch 3).

**What is OOS detection?**
Recognising queries outside the supported intents. A closed-set classifier always picks some intent, so without a gate every unrelated query is misrouted: at τ = 0 the OOS recall is 0.0 (validation sweep, `results/threshold_sweep_*.csv`).

**Why a confidence threshold, and how was it chosen?**
Maximum softmax probability is the standard out-of-distribution baseline (Hendrycks & Gimpel, 2017). τ maximised the mean of in-scope accuracy with rejection and OOS recall on validation data only (1,001 banking + 100 CLINC OOS queries). The test set was scored once afterwards. Tuned τ: M1 0.49, M3 0.58, M4 0.97. M4's τ is so high because the fine-tuned model is confident even on queries it gets wrong.

**What does the gate cost?**
For M3, in-scope accuracy drops from 0.9269 to 0.8617 = 6.5 points. Those are correct answers handed to a human. M3 rejects 11.30% of banking queries in total: 4.77% would have been wrong anyway and 6.53% were correct. In return it catches 92.5% of OOS test queries.

**What about OOS precision?**
M3's OOS precision is 0.7266, but it depends on the test mix (1,000 OOS : 3,080 in-scope). With a different share of unsupported messages it would change. Recall, false-rejection rate and AUROC do not depend on the mix.

**Which model is best at OOS?**
In this single run OOS recall was 0.955 (M4), 0.943 (M1) and 0.925 (M3), and AUROC was 0.9742 (M1), 0.9705 (M4) and 0.9692 (M3). That is one run, one seed and one mostly out-of-domain OOS set, and the gaps are small, so I don't claim any is better.

**Is confidence the same as the probability of being right?**
No. Models can be over-confident: 12 of M3's test errors have confidence above 0.9, and 117 of M4's. That is a stated limitation, and calibration is future work.

**Why macro-F1?**
It weights all 77 intents equally, so rare intents count as much as common ones, and F1 balances precision and recall. The training data is imbalanced (35 to 187 per intent). The test set is balanced, so weighted-F1 would equal macro-F1 and is not reported.

**What does the confusion matrix show?**
True vs predicted counts. Off-diagonal cells show which intents are mistaken for which. We show a zoomed 15-intent version (`figures/f7_confusion_zoom.png`), because a 77×77 matrix is unreadable.

**Why are certain intents confused?**
The top pair is card_arrival ↔ card_delivery_estimate (7 errors). For example, "How long does a card delivery take?" is labelled card_arrival, but it reads like a delivery-estimate question. The two intents differ by context the message doesn't give. The transfer-status intents are the hardest: pending_transfer F1 0.7826, balance_not_updated_after_bank_transfer 0.7838. [TODO: student: add your own label for each pair from `results/confused_pairs.csv`.]

**How did you avoid data leakage?**
A fixed stratified validation split (seed 42). C, the DistilBERT stopping epoch and τ were all chosen on validation. The one caveat: DistilBERT was retrained after the first run's test score was known, although the retrain was decided from the validation curve. I also checked for duplicates: 6 training queries also appear in the test set. They were logged but not removed, to keep the official test set intact.

**Why not just use an LLM?**
Cost, latency, determinism and data-privacy concerns in banking. The aim of the project is a measurable NLP pipeline. For comparison, M3 answers in a median 9.33 ms on a laptop CPU. An LLM comparison is future work.

**Limitations?**
- English only.
- The OOS test set is mostly out-of-domain, and some CLINC "OOS" queries are real banking requests ("i need to update my address" → edit_personal_details at 0.9962 for M1).
- One dataset, one run.
- Uncalibrated confidence (especially the fine-tuned M4).
- Single-intent queries only.
- 6 train/test duplicates kept.

**What would you improve?**
In-domain OOS via held-out intents, confidence calibration, data augmentation for the confused intents, Hinglish queries, and conversation context.
