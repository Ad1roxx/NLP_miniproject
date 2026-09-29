# BankBot Router

Fine-grained banking intent detection (BANKING77, 77 intents) with out-of-scope rejection (CLINC150 OOS queries) and a Streamlit demo. NLP mini-project (IA-II).

## Setup (Windows, Python 3.11)
```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build; use plain `pip install torch` on CPU-only machines
pip install -r requirements.txt
```
.venv\Scripts\python -m streamlit run app/streamlit_app.py

## Run order
```
python run_all.py                     # all phases; DistilBERT only if CUDA is available
python run_all.py --skip-distilbert   # CPU-only machines
python run_all.py --force             # rebuild data, embeddings and models from scratch
streamlit run app/streamlit_app.py    # demo
```
Cached steps (downloads, embeddings, models, scores, latency) are skipped unless `--force` is given. Every run is appended to `results/run_log.md`.

| Step | Script | Main outputs |
|---|---|---|
| 1 Data | `src/data.py` | `data/processed/*.csv`, `labels.json`, `results/dataset_stats.json` |
| 2 EDA | `src/eda.py`, `src/preprocess.py` | F1, F2, `results/preprocessing_examples.csv` |
| 3 TF-IDF (M1, M2) | `src/train_tfidf.py` | `models/m1_tfidf_lr.joblib`, `models/m2_tfidf_svm.joblib` |
| 4 MiniLM (M3) | `src/train_sbert.py` | `models/m3_sbert_lr.joblib`, `models/sbert_encoder/`, `models/embeddings/` |
| 8 DistilBERT (M4) | `src/train_distilbert.py` (standalone, CUDA only; max 20 epochs, early stopping patience 3) | `models/distilbert/` |
| 5 Evaluation | `src/evaluate.py` | T1 `results_table.csv`, T4 `cost_table.csv`, `predictions_*.csv`, `per_class_f1_*.csv`, `model_comparisons.json` |
| 6 OOS gate | `src/oos.py` | `thresholds.json`, `threshold_sweep_*.csv`, T2 `oos_results.csv`, `oos_false_accepts.csv`, `oos_false_rejects.csv` |
| 5 Error analysis | `src/error_analysis.py` | T3 `confused_pairs.csv`, `high_conf_errors.csv`, `error_summary.json`, T5 `example_predictions.csv` |
| Figures | `src/figures.py` | F5–F9 in `figures/` |
| Demo check | `src/demo_check.py` | `results/demo_examples_check.csv` |
| Diagrams | `diagrams/*.mmd` → PNG with mermaid-cli | F3 `pipeline.png`, F4 `architecture.png` |

## Results (single run, seed 42)

In-scope, official BANKING77 test set, 3,080 queries (`results/results_table.csv`, `results/cost_table.csv`):

| Model | Accuracy | Macro-F1 | CPU latency (ms/query) | Size (MB) | Training time (s) |
|---|---|---|---|---|---|
| M1 TF-IDF + LogReg | 0.9130 | 0.9133 | 1.41 | 22.9 | 16.65 (CPU) |
| M2 TF-IDF + LinearSVC | 0.9120 | 0.9122 | 1.41 | 22.9 | 4.55 (CPU) |
| M3 MiniLM + LogReg | **0.9269** | **0.9267** | 9.33 | 91.7 | 0.96 (CPU fit) + 12.84 (GPU encoding) |
| M4 DistilBERT (early stopping, 12 epochs run, best epoch 9) | 0.9266 | 0.9266 | 16.93 | 268.8 | 187.32 (GPU) |

OOS gate, test = 3,080 in-scope + 1,000 CLINC OOS queries; τ chosen on validation only (`results/oos_results.csv`):

| Model | τ | OOS recall | OOS precision* | In-scope acc. with rejection | Gate cost (acc. points) | False-rejection rate | AUROC |
|---|---|---|---|---|---|---|---|
| M1 | 0.49 | 0.943 | 0.7276 | 0.8494 | 6.4 | 0.1146 | 0.9742 |
| M3 | 0.58 | 0.925 | 0.7266 | 0.8617 | 6.5 | 0.1130 | 0.9692 |
| M4 | 0.97 | 0.955 | 0.7449 | 0.8662 | 6.0 | 0.1062 | 0.9705 |

\* OOS precision depends on the 1,000 OOS : 3,080 in-scope test mix. M2 has no probabilities and therefore no gate.

The demo's default model is M3: it has the best validation balanced score (0.9101, vs 0.9076 for M4 and 0.8991 for M1; `results/thresholds.json`).

DistilBERT was first trained for a fixed 3 epochs (validation macro-F1 0.715 → 0.8724 → 0.9079, still rising; test macro-F1 0.9029). Based on that validation curve it was retrained with up to 20 epochs and early stopping (patience 3); the table shows the final run. The initial run is archived in `results/m4_3epoch_*` and `models/archive_m4_3epoch/`.

## Documentation
- `docs/report_draft.md`: report draft (all chapters)
- `docs/slides_outline.md`: 15-slide outline with figure paths
- `docs/viva_notes.md`: likely viva questions with answers
- `docs/demo_script.md`: demo steps, expected outputs, backup screenshots
- `docs/references.md`: reference list
