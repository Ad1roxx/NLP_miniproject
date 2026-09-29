"""Central settings: paths, seed, data URLs, model names and fixed hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SCORES = MODELS / "scores"          # cached score matrices (one .npy per model and split)
EMB_CACHE = MODELS / "embeddings"   # cached MiniLM sentence embeddings
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DIAGRAMS = ROOT / "diagrams"
DOCS = ROOT / "docs"
RUN_LOG = RESULTS / "run_log.md"

SEED = 42

# ---- Data sources ----
BANKING_TRAIN_URL = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                     "master/banking_data/train.csv")
BANKING_TEST_URL = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                    "master/banking_data/test.csv")
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
EXPECTED_TRAIN, EXPECTED_TEST, EXPECTED_LABELS = 10003, 3080, 77
VAL_FRACTION = 0.10
OOS_LABEL = "oos"

# ---- Models (key -> display name). Order is the order used in every table/chart. ----
MODEL_NAMES = {
    "m1_tfidf_lr": "M1 TF-IDF + LogReg",
    "m2_tfidf_svm": "M2 TF-IDF + LinearSVC",
    "m3_sbert_lr": "M3 MiniLM + LogReg",
    "m4_distilbert": "M4 DistilBERT (fine-tuned)",
}
# Same colour per model in every chart
MODEL_COLORS = {   # colour-blind-checked categorical order (blue, orange, aqua, yellow)
    "m1_tfidf_lr": "#2a78d6",
    "m2_tfidf_svm": "#eb6834",
    "m3_sbert_lr": "#1baf7a",
    "m4_distilbert": "#eda100",
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"   # text/axis colours for figures
# LinearSVC gives no probabilities, so it gets no OOS gate
PROB_MODELS = ["m1_tfidf_lr", "m3_sbert_lr", "m4_distilbert"]

# ---- Fixed hyperparameters (only these tiny choices are made on validation) ----
LR_MAX_ITER = 2000
M1_C_GRID = [1, 10]
M2_C_GRID = [0.1, 1]
M3_C_GRID = [1, 10]

SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH = 64

DISTILBERT_NAME = "distilbert-base-uncased"
DB_MAX_LEN = 64
DB_EPOCHS = 3
DB_LR = 5e-5
DB_BATCH = 32
DB_WEIGHT_DECAY = 0.01
DB_WARMUP_RATIO = 0.1

# ---- Evaluation ----
TAU_GRID = [round(i * 0.01, 2) for i in range(100)]   # 0.00 .. 0.99
LATENCY_N = 200                                        # queries timed at batch size 1

for _d in (DATA_RAW, DATA_PROC, MODELS, SCORES, EMB_CACHE, RESULTS, FIGURES, DIAGRAMS, DOCS):
    _d.mkdir(parents=True, exist_ok=True)
