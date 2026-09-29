"""Shared loading + prediction used by evaluation, OOS analysis and the Streamlit app.

Every model is wrapped in a small dict so the rest of the code can call
score_texts(model, texts) and get a (n_texts, 77) matrix whose columns follow labels.json:
  - probabilities for M1, M3 (predict_proba) and M4 (softmax of logits)
  - SVM decision scores for M2 (not probabilities, so M2 gets no OOS gate)
"""
import joblib
import numpy as np

import config
from src.utils import load_json

LABELS = load_json(config.DATA_PROC / "labels.json") if (config.DATA_PROC / "labels.json").exists() else []
DISTILBERT_DIR = config.MODELS / "distilbert"


def model_files(key):
    """Files that must exist for a model to be usable."""
    if key == "m4_distilbert":
        return [DISTILBERT_DIR / "config.json"]
    files = [config.MODELS / f"{key}.joblib"]
    if key == "m3_sbert_lr":
        files.append(config.MODELS / "sbert_encoder")
    return files


def available_models():
    """Model keys (in the standard order) whose saved files are on disk."""
    return [k for k in config.MODEL_NAMES if all(f.exists() for f in model_files(k))]


def load_model(key, device="cpu"):
    """Load one saved model. device matters only for M3's encoder and M4."""
    m = {"key": key, "name": config.MODEL_NAMES[key], "has_proba": key in config.PROB_MODELS}
    if key in ("m1_tfidf_lr", "m2_tfidf_svm"):
        m["pipe"] = joblib.load(config.MODELS / f"{key}.joblib")
        assert list(m["pipe"].classes_) == LABELS, "class order must match labels.json"
    elif key == "m3_sbert_lr":
        from sentence_transformers import SentenceTransformer
        m["encoder"] = SentenceTransformer(str(config.MODELS / "sbert_encoder"), device=device)
        m["clf"] = joblib.load(config.MODELS / f"{key}.joblib")
        assert list(m["clf"].classes_) == LABELS
    elif key == "m4_distilbert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        m["tokenizer"] = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
        m["net"] = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR).to(device).eval()
        m["device"] = device
        assert [m["net"].config.id2label[i] for i in range(len(LABELS))] == LABELS
        m["torch"] = torch
    return m


def score_texts(m, texts, batch_size=64, embeddings=None):
    """Return an (n, 77) score matrix. For M3, precomputed embeddings may be passed in."""
    texts = list(texts)
    key = m["key"]
    if key == "m1_tfidf_lr":
        return m["pipe"].predict_proba(texts)
    if key == "m2_tfidf_svm":
        return m["pipe"].decision_function(texts)
    if key == "m3_sbert_lr":
        if embeddings is None:
            embeddings = m["encoder"].encode(texts, batch_size=config.SBERT_BATCH,
                                             normalize_embeddings=True, show_progress_bar=False)
        return m["clf"].predict_proba(embeddings)
    # M4: softmax over the 77 logits
    torch = m["torch"]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = m["tokenizer"](texts[i:i + batch_size], truncation=True, max_length=config.DB_MAX_LEN,
                                 padding=True, return_tensors="pt").to(m["device"])
            out.append(torch.softmax(m["net"](**enc).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def load_thresholds():
    """Per-model tuned tau and its validation scores (written by src/oos.py)."""
    path = config.RESULTS / "thresholds.json"
    return load_json(path)["models"] if path.exists() else {}


def default_model():
    """Demo default: the gated model with the highest VALIDATION balanced score, i.e. the same
    criterion used to pick tau (mean of in-scope accuracy with rejection and OOS recall)."""
    th = {k: v for k, v in load_thresholds().items() if k in available_models()}
    if not th:
        return available_models()[0]
    return max(th, key=lambda k: th[k]["val_balanced_score"])


def readable(label):
    """card_arrival -> 'Card arrival'"""
    return label.replace("_", " ").capitalize()


def classify(m, text, tau):
    """Classify one query and apply the OOS gate (only for models with probabilities)."""
    scores = score_texts(m, [text])[0]
    top = np.argsort(scores)[::-1][:3]
    conf = float(scores[top[0]]) if m["has_proba"] else None
    accepted = True if conf is None else conf >= tau
    return {
        "intent": LABELS[top[0]],
        "intent_readable": readable(LABELS[top[0]]),
        "confidence": conf,
        "accepted": accepted,
        "routed_to": f"{readable(LABELS[top[0]])} workflow" if accepted else "Human agent",
        "top3": [(LABELS[i], float(scores[i])) for i in top],
    }
