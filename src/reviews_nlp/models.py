"""Classification de sentiment : baseline, TF-IDF + régression logistique, embeddings + régression logistique."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .config import EMBEDDING_MODEL, RANDOM_STATE, SENTIMENT_LABELS  # noqa: E402

SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}


def make_baseline() -> Pipeline:
    """Baseline : prédit toujours la classe majoritaire (référence à dépasser)."""
    return Pipeline([("tfidf", TfidfVectorizer(max_features=50)), ("clf", DummyClassifier(strategy="most_frequent"))])


def make_tfidf_logreg() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def make_embedding_logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def compute_embeddings(texts, model_name: str = EMBEDDING_MODEL, batch_size: int = 64) -> np.ndarray:
    """Embeddings de phrases (sentence-transformers). Import différé : dépendance lourde."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install sentence-transformers (ou utilisez --skip-embeddings).") from exc
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


def cross_validate_models(
    df: pd.DataFrame,
    embeddings: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    """Validation croisée stratifiée. Renvoie (métriques moyennes ± écart-type, matrices de confusion)."""
    y = df["sentiment"].to_numpy()
    texts = df["text"].to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    candidates = {
        "baseline_majority": (make_baseline(), texts),
        "tfidf_logreg": (make_tfidf_logreg(), texts),
    }
    if embeddings is not None:
        candidates["embeddings_logreg"] = (make_embedding_logreg(), embeddings)

    rows, confusions = [], {}
    for name, (model, X) in candidates.items():
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            scores = res[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(scores.mean()), 4)
            row[f"{metric}_std"] = round(float(scores.std()), 4)
        rows.append(row)
        pred = cross_val_predict(model, X, y, cv=cv)
        confusions[name] = confusion_matrix(y, pred, labels=SENTIMENT_LABELS).tolist()
    return pd.DataFrame(rows), confusions


def plot_confusion(cm: list[list[int]], path: Path, title: str) -> Path:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(SENTIMENT_LABELS)), SENTIMENT_LABELS, rotation=30)
    ax.set_yticks(range(len(SENTIMENT_LABELS)), SENTIMENT_LABELS)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, cm_arr[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def fit_and_save_final(df: pd.DataFrame, path: Path) -> Path:
    """Entraîne TF-IDF + régression logistique sur tout le corpus (pour l'application Streamlit)."""
    model = make_tfidf_logreg().fit(df["text"], df["sentiment"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return Path(path)
