"""Extraction de thèmes : termes TF-IDF par sentiment et LDA."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .config import RANDOM_STATE


def top_terms_by_sentiment(df: pd.DataFrame, n_terms: int = 12, min_df: int = 2) -> pd.DataFrame:
    """Termes dont le poids TF-IDF moyen est le plus élevé dans chaque classe de sentiment."""
    vec = TfidfVectorizer(stop_words="english", min_df=min_df, max_df=0.7, ngram_range=(1, 2))
    X = vec.fit_transform(df["text"])
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for label in sorted(df["sentiment"].unique()):
        mask = (df["sentiment"] == label).to_numpy()
        mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
        for rank, idx in enumerate(mean_tfidf.argsort()[::-1][:n_terms], start=1):
            rows.append({"sentiment": label, "rank": rank, "term": terms[idx], "mean_tfidf": float(mean_tfidf[idx])})
    return pd.DataFrame(rows)


def fit_lda(
    df: pd.DataFrame,
    n_topics: int = 6,
    n_top_words: int = 10,
    min_df: int = 5,
    max_df: float = 0.5,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LDA sur un sac de mots. Renvoie (thèmes avec mots-clés, affectation des avis)."""
    vec = CountVectorizer(stop_words="english", min_df=min_df, max_df=max_df)
    X = vec.fit_transform(df["text"])
    lda = LatentDirichletAllocation(
        n_components=n_topics, learning_method="online", random_state=seed, max_iter=15
    )
    doc_topic = lda.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())

    topics = pd.DataFrame(
        {
            "topic_id": range(n_topics),
            "top_words": [", ".join(terms[comp.argsort()[::-1][:n_top_words]]) for comp in lda.components_],
        }
    )
    assignments = pd.DataFrame(
        {
            "review_id": df["review_id"].to_numpy(),
            "topic_id": doc_topic.argmax(axis=1),
            "topic_prob": doc_topic.max(axis=1),
        }
    )
    topics["n_reviews"] = topics["topic_id"].map(assignments["topic_id"].value_counts()).fillna(0).astype(int)
    return topics, assignments
