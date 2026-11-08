"""Collecte, nettoyage et stockage SQL des avis."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_SAMPLE_SIZE,
    HF_DATASET,
    RANDOM_STATE,
    stars_to_sentiment,
)

HTML_TAG = re.compile(r"<[^>]+>")
MIN_CHARS = 10


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_yelp(sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Charge un échantillon aléatoire de Yelp Review Full (Hugging Face)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Le paquet 'datasets' est requis pour charger Yelp : pip install datasets "
            "(ou lancez le mode --demo)."
        ) from exc

    ds = load_dataset(HF_DATASET, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
    df = ds.to_pandas()
    df["stars"] = df["label"] + 1  # labels 0..4 -> 1..5 étoiles
    df = df.drop(columns="label").reset_index(drop=True)
    df.insert(0, "review_id", np.arange(1, len(df) + 1))
    return df


_SENTENCES = {
    "food": {
        "pos": ["The food was delicious and the portions were generous.",
                "Fresh ingredients and amazing flavors, the pasta was perfect."],
        "neg": ["The food was cold and tasteless.",
                "My burger was overcooked and the fries were soggy."],
        "neu": ["The food was okay, nothing special."],
    },
    "service": {
        "pos": ["The staff were friendly and very attentive.",
                "Our waiter was quick and polite."],
        "neg": ["The service was slow and the waiter was rude.",
                "We waited forty minutes for our drinks."],
        "neu": ["Service was average."],
    },
    "price": {
        "pos": ["Great value for the price.", "Prices are very reasonable."],
        "neg": ["Way too expensive for what you get.", "The bill was a nasty surprise."],
        "neu": ["Prices are about what you would expect."],
    },
    "location": {
        "pos": ["Perfect location right in the city centre.",
                "Easy parking and close to the station."],
        "neg": ["Hard to find and no parking nearby."],
        "neu": ["It is located on a quiet street."],
    },
    "cleanliness": {
        "pos": ["The place was spotless.", "Very clean tables and restrooms."],
        "neg": ["The tables were dirty and the restroom smelled bad."],
        "neu": ["Cleanliness was acceptable."],
    },
    "atmosphere": {
        "pos": ["Cozy atmosphere and great music."],
        "neg": ["Too noisy, we could barely talk."],
        "neu": ["The atmosphere was fine."],
    },
}


def make_demo_data(n: int = 600, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Jeu synthétique pour tester le pipeline hors ligne (avec défauts de qualité injectés)."""
    rng = np.random.default_rng(seed)
    stars = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.12, 0.18, 0.30, 0.30])
    topics = list(_SENTENCES)
    texts = []
    for s in stars:
        tone = "neg" if s <= 2 else "neu" if s == 3 else "pos"
        parts = []
        for topic in rng.choice(topics, size=rng.integers(2, 4), replace=False):
            use_tone = tone
            if tone != "neu" and rng.random() < 0.15:  # un peu de bruit réaliste
                use_tone = "pos" if tone == "neg" else "neg"
            options = _SENTENCES[topic].get(use_tone) or _SENTENCES[topic]["neu"]
            parts.append(str(rng.choice(options)))
        texts.append(" ".join(parts))

    df = pd.DataFrame({"review_id": np.arange(1, n + 1), "text": texts, "stars": stars})

    # Défauts de qualité injectés volontairement
    k = max(2, int(0.03 * n))
    dup_idx = rng.choice(n, size=k, replace=False)
    df.loc[dup_idx, "text"] = df.loc[0, "text"]                     # doublons
    df.loc[rng.choice(n, size=k, replace=False), "text"] = None     # textes manquants
    df.loc[rng.choice(n, size=k, replace=False), "text"] = "ok"     # textes trop courts
    html_idx = rng.choice(n, size=k, replace=False)
    df.loc[html_idx, "text"] = df.loc[html_idx, "text"].fillna("") + "\\n<br>Great!"
    return df


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #
def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie les textes et renvoie (df propre, journal des suppressions)."""
    log = {"n_input": int(len(df))}
    out = df.copy()

    out = out[out["text"].notna()]
    log["removed_missing_text"] = log["n_input"] - len(out)

    out["text"] = (
        out["text"]
        .astype(str)
        .str.replace("\\n", " ", regex=False)   # séquences "\n" littérales présentes dans Yelp
        .str.replace(HTML_TAG, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    before = len(out)
    out = out[out["text"].str.len() >= MIN_CHARS]
    log["removed_too_short"] = int(before - len(out))

    before = len(out)
    out = out.drop_duplicates(subset="text", keep="first")
    log["removed_duplicates"] = int(before - len(out))

    out = out.copy()
    out["sentiment"] = stars_to_sentiment(out["stars"])
    out["n_chars"] = out["text"].str.len()
    out["n_words"] = out["text"].str.split().str.len()
    out = out.reset_index(drop=True)
    log["n_output"] = int(len(out))
    return out, log


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
def save_to_sqlite(df: pd.DataFrame, db_path: Path, table: str = "reviews") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def read_from_sqlite(db_path: Path, query: str = "SELECT * FROM reviews") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
