"""Paramètres centraux du projet."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
SQL_DIR = ROOT / "sql"
ANNOTATION_DIR = ROOT / "annotations"
DB_PATH = DATA_DIR / "reviews.db"

RANDOM_STATE = 42

# Données : Yelp Review Full (avis en anglais, notes de 1 à 5 étoiles), via Hugging Face
HF_DATASET = "Yelp/yelp_review_full"
DEFAULT_SAMPLE_SIZE = 20_000

# Embeddings de phrases (modèle léger, anglais)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (modifiable via la variable d'environnement LLM_MODEL)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def stars_to_sentiment(stars: pd.Series) -> pd.Series:
    """1-2 étoiles -> negative, 3 -> neutral, 4-5 -> positive."""
    return pd.cut(
        stars,
        bins=[0, 2, 3, 5],
        labels=SENTIMENT_LABELS,
    ).astype(str)
