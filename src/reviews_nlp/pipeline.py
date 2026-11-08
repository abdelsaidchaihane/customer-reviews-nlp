"""Pipeline complet : python -m reviews_nlp.pipeline [--demo] [--sample-size N] ..."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import clean_reviews, load_yelp, make_demo_data, read_from_sqlite, save_to_sqlite
from .evaluate_llm import build_annotation_sheet
from .explore import make_figures, rating_distribution, statistical_tests
from .llm import select_reviews_for_topic, summarize_topic
from .models import (
    compute_embeddings,
    cross_validate_models,
    fit_and_save_final,
    plot_confusion,
)
from .quality import quality_report, run_sql_checks
from .topics import fit_lda, top_terms_by_sentiment


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Non sérialisable : {type(obj)}")


def _dump(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 1. Collecte
    print("[1/7] Chargement des données…")
    raw = make_demo_data() if args.demo else load_yelp(args.sample_size)
    print(f"      {len(raw)} avis chargés ({'démo synthétique' if args.demo else 'Yelp Review Full'}).")

    # 2. Qualité (brut) puis nettoyage
    print("[2/7] Contrôle qualité et nettoyage…")
    _dump(quality_report(raw), out / "quality_report_raw.json")
    df, cleaning_log = clean_reviews(raw)
    _dump(cleaning_log, out / "cleaning_log.json")
    save_to_sqlite(df, config.DB_PATH)
    _dump(run_sql_checks(config.DB_PATH, config.SQL_DIR / "quality_checks.sql"), out / "sql_checks.json")
    df = read_from_sqlite(config.DB_PATH)  # on repart de la base SQL

    # 3. Exploration statistique
    print("[3/7] Exploration statistique…")
    rating_distribution(df).to_csv(out / "rating_distribution.csv", index=False)
    _dump(statistical_tests(df), out / "statistical_tests.json")
    make_figures(df, config.FIG_DIR)

    # 4. Thèmes
    print("[4/7] Extraction de thèmes (TF-IDF, LDA)…")
    top_terms_by_sentiment(df).to_csv(out / "top_terms_by_sentiment.csv", index=False)
    min_df = 2 if args.demo else 5
    topics, assignments = fit_lda(df, n_topics=args.n_topics, min_df=min_df)
    topics.to_csv(out / "lda_topics.csv", index=False)
    assignments.to_csv(out / "lda_assignments.csv", index=False)

    # 5. Modèles de sentiment
    print("[5/7] Classification de sentiment (validation croisée)…")
    embeddings = None
    if not args.skip_embeddings:
        try:
            embeddings = compute_embeddings(df["text"])
        except ImportError as err:
            print(f"      Embeddings ignorés : {err}")
    metrics, confusions = cross_validate_models(df, embeddings=embeddings, n_splits=args.cv_folds)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    _dump(confusions, out / "confusion_matrices.json")
    for name, cm in confusions.items():
        plot_confusion(cm, config.FIG_DIR / f"confusion_{name}.png", name)
    fit_and_save_final(df, out / "model_tfidf_logreg.joblib")
    print(metrics.to_string(index=False))

    # 6. Synthèses LLM
    provider = args.llm_provider
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("      ANTHROPIC_API_KEY absent : bascule sur le fournisseur 'mock'.")
        provider = "mock"
    print(f"[6/7] Synthèses LLM (fournisseur : {provider})…")
    summaries, reviews_by_id = [], {}
    for row in topics.itertuples():
        sample = select_reviews_for_topic(df, assignments, row.topic_id, k=args.reviews_per_topic)
        if sample.empty:
            continue
        reviews_by_id.update(dict(zip(sample["review_id"], sample["text"])))
        result = summarize_topic(sample, row.top_words, provider=provider)
        summaries.append({"topic_id": int(row.topic_id), "top_words": row.top_words, "summary_json": result})
    _dump(summaries, out / "summaries.json")

    # 7. Feuille d'annotation manuelle
    print("[7/7] Création de la feuille d'annotation…")
    sheet = build_annotation_sheet(summaries, reviews_by_id, config.ANNOTATION_DIR / "llm_claims_to_annotate.csv")
    print(f"      À annoter à la main : {sheet}")
    print("Terminé. Résultats dans le dossier outputs/.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="jeu synthétique, sans téléchargement")
    p.add_argument("--sample-size", type=int, default=config.DEFAULT_SAMPLE_SIZE)
    p.add_argument("--n-topics", type=int, default=6)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--reviews-per-topic", type=int, default=25)
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--llm-provider", choices=["anthropic", "mock"], default="anthropic")
    run(p.parse_args())


if __name__ == "__main__":
    main()
