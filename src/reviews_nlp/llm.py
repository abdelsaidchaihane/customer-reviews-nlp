"""Synthèse d'avis par un LLM, avec sortie structurée en affirmations vérifiables.

Deux fournisseurs :
- "anthropic" : appel à l'API (variable d'environnement ANTHROPIC_API_KEY) ;
- "mock"      : synthèse extractive locale, pour tester le pipeline hors ligne.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from .config import LLM_MODEL, RANDOM_STATE

MAX_REVIEW_CHARS = 600


def select_reviews_for_topic(
    df: pd.DataFrame, assignments: pd.DataFrame, topic_id: int, k: int = 25, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Tire k avis parmi les 200 plus représentatifs du thème (pour garder de la diversité)."""
    merged = df.merge(assignments, on="review_id")
    pool = merged[merged["topic_id"] == topic_id].nlargest(200, "topic_prob")
    sample = pool.sample(n=min(k, len(pool)), random_state=seed) if len(pool) else pool
    return sample[["review_id", "text"]].reset_index(drop=True)


def build_prompt(reviews: pd.DataFrame, topic_words: str) -> str:
    lines = [f"[id={r.review_id}] {r.text[:MAX_REVIEW_CHARS]}" for r in reviews.itertuples()]
    schema = (
        '{"summary": "...", "claims": '
        '[{"claim_id": 1, "text": "...", "supporting_review_ids": [12, 40]}]}'
    )
    return (
        "You are assisting an analyst who reads customer reviews.\n"
        f"Below are {len(reviews)} reviews grouped into one topic (top words: {topic_words}).\n\n"
        "Write a faithful synthesis. Rules:\n"
        "- Use ONLY information present in the reviews. Do not add facts, figures or examples that are not in them.\n"
        "- Give a one-sentence \"summary\" and 3 to 5 \"claims\". Each claim is one verifiable statement and "
        "lists, in \"supporting_review_ids\", the ids of the reviews that support it.\n"
        "- If the reviews disagree, say so in a claim.\n"
        f"Return ONLY valid JSON with this shape: {schema}\n\n"
        "Reviews:\n" + "\n".join(lines)
    )


def parse_llm_json(raw: str) -> dict:
    """Extrait et valide le JSON renvoyé par le modèle (tolère les balises ```json)."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Aucun objet JSON dans la réponse du modèle.")
    data = json.loads(cleaned[start : end + 1])
    if "summary" not in data or not isinstance(data.get("claims"), list):
        raise ValueError("JSON invalide : clés 'summary' et 'claims' attendues.")
    return data


def _call_anthropic(prompt: str, model: str) -> str:  # pragma: no cover (appel réseau)
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("pip install anthropic (ou utilisez --llm-provider mock).") from exc
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY n'est pas défini (voir .env.example).")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=1200, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def _mock_summary(reviews: pd.DataFrame) -> dict:
    """Synthèse extractive déterministe : première phrase des 3 premiers avis."""
    claims = []
    for i, r in enumerate(reviews.head(3).itertuples(), start=1):
        first_sentence = re.split(r"(?<=[.!?])\s", r.text.strip())[0]
        claims.append({"claim_id": i, "text": first_sentence, "supporting_review_ids": [int(r.review_id)]})
    return {"summary": "[mock] Synthèse extractive de démonstration.", "claims": claims}


def summarize_topic(
    reviews: pd.DataFrame, topic_words: str, provider: str = "mock", model: str = LLM_MODEL
) -> dict:
    if provider == "mock":
        return _mock_summary(reviews)
    if provider == "anthropic":
        return parse_llm_json(_call_anthropic(build_prompt(reviews, topic_words), model))
    raise ValueError(f"Fournisseur inconnu : {provider}")
