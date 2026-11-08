"""Évaluation de la fiabilité des synthèses LLM sur un échantillon annoté manuellement.

Principe : chaque affirmation produite par le LLM est relue par un humain avec les avis cités,
puis étiquetée "supported" / "partial" / "unsupported". Un score lexical automatique sert
uniquement à repérer les affirmations suspectes à relire en priorité ; il ne remplace pas
l'annotation humaine.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

VERDICTS = {"supported", "partial", "unsupported"}
ERROR_TYPES = ["hallucination", "overgeneralisation", "wrong_attribution", "omission", "other"]
_STOP = set(
    "that this with from they have were been very would could their there which about "
    "what when also just than then them into only over some such more most much".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z']+", text.lower()) if len(t) > 3 and t not in _STOP}


def lexical_support(claim_text: str, cited_texts: list[str]) -> float:
    """Part des mots significatifs de l'affirmation retrouvés dans les avis cités (0 à 1)."""
    claim_tokens = _tokens(claim_text)
    if not claim_tokens:
        return 0.0
    cited_tokens = set().union(*(_tokens(t) for t in cited_texts)) if cited_texts else set()
    return round(len(claim_tokens & cited_tokens) / len(claim_tokens), 2)


def build_annotation_sheet(summaries: list[dict], reviews_by_id: dict[int, str], path: Path) -> Path:
    """Crée le CSV à remplir à la main (colonnes verdict / error_type / comment)."""
    rows = []
    for item in summaries:
        for claim in item["summary_json"]["claims"]:
            ids = [int(i) for i in claim.get("supporting_review_ids", [])]
            cited = [reviews_by_id[i] for i in ids if i in reviews_by_id]
            rows.append(
                {
                    "topic_id": item["topic_id"],
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("text", ""),
                    "supporting_review_ids": ";".join(map(str, ids)),
                    "cited_reviews": " || ".join(t[:300] for t in cited),
                    "auto_lexical_support": lexical_support(claim.get("text", ""), cited),
                    "verdict": "",       # supported | partial | unsupported
                    "error_type": "",    # hallucination | overgeneralisation | wrong_attribution | omission | other
                    "comment": "",
                }
            )
    sheet = pd.DataFrame(rows).sort_values("auto_lexical_support")  # les plus suspectes d'abord
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return Path(path)


def score_annotations(path: Path) -> dict:
    """Calcule les taux de fiabilité à partir du CSV annoté."""
    sheet = pd.read_csv(path, dtype={"verdict": str, "error_type": str, "comment": str})
    sheet["verdict"] = sheet["verdict"].astype(str).str.strip().str.lower()
    annotated = sheet[sheet["verdict"].isin(VERDICTS)]
    if annotated.empty:
        raise ValueError("Aucune ligne annotée : remplissez la colonne 'verdict' du CSV.")
    n = len(annotated)
    rates = annotated["verdict"].value_counts(normalize=True).round(3).to_dict()
    return {
        "n_annotated": int(n),
        "n_total": int(len(sheet)),
        "supported_rate": float(rates.get("supported", 0.0)),
        "partial_rate": float(rates.get("partial", 0.0)),
        "unsupported_rate": float(rates.get("unsupported", 0.0)),
        "by_topic": annotated.groupby("topic_id")["verdict"]
        .apply(lambda s: round(float((s == "supported").mean()), 3))
        .to_dict(),
        "error_types": annotated["error_type"].fillna("").replace("", pd.NA).dropna().value_counts().to_dict(),
        "lexical_support_vs_verdict": annotated.groupby("verdict")["auto_lexical_support"].mean().round(3).to_dict(),
    }


def main() -> None:
    """python -m reviews_nlp.evaluate_llm [chemin_csv] : calcule les taux de fiabilité et les enregistre."""
    import argparse
    import json

    from .config import ANNOTATION_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("path", nargs="?", default=str(ANNOTATION_DIR / "llm_claims_to_annotate.csv"))
    args = parser.parse_args()
    scores = score_annotations(Path(args.path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "llm_evaluation.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
