import pandas as pd
import pytest

from reviews_nlp.evaluate_llm import build_annotation_sheet, lexical_support, score_annotations


def test_lexical_support_bounds():
    assert lexical_support("delicious pasta", ["The pasta was delicious"]) == 1.0
    assert lexical_support("expensive wine", ["The pasta was delicious"]) == 0.0


def test_annotation_roundtrip(tmp_path):
    summaries = [{
        "topic_id": 0,
        "summary_json": {"claims": [
            {"claim_id": 1, "text": "Pasta is delicious", "supporting_review_ids": [1]},
            {"claim_id": 2, "text": "Wine is expensive", "supporting_review_ids": [1]},
        ]},
    }]
    path = build_annotation_sheet(summaries, {1: "The pasta was delicious"}, tmp_path / "sheet.csv")
    sheet = pd.read_csv(path, dtype={"verdict": str, "error_type": str, "comment": str})
    sheet.loc[sheet["claim_id"] == 1, "verdict"] = "supported"
    sheet.loc[sheet["claim_id"] == 2, "verdict"] = "unsupported"
    sheet.loc[sheet["claim_id"] == 2, "error_type"] = "hallucination"
    sheet.to_csv(path, index=False)
    scores = score_annotations(path)
    assert scores["n_annotated"] == 2
    assert scores["supported_rate"] == 0.5 and scores["unsupported_rate"] == 0.5
    assert scores["error_types"] == {"hallucination": 1}


def test_score_requires_annotations(tmp_path):
    summaries = [{"topic_id": 0, "summary_json": {"claims": [{"claim_id": 1, "text": "x", "supporting_review_ids": []}]}}]
    path = build_annotation_sheet(summaries, {}, tmp_path / "empty.csv")
    with pytest.raises(ValueError):
        score_annotations(path)
