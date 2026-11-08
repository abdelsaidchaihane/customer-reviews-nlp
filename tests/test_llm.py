import pandas as pd
import pytest

from reviews_nlp.llm import build_prompt, parse_llm_json, summarize_topic


def test_parse_llm_json_accepts_code_fences():
    raw = '```json\n{"summary": "ok", "claims": [{"claim_id": 1, "text": "x", "supporting_review_ids": [1]}]}\n```'
    assert parse_llm_json(raw)["summary"] == "ok"


def test_parse_llm_json_rejects_invalid_payload():
    with pytest.raises(ValueError):
        parse_llm_json("pas de json ici")
    with pytest.raises(ValueError):
        parse_llm_json('{"summary": "ok"}')


def test_prompt_contains_ids_and_rules():
    reviews = pd.DataFrame({"review_id": [7], "text": ["Great pasta."]})
    prompt = build_prompt(reviews, "food, pasta")
    assert "[id=7]" in prompt and "ONLY information" in prompt


def test_mock_provider_returns_structured_claims():
    reviews = pd.DataFrame({"review_id": [1, 2], "text": ["Good food. Nice.", "Bad service! Slow."]})
    result = summarize_topic(reviews, "food", provider="mock")
    assert result["claims"] and result["claims"][0]["supporting_review_ids"] == [1]
