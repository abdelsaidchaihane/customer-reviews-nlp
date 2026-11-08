from reviews_nlp.config import SENTIMENT_LABELS
from reviews_nlp.data import clean_reviews, make_demo_data


def test_demo_data_contains_injected_quality_issues():
    raw = make_demo_data()
    assert raw["text"].isna().any()
    assert raw["text"].dropna().duplicated().any()


def test_clean_reviews_removes_issues_and_adds_features():
    raw = make_demo_data()
    clean, log = clean_reviews(raw)
    assert clean["text"].notna().all()
    assert not clean["text"].duplicated().any()
    assert (clean["text"].str.len() >= 10).all()
    assert not clean["text"].str.contains("<br>", regex=False).any()
    assert not clean["text"].str.contains("\\n", regex=False).any()
    assert set(clean["sentiment"]) <= set(SENTIMENT_LABELS)
    assert log["n_output"] == len(clean)
    assert log["n_input"] == len(raw)
