from reviews_nlp.data import clean_reviews, make_demo_data
from reviews_nlp.models import cross_validate_models


def test_tfidf_logreg_beats_majority_baseline():
    clean, _ = clean_reviews(make_demo_data())
    metrics, confusions = cross_validate_models(clean, n_splits=3)
    scores = metrics.set_index("model")["f1_macro_mean"]
    assert scores["tfidf_logreg"] > scores["baseline_majority"]
    assert set(confusions) == {"baseline_majority", "tfidf_logreg"}
