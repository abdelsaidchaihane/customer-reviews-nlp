from reviews_nlp.data import clean_reviews, make_demo_data
from reviews_nlp.topics import fit_lda, top_terms_by_sentiment


def test_lda_returns_one_row_per_topic_and_review():
    clean, _ = clean_reviews(make_demo_data())
    topics, assignments = fit_lda(clean, n_topics=4, min_df=2)
    assert len(topics) == 4
    assert len(assignments) == len(clean)
    assert assignments["topic_prob"].between(0, 1).all()


def test_top_terms_cover_every_sentiment():
    clean, _ = clean_reviews(make_demo_data())
    terms = top_terms_by_sentiment(clean, n_terms=5)
    assert set(terms["sentiment"]) == set(clean["sentiment"])
