from reviews_nlp.config import SQL_DIR
from reviews_nlp.data import clean_reviews, make_demo_data, save_to_sqlite
from reviews_nlp.quality import quality_report, run_sql_checks


def test_quality_report_detects_defects():
    report = quality_report(make_demo_data())
    assert report["n_missing_text"] > 0
    assert report["n_duplicated_texts"] > 0
    assert report["imbalance_ratio_max_over_min"] >= 1


def test_sql_checks_run_on_clean_table(tmp_path):
    clean, _ = clean_reviews(make_demo_data())
    db = tmp_path / "reviews.db"
    save_to_sqlite(clean, db)
    results = run_sql_checks(db, SQL_DIR / "quality_checks.sql")
    assert results["duplicated_texts"][0]["n_duplicated_texts"] == 0
    assert results["null_or_empty_texts"][0]["n_null_or_empty"] == 0
    assert results["invalid_stars"][0]["n_invalid_stars"] == 0
