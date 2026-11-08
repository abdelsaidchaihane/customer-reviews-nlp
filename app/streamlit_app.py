"""Tableau de bord Streamlit. Générez d'abord les résultats : `make demo` ou `make run`."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
FIG = OUT / "figures"

st.set_page_config(page_title="Analyse d'avis clients", layout="wide")
st.title("Analyse automatique d'avis clients")

if not (OUT / "model_metrics.csv").exists():
    st.warning("Aucun résultat trouvé. Lancez `make demo` ou `make run`, puis rechargez la page.")
    st.stop()


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    path = OUT / "model_tfidf_logreg.joblib"
    return joblib.load(path) if path.exists() else None


tab_quality, tab_explore, tab_topics, tab_models, tab_llm, tab_try = st.tabs(
    ["Qualité des données", "Exploration", "Thèmes", "Modèles", "Synthèses LLM", "Tester un avis"]
)

with tab_quality:
    report, log = read_json("quality_report_raw.json"), read_json("cleaning_log.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avis bruts", report["n_rows"])
    c2.metric("Textes manquants", report["n_missing_text"])
    c3.metric("Doublons", report["n_duplicated_texts"])
    c4.metric("Avis conservés", log["n_output"])
    st.subheader("Déséquilibre des classes")
    st.json(report["sentiment_counts"])
    st.subheader("Journal de nettoyage")
    st.json(log)
    st.subheader("Contrôles SQL (après nettoyage)")
    for name, rows in read_json("sql_checks.json").items():
        st.caption(name)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_explore:
    dist = pd.read_csv(OUT / "rating_distribution.csv")
    st.bar_chart(dist.set_index("stars")["n"])
    st.subheader("Tests statistiques")
    st.json(read_json("statistical_tests.json"))
    for img in ("rating_distribution.png", "length_by_stars.png"):
        if (FIG / img).exists():
            st.image(str(FIG / img))

with tab_topics:
    st.subheader("Thèmes (LDA)")
    st.dataframe(pd.read_csv(OUT / "lda_topics.csv"), use_container_width=True)
    st.subheader("Termes les plus caractéristiques par sentiment (TF-IDF)")
    terms = pd.read_csv(OUT / "top_terms_by_sentiment.csv")
    st.dataframe(terms.pivot(index="rank", columns="sentiment", values="term"), use_container_width=True)

with tab_models:
    st.dataframe(pd.read_csv(OUT / "model_metrics.csv"), use_container_width=True)
    cols = st.columns(3)
    for col, img in zip(cols, sorted(FIG.glob("confusion_*.png"))):
        col.image(str(img), caption=img.stem)

with tab_llm:
    if (OUT / "llm_evaluation.json").exists():
        st.subheader("Fiabilité mesurée sur l'échantillon annoté")
        st.json(read_json("llm_evaluation.json"))
    else:
        st.info("Annotez annotations/llm_claims_to_annotate.csv puis lancez `make score`.")
    for item in read_json("summaries.json"):
        with st.expander(f"Thème {item['topic_id']} : {item['top_words']}"):
            st.write(item["summary_json"]["summary"])
            for claim in item["summary_json"]["claims"]:
                st.markdown(f"- {claim['text']}  \n  _avis : {claim.get('supporting_review_ids')}_")

with tab_try:
    model = load_model()
    text = st.text_area("Saisissez un avis (en anglais)", "The pasta was delicious but the service was slow.")
    if model is not None and text.strip():
        proba = pd.Series(model.predict_proba([text])[0], index=model.classes_, name="probabilité")
        st.bar_chart(proba)
        st.write(f"Sentiment prédit : **{proba.idxmax()}**")
