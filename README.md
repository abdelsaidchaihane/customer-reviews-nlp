# Analyse automatique d'avis clients : sentiment, thèmes et synthèse par LLM

> **In short.** End-to-end NLP project on Yelp reviews: data-quality checks (pandas + SQL), exploratory statistics,
> cross-validated sentiment classification (TF-IDF and sentence embeddings), topic extraction (TF-IDF, LDA) and an LLM
> summariser whose faithfulness is **measured on a hand-annotated sample**. Reproducible, tested, with a Streamlit dashboard
> and a Quarto report.

## Objectif

Exploiter automatiquement un grand volume d'avis clients : fiabiliser le corpus, prédire le sentiment, faire émerger les
thèmes récurrents et tester une synthèse générée par un LLM **dont on mesure la fiabilité** plutôt que de la supposer.
Le besoin et les critères d'évaluation sont posés **avant** la modélisation dans [`docs/cadrage.md`](docs/cadrage.md).

## Démarche

| Étape | Contenu | Code |
|---|---|---|
| 1. Cadrage | Besoin, priorisation des cas d'usage, critères d'évaluation fixés a priori | `docs/cadrage.md` |
| 2. Qualité des données | Valeurs manquantes, doublons, textes courts, artefacts, déséquilibre ; nettoyage ; contrôles SQL | `data.py`, `quality.py`, `sql/quality_checks.sql` |
| 3. Exploration statistique | Distribution des notes, Kruskal-Wallis, Spearman, chi-deux | `explore.py` |
| 4. Thèmes | Termes TF-IDF par sentiment, LDA | `topics.py` |
| 5. Sentiment | Baseline, TF-IDF + régression logistique, embeddings + régression logistique, validation croisée | `models.py` |
| 6. Synthèse LLM | Sortie JSON structurée (affirmations reliées aux avis), évaluation sur échantillon annoté | `llm.py`, `evaluate_llm.py` |
| 7. Restitution | Tableau de bord Streamlit, rapport Quarto reproductible | `app/`, `report/` |

## Résultats

> À compléter après exécution (`make run`). Ne reportez ici que des chiffres issus de `outputs/`.

**Sentiment (validation croisée 5 plis)** : voir `outputs/model_metrics.csv`

| Modèle | F1 macro (moyenne ± écart-type) | Exactitude |
|---|---|---|
| Baseline classe majoritaire | _à compléter_ | _à compléter_ |
| TF-IDF + régression logistique | _à compléter_ | _à compléter_ |
| Embeddings + régression logistique | _à compléter_ | _à compléter_ |

**Fiabilité des synthèses LLM** : voir `outputs/llm_evaluation.json` après annotation

| Indicateur | Valeur |
|---|---|
| Affirmations annotées | _à compléter_ |
| Soutenues / partielles / non soutenues | _à compléter_ |
| Principaux types d'erreurs | _à compléter_ |

## Installation

```bash
git clone <url-du-depot> && cd customer-reviews-nlp
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # puis renseigner ANTHROPIC_API_KEY (facultatif)
```

## Utilisation

```bash
make demo      # test rapide hors ligne sur données synthétiques (aucun téléchargement, LLM simulé)
make run       # pipeline complet sur 20 000 avis Yelp (nécessite internet ; LLM réel si ANTHROPIC_API_KEY est défini)
make test      # tests unitaires
make app       # tableau de bord Streamlit
make report    # rapport Quarto (nécessite Quarto installé)
```

Après `make run`, ouvrez `annotations/llm_claims_to_annotate.csv`, renseignez la colonne `verdict`
(`supported`, `partial` ou `unsupported`) pour un échantillon d'affirmations, puis lancez `make score`.
Les affirmations au plus faible score lexical automatique sont listées en premier : ce score sert à
prioriser la relecture, il ne remplace pas l'annotation humaine.

Options utiles : `python -m reviews_nlp.pipeline --help` (taille de l'échantillon, nombre de thèmes, plis, etc.).

## Structure du dépôt

```
├── docs/cadrage.md          # besoin, priorisation, critères d'évaluation
├── sql/quality_checks.sql   # contrôles qualité SQL
├── src/reviews_nlp/         # code du pipeline
├── app/streamlit_app.py     # tableau de bord
├── report/report.qmd        # rapport reproductible
├── tests/                   # tests pytest
├── annotations/             # feuille d'annotation du LLM
├── outputs/                 # résultats générés (figures, métriques)
└── data/                    # base SQLite générée (non versionnée)
```

## Choix méthodologiques

- **Trois classes** (négatif / neutre / positif) dérivées des notes : plus lisible pour un usage métier que cinq étoiles, mais la classe neutre est ambiguë.
- **F1 macro** comme métrique principale, car les classes sont déséquilibrées, et **baseline majoritaire** comme référence à dépasser.
- **Validation croisée stratifiée**, graine fixée. Les embeddings viennent d'un modèle pré-entraîné figé : ils ne sont pas ajustés sur les plis de test.
- **LDA** pour les thèmes : interprétable mais le nombre de thèmes est fixé a priori.
- **LLM avec sortie structurée** : chaque affirmation cite les avis qui la soutiennent, ce qui rend la vérification possible.
- **Mode `mock`** : synthèse extractive locale pour tester le pipeline sans clé d'API ni réseau.

## Limites

- Corpus en anglais, restaurants et commerces américains (Yelp 2015) : la généralisation à d'autres domaines n'est pas testée.
- Échantillon d'annotation limité et relu par une seule personne : pas d'accord inter-annotateurs.
- Résultats du LLM variables d'un appel à l'autre et d'un modèle à l'autre.
- Les avis envoyés à l'API sont transmis à un tiers : à ne pas faire avec des données confidentielles.

## Données et licence

Aucune donnée n'est versionnée. Le jeu **Yelp Review Full** est chargé depuis Hugging Face et soumis à l'accord d'utilisation
du jeu de données Yelp (usage académique) : vérifiez ses conditions avant toute réutilisation.
Le code est sous licence MIT (voir `LICENSE`).

## Auteur

Abdel Said Chaihane : [GitHub](https://github.com/abdelsaidchaihane) · [LinkedIn](https://www.linkedin.com/in/abdel-said-chaihane-5a55b4234/)
