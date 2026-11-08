# Cadrage du projet

> Ce document a été rédigé **avant** la modélisation : les critères de réussite ne sont pas ajustés a posteriori.

## 1. Contexte et besoin

Une équipe reçoit un grand nombre d'avis clients en texte libre. La lecture manuelle ne passe pas à l'échelle.
Le besoin exprimé : **comprendre rapidement ce que disent les clients** (ce qui est positif ou négatif, sur quels sujets)
et **disposer de synthèses fiables**, c'est-à-dire que l'on peut relier aux avis d'origine.

## 2. Utilisateurs cibles

| Utilisateur | Besoin | Livrable |
|---|---|---|
| Analyste / chargé d'études | Vue d'ensemble, thèmes, tendances | Tableau de bord Streamlit, rapport |
| Équipe métier | Synthèse lisible par thème | Synthèses LLM avec avis sources |
| Data scientist (relecteur) | Savoir si l'on peut se fier au modèle | Métriques, limites, taux de fiabilité du LLM |

## 3. Cas d'usage envisagés et priorisation

| Cas d'usage | Valeur attendue | Faisabilité | Qualité / disponibilité des données | Priorité |
|---|---|---|---|---|
| Contrôle qualité du corpus | Élevée (conditionne tout le reste) | Élevée | Données disponibles | 1 |
| Classification du sentiment | Élevée | Élevée (étiquettes = notes) | Bonne, classes déséquilibrées | 2 |
| Extraction de thèmes | Moyenne à élevée | Moyenne (non supervisé, interprétation humaine) | Bonne | 3 |
| Synthèse par LLM | Élevée mais risquée | Moyenne (fiabilité à prouver) | Dépend des thèmes | 4 |

## 4. Données

- **Source** : Yelp Review Full (avis en anglais, notes de 1 à 5), via Hugging Face (`Yelp/yelp_review_full`).
- **Volume utilisé** : échantillon aléatoire de 20 000 avis (paramétrable).
- **Cible de sentiment** (dérivée de la note) : 1-2 étoiles = négatif, 3 = neutre, 4-5 = positif.
- **Risques identifiés** : doublons, textes très courts, artefacts de format (séquences `\n`), classe « neutre » ambiguë,
  corpus daté (2015) et centré sur les États-Unis.

## 5. Critères d'évaluation (fixés avant modélisation)

**Sentiment**
- Validation croisée stratifiée à 5 plis, graine fixée.
- Métrique principale : **F1 macro** (classes déséquilibrées) ; métriques secondaires : exactitude, F1 pondéré, matrice de confusion.
- Référence à dépasser : baseline « classe majoritaire ». Un modèle n'est retenu que s'il la dépasse nettement et si son
  écart-type entre plis reste faible.

**Thèmes**
- Interprétabilité jugée par relecture des mots-clés et d'avis représentatifs de chaque thème.

**Synthèses LLM**
- Sortie structurée : une synthèse + 3 à 5 affirmations, chacune reliée aux avis qui la soutiennent.
- Chaque affirmation est relue à la main avec les avis cités et étiquetée `supported`, `partial` ou `unsupported`.
- Indicateurs : taux d'affirmations soutenues et taux d'affirmations non soutenues (hallucinations), par thème.
- Seuils visés à l'avance (hypothèses de travail, à discuter) : au moins 85 % d'affirmations soutenues et au plus 5 % non soutenues.
  Si ces seuils ne sont pas atteints, le résultat est documenté tel quel avec les types d'erreurs observés.

## 6. Contraintes

- Reproductibilité : graines fixées, dépendances listées, pipeline exécutable en une commande.
- Pas de donnée personnelle ; respect de l'accord d'utilisation du jeu de données Yelp (usage académique).
- Coût et confidentialité de l'API LLM : appels limités à quelques dizaines d'avis par thème.

## 7. Livrables

Pipeline Python, base SQLite et requêtes de contrôle, tests, tableau de bord Streamlit, rapport Quarto,
feuille d'annotation du LLM, et ce document de cadrage.
