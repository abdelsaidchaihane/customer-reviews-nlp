-- Contrôles qualité exécutés sur la table `reviews` (après nettoyage).
-- Chaque requête est précédée d'un en-tête "-- name: <nom>" lu par quality.run_sql_checks().

-- name: rows_per_sentiment
SELECT sentiment,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 1) AS pct
FROM reviews
GROUP BY sentiment
ORDER BY n DESC;

-- name: duplicated_texts
SELECT COUNT(*) AS n_duplicated_texts
FROM (SELECT text FROM reviews GROUP BY text HAVING COUNT(*) > 1);

-- name: null_or_empty_texts
SELECT COUNT(*) AS n_null_or_empty
FROM reviews
WHERE text IS NULL OR TRIM(text) = '';

-- name: invalid_stars
SELECT COUNT(*) AS n_invalid_stars
FROM reviews
WHERE stars IS NULL OR stars NOT BETWEEN 1 AND 5;

-- name: length_by_stars
SELECT stars,
       COUNT(*) AS n,
       ROUND(AVG(n_words), 1) AS avg_words,
       MIN(n_words) AS min_words,
       MAX(n_words) AS max_words
FROM reviews
GROUP BY stars
ORDER BY stars;
