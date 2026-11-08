"""Analyse exploratoire : distribution des notes, tests statistiques, figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["stars"].value_counts().sort_index()
    out = counts.rename("n").reset_index()
    out["pct"] = (100 * out["n"] / out["n"].sum()).round(1)
    return out


def statistical_tests(df: pd.DataFrame) -> dict:
    """Trois tests simples reliant longueur, note et ponctuation."""
    # 1) La longueur des avis diffère-t-elle selon la note ? (Kruskal-Wallis, non paramétrique)
    groups = [g["n_words"].to_numpy() for _, g in df.groupby("stars")]
    h_stat, p_kw = stats.kruskal(*groups)

    # 2) Corrélation de rang longueur / note (Spearman)
    rho, p_sp = stats.spearmanr(df["n_words"], df["stars"])

    # 3) L'usage du "!" dépend-il du sentiment ? (chi-deux d'indépendance)
    table = pd.crosstab(df["sentiment"], df["text"].str.contains("!", regex=False))
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)

    return {
        "kruskal_wallis_length_by_stars": {"H": float(h_stat), "p_value": float(p_kw)},
        "spearman_length_vs_stars": {"rho": float(rho), "p_value": float(p_sp)},
        "chi2_exclamation_vs_sentiment": {"chi2": float(chi2), "dof": int(dof), "p_value": float(p_chi)},
    }


def make_figures(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    dist = rating_distribution(df)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(dist["stars"].astype(str), dist["n"])
    ax.set_xlabel("Étoiles")
    ax.set_ylabel("Nombre d'avis")
    ax.set_title("Distribution des notes")
    fig.tight_layout()
    path = fig_dir / "rating_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    data = [g["n_words"].to_numpy() for _, g in df.groupby("stars")]
    ax.boxplot(data, tick_labels=sorted(df["stars"].unique()), showfliers=False)
    ax.set_xlabel("Étoiles")
    ax.set_ylabel("Nombre de mots")
    ax.set_title("Longueur des avis selon la note")
    fig.tight_layout()
    path = fig_dir / "length_by_stars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths
