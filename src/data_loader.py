"""Chargement du dataset UCI Fertility et conversions d'unités."""
import pandas as pd
from ucimlrepo import fetch_ucirepo

FEATURES = ["season", "age", "child_diseases", "accident",
            "surgical_intervention", "high_fevers", "alcohol",
            "smoking", "hrs_sitting"]


def load_fertility():
    """Retourne X (9 colonnes normalisées) et y (N=0, O=1)."""
    ds = fetch_ucirepo(id=244)
    X = ds.data.features[FEATURES].copy()
    y = (ds.data.targets.iloc[:, 0] == "O").astype(int)
    return X, y


def age_to_years(age_norm):
    return age_norm * 18 + 18


def hours_from_norm(hrs_norm):
    return hrs_norm * 16