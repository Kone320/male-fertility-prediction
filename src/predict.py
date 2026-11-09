"""Prédiction à partir de valeurs en unités réelles."""
import warnings
import joblib
import pandas as pd


def load_model(path="models/fertility_gb.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["columns"]


def to_normalized(age_years, hrs_sitting_hours):
    """Années vers [0,1] (18 à 36 ans) et heures vers [0,1] (0 à 16 h)."""
    return (age_years - 18) / 18, hrs_sitting_hours / 16


def predict_fertility(model, columns, season, age_years, child_diseases,
                      accident, surgical_intervention, high_fevers, alcohol,
                      smoking, hrs_sitting_hours, threshold=0.5):
    if not 27 <= age_years <= 36:
        warnings.warn("Âge hors de la plage d'entraînement (27 à 36 ans).",
                      UserWarning)
    if not 0 <= hrs_sitting_hours <= 16:
        warnings.warn("Heures assises hors de la plage 0 à 16.", UserWarning)

    age_norm, hrs_norm = to_normalized(age_years, hrs_sitting_hours)
    row = pd.DataFrame([{
        "season": season, "age": age_norm,
        "child_diseases": child_diseases, "accident": accident,
        "surgical_intervention": surgical_intervention,
        "high_fevers": high_fevers, "alcohol": alcohol,
        "smoking": smoking, "hrs_sitting": hrs_norm,
    }])[columns]

    p_altered = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": "Altéré" if p_altered >= threshold else "Normal",
        "probability_altered": p_altered,
        "probability_normal": 1 - p_altered,
        "threshold": threshold,
    }