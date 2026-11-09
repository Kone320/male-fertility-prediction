"""Construction, évaluation et sauvegarde du modèle."""
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

SEED = 42
# Paramètres issus du GridSearchCV du notebook (sélection optimiste, voir README)
BEST_PARAMS = {"n_estimators": 200, "learning_rate": 0.01,
               "max_depth": 2, "subsample": 1.0}


def build_model(seed=SEED):
    return GradientBoostingClassifier(random_state=seed, **BEST_PARAMS)


def repeated_cv_auc(X, y, n_splits=5, n_repeats=10, seed=SEED):
    """AUC moyenne et écart-type en CV stratifiée répétée."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    scores = cross_val_score(build_model(seed), X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def train_and_save(X, y, path="models/fertility_gb.joblib"):
    model = build_model().fit(X, y)
    joblib.dump({"model": model, "columns": list(X.columns)}, path)
    return model