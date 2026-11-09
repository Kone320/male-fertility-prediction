import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.predict import predict_fertility, to_normalized

COLUMNS = ["season", "age", "child_diseases", "accident",
           "surgical_intervention", "high_fevers", "alcohol",
           "smoking", "hrs_sitting"]


def _fake_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((60, 9)), columns=COLUMNS)
    y = np.array([0] * 50 + [1] * 10)
    return GradientBoostingClassifier(random_state=0).fit(X, y)


def _call(model, age=30, hrs=8):
    return predict_fertility(model, COLUMNS, season=-0.33, age_years=age,
                             child_diseases=1, accident=0,
                             surgical_intervention=0, high_fevers=0,
                             alcohol=0.8, smoking=-1, hrs_sitting_hours=hrs)


def test_normalization():
    assert to_normalized(18, 0) == (0, 0)
    assert to_normalized(36, 16) == (1, 1)
    assert to_normalized(27, 8) == (0.5, 0.5)


def test_probabilities_sum_to_one():
    out = _call(_fake_model())
    assert out["probability_altered"] + out["probability_normal"] == pytest.approx(1.0)
    assert out["prediction"] in ("Normal", "Altéré")


def test_out_of_range_age_warns():
    with pytest.warns(UserWarning):
        _call(_fake_model(), age=20)