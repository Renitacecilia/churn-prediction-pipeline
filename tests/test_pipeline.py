import numpy as np
import pandas as pd
import pytest

from src import config, etl, make_sample_data, train


@pytest.fixture(scope="module")
def clean_df():
    return etl.transform(make_sample_data.generate(n=800, seed=1))


def test_transform_fixes_total_charges_and_target(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].notna().all()
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}


def test_transform_removes_duplicates():
    raw = make_sample_data.generate(n=50, seed=2)
    doubled = pd.concat([raw, raw], ignore_index=True)
    assert len(etl.transform(doubled)) == 50


def test_transform_rejects_bad_target():
    raw = make_sample_data.generate(n=20, seed=3)
    raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        etl.transform(raw)


def test_validate_catches_missing_columns(clean_df):
    with pytest.raises(ValueError):
        etl.validate(clean_df.drop(columns=["Contract"]))


def test_model_beats_random_guessing(clean_df):
    _, metrics = train.fit_and_evaluate(clean_df)
    assert metrics["test"]["roc_auc"] > 0.6
    assert metrics["best_model"] in train.candidate_models()


def test_api_predict(tmp_path, monkeypatch, clean_df):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import joblib
    from fastapi.testclient import TestClient

    model, _ = train.fit_and_evaluate(clean_df)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))

    from src.api import Customer, app

    payload = Customer.model_config["json_schema_extra"]["example"]
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert 0 <= body["churn_probability"] <= 1
        assert body["risk_level"] in {"low", "medium", "high"}
        bad = client.post("/predict", json={**payload, "tenure": -1})
        assert bad.status_code == 422
