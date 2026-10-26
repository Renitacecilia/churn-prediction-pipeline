"""Train, compare and save a churn model."""
import json
import logging
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

log = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC), ("cat", categorical, config.CATEGORICAL)]
    )


def candidate_models() -> dict:
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=rs
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=rs),
    }


def fit_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Compare candidate models with cross-validation, refit the best, score on a held-out test set."""
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = {"mean_roc_auc": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        log.info("%-22s CV ROC-AUC = %.4f (+/- %.4f)", name, scores.mean(), scores.std())

    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean_roc_auc"])
    best = Pipeline([("prep", build_preprocessor()), ("model", candidate_models()[best_name])])
    best.fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    importance = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=config.RANDOM_STATE
    )
    top = sorted(zip(config.FEATURES, importance.importances_mean), key=lambda t: -t[1])[:10]

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "cv": cv_scores,
        "test": {
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "avg_precision": round(average_precision_score(y_test, proba), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        },
        "top_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in top],
    }
    return best, metrics


def run() -> dict:
    with sqlite3.connect(config.DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM customers", conn)
    model, metrics = fit_and_evaluate(df)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Best model: %s | test ROC-AUC: %s", metrics["best_model"], metrics["test"]["roc_auc"])
    log.info("Saved model to %s and metrics to %s", config.MODEL_PATH, config.METRICS_PATH)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
