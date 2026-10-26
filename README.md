# Customer Churn Prediction Pipeline

An end-to-end data project: raw CSV → cleaned SQLite database → SQL business analysis → model comparison → REST API → Docker container, with automated tests and CI.

**Business question:** Which customers are likely to cancel, and how much recurring revenue is at risk?

## Architecture

```
raw CSV ──► ETL (pandas) ──► SQLite ──┬─► SQL analysis ──► reports/*.csv
                                      └─► Model training ──► models/churn_model.joblib
                                                                   │
                                                  FastAPI /predict ◄┘ ──► Docker
```

## Project structure

```
src/
  config.py           paths, column lists, constants
  download_data.py    fetch the public Telco churn dataset
  make_sample_data.py synthetic data with the same schema (offline dev + tests)
  etl.py              extract / transform / validate / load to SQLite
  analysis.py         SQL queries: GROUP BY, CASE, CTE, window functions
  train.py            sklearn pipelines, cross-validation, model comparison, metrics
  api.py              FastAPI service with input validation
tests/test_pipeline.py  unit + API tests (pytest)
.github/workflows/ci.yml  runs tests on every push
Dockerfile
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.download_data      # real dataset (or: python -m src.make_sample_data)
python -m src.etl                # clean + load into data/churn.db
python -m src.analysis           # SQL insights -> reports/
python -m src.train              # trains 3 models, saves best + metrics.json
pytest -q                        # run tests

uvicorn src.api:app --reload     # API docs at http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "SeniorCitizen": 0, "tenure": 5, "MonthlyCharges": 85.5, "TotalCharges": 420.0,
  "gender": "Female", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
  "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
  "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}'
```

Docker (after training, so `models/` contains the model):

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Results

> Run on the real dataset, then replace this section with your actual numbers from `models/metrics.json`.

| Model | CV ROC-AUC |
|---|---|
| Logistic regression | _fill in_ |
| Random forest | _fill in_ |
| Gradient boosting | _fill in_ |

Key findings (from `reports/`): _write 3 insights, e.g. which contract type churns most, how churn changes with tenure, how much monthly revenue the churned customers represent._

## Design decisions worth explaining in interviews

- **Preprocessing lives inside the sklearn `Pipeline`**, so imputation and scaling are fit only on training folds. This prevents data leakage.
- **ROC-AUC for model selection**, because churn is imbalanced and accuracy would be misleading. Precision, recall, and F1 are reported on a held-out test set.
- **`class_weight="balanced"`** to handle the imbalance without resampling.
- **`transform()` is a pure function**, so it is easy to unit test. Validation fails loudly on bad data.
- **Pydantic validates API input**, so bad requests return a 422 instead of a crash.
- **Permutation importance** on the test set explains which features drive predictions.

## Ideas to extend it (do at least one)

1. Tune hyperparameters with `RandomizedSearchCV` and report the gain.
2. Choose the decision threshold using a cost assumption (cost of a retention offer vs. revenue lost) instead of 0.5.
3. Add SHAP explanations to the API response.
4. Swap SQLite for PostgreSQL running in Docker Compose.
5. Schedule `etl` + `train` with Airflow or Prefect.
6. Build a Power BI / Tableau / Streamlit dashboard on top of `data/churn.db`.
7. Track experiments with MLflow.
