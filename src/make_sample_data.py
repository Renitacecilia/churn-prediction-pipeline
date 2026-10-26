"""Generate a synthetic dataset with the same schema as the Telco churn data.

Useful for offline development and for fast, deterministic tests.
"""
import numpy as np
import pandas as pd

from src import config


def generate(n: int = 3000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    base = pd.Series(internet).map({"DSL": 50.0, "Fiber optic": 80.0, "No": 21.0}).to_numpy()
    monthly = np.clip(base + rng.normal(0, 8, n), 18.25, None).round(2)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])

    def internet_dependent() -> np.ndarray:
        vals = rng.choice(["Yes", "No"], n)
        return np.where(internet == "No", "No internet service", vals)

    tech_support = internet_dependent()

    logit = (
        -1.0
        + 1.2 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        - 0.03 * tenure
        + 0.015 * (monthly - 65)
        + 0.6 * (internet == "Fiber optic")
        + 0.4 * (payment == "Electronic check")
        + 0.3 * senior
        - 0.4 * (tech_support == "Yes")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    total = (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2).astype(str)
    total = np.where(tenure == 0, " ", total)  # the real dataset has blank strings here

    return pd.DataFrame(
        {
            "customerID": [f"{i:05d}-SYNTH" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": senior,
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": internet_dependent(),
            "OnlineBackup": internet_dependent(),
            "DeviceProtection": internet_dependent(),
            "TechSupport": tech_support,
            "StreamingTV": internet_dependent(),
            "StreamingMovies": internet_dependent(),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


def main() -> None:
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(config.RAW_CSV, index=False)
    print(f"Synthetic data written to {config.RAW_CSV}")


if __name__ == "__main__":
    main()
