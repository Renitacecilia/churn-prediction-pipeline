"""Business analysis in SQL. Results are printed and saved to reports/*.csv."""
import sqlite3

import pandas as pd

from src import config

QUERIES = {
    "01_overall": """
        SELECT COUNT(*)                                   AS customers,
               ROUND(100.0 * AVG(Churn), 2)               AS churn_rate_pct,
               ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2)
                                                          AS monthly_revenue_lost
        FROM customers
    """,
    "02_churn_by_contract": """
        SELECT Contract,
               COUNT(*)                         AS customers,
               ROUND(100.0 * AVG(Churn), 2)     AS churn_rate_pct,
               ROUND(AVG(MonthlyCharges), 2)    AS avg_monthly_charge
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """,
    "03_churn_by_tenure_bucket": """
        SELECT CASE WHEN tenure < 12 THEN '0-11 months'
                    WHEN tenure < 24 THEN '12-23 months'
                    WHEN tenure < 48 THEN '24-47 months'
                    ELSE '48+ months' END        AS tenure_bucket,
               COUNT(*)                          AS customers,
               ROUND(100.0 * AVG(Churn), 2)      AS churn_rate_pct
        FROM customers
        GROUP BY tenure_bucket
        ORDER BY MIN(tenure)
    """,
    "04_payment_method_ranking": """
        WITH rates AS (
            SELECT PaymentMethod,
                   COUNT(*)                     AS customers,
                   100.0 * AVG(Churn)           AS churn_rate_pct
            FROM customers
            GROUP BY PaymentMethod
        )
        SELECT PaymentMethod, customers,
               ROUND(churn_rate_pct, 2)                          AS churn_rate_pct,
               RANK() OVER (ORDER BY churn_rate_pct DESC)        AS risk_rank
        FROM rates
    """,
    "05_top_revenue_at_risk": """
        SELECT customerID, Contract, tenure, MonthlyCharges
        FROM customers
        WHERE Churn = 0 AND Contract = 'Month-to-month' AND tenure < 12
        ORDER BY MonthlyCharges DESC
        LIMIT 20
    """,
}


def run() -> dict[str, pd.DataFrame]:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    with sqlite3.connect(config.DB_PATH) as conn:
        for name, sql in QUERIES.items():
            df = pd.read_sql_query(sql, conn)
            df.to_csv(config.REPORTS_DIR / f"{name}.csv", index=False)
            results[name] = df
            print(f"\n=== {name} ===\n{df.to_string(index=False)}")
    return results


if __name__ == "__main__":
    run()
