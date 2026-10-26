"""Download the public Telco Customer Churn dataset (IBM sample data)."""
import urllib.request

from src import config

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, config.RAW_CSV)
    print(f"Saved to {config.RAW_CSV}")


if __name__ == "__main__":
    main()
