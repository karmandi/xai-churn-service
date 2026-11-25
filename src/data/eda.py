from pathlib import Path
import sys
import pandas as pd


DEFAULT_DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
KEY_SEGMENTS = ["Contract", "InternetService", "PaymentMethod"]


def load_telco_data(filepath: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset and coerce TotalCharges to numeric."""
    df = pd.read_csv(filepath)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def run_eda(filepath: str | Path = DEFAULT_DATA_PATH) -> None:
    """Run exploratory data analysis on the Telco Customer Churn dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = load_telco_data(path)
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    print(f"=== Dataset Overview ({df.shape[0]} rows, {df.shape[1] - 1} columns) ===")

    # 1. Missing values (including blank strings coerced from TotalCharges)
    missing = df.drop(columns=["Churn_Binary"]).isna().sum()
    missing = missing[missing > 0]
    print("\n--- Missing Values ---")
    if missing.empty:
        print("No missing values detected.")
    else:
        for col, count in missing.items():
            print(f"{col}: {count} ({count / len(df):.2%})")

    # 2. Target class balance
    print("\n--- Target Class Balance (Churn) ---")
    counts = df["Churn"].value_counts()
    props = df["Churn"].value_counts(normalize=True)
    balance_df = pd.DataFrame({"Count": counts, "Proportion": props.map("{:.2%}".format)})
    print(balance_df.to_string())

    # 3. Churn rates across key categorical segments
    print("\n--- Churn Rates by Segment ---")
    for col in KEY_SEGMENTS:
        if col in df.columns:
            summary = (
                df.groupby(col, observed=False)["Churn_Binary"]
                .agg(Customers="count", Churn_Rate="mean")
                .sort_values(by="Churn_Rate", ascending=False)
            )
            summary["Churn_Rate"] = summary["Churn_Rate"].map("{:.2%}".format)
            print(f"\n[{col}]")
            print(summary.to_string())


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    run_eda(data_path)
