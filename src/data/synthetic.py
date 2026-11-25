from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_telco_csv(
    output_path: str = "data/churn.csv",
    n_rows: int = 600,
    seed: int = 42,
) -> Path:
    """Generate a deterministic synthetic Telco Churn CSV when the IBM dataset is absent (e.g., CI)."""
    rng = np.random.default_rng(seed)

    contracts = np.array(["Month-to-month", "One year", "Two year"])
    internet_services = np.array(["Fiber optic", "DSL", "No"])
    payment_methods = np.array(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    contract = rng.choice(contracts, size=n_rows, p=[0.55, 0.22, 0.23])
    internet = rng.choice(internet_services, size=n_rows, p=[0.45, 0.35, 0.20])
    payment = rng.choice(payment_methods, size=n_rows, p=[0.35, 0.22, 0.22, 0.21])

    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(0, 36, size=n_rows),
        rng.integers(12, 72, size=n_rows),
    )
    monthly_charges = np.where(
        internet == "Fiber optic",
        rng.uniform(70.0, 115.0, size=n_rows),
        np.where(
            internet == "DSL",
            rng.uniform(35.0, 75.0, size=n_rows),
            rng.uniform(18.0, 30.0, size=n_rows),
        ),
    ).round(2)
    total_charges = (tenure * monthly_charges).round(2)

    # Compute realistic churn log-odds so Low, Medium, and High risk tiers are well-populated
    logit = (
        -0.5
        + (contract == "Month-to-month") * 1.6
        - (contract == "Two year") * 1.8
        + (internet == "Fiber optic") * 1.1
        - (internet == "No") * 1.0
        + (payment == "Electronic check") * 0.7
        - (tenure / 24.0) * 1.2
        + ((monthly_charges - 65.0) / 30.0) * 0.4
    )
    probs = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.uniform(0.0, 1.0, size=n_rows) < probs, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"SYNTH-{i:04d}" for i in range(n_rows)],
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "Churn": churn,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    path = generate_synthetic_telco_csv()
    print(f"Generated synthetic dataset at {path}")
