import json
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.explainer import ChurnExplainer

FEATURE_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]

SYSTEM_PROMPT = (
    "You are a telco retention specialist. Given a customer's churn prediction "
    "and top 3 SHAP feature attributions, write a concise 2-sentence retention "
    "strategy grounded strictly in those SHAP drivers."
)


def _format_driver(feature_key: str, shap_val: float) -> str:
    clean = feature_key.replace("num__", "").replace("cat__", "").replace("_", " ")
    direction = "increases" if shap_val >= 0 else "reduces"
    return f"{clean} ({shap_val:+.4f}, {direction} risk)"


def _build_strategy(prediction: dict[str, Any]) -> str:
    prob = prediction["churn_probability"]
    risk = prediction["risk_category"]
    top_attr = prediction["top_attributions"]
    drivers = list(top_attr.items())

    d1 = _format_driver(drivers[0][0], drivers[0][1])
    d2 = _format_driver(drivers[1][0], drivers[1][1]) if len(drivers) > 1 else ""
    driver_phrase = f"{d1} and {d2}" if d2 else d1

    if risk == "High":
        action = (
            "Offer a 10% loyalty discount paired with a 1-year contract upgrade "
            "and automated billing enrollment to mitigate immediate churn."
        )
    elif risk == "Medium":
        action = (
            "Proactively send a personalized plan-optimization review and highlight "
            "annual-billing savings before the next billing cycle."
        )
    else:
        action = (
            "Maintain standard engagement and enroll the customer in periodic "
            "loyalty appreciation check-ins without unnecessary discounting."
        )

    return (
        f"Churn risk is {risk} ({prob:.1%}), driven primarily by {driver_phrase}. "
        f"{action}"
    )


def generate_sft_dataset(
    csv_path: str = "data/churn.csv",
    output_path: str = "data/sft_shap_pairs.jsonl",
    n_samples: int = 450,
) -> list[dict[str, Any]]:
    """Generate chat-style instruction JSONL pairs from stratified training rows."""
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    X = df[FEATURE_COLS]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sample_count = min(n_samples, len(X_train))
    X_sample, _, _, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_count,
        stratify=y_train,
        random_state=42,
    )

    explainer = ChurnExplainer()
    records: list[dict[str, Any]] = []

    for _, row in X_sample.iterrows():
        payload = {
            "tenure": int(row["tenure"]),
            "MonthlyCharges": float(row["MonthlyCharges"]),
            "TotalCharges": float(row["TotalCharges"]),
            "Contract": str(row["Contract"]),
            "InternetService": str(row["InternetService"]),
            "PaymentMethod": str(row["PaymentMethod"]),
        }
        prediction = explainer.predict_and_explain(payload)
        user_content = json.dumps(
            {
                "churn_probability": prediction["churn_probability"],
                "risk_category": prediction["risk_category"],
                "top_attributions": prediction["top_attributions"],
            }
        )
        assistant_content = _build_strategy(prediction)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return records


if __name__ == "__main__":
    generated = generate_sft_dataset()
    print(f"Generated {len(generated)} SFT pairs in data/sft_shap_pairs.jsonl")
