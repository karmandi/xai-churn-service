import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from src.models.explainer import ChurnExplainer


def evaluate() -> None:
    df = pd.read_csv("data/churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]
    X = df[features]
    y = (df["Churn"] == "Yes").astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    explainer = ChurnExplainer()

    # 1. Holdout Metrics
    probs = explainer.pipeline.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}\n")
    print(classification_report(y_test, (probs >= 0.5).astype(int), target_names=["Retained", "Churned"]))

    # 2. Sample SHAP Predictions
    samples = [
        {"tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0, "Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"},
        {"tenure": 60, "MonthlyCharges": 55.0, "TotalCharges": 3300.0, "Contract": "Two year", "InternetService": "DSL", "PaymentMethod": "Bank transfer (automatic)"},
    ]
    for s in samples:
        print(explainer.predict_and_explain(s))


if __name__ == "__main__":
    evaluate()
