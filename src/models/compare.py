import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def compare_models(csv_path: str = "data/churn.csv") -> None:
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract", "InternetService", "PaymentMethod"]
    X = df[num_cols + cat_cols]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression())])

    param_grid = [
        {
            "clf": [LogisticRegression(max_iter=1000, random_state=42)],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf": [RandomForestClassifier(random_state=42)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [4, 6, 8],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf": [HistGradientBoostingClassifier(random_state=42)],
            "clf__max_iter": [100, 200],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"},
        refit="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    results["model"] = results["param_clf"].apply(lambda m: m.__class__.__name__)
    best_per_model = (
        results.sort_values("mean_test_roc_auc", ascending=False)
        .groupby("model", as_index=False)
        .first()[["model", "mean_test_roc_auc", "mean_test_f1", "mean_test_recall", "mean_test_precision", "params"]]
        .sort_values("mean_test_roc_auc", ascending=False)
    )

    print("\n=== Stratified 5-Fold CV Comparison (Best per Model) ===")
    print(best_per_model.drop(columns=["params"]).to_string(index=False, float_format="{:.4f}".format))
    print(f"\nBest Overall Estimator: {search.best_params_['clf'].__class__.__name__}")
    print(f"Best Params: { {k: v for k, v in search.best_params_.items() if k != 'clf'} }")

    probs = search.best_estimator_.predict_proba(X_test)[:, 1]
    print(f"\nHoldout ROC-AUC: {roc_auc_score(y_test, probs):.4f}\n")
    print(classification_report(y_test, (probs >= 0.5).astype(int), target_names=["Retained", "Churned"]))


if __name__ == "__main__":
    compare_models()
