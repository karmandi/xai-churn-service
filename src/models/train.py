import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load and clean
df = pd.read_csv("data/churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = ["Contract", "InternetService", "PaymentMethod"]
X = df[num_cols + cat_cols]
y = (df["Churn"] == "Yes").astype(int)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Build preprocessing + tree pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
])

# 3. Fit and save artifact
model.fit(X_train, y_train)
joblib.dump(model, "artifacts/model.joblib")
print("Saved model to artifacts/model.joblib")
