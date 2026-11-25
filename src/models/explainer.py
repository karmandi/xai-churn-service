import joblib
import pandas as pd
import shap

class ChurnExplainer:
    def __init__(self, model_path: str = "artifacts/model.joblib"):
        self.pipeline = joblib.load(model_path)
        self.preprocessor = self.pipeline.named_steps["prep"]
        self.classifier = self.pipeline.named_steps["clf"]
        self.explainer = shap.TreeExplainer(self.classifier)
        self.feature_names = self.preprocessor.get_feature_names_out()

    def predict_and_explain(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])
        prob = float(self.pipeline.predict_proba(df)[0][1])

        # Transform input and extract SHAP values for class 1 (Churn = Yes)
        X_transformed = self.preprocessor.transform(df)
        shap_vals = self.explainer.shap_values(X_transformed)
        
        # Handle binary classification output format across SHAP versions
        class_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0, :, 1]

        # Pair feature names with SHAP impacts and pick top 3 by absolute magnitude
        attributions = dict(zip(self.feature_names, [round(float(v), 4) for v in class_shap]))
        top_3 = dict(sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3])

        risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
        return {
            "churn_probability": round(prob, 4),
            "risk_category": risk,
            "top_attributions": top_3,
        }
