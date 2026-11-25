from fastapi import FastAPI
from src.agent.harness import RemediationHarness
from src.agent.tools import ToolExecutor
from src.api.schemas import (
    CustomerPayload,
    PredictionResponse,
    RemediationRequest,
    RemediationResponse,
)
from src.models.explainer import ChurnExplainer

app = FastAPI(title="Churn XAI Prediction Service")
explainer = ChurnExplainer()
tool_executor = ToolExecutor(explainer=explainer)
remediation_harness = RemediationHarness(executor=tool_executor)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict-explain", response_model=PredictionResponse)
def predict_explain(payload: CustomerPayload) -> PredictionResponse:
    result = explainer.predict_and_explain(payload.model_dump())
    return PredictionResponse(**result)


@app.post("/remediate", response_model=RemediationResponse)
def remediate(request: RemediationRequest) -> RemediationResponse:
    payload_dict = request.customer.model_dump()
    prediction = explainer.predict_and_explain(payload_dict)

    if prediction["risk_category"] != "High":
        return RemediationResponse(
            customer_id=request.customer_id,
            prediction=PredictionResponse(**prediction),
            remediation_triggered=False,
            trace=None,
        )

    trace = remediation_harness.run(request.customer_id, payload_dict)
    return RemediationResponse(
        customer_id=request.customer_id,
        prediction=PredictionResponse(**prediction),
        remediation_triggered=True,
        trace=trace,
    )
