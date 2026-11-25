from pydantic import BaseModel, Field
from src.agent.tracing import AgentTrace


class CustomerPayload(BaseModel):
    tenure: int = Field(..., ge=0, examples=[1])
    MonthlyCharges: float = Field(..., gt=0, examples=[95.0])
    TotalCharges: float = Field(..., ge=0, examples=[95.0])
    Contract: str = Field(..., examples=["Month-to-month"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., examples=[0.7421])
    risk_category: str = Field(..., examples=["High"])
    top_attributions: dict[str, float] = Field(
        ...,
        examples=[
            {
                "cat__Contract_Month-to-month": 0.1823,
                "num__tenure": 0.1412,
                "cat__InternetService_Fiber optic": 0.0954,
            }
        ],
    )


class RemediationRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, examples=["CUST-1001"])
    customer: CustomerPayload


class RemediationResponse(BaseModel):
    customer_id: str
    prediction: PredictionResponse
    remediation_triggered: bool
    trace: AgentTrace | None = None
