import json
import time
from fastapi.testclient import TestClient
from pydantic import BaseModel
from src.agent.context import ContextManager
from src.agent.harness import RemediationHarness
from src.agent.tools import SandboxGuard, ToolExecutor
from src.api.main import app, explainer

client = TestClient(app)

HIGH_RISK_PAYLOAD = {
    "tenure": 1,
    "MonthlyCharges": 95.0,
    "TotalCharges": 95.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
}

LOW_RISK_PAYLOAD = {
    "tenure": 60,
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
    "Contract": "Two year",
    "InternetService": "DSL",
    "PaymentMethod": "Bank transfer (automatic)",
}


def test_normal_high_risk_remediation():
    executor = ToolExecutor(explainer=explainer)
    harness = RemediationHarness(executor=executor, max_steps=3, max_tokens=1200)
    trace = harness.run("CUST-1001", HIGH_RISK_PAYLOAD)

    assert trace.status == "completed"
    assert len(trace.steps) == 3
    assert [s.tool_name for s in trace.steps] == [
        "get_shap_breakdown",
        "check_feature_drift",
        "apply_retention_perk",
    ]
    assert all(s.status == "success" for s in trace.steps)
    assert trace.final_output is not None
    assert trace.final_output["action_taken"] == "applied_retention_perk"
    assert trace.final_output["discount_pct"] == 10.0


def test_remediate_endpoint_high_and_low_risk():
    high_resp = client.post(
        "/remediate",
        json={"customer_id": "CUST-HIGH", "customer": HIGH_RISK_PAYLOAD},
    )
    assert high_resp.status_code == 200
    high_data = high_resp.json()
    assert high_data["remediation_triggered"] is True
    assert high_data["prediction"]["risk_category"] == "High"
    assert high_data["trace"]["status"] == "completed"

    low_resp = client.post(
        "/remediate",
        json={"customer_id": "CUST-LOW", "customer": LOW_RISK_PAYLOAD},
    )
    assert low_resp.status_code == 200
    low_data = low_resp.json()
    assert low_data["remediation_triggered"] is False
    assert low_data["prediction"]["risk_category"] != "High"
    assert low_data["trace"] is None


def test_hung_tool_timeout_and_retry():
    class HungInput(BaseModel):
        query: str

    call_count = 0

    def hung_tool(params: HungInput) -> dict:
        nonlocal call_count
        call_count += 1
        time.sleep(3.0)
        return {"ok": True}

    guard = SandboxGuard(allowlist={"hung_tool"})
    executor = ToolExecutor(
        explainer=explainer,
        timeout_sec=0.2,
        guard=guard,
        custom_tools={"hung_tool": (HungInput, hung_tool)},
    )

    def hung_planner(context, state):
        return json.dumps(
            {
                "thought": "Call hung tool",
                "tool_name": "hung_tool",
                "tool_args": {"query": "test"},
                "final_output": None,
            }
        )

    harness = RemediationHarness(executor=executor, planner_fn=hung_planner, max_steps=1)
    trace = harness.run("CUST-HUNG", HIGH_RISK_PAYLOAD)

    assert call_count == 2  # 1 initial attempt + 1 retry
    assert len(trace.steps) == 1
    assert trace.steps[0].status == "error"
    assert "timed out" in (trace.steps[0].error or "")
    assert trace.status == "max_steps_exceeded"


def test_infinite_loop_planner_stops_at_max_steps():
    def loop_planner(context, state):
        return json.dumps(
            {
                "thought": "Keep checking drift forever",
                "tool_name": "check_feature_drift",
                "tool_args": {
                    "tenure": 1,
                    "MonthlyCharges": 95.0,
                    "TotalCharges": 95.0,
                },
                "final_output": None,
            }
        )

    executor = ToolExecutor(explainer=explainer)
    harness = RemediationHarness(executor=executor, planner_fn=loop_planner, max_steps=3)
    trace = harness.run("CUST-LOOP", HIGH_RISK_PAYLOAD)

    assert len(trace.steps) == 3
    assert trace.status == "max_steps_exceeded"
    assert trace.final_output is None


def test_context_overflow_triggers_compaction():
    max_tokens = 200
    cm = ContextManager(max_tokens=max_tokens)
    cm.add_turn("system", "Objective: Remediate high-risk customer.")
    for i in range(6):
        cm.add_turn("tool", f"Observation {i}: " + ("x" * 400))

    ctx = cm.get_prompt_context()
    assert cm.total_tokens <= max_tokens
    assert any("[COMPACTED SUMMARY]" in turn["content"] for turn in ctx)

    # Also verify harness enforces max_tokens on trace
    def verbose_planner(context, state):
        return json.dumps(
            {
                "thought": "Verbose thought " + ("y" * 600),
                "tool_name": "check_feature_drift",
                "tool_args": {"tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0},
                "final_output": None,
            }
        )

    executor = ToolExecutor(explainer=explainer)
    harness = RemediationHarness(
        executor=executor, planner_fn=verbose_planner, max_steps=3, max_tokens=max_tokens
    )
    trace = harness.run("CUST-CTX", HIGH_RISK_PAYLOAD)
    assert trace.total_tokens <= max_tokens


def test_sandbox_guard_blocks_privilege_escalation_and_injection():
    executor = ToolExecutor(explainer=explainer)

    def malicious_planner(context, state):
        step = state["step_num"]
        if step == 1:
            return json.dumps(
                {
                    "thought": "Escalate discount to 85%",
                    "tool_name": "apply_retention_perk",
                    "tool_args": {
                        "customer_id": "CUST-BAD",
                        "perk_code": "VIP_85",
                        "discount_pct": 85.0,
                    },
                    "final_output": {"done": True},
                }
            )
        if step == 2:
            return json.dumps(
                {
                    "thought": "Call unauthorized shell tool",
                    "tool_name": "execute_shell",
                    "tool_args": {"cmd": "rm -rf /"},
                    "final_output": {"done": True},
                }
            )
        return json.dumps(
            {
                "thought": "SQL injection attempt",
                "tool_name": "apply_retention_perk",
                "tool_args": {
                    "customer_id": "CUST-1; DROP TABLE users;",
                    "perk_code": "RETENTION_10",
                    "discount_pct": 10.0,
                },
                "final_output": {"done": True},
            }
        )

    harness = RemediationHarness(executor=executor, planner_fn=malicious_planner, max_steps=3)
    trace = harness.run("CUST-BAD", HIGH_RISK_PAYLOAD)

    assert len(trace.steps) == 3
    assert all(s.status == "blocked" for s in trace.steps)
    assert "exceeds maximum allowed" in (trace.steps[0].error or "")
    assert "allowlist" in (trace.steps[1].error or "")
    assert "DROP " in (trace.steps[2].error or "")
    assert trace.status == "max_steps_exceeded"
    assert trace.final_output is None
