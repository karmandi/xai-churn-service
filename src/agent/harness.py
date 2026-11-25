import json
import time
import uuid
from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError
from src.agent.context import ContextManager, estimate_tokens
from src.agent.tools import ToolExecutor
from src.agent.tracing import AgentTrace, TraceStep


class AgentAction(BaseModel):
    thought: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    final_output: dict[str, Any] | str | None = None


def rule_based_planner(
    context: list[dict[str, str]],
    state: dict[str, Any],
) -> str:
    """Deterministic offline planner for the 3-step remediation workflow."""
    step_num: int = state["step_num"]
    customer_id: str = state["customer_id"]
    payload: dict[str, Any] = state["customer_payload"]
    observations: list[dict[str, Any]] = state["observations"]

    if step_num == 1:
        return json.dumps(
            {
                "thought": "Step 1: Retrieve SHAP feature attributions for high-risk customer.",
                "tool_name": "get_shap_breakdown",
                "tool_args": payload,
                "final_output": None,
            }
        )

    if step_num == 2:
        return json.dumps(
            {
                "thought": "Step 2: Check numeric features for training-bound drift.",
                "tool_name": "check_feature_drift",
                "tool_args": {
                    "tenure": payload["tenure"],
                    "MonthlyCharges": payload["MonthlyCharges"],
                    "TotalCharges": payload["TotalCharges"],
                },
                "final_output": None,
            }
        )

    # Step 3: Decide retention perk based on drift observation
    drift_obs = next(
        (obs for obs in observations if obs.get("tool_name") == "check_feature_drift"),
        {},
    )
    drift_result = drift_obs.get("result", {})
    has_severe_drift = drift_result.get("severity") == "severe"

    if has_severe_drift:
        return json.dumps(
            {
                "thought": "Severe feature drift detected; skipping automated perk.",
                "tool_name": None,
                "tool_args": {},
                "final_output": {
                    "customer_id": customer_id,
                    "action_taken": "manual_review_required",
                    "reason": "Severe feature drift detected",
                    "drift_violations": drift_result.get("violations", []),
                },
            }
        )

    shap_obs = next(
        (obs for obs in observations if obs.get("tool_name") == "get_shap_breakdown"),
        {},
    )
    top_drivers = shap_obs.get("result", {}).get("top_attributions", {})

    return json.dumps(
        {
            "thought": "No severe drift detected; applying 10% retention perk.",
            "tool_name": "apply_retention_perk",
            "tool_args": {
                "customer_id": customer_id,
                "perk_code": "RETENTION_10",
                "discount_pct": 10.0,
            },
            "final_output": {
                "customer_id": customer_id,
                "action_taken": "applied_retention_perk",
                "perk_code": "RETENTION_10",
                "discount_pct": 10.0,
                "top_drivers": top_drivers,
            },
        }
    )


class RemediationHarness:
    """Bounded controller loop for autonomous high-risk churn remediation."""

    def __init__(
        self,
        executor: ToolExecutor,
        planner_fn: Callable[[list[dict[str, str]], dict[str, Any]], str | dict[str, Any]] = rule_based_planner,
        max_steps: int = 3,
        max_tokens: int = 1200,
    ) -> None:
        self.executor = executor
        self.planner_fn = planner_fn
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    def run(self, customer_id: str, customer_payload: dict[str, Any]) -> AgentTrace:
        run_start = time.perf_counter()
        trace_id = str(uuid.uuid4())
        context = ContextManager(max_tokens=self.max_tokens)
        context.add_turn(
            "system",
            f"Objective: Investigate churn drivers and safely remediate customer {customer_id}.",
        )

        steps: list[TraceStep] = []
        observations: list[dict[str, Any]] = []
        final_output: dict[str, Any] | str | None = None
        trace_status = "max_steps_exceeded"

        for step_num in range(1, self.max_steps + 1):
            step_start = time.perf_counter()
            prompt_context = context.get_prompt_context()
            state = {
                "step_num": step_num,
                "customer_id": customer_id,
                "customer_payload": customer_payload,
                "observations": observations,
            }

            raw_plan = self.planner_fn(prompt_context, state)
            raw_plan_str = raw_plan if isinstance(raw_plan, str) else json.dumps(raw_plan)

            try:
                parsed_json = json.loads(raw_plan_str) if isinstance(raw_plan, str) else raw_plan
                action = AgentAction.model_validate(parsed_json)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                step_latency = (time.perf_counter() - step_start) * 1000.0
                context.add_turn("assistant", raw_plan_str)
                context.add_turn("tool", f"Planner parse error: {exc}")
                steps.append(
                    TraceStep(
                        step_num=step_num,
                        action="parse_error",
                        tool_name=None,
                        latency_ms=round(step_latency, 2),
                        tokens_used=estimate_tokens(raw_plan_str),
                        status="error",
                        error=str(exc),
                    )
                )
                continue

            context.add_turn("assistant", raw_plan_str)

            tool_status = "success"
            tool_error: str | None = None
            step_action_label = "final_response"

            if action.tool_name is not None:
                step_action_label = "tool_call"
                tool_result = self.executor.execute(action.tool_name, action.tool_args)
                observations.append(tool_result)
                obs_str = json.dumps(tool_result)
                context.add_turn("tool", obs_str)

                tool_status = tool_result.get("status", "error")
                tool_error = tool_result.get("error")
                tokens_used = estimate_tokens(raw_plan_str) + estimate_tokens(obs_str)
            else:
                tokens_used = estimate_tokens(raw_plan_str)

            step_latency = (time.perf_counter() - step_start) * 1000.0
            steps.append(
                TraceStep(
                    step_num=step_num,
                    action=step_action_label,
                    tool_name=action.tool_name,
                    latency_ms=round(step_latency, 2),
                    tokens_used=tokens_used,
                    status=tool_status,
                    error=tool_error,
                )
            )

            if action.final_output is not None and tool_status == "success":
                final_output = action.final_output
                trace_status = "completed"
                break

        total_latency = (time.perf_counter() - run_start) * 1000.0
        return AgentTrace(
            trace_id=trace_id,
            total_latency_ms=round(total_latency, 2),
            total_tokens=context.total_tokens,
            status=trace_status,
            steps=steps,
            final_output=final_output,
        )
