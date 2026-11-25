from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError
from src.models.explainer import ChurnExplainer


class SandboxViolationError(ValueError):
    """Raised when a tool call violates SandboxGuard policy."""


class GetShapBreakdownInput(BaseModel):
    tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    Contract: str
    InternetService: str
    PaymentMethod: str


class CheckFeatureDriftInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


class ApplyRetentionPerkInput(BaseModel):
    customer_id: str = Field(..., min_length=1)
    perk_code: str = Field(..., min_length=1)
    discount_pct: float = Field(..., gt=0, le=100)


class SandboxGuard:
    """Enforces tool allowlist, parameter caps, and prompt-injection filtering."""

    DEFAULT_ALLOWLIST = frozenset(
        {"get_shap_breakdown", "check_feature_drift", "apply_retention_perk"}
    )
    SUSPICIOUS_TOKENS = ("DROP ", "rm -rf", "__import__", "IGNORE PREVIOUS")

    def __init__(
        self,
        allowlist: frozenset[str] | set[str] | None = None,
        max_discount_pct: float = 20.0,
    ) -> None:
        self.allowlist = frozenset(allowlist) if allowlist is not None else self.DEFAULT_ALLOWLIST
        self.max_discount_pct = max_discount_pct

    def _scan_strings(self, value: Any) -> str | None:
        if isinstance(value, str):
            upper_val = value.upper()
            for token in self.SUSPICIOUS_TOKENS:
                if token.upper() in upper_val or token in value:
                    return token
        elif isinstance(value, dict):
            for v in value.values():
                found = self._scan_strings(v)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                found = self._scan_strings(item)
                if found is not None:
                    return found
        return None

    def validate(self, tool_name: str, args: dict[str, Any]) -> None:
        if tool_name not in self.allowlist:
            raise SandboxViolationError(
                f"Tool '{tool_name}' is not in the SandboxGuard allowlist."
            )

        if tool_name == "apply_retention_perk":
            discount = args.get("discount_pct")
            if isinstance(discount, (int, float)) and discount > self.max_discount_pct:
                raise SandboxViolationError(
                    f"discount_pct={discount} exceeds maximum allowed {self.max_discount_pct}%."
                )

        bad_token = self._scan_strings(args)
        if bad_token is not None:
            raise SandboxViolationError(
                f"Suspicious injection token '{bad_token}' detected in arguments."
            )


class ToolExecutor:
    """Validates, sandboxes, and executes tools with timeout and 1-retry resilience."""

    def __init__(
        self,
        explainer: ChurnExplainer | None = None,
        timeout_sec: float = 2.0,
        guard: SandboxGuard | None = None,
        custom_tools: dict[str, tuple[type[BaseModel], Callable[..., dict[str, Any]]]] | None = None,
    ) -> None:
        self._explainer = explainer
        self.timeout_sec = timeout_sec
        self.guard = guard or SandboxGuard()
        self.registry: dict[str, tuple[type[BaseModel], Callable[..., dict[str, Any]]]] = {
            "get_shap_breakdown": (GetShapBreakdownInput, self._get_shap_breakdown),
            "check_feature_drift": (CheckFeatureDriftInput, self._check_feature_drift),
            "apply_retention_perk": (ApplyRetentionPerkInput, self._apply_retention_perk),
        }
        if custom_tools:
            self.registry.update(custom_tools)

    @property
    def explainer(self) -> ChurnExplainer:
        if self._explainer is None:
            self._explainer = ChurnExplainer()
        return self._explainer

    def _get_shap_breakdown(self, params: GetShapBreakdownInput) -> dict[str, Any]:
        return self.explainer.predict_and_explain(params.model_dump())

    def _check_feature_drift(self, params: CheckFeatureDriftInput) -> dict[str, Any]:
        out_of_bounds: list[str] = []
        if params.tenure < 0 or params.tenure > 72:
            out_of_bounds.append(f"tenure={params.tenure} outside [0, 72]")
        if params.MonthlyCharges <= 0 or params.MonthlyCharges > 120:
            out_of_bounds.append(f"MonthlyCharges={params.MonthlyCharges} outside (0, 120]")
        if params.TotalCharges < 0 or params.TotalCharges > 9000:
            out_of_bounds.append(f"TotalCharges={params.TotalCharges} outside [0, 9000]")

        return {
            "has_drift": len(out_of_bounds) > 0,
            "severity": "severe" if len(out_of_bounds) > 0 else "none",
            "violations": out_of_bounds,
        }

    def _apply_retention_perk(self, params: ApplyRetentionPerkInput) -> dict[str, Any]:
        return {
            "applied": True,
            "customer_id": params.customer_id,
            "perk_code": params.perk_code,
            "discount_pct": params.discount_pct,
        }

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        # 1. SandboxGuard check (no retries on policy violations)
        try:
            self.guard.validate(tool_name, args)
        except SandboxViolationError as exc:
            return {
                "status": "blocked",
                "tool_name": tool_name,
                "error": str(exc),
                "attempts": 0,
            }

        if tool_name not in self.registry:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Unknown tool: '{tool_name}'.",
                "attempts": 0,
            }

        schema_cls, fn = self.registry[tool_name]

        # 2. Pydantic input validation (no retries on schema errors)
        try:
            validated = schema_cls(**args)
        except ValidationError as exc:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Input validation failed: {exc}",
                "attempts": 0,
            }

        # 3. Execute with timeout and 1 retry on timeout/transient failure
        max_attempts = 2
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(fn, validated)
            try:
                result = future.result(timeout=self.timeout_sec)
                return {
                    "status": "success",
                    "tool_name": tool_name,
                    "result": result,
                    "attempts": attempt,
                }
            except FuturesTimeoutError:
                future.cancel()
                last_error = f"Tool '{tool_name}' timed out after {self.timeout_sec}s."
            except Exception as exc:
                last_error = f"Tool '{tool_name}' failed: {exc}"
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        return {
            "status": "error",
            "tool_name": tool_name,
            "error": last_error,
            "attempts": max_attempts,
        }
