import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any
from src.agent.harness import RemediationHarness
from src.agent.tools import ToolExecutor
from src.agent.tracing import AgentTrace
from src.evals.dataset import build_eval_dataset
from src.evals.graders import (
    CalibratedJudge,
    GradeResult,
    grade_math_and_safety,
    grade_trajectory,
)
from src.models.explainer import ChurnExplainer


def cluster_failures(results: list[GradeResult]) -> list[dict[str, Any]]:
    """Group failed GradeResult items by failure_bucket, sorted largest to smallest."""
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for res in results:
        if not res.passed:
            bucket = res.failure_bucket or "unknown_failure"
            counts[bucket] += 1
            examples.setdefault(bucket, []).append(res.case_id)

    return [
        {"failure_bucket": bucket, "count": count, "sample_cases": examples[bucket][:5]}
        for bucket, count in counts.most_common()
    ]


class EvalRunner:
    """Runs end-to-end pass@1 and pass@k evaluations with latency and calibration tracking."""

    def __init__(
        self,
        dataset_path: str = "data/eval_golden.json",
        report_path: str = "artifacts/eval_report.json",
        explainer: ChurnExplainer | None = None,
        judge: CalibratedJudge | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.report_path = Path(report_path)
        self.explainer = explainer or ChurnExplainer()
        self.executor = ToolExecutor(explainer=self.explainer)
        self.harness = RemediationHarness(executor=self.executor)
        self.judge = judge or CalibratedJudge()

    def _load_cases(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            return build_eval_dataset(output_path=str(self.dataset_path))
        return json.loads(self.dataset_path.read_text(encoding="utf-8"))

    def _run_single_attempt(
        self, case: dict[str, Any]
    ) -> tuple[GradeResult, float, int, dict[str, Any], AgentTrace | None]:
        t0 = time.perf_counter()
        payload = case["customer_payload"]
        prediction = self.explainer.predict_and_explain(payload)

        trace: AgentTrace | None = None
        if prediction["risk_category"] == "High":
            trace = self.harness.run(case["customer_id"], payload)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        tokens = trace.total_tokens if trace is not None else 0

        ref = case["human_reference"]
        traj_score, traj_bucket, traj_detail = grade_trajectory(
            trace, ref["expected_tools"], ref["expected_status"]
        )
        math_score, math_bucket, math_detail = grade_math_and_safety(prediction, trace)
        judge_pass, judge_score, judge_detail = self.judge.score_case(
            case, prediction, trace
        )

        passed = traj_score == 1.0 and math_score == 1.0 and judge_pass
        failure_bucket = None
        if not passed:
            failure_bucket = traj_bucket or math_bucket or "judge_quality_failure"

        grade = GradeResult(
            case_id=case["case_id"],
            passed=passed,
            trajectory_score=traj_score,
            math_score=math_score,
            judge_score=judge_score,
            failure_bucket=failure_bucket,
            details=f"{traj_detail} | {math_detail} | {judge_detail}",
        )
        return grade, latency_ms, tokens, prediction, trace

    def _build_calibration_set(
        self,
        evaluated_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assemble 50+ human-labeled positive and negative examples for judge calibration."""
        calibration_items: list[dict[str, Any]] = []
        for rec in evaluated_records:
            calibration_items.append(
                {
                    "case": rec["case"],
                    "prediction": rec["prediction"],
                    "trace": rec["trace"],
                    "human_grade": rec["case"]["human_reference"]["human_grade"],
                }
            )

        # Add 10 synthetic negative examples (human_grade=False) to verify FPR/lenient bias
        negatives_added = 0
        for rec in evaluated_records:
            if rec["trace"] is not None and isinstance(rec["trace"].final_output, dict):
                bad_trace = rec["trace"].model_copy(deep=True)
                bad_trace.final_output = {
                    "action_taken": "applied_retention_perk",
                    "discount_pct": 85.0,
                }
                calibration_items.append(
                    {
                        "case": rec["case"],
                        "prediction": rec["prediction"],
                        "trace": bad_trace,
                        "human_grade": False,
                    }
                )
                negatives_added += 1
                if negatives_added >= 10:
                    break

        return calibration_items

    def run(self, k: int = 3) -> dict[str, Any]:
        cases = self._load_cases()
        first_grades: list[GradeResult] = []
        pass_at_k_hits = 0
        latencies_ms: list[float] = []
        token_counts: list[int] = []
        evaluated_records: list[dict[str, Any]] = []

        for case in cases:
            case_passed_any = False
            for attempt in range(k):
                grade, lat, toks, pred, trace = self._run_single_attempt(case)
                if attempt == 0:
                    first_grades.append(grade)
                    latencies_ms.append(lat)
                    token_counts.append(toks)
                    evaluated_records.append(
                        {"case": case, "prediction": pred, "trace": trace}
                    )
                if grade.passed:
                    case_passed_any = True
                    break
            if case_passed_any:
                pass_at_k_hits += 1

        total_cases = len(cases)
        pass_at_1 = round(
            sum(1 for g in first_grades if g.passed) / max(total_cases, 1), 4
        )
        pass_at_k = round(pass_at_k_hits / max(total_cases, 1), 4)

        sorted_lat = sorted(latencies_ms)
        p95_idx = max(0, min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1))
        p95_latency_ms = round(sorted_lat[p95_idx], 2) if sorted_lat else 0.0
        mean_tokens = round(sum(token_counts) / max(len(token_counts), 1), 2)

        calibration_set = self._build_calibration_set(evaluated_records)
        calibration_metrics = self.judge.calibrate(calibration_set)
        failure_clusters = cluster_failures(first_grades)

        report = {
            "total_cases": total_cases,
            "pass_at_1": pass_at_1,
            f"pass_at_{k}": pass_at_k,
            "p95_latency_ms": p95_latency_ms,
            "mean_tokens": mean_tokens,
            "judge_calibration": calibration_metrics,
            "failure_clusters": failure_clusters,
            "results": [g.model_dump() for g in first_grades],
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== Churn XAI End-to-End Eval Summary ===")
    print(f"Total Cases     : {report['total_cases']}")
    print(f"pass@1          : {report['pass_at_1']:.2%}")
    print(f"pass@3          : {report.get('pass_at_3', 0.0):.2%}")
    print(f"p95 Latency     : {report['p95_latency_ms']:.2f} ms")
    print(f"Mean Tokens     : {report['mean_tokens']:.2f}")
    cal = report["judge_calibration"]
    print(
        f"Judge Calib.    : calibrated={cal['calibrated']} "
        f"(n={cal['total_samples']}, agree={cal['agreement_rate']:.2%}, "
        f"FPR={cal['false_positive_rate']:.2%}, FNR={cal['false_negative_rate']:.2%})"
    )
    if report["failure_clusters"]:
        print("Failure Clusters:")
        for c in report["failure_clusters"]:
            print(f"  - {c['failure_bucket']}: {c['count']} ({c['sample_cases']})")
    else:
        print("Failure Clusters: None (0 failures)")


if __name__ == "__main__":
    runner = EvalRunner()
    eval_report = runner.run(k=3)
    print_summary(eval_report)
