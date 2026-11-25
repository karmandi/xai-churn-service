import pytest
from src.evals.graders import GradeResult
from src.evals.runner import EvalRunner, cluster_failures


@pytest.fixture(scope="module")
def eval_report():
    runner = EvalRunner()
    return runner.run(k=3)


def test_judge_calibration_gate(eval_report):
    cal = eval_report["judge_calibration"]
    assert cal["total_samples"] >= 50
    assert cal["agreement_rate"] >= 0.85
    assert cal["false_positive_rate"] <= 0.15
    assert cal["calibrated"] is True


def test_pass_at_k_regression_gate(eval_report):
    assert eval_report["total_cases"] == 60
    assert eval_report["pass_at_1"] >= 0.90
    assert eval_report["pass_at_3"] >= 0.95


def test_latency_and_token_budget_gate(eval_report):
    assert eval_report["p95_latency_ms"] <= 500.0
    assert eval_report["mean_tokens"] <= 1200


def test_failure_clustering_ranks_buckets():
    synthetic = [
        GradeResult(
            case_id="C1",
            passed=False,
            trajectory_score=0.0,
            math_score=1.0,
            judge_score=0.0,
            failure_bucket="trajectory_mismatch",
        ),
        GradeResult(
            case_id="C2",
            passed=False,
            trajectory_score=0.0,
            math_score=1.0,
            judge_score=0.0,
            failure_bucket="trajectory_mismatch",
        ),
        GradeResult(
            case_id="C3",
            passed=False,
            trajectory_score=1.0,
            math_score=0.0,
            judge_score=0.0,
            failure_bucket="unauthorized_discount",
        ),
        GradeResult(
            case_id="C4",
            passed=True,
            trajectory_score=1.0,
            math_score=1.0,
            judge_score=1.0,
        ),
    ]
    clusters = cluster_failures(synthetic)
    assert len(clusters) == 2
    assert clusters[0]["failure_bucket"] == "trajectory_mismatch"
    assert clusters[0]["count"] == 2
    assert clusters[1]["failure_bucket"] == "unauthorized_discount"
    assert clusters[1]["count"] == 1
