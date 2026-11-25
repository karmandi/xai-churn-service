import asyncio
import json
from pathlib import Path
from src.finetune.benchmark_vllm import run_mock_comparison
from src.finetune.kv_math import compute_kv_cache_profile
from src.finetune.prepare_sft import generate_sft_dataset
from src.finetune.train_qlora import run_dry_run


def test_generate_sft_dataset_grounded_in_shap(tmp_path: Path):
    out_file = tmp_path / "sft_test.jsonl"
    records = generate_sft_dataset(output_path=str(out_file), n_samples=25)

    assert len(records) == 25
    assert out_file.exists()

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 25

    for line in lines:
        rec = json.loads(line)
        msgs = rec["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        user_payload = json.loads(msgs[1]["content"])
        assert "churn_probability" in user_payload
        assert "risk_category" in user_payload
        assert len(user_payload["top_attributions"]) == 3

        # Verify assistant response references top SHAP driver name
        top_key = next(iter(user_payload["top_attributions"]))
        clean_key = top_key.replace("num__", "").replace("cat__", "").replace("_", " ")
        assert clean_key in msgs[2]["content"]


def test_train_qlora_dry_run(tmp_path: Path):
    out_file = tmp_path / "sft_dry.jsonl"
    generate_sft_dataset(output_path=str(out_file), n_samples=15)
    summary = run_dry_run(
        dataset_path=str(out_file),
        model_name="meta-llama/Llama-3.2-3B-Instruct",
    )
    assert summary["mode"] == "dry_run"
    assert summary["num_records"] == 15
    assert summary["trainable_pct"] < 2.0


def test_kv_math_exact_calculations():
    # Llama-3.2-3B: 28 layers, 8 kv_heads, 128 head_dim, FP16 (2 bytes)
    # 2 * 28 * 8 * 128 * 2 = 114,688 bytes/token
    prof_fp16 = compute_kv_cache_profile(
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        free_vram_gb=18.2,
        context_length=2048,
    )
    assert prof_fp16["kv_bytes_per_token"] == 114_688
    expected_mb = (114_688 * 2048) / (1024**2)
    assert abs(prof_fp16["kv_mb_per_sequence"] - expected_mb) < 1e-3
    expected_conc = int((18.2 * (1024**3)) // (114_688 * 2048))
    assert prof_fp16["max_concurrency"] == expected_conc

    # FP8 (1 byte) halves bytes/token and doubles concurrency
    prof_fp8 = compute_kv_cache_profile(
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=1,
        free_vram_gb=18.2,
        context_length=2048,
    )
    assert prof_fp8["kv_bytes_per_token"] == 57_344
    assert prof_fp8["max_concurrency"] >= prof_fp16["max_concurrency"] * 2


def test_benchmark_vllm_mock_server():
    report = asyncio.run(run_mock_comparison(concurrency=4, sla_ms=250.0))
    assert "baseline" in report and "qlora" in report

    for key in ("baseline", "qlora"):
        metrics = report[key]
        assert metrics["ttft_p50_ms"] > 0
        assert metrics["ttft_p95_ms"] >= metrics["ttft_p50_ms"]
        assert metrics["itl_mean_ms"] > 0
        assert metrics["e2e_p95_ms"] >= metrics["ttft_p95_ms"]
        assert metrics["goodput_rps"] >= 0
        assert metrics["estimated_cost_per_1k_req"] > 0

    assert report["qlora"]["ttft_p95_ms"] < report["baseline"]["ttft_p95_ms"]
    assert report["qlora"]["e2e_p95_ms"] < report["baseline"]["e2e_p95_ms"]
