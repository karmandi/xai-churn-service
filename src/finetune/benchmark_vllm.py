import argparse
import asyncio
import json
import math
import time
from typing import Any
import httpx


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 2)


def _build_mock_transport(
    ttft_ms: float = 42.0,
    itl_ms: float = 8.0,
    num_tokens: int = 24,
) -> httpx.MockTransport:
    """Simulate an OpenAI-compatible vLLM SSE streaming response."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(ttft_ms / 1000.0)
        chunks: list[bytes] = []
        for i in range(num_tokens):
            payload = {
                "id": "cmpl-mock",
                "choices": [{"index": 0, "delta": {"content": f"tok{i} "}}],
            }
            chunks.append(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        chunks.append(b"data: [DONE]\n\n")

        class _DelayedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for idx, chunk in enumerate(chunks):
                    if idx > 0 and idx < len(chunks) - 1:
                        await asyncio.sleep(itl_ms / 1000.0)
                    yield chunk

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DelayedStream(),
        )

    return httpx.MockTransport(handler)


async def _single_stream_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    t_first_token: float | None = None
    token_timestamps: list[float] = []

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }

    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
            token_timestamps.append(now)

    t_end = time.perf_counter()
    ttft_ms = ((t_first_token or t_end) - t_start) * 1000.0
    e2e_ms = (t_end - t_start) * 1000.0

    itls = [
        (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
        for i in range(1, len(token_timestamps))
    ]
    itl_mean_ms = sum(itls) / len(itls) if itls else 0.0

    return {
        "ttft_ms": ttft_ms,
        "itl_mean_ms": itl_mean_ms,
        "e2e_ms": e2e_ms,
        "tokens": len(token_timestamps),
    }


async def run_benchmark(
    base_url: str = "http://127.0.0.1:8001",
    model: str = "qlora-churn-3b",
    concurrency: int = 8,
    num_requests: int = 16,
    sla_ms: float = 250.0,
    mock_server: bool = False,
    mock_ttft_ms: float = 42.0,
    mock_itl_ms: float = 6.0,
    mock_num_tokens: int = 20,
    gpu_hourly_usd: float = 0.75,
) -> dict[str, Any]:
    """Run concurrent streaming benchmark against vLLM or mock transport."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    prompt = json.dumps(
        {
            "churn_probability": 0.8006,
            "risk_category": "High",
            "top_attributions": {
                "num__tenure": 0.1375,
                "cat__InternetService_Fiber optic": 0.0868,
                "cat__Contract_Month-to-month": 0.0852,
            },
        }
    )

    transport = (
        _build_mock_transport(
            ttft_ms=mock_ttft_ms, itl_ms=mock_itl_ms, num_tokens=mock_num_tokens
        )
        if mock_server
        else None
    )

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def _worker(client: httpx.AsyncClient) -> None:
        async with semaphore:
            res = await _single_stream_request(client, url, model, prompt)
            results.append(res)

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
        await asyncio.gather(*[_worker(client) for _ in range(num_requests)])
    wall_elapsed_sec = max(time.perf_counter() - wall_start, 1e-6)

    ttfts = [r["ttft_ms"] for r in results]
    itls = [r["itl_mean_ms"] for r in results]
    e2es = [r["e2e_ms"] for r in results]

    within_sla = sum(1 for e2e in e2es if e2e <= sla_ms)
    goodput_rps = round(within_sla / wall_elapsed_sec, 2)
    throughput_rps = max(num_requests / wall_elapsed_sec, 1e-6)
    cost_per_1k = round((gpu_hourly_usd / 3600.0) * (1000.0 / throughput_rps), 4)

    return {
        "model": model,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "sla_ms": sla_ms,
        "ttft_p50_ms": _percentile(ttfts, 50),
        "ttft_p95_ms": _percentile(ttfts, 95),
        "itl_mean_ms": round(sum(itls) / len(itls), 2) if itls else 0.0,
        "e2e_p95_ms": _percentile(e2es, 95),
        "goodput_rps": goodput_rps,
        "sla_attainment_pct": round((within_sla / max(num_requests, 1)) * 100.0, 2),
        "estimated_cost_per_1k_req": cost_per_1k,
    }


async def run_mock_comparison(
    concurrency: int = 8,
    sla_ms: float = 250.0,
) -> dict[str, Any]:
    """Compare prompting-only 8B baseline vs QLoRA-tuned 3B model under identical SLA."""
    baseline = await run_benchmark(
        model="Llama-3.1-8B-Instruct (Prompt Baseline)",
        concurrency=concurrency,
        num_requests=16,
        sla_ms=sla_ms,
        mock_server=True,
        mock_ttft_ms=115.0,
        mock_itl_ms=11.0,
        mock_num_tokens=16,
        gpu_hourly_usd=1.50,
    )
    qlora = await run_benchmark(
        model="Llama-3.2-3B-QLoRA (Fine-Tuned)",
        concurrency=concurrency,
        num_requests=16,
        sla_ms=sla_ms,
        mock_server=True,
        mock_ttft_ms=42.0,
        mock_itl_ms=5.0,
        mock_num_tokens=10,
        gpu_hourly_usd=0.75,
    )
    return {"baseline": baseline, "qlora": qlora}


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM streaming latency & goodput benchmark")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--model", type=str, default="qlora-churn-3b")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--sla-ms", type=float, default=250.0)
    parser.add_argument("--mock-server", action="store_true")
    args = parser.parse_args()

    if args.mock_server:
        report = asyncio.run(
            run_mock_comparison(concurrency=args.concurrency, sla_ms=args.sla_ms)
        )
        print(json.dumps(report, indent=2))
    else:
        report = asyncio.run(
            run_benchmark(
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                num_requests=args.num_requests,
                sla_ms=args.sla_ms,
                mock_server=False,
            )
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
