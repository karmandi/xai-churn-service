from typing import Any


def compute_kv_cache_profile(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    free_vram_gb: float,
    context_length: int,
) -> dict[str, Any]:
    """Compute analytical KV-cache memory per token, per sequence, and max concurrency."""
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    kv_mb_per_sequence = (kv_bytes_per_token * context_length) / (1024**2)
    max_concurrency = int(
        (free_vram_gb * (1024**3)) // (kv_bytes_per_token * context_length)
    )
    return {
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_mb_per_sequence": round(kv_mb_per_sequence, 4),
        "max_concurrency": max_concurrency,
    }


def print_kv_comparison(context_length: int = 2048) -> list[dict[str, Any]]:
    """Compare FP16 (2 bytes) vs FP8 (1 byte) KV cache concurrency on L4 and A100 GPUs."""
    models = {
        "Llama-3.2-3B": {"num_layers": 28, "num_kv_heads": 8, "head_dim": 128, "weights_4bit_gb": 2.2},
        "Llama-3.1-8B": {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "weights_4bit_gb": 5.2},
    }
    gpus = {
        "L4 (24 GB)": 24.0,
        "A100 (80 GB)": 80.0,
    }
    dtypes = {"FP16": 2, "FP8": 1}

    rows: list[dict[str, Any]] = []
    print(f"\n=== KV-Cache Memory & Max Concurrency (context_length={context_length}) ===")
    print(
        f"{'Model':<14} {'GPU':<13} {'KV Dtype':<9} {'Free VRAM':<10} "
        f"{'KB/tok':<9} {'MB/seq':<10} {'Max Conc.':<10}"
    )
    print("-" * 80)

    for model_name, cfg in models.items():
        for gpu_name, total_vram in gpus.items():
            # Reserve 15% overhead + 4-bit model weights
            free_vram = round(total_vram * 0.85 - cfg["weights_4bit_gb"], 2)
            for dtype_name, dtype_bytes in dtypes.items():
                prof = compute_kv_cache_profile(
                    num_layers=cfg["num_layers"],
                    num_kv_heads=cfg["num_kv_heads"],
                    head_dim=cfg["head_dim"],
                    dtype_bytes=dtype_bytes,
                    free_vram_gb=free_vram,
                    context_length=context_length,
                )
                row = {
                    "model": model_name,
                    "gpu": gpu_name,
                    "kv_dtype": dtype_name,
                    "free_vram_gb": free_vram,
                    **prof,
                }
                rows.append(row)
                print(
                    f"{model_name:<14} {gpu_name:<13} {dtype_name:<9} "
                    f"{free_vram:<10.2f} {prof['kv_bytes_per_token'] / 1024:<9.1f} "
                    f"{prof['kv_mb_per_sequence']:<10.2f} {prof['max_concurrency']:<10d}"
                )
    return rows


if __name__ == "__main__":
    print_kv_comparison()
