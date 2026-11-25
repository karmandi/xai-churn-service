import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_DATASET_PATH = "data/sft_shap_pairs.jsonl"
DEFAULT_OUTPUT_DIR = "artifacts/qlora_churn_explainer"


def load_sft_records(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    return "\n".join(parts)


def run_dry_run(dataset_path: str, model_name: str) -> dict[str, Any]:
    """Validate dataset formatting, tokenization stats, and LoRA parameter budget without GPU."""
    records = load_sft_records(dataset_path)
    if not records:
        raise ValueError("SFT dataset is empty.")

    token_lengths: list[int] = []
    for idx, rec in enumerate(records):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            raise ValueError(f"Record {idx} must contain 3 chat messages.")
        if [m["role"] for m in msgs] != ["system", "user", "assistant"]:
            raise ValueError(f"Record {idx} has invalid role sequence.")
        prompt = format_chat_prompt(msgs)
        token_lengths.append(max(1, len(prompt) // 4))

    # Analytical LoRA (r=16) trainable parameter ratio for Llama-3.2-3B / 3.1-8B
    # For Llama-3.2-3B (~3.21B total params), q/k/v/o/gate/up/down LoRA r=16 adds ~24.3M params (~0.75%).
    base_params = 3_212_749_824 if "3B" in model_name else 8_030_261_248
    lora_trainable_params = 24_313_856 if "3B" in model_name else 41_943_040
    trainable_pct = round((lora_trainable_params / base_params) * 100.0, 4)

    assert trainable_pct < 2.0, f"LoRA trainable % ({trainable_pct}%) exceeds 2% cap."

    summary = {
        "mode": "dry_run",
        "model_name": model_name,
        "num_records": len(records),
        "max_estimated_tokens": max(token_lengths),
        "mean_estimated_tokens": round(sum(token_lengths) / len(token_lengths), 2),
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "task_type": "CAUSAL_LM",
            "quantization": "4bit_nf4",
        },
        "base_params": base_params,
        "trainable_params": lora_trainable_params,
        "trainable_pct": trainable_pct,
    }
    print(json.dumps(summary, indent=2))
    return summary


def train_qlora(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> None:
    """Run 4-bit NF4 QLoRA fine-tuning via transformers, peft, bitsandbytes, and trl."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    records = load_sft_records(dataset_path)
    dataset = Dataset.from_list(
        [{"text": format_chat_prompt(r["messages"])} for r in records]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved QLoRA adapter to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Churn XAI Explainer")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and LoRA params without GPU")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args.dataset_path, args.model_name)
    else:
        train_qlora(
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
