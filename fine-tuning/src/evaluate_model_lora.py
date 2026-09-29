import argparse
import json
import os
from pathlib import Path

import torch  # type: ignore
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

EVAL_FILE = ROOT / "data" / "eval.jsonl"
OUTPUT_DIR = ROOT / "data" / "evaluation"

LORA_DIR = ROOT / "outputs" / "lora_adapter"

SYSTEM_PROMPT = (
    "You are Rosa, an industrial automation assistant. "
    "Be accurate. Do not invent technical specifications. "
    "If information is missing, clearly state what is unknown."
)

MAX_NEW_TOKENS = 160

parser = argparse.ArgumentParser(description="Evaluate Rosa base or fine-tuned model.")

parser.add_argument(
    "--mode",
    choices=["base", "lora-init", "finetuned"],
    default="base",
    help="Evaluation mode.",
)

parser.add_argument(
    "--adapter",
    type=str,
    default=str(LORA_DIR),
    help="Path to the trained LoRA adapter.",
)

args = parser.parse_args()

if HF_TOKEN:
    print("Hugging Face token: Loaded")
else:
    print("Hugging Face token: Not configured")

if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Running on CPU.")
else:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print(f"\nLoading tokenizer: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model: {MODEL_ID}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

if args.mode == "lora-init":
    print("\nAttaching a new, untrained LoRA adapter.")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

elif args.mode == "finetuned":
    adapter_path = Path(args.adapter)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    print(f"\nLoading trained adapter: {adapter_path}")

    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
    )

model.eval()


def generate_answer(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # Use the embedding layer's device, which is suitable for
    # ordinary single-GPU loading and common device_map setups.
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


if not EVAL_FILE.exists():
    raise FileNotFoundError(f"Evaluation dataset not found: {EVAL_FILE}")

eval_cases = []

with open(EVAL_FILE, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

        required_fields = [
            "question",
            "expected",
            "category",
            "checks",
        ]

        missing = [field for field in required_fields if field not in case]

        if missing:
            raise ValueError(f"Line {line_number} missing fields: {missing}")

        eval_cases.append(case)

if not eval_cases:
    raise ValueError("Evaluation dataset is empty.")

print(f"\nEvaluation cases: {len(eval_cases)}")

results = []

for index, case in enumerate(eval_cases, start=1):
    question = case["question"]

    print(f"\n[{index}/{len(eval_cases)}] {question}")

    answer = generate_answer(question)

    print(f"Answer: {answer}")

    results.append(
        {
            "question": question,
            "expected": case["expected"],
            "answer": answer,
            "category": case["category"],
            "checks": case["checks"],
            "mode": args.mode,
        }
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_names = {
    "base": "eval_results_base.json",
    "lora-init": "eval_results_lora_init.json",
    "finetuned": "eval_results_finetuned.json",
}

output_file = OUTPUT_DIR / output_names[args.mode]

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"Mode: {args.mode}")
print(f"Cases evaluated: {len(results)}")
print(f"Results saved to: {output_file}")
