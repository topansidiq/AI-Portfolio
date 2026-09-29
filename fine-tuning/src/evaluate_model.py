import json
import os
from pathlib import Path

import torch  # type: ignore
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    print("Hugging Face token: Loaded")
else:
    print("Hugging Face token: Not configured")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "eval.jsonl"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)


def generate_answer(question):
    messages = [
        {
            "role": "system",
            "content": "You are Rosa, an industrial automation assistant. "
            "Be accurate. Do not invent technical specifications.",
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


results = []

with open(EVAL_FILE, encoding="utf-8") as file:
    for line in file:
        case = json.loads(line)
        answer = generate_answer(case["question"])

        result = {
            "question": case["question"],
            "expected": case["expected"],
            "answer": answer,
            "category": case["category"],
            "checks": case["checks"],
        }
        results.append(result)

output_file = ROOT / "data" / "evaluation" / "eval_results.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

print(f"Evaluated {len(results)} cases")
print(f"Results saved to: {output_file}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Q = Query
# K = Key
# V = Value

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()