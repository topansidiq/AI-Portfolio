import os
from pathlib import Path

import torch  # type: ignore
from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_FILE = ROOT / "data" / "industrial_qa.jsonl"

OUTPUT_DIR = ROOT / "data" / "outputs" / "lora_adapter"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=str(DATA_FILE),
    split="train",
)

print(f"Dataset size: {len(dataset)}")


def format_example(example):
    messages = [
        {
            "role": "system",
            "content": (
                "You are Rosa, an industrial automation assistant. "
                "Be accurate and do not invent technical specifications."
            ),
        },
        {
            "role": "user",
            "content": example["instruction"],
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text,
    }


dataset = dataset.map(format_example)

print("Dataset formatted.")


def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )

    result["labels"] = result["input_ids"].copy()

    return result


tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
)

print("Dataset tokenized.")

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

model.config.use_cache = False

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

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    bf16=False,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\nStarting LoRA training...")

trainer.train()

print("\nSaving LoRA adapter...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    OUTPUT_DIR,
)

tokenizer.save_pretrained(
    OUTPUT_DIR,
)

print("\nTraining complete.")
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
