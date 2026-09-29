import os
from pathlib import Path

from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv
from transformers import AutoTokenizer  # type: ignore

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    print("Hugging Face token: Loaded")
else:
    print("Hugging Face token: Not configured")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = "You are an industrial automation assistant."

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "industrial_qa.jsonl"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
)

train_dataset = dataset["train"]


def create_messages(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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
    }


dataset = train_dataset.map(create_messages)


def apply_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


dataset = dataset.map(apply_template)

print("Chat-formatted example:")
print(dataset[0]["text"])

example_text = dataset[0]["text"]

tokens = tokenizer(example_text)

print("\nToken IDs:")
print(tokens["input_ids"])

print("\nNumber of tokens:")
print(len(tokens["input_ids"]))

token_ids = tokens["input_ids"]

token_strings = tokenizer.convert_ids_to_tokens(token_ids)

print("\nTokens:")
print(token_strings)
