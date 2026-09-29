from pathlib import Path

from datasets import load_dataset  # type: ignore

SYSTEM_PROMPT = "You are an industrial automation assistant."

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "industrial_qa.jsonl"

dataset = load_dataset(
    "json",
    data_files=str(INPUT_PATH),
)

train_dataset = dataset["train"]


def format_example(example):
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


formatted_dataset = train_dataset.map(format_example)

print("Original example:")
print(train_dataset[0])

print("\nFormatted example:")
print(formatted_dataset[0])
