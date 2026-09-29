from pathlib import Path

from datasets import load_dataset  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "industrial_qa.jsonl"

print("Loading from:", DATASET_PATH)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
)

print("\nDataset:")
print(dataset)

print("\nTraining dataset:")
print(dataset["train"])

print("\nNumber of examples:")
print(len(dataset["train"]))

print("\nFirst example:")
print(dataset["train"][0])

print("\nAll examples:")

for index, example in enumerate(dataset["train"]):
    print(f"\n--- Example {index + 1} ---")
    print("Instruction:", example["instruction"])
    print("Response:", example["response"])
