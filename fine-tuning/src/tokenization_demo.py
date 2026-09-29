import os

from dotenv import load_dotenv
from transformers import AutoTokenizer  # type: ignore

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

# Qwen tokenizer may not have a separate pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


messages = [
    {"role": "system", "content": "You are an industrial automation assistant."},
    {"role": "user", "content": "What is Modbus TCP?"},
    {
        "role": "assistant",
        "content": (
            "Modbus TCP is an industrial communication protocol "
            "that uses TCP/IP to exchange data between industrial devices."
        ),
    },
]


text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)


encoded = tokenizer(text, return_tensors="pt")


print("Formatted text:")
print(text)

print("\nTokenizer output:")
print(encoded)

print("\nKeys:")
print(encoded.keys())

print("\ninput_ids shape:")
print(encoded["input_ids"].shape)

print("\nattention_mask shape:")
print(encoded["attention_mask"].shape)

print("\nNumber of tokens:")
print(encoded["input_ids"].shape[1])

messages_2 = [
    {"role": "system", "content": "You are an industrial automation assistant."},
    {"role": "user", "content": "What is a PLC?"},
    {
        "role": "assistant",
        "content": (
            "A PLC is a programmable logic controller used "
            "to control industrial machines and processes."
        ),
    },
]

text_2 = tokenizer.apply_chat_template(
    messages_2, tokenize=False, add_generation_prompt=False
)


batch = tokenizer(
    [text, text_2], padding=True, truncation=True, max_length=64, return_tensors="pt"
)


print("\nBatch:")
print(batch)

print("\nBatch input_ids shape:")
print(batch["input_ids"].shape)

print("\nBatch attention_mask:")
print(batch["attention_mask"])
