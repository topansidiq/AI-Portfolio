import os

import torch  # type: ignore
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    print("Hugging Face token: Loaded")
else:
    print("Hugging Face token: Not configured")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    dtype=(torch.float16 if device == "cuda" else torch.float32),
)

model = model.to(device)
model.eval()

parameter_count = sum(parameter.numel() for parameter in model.parameters())

print("Model:", MODEL_ID)
print("Parameters:", parameter_count)
print("Tokenizer vocabulary:", tokenizer.vocab_size)
print("Device:", next(model.parameters()).device)

messages = [
    {
        "role": "system",
        "content": "You are an industrial automation assistant.",
    },
    {
        "role": "user",
        "content": "What is Modbus TCP?",
    },
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\nFormatted prompt:")
print(formatted_prompt)

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1] :]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)

print("\nAssistant:")
print(response)
