import os

import torch  # type: ignore
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16 if device == "cuda" else torch.float32
)

model.to(device)
model.train()


messages = [
    {"role": "system", "content": "You are an industrial automation assistant."},
    {"role": "user", "content": "What is a PLC?"},
    {
        "role": "assistant",
        "content": (
            "A PLC is a programmable logic controller "
            "used to control industrial machines and processes."
        ),
    },
]


text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)


inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)


outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)


print("Input shape:", input_ids.shape)
print("Logits shape:", outputs.logits.shape)
print("Loss:", outputs.loss.item())
