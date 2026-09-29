import os

import torch  # type: ignore
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model  # type: ignore
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16 if device == "cuda" else torch.float32
)

model.to(device)

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
