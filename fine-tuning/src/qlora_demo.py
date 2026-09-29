import os

import torch  # type: ignore
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this QLoRA demo.")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

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
    MODEL_ID, quantization_config=quantization_config, device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
