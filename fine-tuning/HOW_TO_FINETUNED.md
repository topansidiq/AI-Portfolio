# How To Finetune Model with LoRA

Create <code>eval.jsonl</code>

```text
fine-tuning/
├── data/
│ ├── eval.jsonl
│ └── evaluation/
├── src/
│ └── evaluate_model_lora.py
└── outputs/
```

Run evaluate base mode:

```sh
python .\src\evaluate_model.py --mode base
```

Run evaluate model lora:

```sh
python .\src\evaluate_model.py --mode base
```

Create dataset/raw:

```text
fine-tuning/
├── data/
│ ├── eval.jsonl
│ ├── raw.jsonl
│ └── evaluation/
├── src/
│ └── evaluate_model_lora.py
└── outputs/
```

Training base model:

```sh
python .\src\train_lora.py
```

Expect result:

```text
data/
└── outputs/
    └── lora_adapter/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── README.md
```

Run evaluate fine-tuning LoRA:

```sh
python .\src\evaluate_model_lora.py --mode finetuned --adapter ".\data\outputs\lora_adapter"
```
