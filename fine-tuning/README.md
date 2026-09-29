# Topan AI-Portfolio: Fine-tuning Model

In this phase, I executed a pre-trained machine learning model using Python to establish a baseline performance. Following the initial run, I extended the model's capabilities by performing targeted fine-tuning on a specialized dataset.

Make sure you have:

1. python 3.11+
2. CUDA 12.3+
3. Good internet connection
4. HuggingFace account

Create python virtual environment:

```sh
python -m venv venv
```

Enter to virtual environment:

```sh
.\venv\Scripts\Activate.ps1
.\venv\bin\activate # Linux/macOS
```

Exit from virtual environment:

```sh
deactivate
```

Install the requirements:

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have CUDA right now in your device, run this in your active virtual environment:

```sh
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
