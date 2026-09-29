import time

import torch  # type: ignore

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check your PyTorch installation.")

print("GPU:", torch.cuda.get_device_name(0))

props = torch.cuda.get_device_properties(0)
print(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")

# Create tensors on the GPU
device = "cuda"
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)

torch.cuda.synchronize()
start = time.time()

# Matrix multiplication
c = torch.matmul(a, b)

torch.cuda.synchronize()
elapsed = time.time() - start

print("Tensor device:", c.device)
print("Result shape:", c.shape)
print(f"Execution time: {elapsed:.4f} seconds")
print("GPU memory allocated:", f"{torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
print("GPU test successful.")
