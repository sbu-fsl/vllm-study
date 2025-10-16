import torch
import time

device = torch.device("cuda")

# Matrix multiplication in loop
for _ in range(10000):
    a = torch.randn(8192, 8192, device=device)
    b = torch.randn(8192, 8192, device=device)
    c = torch.matmul(a, b)
    del a, b, c
    torch.cuda.empty_cache()
    time.sleep(0.1)  # To vary the pattern

