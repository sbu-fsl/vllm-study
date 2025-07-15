import torch
import time

def stress_gpu_normal_load(num_tensors=4, sleep_time=0.5):
    # Allocate a fixed number of tensors (each ~512MB)
    tensors = [torch.randn(128, 1024, 1024, device='cuda') for _ in range(num_tensors)]
    while True:
        # Perform a simple operation to keep the GPU busy
        for t in tensors:
            t = t * 1.0001
        time.sleep(sleep_time)

stress_gpu_normal_load()
