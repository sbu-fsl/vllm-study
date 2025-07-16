import torch
import time

device = torch.device("cuda")

# Create large CPU tensors
cpu_tensor = torch.randn((10000, 10000))

while True:
    gpu_tensor = cpu_tensor.to(device, non_blocking=True)
    cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    time.sleep(0.01)  # Tune delay to balance pressure

