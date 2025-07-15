import torch
import time

def stress_gpu():
    tensors = []
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * 0.92)  # Use 92% of GPU memory

    chunk_shape = (128, 1024, 1024)
    # Create a sample tensor to get the actual size in bytes
    sample_tensor = torch.randn(*chunk_shape, device=device)
    tensor_size = sample_tensor.element_size() * sample_tensor.numel()
    del sample_tensor
    torch.cuda.empty_cache()

    allocated = 0

    # Allocate up to 92% of GPU memory
    while allocated + tensor_size < target_mem:
        tensors.append(torch.randn(*chunk_shape, device=device))
        allocated += tensor_size

    print(f"Allocated ~{allocated/1e9:.2f} GB, holding for 1 minute...")
    time.sleep(60)

    print("Triggering OOM...")
    while True:
        tensors.append(torch.randn(*chunk_shape, device=device))

stress_gpu()
