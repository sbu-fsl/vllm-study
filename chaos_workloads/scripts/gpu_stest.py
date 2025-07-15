import torch
import time
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure workload
duration_seconds = 600  # Run for 10 minutes
tensor_size = (4096, 4096)  # Size of tensor to multiply (larger = more stress)
log_interval = 5  # seconds

def stress_gpu():
    a = torch.randn(tensor_size, device=device)
    b = torch.randn(tensor_size, device=device)

    for _ in range(100):
        torch.matmul(a, b)  # Matrix multiplication to load GPU
        torch.cuda.synchronize()  # Wait for ops to finish

# Warm-up GPU
print("Warming up GPU...")
for _ in range(10):
    stress_gpu()

print("Starting GPU stress test...")
start_time = time.time()

try:
    while time.time() - start_time < duration_seconds:
        stress_gpu()
        print(f"{datetime.now()}: GPU stress test running...")
        time.sleep(log_interval)
except KeyboardInterrupt:
    print("Stress test interrupted.")

print("Test complete.")
