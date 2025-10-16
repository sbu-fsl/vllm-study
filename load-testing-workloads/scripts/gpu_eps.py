import torch
import os
import time

def stress_gpu_and_ephemeral_storage(
    num_files=10,
    file_size_mb=512,
    tensor_size=(128, 1024, 1024),
    sleep_time=0.2
):
    tensors = []
    tmp_dir = "/tmp"
    file_paths = [os.path.join(tmp_dir, f"stress_file_{i}.bin") for i in range(num_files)]
    data = os.urandom(file_size_mb * 1024 * 1024)

    file_idx = 0
    while True:
        # Stress GPU memory
        try:
            tensors.append(torch.randn(*tensor_size, device='cuda'))
        except RuntimeError as e:
            print("CUDA OOM:", e)
            tensors = []  # Free up GPU memory
            torch.cuda.empty_cache()
            time.sleep(1)

        # Stress ephemeral storage
        try:
            with open(file_paths[file_idx % num_files], "wb") as f:
                f.write(data)
            file_idx += 1
        except Exception as e:
            print("Ephemeral storage error:", e)
            # Optionally clean up files
            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)
            time.sleep(1)

        time.sleep(sleep_time)

stress_gpu_and_ephemeral_storage()
