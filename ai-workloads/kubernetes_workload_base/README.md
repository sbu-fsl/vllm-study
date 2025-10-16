å# Kubernetes Workload Base Setup

This folder contains the Kubernetes manifest used to deploy the base environment for running distributed AI/ML experiments.

## Deployment Overview

We use a Kubernetes `Job` to deploy a container based on the `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime` image.

During startup, the container:
- Installs system tools (`tmux`, `nano`, `curl`, `micro`)
- Installs Python packages (`transformers`, `accelerate`)
- Prepares an environment ready for training experiments.

The pod mounts a Persistent Volume Claim (PVC) to provide access to necessary datasets and shared storage.

## Storage Details

- The `/home` directory inside the container is mounted from a Kubernetes **Persistent Volume Claim (PVC)** named `pvc-workspace`.
- **Dataset Access:**  
  The ImageNet-1K dataset is expected to be available inside the pod at:
  ```
  /home/ImageNet-1k/
  ```
- **Usage:**  
  All training scripts refer to the dataset path `/home/ImageNet-1k/train` and `/home/ImageNet-1k/val` for training and validation respectively.
- **Advantages of PVC:**
  - Dataset survives across pod restarts or recreations.
  - Multiple pods can access the same dataset volume if needed.
  - No need to re-copy datasets inside containers manually.

## How to Deploy

Apply the Kubernetes Job manifest:

```bash
kubectl apply -f ml_workload_job.yaml
```

This will create a Job named `ml-workload` in the `ml-scheduling` namespace.

## How to Access the Pod

After the pod is created, you can attach into the container:

```bash
kubectl exec -it <pod-name> -n ml-scheduling -- bash
```

(Replace `<pod-name>` with the actual running pod name, e.g., `ml-workload-xxxxx`.)

You will have an interactive shell inside the container.

## Running Experiments

Once inside the pod:
- Navigate to the `/home` directory if needed.
- Ensure that the ImageNet-1k dataset is accessible at `/home/ImageNet-1k/`.
- Launch your distributed experiments manually by running your Python scripts (e.g., `imagenet_distributed_benchmark.py`, `timm_imagenet_distributed_benchmark.py`, etc.).

## Resource Requests

The pod requests:
- 4 GPUs (`nvidia.com/gpu: 4`)
- 20Gi ephemeral storage for both `/home` and `/dev/shm` volumes.

Ensure that the cluster has sufficient available resources before creating the pod.

## Notes

- Environment variables like `NUM_EPOCHS`, `MASTER_ADDR`, and `MASTER_PORT` are configurable inside the YAML.
- The pod runs `sleep infinity` after setup, allowing users to attach at any time without exiting.
- Use `kubectl logs` and `kubectl exec` as needed to monitor and control the workload.

## Folder Structure

```
suny-ibm-multicloud-gpus/
├── kubernetes_workload/
│   ├── ml_workload_job.yaml
│   └── README.md
├── (other files)
```
