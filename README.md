# vLLM study: I/O perspective

## Models

1. OPT by Facebook
2. Qwen by QwenAI
3. Granite by iBM
4. LlaMa by Meta

> Models larger than 10B don't fit within our GPUs (24 GB limited VRAM space)!

| INX |                 Model                 |                            Access Link                             | Number of Parameters | Type                                                      |
|:---:|:-------------------------------------:|:------------------------------------------------------------------:|:--------------------:|-----------------------------------------------------------|
|1    |          `facebook/opt-125m`          |          [link](https://huggingface.co/facebook/opt-125m)          |         125M         | Text generation (base LLM)                                |
|2    |          `facebook/opt-350m`          |          [link](https://huggingface.co/facebook/opt-350m)          |         350M         | Text generation                                           |
|3    |          `facebook/opt-1.3b`          |          [link](https://huggingface.co/facebook/opt-1.3b)          |          3B          | Text generation                                           |
|4    |          `facebook/opt-6.7b`          |          [link](https://huggingface.co/facebook/opt-6.7b)          |          7B          | Text generation                                           |
|5    |          `facebook/opt-13b`           |          [link](https://huggingface.co/facebook/opt-13b)           |       **13B**        | Text generation                                           |
|6    | `ibm-granite/granite-4.0-h-350m-base` | [link](https://huggingface.co/ibm-granite/granite-4.0-h-350m-base) |         350M         | Text generation                                           |
|7    |    `ibm-granite/granite-4.0-h-1b`     |    [link](https://huggingface.co/ibm-granite/granite-4.0-h-1b)     |          1B          | Text generation / enterprise LLM                          |
|8    |   `ibm-granite/granite-4.0-h-micro`   |   [link](https://huggingface.co/ibm-granite/granite-4.0-h-micro)   |          3B          | Text generation / lightweight assistant                   |
|9    |   `ibm-granite/granite-4.0-h-tiny`    |   [link](https://huggingface.co/ibm-granite/granite-4.0-h-tiny)    |          7B          | Text generation / reasoning                               |
|10   | `ibm-granite/granite-3.3-2b-instruct` | [link](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) |          2B          | Instruction-following assistant                           |
|11   | `ibm-granite/granite-3.3-8b-instruct` | [link](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) |          8B          | Instruction-following assistant                           |
|12   |           `Qwen/Qwen3-0.6B`           |           [link](https://huggingface.co/Qwen/Qwen3-0.6B)           |         800M         | Text generation                                           |
|13   |           `Qwen/Qwen3-1.7B`           |           [link](https://huggingface.co/Qwen/Qwen3-1.7B)           |         1.7B         | Text generation                                           |
|14   |            `Qwen/Qwen3-4B`            |            [link](https://huggingface.co/Qwen/Qwen3-4B)            |          4B          | Text generation / reasoning                               |
|15   |            `Qwen/Qwen3-8B`            |            [link](https://huggingface.co/Qwen/Qwen3-8B)            |          8B          | Text generation / reasoning                               |
|16   |           `Qwen/Qwen3-14B`            |           [link](https://huggingface.co/Qwen/Qwen3-14B)            |       **14B**        | Text generation / reasoning                               |
|17   |           `Qwen/Qwen3-32B`            |           [link](https://huggingface.co/Qwen/Qwen3-32B)            |       **32B**        | Text generation / reasoning                               |
|18   |  `meta-llama/Llama-3.2-1B-Instruct`   |  [link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)   |          1B          | Instruction-following assistant (chat, code, reasoning)   |
|19   |  `meta-llama/Llama-3.2-3B-Instruct`   |  [link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)   |          3B          | Instruction-following assistant (chat, code, reasoning)   |
|20   |  `meta-llama/Llama-3.1-8B-Instruct`   |  [link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)   |          8B          | Instruction-following assistant (chat, code, reasoning)   |

## Parameters for study

- Parallel Config:
  - tensor-parallel-size
- Model Config:
  - model
  - max-model-len
  - quantization
- Load Config:
  - download-dir
  - safetensors-load-strategy
- Cache Config:
  - gpu-memory-utilization
  - swap-space
  - enable-prefix-caching
  - cpu-offload-gb
  - block-size
  - kv-cache-memory-bytes
  - kv-cache-dtype
  - kv-offloading-size
  - kv-offloading-backend
- Compilation Config:
  - cudagraph-capture-sizes
  - max-cudagraph-capture-size

## LLM Benchmarks

Using OpenAI API exported by vLLM to send our requests.

### Datasets

- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
  - A dataset of 52,000 instructions and demonstrations generated by OpenAI's text-davinci-003 engine.
- [LongBench](https://huggingface.co/datasets/zai-org/LongBench)
  - The first benchmark for bilingual, multitask, and comprehensive assessment of long context understanding capabilities of large language models.
- [WMT16](https://huggingface.co/datasets/wmt/wmt16)
  - German to English translation dataset for shared-prefix tasks.
- [ShareGPT](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)
  - ShareGPT-Chinese-English-90k bilingual human-machine QA dataset.

### Tasks

- Single Prompt Single Response
  - Alpaca
- Shared Prefix
  - WMT16
- Chatbot Evaluation
  - ShareGPT
- Question Answering
  - LongBench/NarrativeQA
- Summarization
  - LongBench/QMSum

## Metrics

39 metrics to investigate.

- File System:
    - container_fs_reads_bytes_total{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_fs_writes_bytes_total{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_fs_reads_total{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_fs_writes_total{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_file_descriptors{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
- Disks:
    - container_blkio_device_usage_total{namespace="llm-servings", pod="facebook-opt", operation="Read"}
    - container_blkio_device_usage_total{namespace="llm-servings", pod="facebook-opt", operation="Write"}
- Memory:
    - container_memory_usage_bytes{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_cache{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_mapped_file{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_rss{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_working_set_bytes{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_swap{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_failcnt{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_max_usage_bytes{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_oom_events_total{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_memory_total_active_file_bytes{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
- Network:
    - container_network_receive_bytes_total{namespace="llm-servings", pod="facebook-opt", interface="eth0"}
    - container_network_transmit_bytes_total{namespace="llm-servings", pod="facebook-opt", interface="eth0"}
    - container_network_receive_packets_total{namespace="llm-servings", pod="facebook-opt", interface="eth0"}
    - container_network_transmit_packets_total{namespace="llm-servings", pod="facebook-opt", interface="eth0"}
- CPU:
    - container_processes{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_sockets{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
    - container_threads{namespace="llm-servings", pod="facebook-opt", container="vllm-container"}
- GPU:
    - DCGM_FI_DEV_FB_USED{exported_namespace="llm-servings", exported_pod="facebook-opt"}
    - DCGM_FI_DEV_FB_FREE{exported_namespace="llm-servings", exported_pod="facebook-opt"}
    - DCGM_FI_DEV_MEM_COPY_UTIL{exported_namespace="llm-servings", exported_pod="facebook-opt"}
    - DCGM_FI_PROF_GR_ENGINE_ACTIVE{exported_namespace="llm-servings", exported_pod="facebook-opt"}
    - DCGM_FI_DEV_SM_CLOCK{exported_namespace="llm-servings", exported_pod="facebook-opt"}
    - DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{exported_namespace="llm-servings", exported_pod="facebook-opt"}
- PCIe:
    - DCGM_FI_PROF_PCIE_RX_BYTES{exported_namespace="llm-servings", exported_pod="facebook-opt"}
    - DCGM_FI_PROF_PCIE_TX_BYTES{exported_namespace="llm-servings", exported_pod="facebook-opt"}
- vLLM:
    - vllm:time_to_first_token_seconds_sum{model_name="facebook/opt-125m"}
    - vllm:request_time_per_output_token_seconds_sum{model_name="facebook/opt-125m"}
    - vllm:prefix_cache_hits_total{model_name="facebook/opt-125m"}
    - vllm:e2e_request_latency_seconds_sum{model_name="facebook/opt-125m"}
    - vllm:mm_cache_hits_total{model_name="facebook/opt-125m"}
    - vllm:request_queue_time_seconds_sum{model_name="facebook/opt-125m"}
- vLLM startup latency (using its logs)

## NOTES

### helm

Deploy:

```sh
helm install -f models/qwen/2b.yaml vllm-qwen3-2b models/
```

List deployments:

```sh
helm list
```

Uninstall deployment:

```sh
helm uninstall vllm-qwen3-2b
```

### logs & metrics

List pods:

```sh
kubectl -n llm-servings get pods
```

Export pod logs:

```sh
kubectl -n llm-servings logs $POD_NAME > "$POD_NAME.log"
```

Extract timestamps:

```sh
./scripts/extract_ts.sh "$POD_NAME.log"
```

Get metrics (make sure to edit the values in the `collect_all.sh` script):

```sh
./scripts/metrics/collect_all.sh
```
