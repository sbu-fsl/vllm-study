# vLLM study: I/O perspective

## Models

> Models larger than 10B don't fit with our GPUs (24 GB limited VRAM space)!

|                 Model                 |                            Access Link                             | Number of Parameters | Type                                                      |
|:-------------------------------------:|:------------------------------------------------------------------:|:--------------------:|-----------------------------------------------------------|
|          `facebook/opt-125m`          |          [link](https://huggingface.co/facebook/opt-125m)          |         125M         | Text generation (base LLM)                                |
|          `facebook/opt-350m`          |          [link](https://huggingface.co/facebook/opt-350m)          |         350M         | Text generation                                           |
|           `Qwen/Qwen3-0.6B`           |           [link](https://huggingface.co/Qwen/Qwen3-0.6B)           |         800M         | Text generation                                           |
|    `ibm-granite/granite-4.0-h-1b`     |    [link](https://huggingface.co/ibm-granite/granite-4.0-h-1b)     |          1B          | Text generation / enterprise LLM                          |
|    `ibm-granite/granite-4.0-micro`    |    [link](https://huggingface.co/ibm-granite/granite-4.0-micro)    |          3B          | Text generation / lightweight assistant                   |
|           `google/gemma-2b`           |           [link](https://huggingface.co/google/gemma-2b)           |          3B          | Text generation                                           |
|   `ibm-granite/granite-4.0-h-tiny`    |   [link](https://huggingface.co/ibm-granite/granite-4.0-h-tiny)    |          7B          | Text generation / reasoning                               |
|  `meta-llama/Llama-3.1-8B-Instruct`   |  [link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)   |          8B          | Instruction-following assistant (chat, code, reasoning)   |
| `ibm-granite/granite-3.3-8b-instruct` | [link](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) |          8B          | Instruction-following assistant                           |
|            `Qwen/Qwen3-8B`            |            [link](https://huggingface.co/Qwen/Qwen3-8B)            |          8B          | Text generation / reasoning                               |

### Too big

- https://huggingface.co/google/gemma-2-9b

## All Parameters

- Model Config:
    - I/O related
        - From document
            - model
            - tokenizer
            - tokenizer mode
            - hf-config-path
            - allowed-local-media-path
            - allowed-media-domains
            - generation-config
            - max-model-len
        - Needs investigation
            - convert
            - dtype
            - quantization
            - enforce-eager
            - config-format
            - enable-sleep-mode
            - model-impl
    - Not I/O related
        - trust-remote-code
        - seed
        - revisions
        - logprobs
        - sliding-window
        - cascade-attention
        - served-model-name
        - hf-token
        - hf-overrides
        - pooler-config
        - logits-processor-pattern
        - runner
- Load Config:
    - I/O related
        - From document
            - download-dir
            - safetensors-load-strategy
        - Needs investigation
            - load-format
            - ignore-patterns
            - pt-load-map-location
    - Not I/O related
        - model-loader-extra-config
        - use-tqdm-on-load
- Cache Config:
    - I/O related
        - From document
            - gpu-memory-utilization
            - swap-space
            - enable-prefix-caching
            - cpu-offload-gb
        - Needs investigation
            - block-size
            - kv-cache-memory-bytes
            - kv-cache-dtype
            - kv-offloading-size
            - kv-offloading-backend
    - Not I/O related
        - num-gpu-blocks-override
        - prefix-caching-hash-algo
        - calculate-kv-scales
- Compilation Config:
    - I/O related
        - Needs investigation
            - cudagraph-capture-sizes
            - max-cudagraph-capture-size

## Helm

```sh
helm install -f models/facebook/opt-125m.yaml vllm-opt-125m models/
helm install -f models/facebook/opt-350m.yaml vllm-opt-350m models/
helm install -f models/google/gemma-2b.yaml vllm-gemma-2b models/
helm install -f models/ibm-granite/granite-3.3-8b.yaml vllm-granite-33 models/
helm install -f models/ibm-granite/granite-4.0-h-1b.yaml vllm-granite-40-h models/
helm install -f models/ibm-granite/granite-4.0-h-tiny.yaml vllm-granite-40-h-tiny models/
helm install -f models/ibm-granite/granite-4.0-micro.yaml vllm-granite-40-micro models/
helm install -f models/meta-llama/llama-3.1-8b.yaml vllm-meta-llama models/
helm install -f models/qwen/qwen3-0.6b.yaml vllm-qwen3 models/
helm install -f models/qwen/qwen3-8b.yaml vllm-qwen3-8b models/
```

## LLM Benchmarks

6 separate tasks, using 8 different datasets. Using OpenAI API to send our requests. Sampling `1,000` data from these datasets for our benchmarks.

### Datasets

- [MMLU (Massive Multitask Language Understanding)](https://huggingface.co/datasets/cais/mmlu)
    - Benchmark for GPT-scale models; used in many evaluation papers.
- [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
    - By OpenAI.
- [Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions)
    - By Google research, Long-form question answering from real Google search queries + Wikipedia.
- [LooGLE](https://huggingface.co/datasets/bigai-nlco/LooGLE)
    - [Long documents + QA designed to test context >24k tokens.](https://arxiv.org/abs/2311.04939)
- [QMSum](https://huggingface.co/datasets/pszemraj/qmsum-cleaned)
    - Academic/industry collaborations for meeting summarization.
- [OpenChat](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)
    - Cleaned GPT-4-based ShareGPT data used for training OpenChat.
- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
    - A dataset of 52,000 instructions and demonstrations generated by OpenAI's text-davinci-003 engine.
- [LongBench](https://huggingface.co/datasets/zai-org/LongBench)
    - The first benchmark for bilingual, multitask, and comprehensive assessment of long context understanding capabilities of large language models.

### Tasks

- Single Prompt Single Response
    - HumanEval
    - Alpaca
- Beam Search Evaluation
    - Natural Questions
- Shared Prefix
    - LooGLE
    - LongBench
- Chatbot Evaluation
    - OpenChat
- Question Answering
    - MMLU
- Summarization
    - QMSum

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

List deployments:

```sh
helm list
```

Uninstall deployment:

```sh
helm uninstall
```

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
