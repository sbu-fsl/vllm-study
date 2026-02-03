# Helm

```sh
helm install -f models/facebook-opt/opt-125m.yaml vllm-opt models/
helm install -f models/google/gemma-2-9b.yaml vllm-gemma-2-9b models/
helm install -f models/google/gemma-2b.yaml vllm-gemma-2b models/
helm install -f models/google/recurrentgemma-9b.yaml vllm-recurrentgemma models/
helm install -f models/ibm-granite/granite-3.3-8b.yaml vllm-granite-3.3 models/
helm install -f models/ibm-granite/granite-4.0-h-1b.yaml vllm-granite-4.0-h models/
helm install -f models/ibm-granite/granite-4.0-h-tiny.yaml vllm-granite-4.0-h-tiny models/
helm install -f models/ibm-granite/granite-4.0-micro.yaml vllm-granite-4.0-micro models/
helm install -f models/meta-llama/llama-3.1-8b.yaml vllm-meta-llama models/
helm install -f models/openai/circuit.yaml vllm-openai-circuit models/
helm install -f models/qwen/qwen3-0.6b.yaml vllm-qwen3 models/
helm install -f models/qwen/qwen3-8b.yaml vllm-qwen3-8b models/
```
