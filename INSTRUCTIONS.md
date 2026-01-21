# Benchmarks instructions

## vLLM Deployment (using HelmCharts)

1. In `models/` directory, you will find 4 models. In each model's directory there is `01-base.yaml` file that deploys the model as a Kubernetes pod (READ NOTE 1.).
2. Deply a model by running `helm install -f models/facebook-opt/values.yaml vllm-<name> models/`, then you will see a pod in the `llm-servings` namespace. Run `kubectl get pods -n llm-servings` to see it.
3. It might take up to 5 minutes until the pod gets ready. Until the pod is not in the ready status, it won't accept any requests. You musdt wait until you see the `Read 1/1` when running `kubectl get pods -n llm-servings` (READ NOTE 2.).
4. After the pod is ready, export its logs into a file. E.g., run `k logs <pod-name> > logs.txt`.

## Running the benchmarks

5. In the `benchmarks/` directory, you see 6 benchmark scripts, each script reads its own dataset from `datasets` directory, and sends a HTTP request to the OpenAI API exported by the vLLM instance.
6. At this time, there must be two NodePort services in the Kubernetes cluster. The NodePort service allows you to access the OpenAI API exported by your vLLM pod using a port on the running machine (kctl.fsl). Run `k get svc -o wide` to find the port number of a service (READ NOTE 3.).
7. Run benchmarks sequentially and store the logs into a file. E.g., run `python3 benchmarks/spsr.py > spsr_bench.txt`.

## Collecting runtime metrics

8. After finishing the benchmarks, run the `collect_all.sh` script in the `prometheus/` directory. There is a `README.md` in `prometheus/` directory. Read it and tune the flags before running the script. You must get a `.csv` output for each metric in the [METRICS.md](METRICS.md) document.
9. After collecting all metrics as csv files, create a new directory with this format `MODEL__CATEGORY__PARAMETER__VALUE` (e.g., facebook-opt__ModelConfig__runner__pooling).
10. Finally, place all the csv files with the exported logs (from steps 4, 7, and 8) inside the directory and compress it into `.zip` format.
11. Make sure to run `helm uninstall vllm` to free the resources.

## NOTES

1. All secrets, services, and volumes are placed in the `llm-servings` namespace.
2. Create an alias such as `alias k="kubectl -n llm-servings"`, so you don't have to type the whole thing for each command.
3. There are two GPU cards on the node that we deploy our vLLM instances. Every vLLM instance in these experiments needs only one GPU card. Therefore, you can run at most two vLLM instances on the target node. There is a `serviceid` in labels section of models deployment manifest file (`spec.template.labels.serviceid`), which can be `1` or `2`. The Kubernetes services will proxy the traffic to your pods based on this serviceid. If your pod serviceid is 1, then you should select the port number of `vllm-service-1` service, same as for serviceid 2. This is important when you have two running vLLM instances at the same time and you want to benchmark one of them.
