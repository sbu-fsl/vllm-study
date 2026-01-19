# Instructions

1. In `models/` directory, you will find 4 models. In each model's directory there is `01-base.yaml` file that deploys the model as a Kubernetes job (READ NOTE 1.).
2. By running `kubectl apply -f 01-base.yaml` you will see a pod in the `llm-servings` namespace. Run `kubectl get pods -n llm-servings` to see it.
3. It might take 2 to 5 minutes until the pod gets ready. Until the pod is not in the ready status, it won't accept any input requests.
4. After the pod gets ready, export its logs into a text file for future analysis. Run `kubectl logs -n llm-servings <pod-name>`.
5. There are two public services in the Kubernetes cluster that allows you to access the OpenAI API exported by your vLLM pod. Run `kubectl get svc -n llm-servings -o wide` to find its port number. You can access that service using `localhost:<port-number>` (READ NOTE 2.).
6. In `benchmarks/` directory, you see a benchmark script for each of the tasks that we have.
7. Run each benchmark sequentially and keep the timestamp of your execution time for each benchmark. You will need that timestamps to get and analysie the runtime metrics.
8. After executing your benchmarks, run the `metric_to_csv.py` script in the `prometheus/` directory. There is a `README.md` in `prometheus/` directory. Make sure to set the flags before running the script.
9. After collecting all metrics, create a new sub-directory in `results/` with this format `MODEL__PARAMETER__VALUE` (e.g., facebook-opt__ModelConfig/runner__pooling). Then, place all the csv files with the vLLM exported logs (from steps 4 and 8) inside it.

## NOTES

1. All services and volumes are created in the `llm-servings` namespace.
2. Since there are two GPU cards on the target node, there is a `serviceid` in the `01-base.yaml` file which can be `1` or `2` (can change it `spec.template.labels.serviceid`). The Kubernetes services will proxy the traffic to your pods based on this serviceid. If your pod serviceid is 1, then you should select the port number of `vllm-service-1` service. We have this serviceid to enable running at most two vLLM instances at the same time.
