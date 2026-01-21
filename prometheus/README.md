# Prometheus

There are two ways to fetch metrics from our Prometheus API.

## One reqeust, one metric

By running `metric_to_csv.py` script, you select a metric with specific labels in a time range and it returns the output as a CSV file.

```sh
python metric_to_csv.py \
  --url http://localhost:9090 \
  --metric node_cpu_seconds_total \
  --labels mode=idle,cpu=0 \
  --start 2026-01-01T00:00:00Z \
  --end 2026-01-01T01:00:00Z \
  --step 1s \
  --output cpu_idle.csv
```

You need to adjust metric, labels, start, end, and output name. The url and step values are the same most of the time. You can find the list of metrics here [METRICS.md](../METRICS.md).

## Collect all

There is a shell script `collect_all.sh`, which runs the `metric_to_csv.py` for all metrics listed in `metrics.list` file. All you have to do is setting the configuration values in the `collect_all.sh` script and it will collect them.

```sh
# metrics labels, must be set to collect your target vLLM pod/container
NAMESPACE="llm-servings"
POD="facebook-opt"
CONTAINER="vllm-container"
MODEL="facebook\/opt-125m"

# time settings, must be set in the benchmark timestamp range (start = the time you deployed the instance, end = the time you finished your benchmarks)
START="2026-01-01T00:00:00Z"
END="2026-01-01T00:00:00Z"

# Prometheus settings stay the same most of the time
PROM_URL="http://localhost:32562"
STEP="1s"

# python/python3
PC="python3"
```
