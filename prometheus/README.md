# Prometheus

By running this script, you can select a metric, with specific labels, in a time range and
get the output as a CSV file.

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

You need to adjust metric, labels, start, end, and output name. The url and step values are the same most of the time. You can find the list of the prometheus metrics in `../METRICS.md`.
