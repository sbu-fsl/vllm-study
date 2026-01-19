# Prometheus

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
