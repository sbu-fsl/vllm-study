import requests
import csv
import argparse
import datetime


def build_promql(metric, labels):
    if not labels:
        return metric

    label_str = ",".join(
        f'{k}="{v}"' for k, v in labels.items()
    )
    return f'{metric}{{{label_str}}}'


def parse_labels(label_str):
    if not label_str:
        return {}

    labels = {}
    for item in label_str.split(","):
        k, v = item.split("=", 1)
        labels[k.strip()] = v.strip()
    return labels


def fetch_metric(
    prom_url,
    promql,
    start,
    end,
    step
):
    endpoint = f"{prom_url}/api/v1/query_range"
    params = {
        "query": promql,
        "start": start,
        "end": end,
        "step": step,
    }

    r = requests.get(endpoint, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    if data["status"] != "success":
        raise RuntimeError(data)

    return data["data"]["result"]


def write_csv(results, output_file, append=False):
    mode = "a" if append else "w"

    with open(output_file, mode, newline="") as f:
        writer = csv.writer(f)

        if not append:
            writer.writerow([
                "timestamp",
                "value",
                "metric_name",
                "labels"
            ])

        for series in results:
            metric_name = series["metric"].get("__name__", "")

            labels = {
                k: v for k, v in series["metric"].items()
                if k != "__name__"
            }

            for ts, value in series["values"]:
                writer.writerow([
                    datetime.datetime.fromtimestamp(float(ts), datetime.UTC).isoformat(),
                    value,
                    metric_name,
                    labels
                ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:32562")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step", default="5s")
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")

    args = parser.parse_args()

    labels = parse_labels(args.labels)
    promql = build_promql(args.metric, labels)

    results = fetch_metric(
        args.url,
        promql,
        args.start,
        args.end,
        args.step
    )

    write_csv(results, args.output, append=args.append)

    print(f"Saved {len(results)} time-series to {args.output}")


if __name__ == "__main__":
    main()
