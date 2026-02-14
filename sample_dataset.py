import argparse
import json
import os
import random
import sys
from pathlib import Path


def flatten_value(val):
    """Flatten complex nested values into strings for CSV compatibility."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, default=str)
    return val


def sample_and_save_csv(dataset, name, num_samples, output_dir, seed):
    """Sample from a HuggingFace dataset and save as CSV."""
    import pandas as pd

    total = len(dataset)
    n = min(num_samples, total)

    if n < total:
        random.seed(seed)
        indices = sorted(random.sample(range(total), n))
        sampled = dataset.select(indices)
    else:
        sampled = dataset

    # Convert to list of dicts and flatten any nested structures
    rows = []
    for item in sampled:
        row = {k: flatten_value(v) for k, v in item.items()}
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"  ✓ {name}: sampled {n}/{total} items -> {output_path}")
    print(f"    columns: {list(df.columns)}")
    return n


def streaming_save_csv(ds_stream, name, num_samples, output_dir):
    """Save from a streaming dataset to CSV."""
    import pandas as pd

    items = []
    for i, item in enumerate(ds_stream):
        row = {k: flatten_value(v) for k, v in item.items()}
        items.append(row)
        if i + 1 >= num_samples:
            break

    df = pd.DataFrame(items)
    output_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"  ✓ {name} (streaming): saved {len(items)} items -> {output_path}")
    print(f"    columns: {list(df.columns)}")
    return len(items)


def main():
    parser = argparse.ArgumentParser(description="Sample benchmark datasets for vllm-study")
    parser.add_argument("--output-dir", type=str, default="./sampled_datasets",
                        help="Output directory for sampled datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples per dataset")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install with: pip install datasets")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: 'pandas' package not found. Install with: pip install pandas")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    datasets_config = [
        {
            "name": "mmlu",
            "path": "cais/mmlu",
            "description": "MMLU - Massive Multitask Language Understanding",
            "split": "test",
            "subset": "all",
        },
        {
            "name": "humaneval",
            "path": "openai/openai_humaneval",
            "description": "HumanEval - Code generation benchmark by OpenAI",
            "split": "test",
            "subset": None,
        },
        {
            "name": "natural_questions",
            "path": "google-research-datasets/natural_questions",
            "description": "Natural Questions - Google research QA",
            "split": "validation",
            "subset": "default",
        },
        {
            "name": "loogle",
            "path": "bigai-nlco/LooGLE",
            "description": "LooGLE - Long context QA (>24k tokens)",
            "split": "test",
            "subset": None,
        },
        {
            "name": "qmsum",
            "path": "pszemraj/qmsum-cleaned",
            "description": "QMSum - Meeting summarization",
            "split": "train",
            "subset": None,
        },
        {
            "name": "openchat",
            "path": "openchat/openchat_sharegpt4_dataset",
            "description": "OpenChat - GPT-4-based ShareGPT data",
            "split": "train",
            "subset": None,
        },
        {
            "name": "alpaca",
            "path": "tatsu-lab/alpaca",
            "description": "Alpaca - 52K instruction-following dataset",
            "split": "train",
            "subset": None,
        },
        {
            "name": "longbench",
            "path": "zai-org/LongBench",
            "description": "LongBench - Bilingual long context benchmark",
            "split": "test",
            "subset": None,
        },
    ]

    print(f"Sampling {args.num_samples} items per dataset (seed={args.seed})")
    print(f"Output directory: {args.output_dir}")
    print(f"Output format: CSV\n")

    results = {}
    for cfg in datasets_config:
        name = cfg["name"]
        print(f"[{name}] Loading {cfg['path']}... ({cfg['description']})")
        try:
            load_kwargs = {"path": cfg["path"], "trust_remote_code": True}
            if cfg.get("subset"):
                load_kwargs["name"] = cfg["subset"]
            if cfg.get("split"):
                load_kwargs["split"] = cfg["split"]

            ds = load_dataset(**load_kwargs)
            n = sample_and_save_csv(ds, name, args.num_samples, args.output_dir, args.seed)
            results[name] = {"status": "ok", "sampled": n, "total": len(ds)}

        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            results[name] = {"status": "error", "error": str(e)}

            # Fallback: streaming for large datasets like Natural Questions
            if name == "natural_questions":
                print(f"  → Trying streaming mode...")
                try:
                    load_kwargs_stream = {"path": cfg["path"], "trust_remote_code": True, "streaming": True}
                    if cfg.get("subset"):
                        load_kwargs_stream["name"] = cfg["subset"]
                    if cfg.get("split"):
                        load_kwargs_stream["split"] = cfg["split"]

                    ds_stream = load_dataset(**load_kwargs_stream)
                    n = streaming_save_csv(ds_stream, name, args.num_samples, args.output_dir)
                    results[name] = {"status": "ok (streaming)", "sampled": n}
                except Exception as e2:
                    print(f"  ✗ {name} (streaming fallback): FAILED - {e2}")
                    results[name] = {"status": "error", "error": str(e2)}

            # Fallback: try alternative splits
            elif name in ("loogle", "longbench"):
                for alt_split in ["test", "train", "validation"]:
                    try:
                        ds = load_dataset(cfg["path"], split=alt_split, trust_remote_code=True)
                        n = sample_and_save_csv(ds, name, args.num_samples, args.output_dir, args.seed)
                        results[name] = {"status": "ok", "sampled": n, "total": len(ds), "split": alt_split}
                        break
                    except Exception:
                        continue

    # Save summary
    summary_path = os.path.join(args.output_dir, "sampling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, info in results.items():
        status = info.get("status", "unknown")
        if "sampled" in info:
            total_str = f"/ {info['total']}" if "total" in info else ""
            print(f"  {name:25s} {status:15s} {info['sampled']} {total_str} samples")
        else:
            print(f"  {name:25s} {status:15s} (error: {info.get('error', 'unknown')[:60]})")
    print(f"\nOutput files:")
    for f in sorted(Path(args.output_dir).glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:35s} {size_mb:8.2f} MB")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()