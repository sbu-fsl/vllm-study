#!/usr/bin/env python3
"""
Sample up to 1,000 items from benchmark datasets and save as CSV.

Usage:
    python sample_datasets.py -d all -o ./sampled_datasets
    python sample_datasets.py -d alpaca sharegpt -o ./sampled_datasets
    python sample_datasets.py -d wmt16 -o ./out -n 500 -s 99
    python sample_datasets.py --list

Each run writes a manifest.json alongside the CSVs with checksums
so repeated sampling runs can be diffed / deduplicated.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "alpaca": {
        "hf_path": "tatsu-lab/alpaca",
        "description": "Alpaca — 52K instruction-following demos (OpenAI text-davinci-003)",
        "split": "train",
    },
    "longbench": {
        "hf_path": "THUDM/LongBench",
        "description": "LongBench — bilingual long-context benchmark (21 subsets combined)",
        "loader": "load_longbench",
    },
    "wmt16": {
        "hf_path": "wmt/wmt16",
        "description": "WMT16 de-en — German to English translation",
        "subset": "de-en",
        "split": "train",
        "streaming": True,
    },
    "sharegpt": {
        "hf_path": "shareAI/ShareGPT-Chinese-English-90k",
        "description": "ShareGPT Chinese-English 90k bilingual QA",
        "loader": "load_sharegpt",
    },
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten(val):
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, default=str)
    return val


def sample_rows(rows, n, seed):
    total = len(rows)
    k = min(n, total)
    if k < total:
        random.seed(seed)
        indices = sorted(random.sample(range(total), k))
        rows = [rows[i] for i in indices]
    return rows, total


def sample_hf_dataset(dataset, n, seed):
    total = len(dataset)
    k = min(n, total)
    if k < total:
        random.seed(seed)
        indices = sorted(random.sample(range(total), k))
        dataset = dataset.select(indices)
    return [dict(row) for row in dataset], total


def save_csv(rows, name, output_dir):
    import pandas as pd
    flat = [{k: flatten(v) for k, v in row.items()} for row in rows]
    path = output_dir / f"{name}.csv"
    pd.DataFrame(flat).to_csv(path, index=False, encoding="utf-8")
    return path


def load_standard(cfg, n, seed):
    from datasets import load_dataset
    kwargs = {"path": cfg["hf_path"]}
    if "subset" in cfg:
        kwargs["name"] = cfg["subset"]
    if "split" in cfg:
        kwargs["split"] = cfg["split"]

    if cfg.get("streaming"):
        kwargs["streaming"] = True
        ds = load_dataset(**kwargs)
        rows = []
        for i, item in enumerate(ds):
            rows.append(dict(item))
            if i + 1 >= n:
                break
        return rows, -1

    ds = load_dataset(**kwargs)
    return sample_hf_dataset(ds, n, seed)


def load_longbench(cfg, n, seed):
    import tempfile
    import zipfile
    from huggingface_hub import hf_hub_download

    print("  downloading data.zip from THUDM/LongBench ...")
    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )
    keep = {"narrativeqa", "qmsum"}
    all_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        data_dir = Path(tmp) / "data"
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            subset_name = jsonl_file.stem
            if subset_name not in keep:
                print(f"  skipping {subset_name} ...")
                continue
            with open(jsonl_file, "r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    row = json.loads(line)
                    row["longbench_subset"] = subset_name
                    all_rows.append(row)
                    count += 1
            print(f"    {subset_name}: {count} rows")

    if not all_rows:
        raise RuntimeError("No data found in LongBench data.zip")

    return sample_rows(all_rows, n, seed)


def load_sharegpt(cfg, n, seed):
    from datasets import load_dataset

    files = [
        "sharegpt_jsonl/common_en_70k.jsonl",
        "sharegpt_jsonl/common_zh_70k.jsonl",
    ]

    all_rows = []
    for f in files:
        print(f"  loading {f} ...")
        ds = load_dataset(
            "shareAI/ShareGPT-Chinese-English-90k",
            data_files=f,
            split="train",
        )
        all_rows.extend([dict(row) for row in ds])
        print(f"    {f}: {len(ds)} rows")

    return sample_rows(all_rows, n, seed)


CUSTOM_LOADERS = {
    "load_longbench": load_longbench,
    "load_sharegpt": load_sharegpt,
}


# ---------------------------------------------------------------------------
# Version tag
# ---------------------------------------------------------------------------
def version_tag(seed, n, names):
    payload = json.dumps({"seed": seed, "n": n, "datasets": sorted(names)}, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"s{seed}_n{n}_{h}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Sample benchmark datasets to CSV.")
    p.add_argument("-d", "--dataset", nargs="+", metavar="NAME",
                   help='Dataset name(s) or "all".  Use --list to see options.')
    p.add_argument("-o", "--output-dir", help="Output directory for CSVs.")
    p.add_argument("-n", "--num-samples", type=int, default=1000, help="Samples per dataset (default: 1000).")
    p.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--list", action="store_true", dest="list_datasets", help="Show available datasets and exit.")
    p.add_argument("--dry-run", action="store_true", help="Show plan without downloading.")
    args = p.parse_args()

    if args.list_datasets:
        for name, cfg in DATASETS.items():
            print(f"  {name:<14} {cfg['hf_path']:<50} {cfg['description']}")
        return

    if not args.dataset:
        p.error("--dataset is required (or use --list)")
    if not args.output_dir:
        p.error("--output-dir is required")

    # resolve names
    if "all" in args.dataset:
        names = list(DATASETS.keys())
    else:
        names = []
        for n in args.dataset:
            key = n.lower().replace("-", "_")
            if key not in DATASETS:
                p.error(f"Unknown dataset '{n}'. Available: {', '.join(DATASETS.keys())}")
            names.append(key)

    tag = version_tag(args.seed, args.num_samples, names)

    if args.dry_run:
        print(f"Would sample {args.num_samples} items (seed={args.seed}) from: {', '.join(names)}")
        print(f"Output: {args.output_dir}   Tag: {tag}")
        return

    # deps
    try:
        import datasets, pandas  # noqa: F401
    except ImportError:
        print("Install dependencies: pip install datasets pandas", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Tag: {tag}  |  Seed: {args.seed}  |  Samples: {args.num_samples}")
    print(f"Output: {out.resolve()}\n")

    results = []
    for name in names:
        cfg = DATASETS[name]
        print(f"[{name}] {cfg['description']}")
        t0 = time.monotonic()
        try:
            loader_name = cfg.get("loader")
            if loader_name:
                loader = CUSTOM_LOADERS[loader_name]
            else:
                loader = load_standard
            rows, total = loader(cfg, args.num_samples, args.seed)
            csv_path = save_csv(rows, name, out)
            checksum = sha256_file(csv_path)
            elapsed = round(time.monotonic() - t0, 1)
            size_mb = csv_path.stat().st_size / (1 << 20)
            total_str = str(total) if total >= 0 else "streamed"
            print(f"  ✓ {len(rows)}/{total_str} rows  {size_mb:.1f} MB  sha256:{checksum[:16]}…  ({elapsed}s)\n")
            results.append({
                "name": name, "status": "ok", "sampled": len(rows), "total": total,
                "file": str(csv_path), "sha256": checksum, "size_bytes": csv_path.stat().st_size,
            })
        except Exception as e:
            elapsed = round(time.monotonic() - t0, 1)
            print(f"  ✗ FAILED: {e}  ({elapsed}s)\n")
            results.append({"name": name, "status": "error", "error": str(e)})

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        # update existing results, keep ones not in this run
        existing = {r["name"]: r for r in manifest.get("results", [])}
        for r in results:
            existing[r["name"]] = r
        manifest["results"] = list(existing.values())
        manifest["version_tag"] = tag
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest["seed"] = args.seed
        manifest["num_samples"] = args.num_samples
    else:
        manifest = {
            "version_tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "num_samples": args.num_samples,
            "results": results,
        }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]
    print(f"Done: {len(ok)}/{len(results)} succeeded  |  Tag: {tag}  |  Manifest: {manifest_path}")
    if fail:
        print(f"Failed: {', '.join(r['name'] for r in fail)}")
        sys.exit(1)


if __name__ == "__main__":
    main()