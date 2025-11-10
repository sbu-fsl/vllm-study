#!/usr/bin/env python3
import fnmatch
import re
import sys
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
import plotly.graph_objects as go

# Usage:
#   ./plot_traces_prefix_timeline_interactive.py <logfile> [--levels N]

if len(sys.argv) < 2:
    print("Usage: ./plot_traces_prefix_timeline_interactive.py <logfile> [--levels N]")
    sys.exit(1)

log_file = Path(sys.argv[1])
levels = 3

if "--levels" in sys.argv:
    idx = sys.argv.index("--levels")
    if idx + 1 < len(sys.argv):
        levels = int(sys.argv[idx + 1])

pattern = re.compile(
    r'^\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?\bfname=(?P<file>[^\s]+)'
)
ex_patterns = [
    "/cpu*", "/vulnerabilities*", "/bus*", "/dev*", "/etc*", "/__pycache__*",
    "/proc*", "/sys*", "/self*", "/sbin*", "/dispatching*", "/kernel*"
]

def get_prefix(path: str, levels: int = 3) -> str:
    parts = path.strip().split('/')
    if parts and parts[0] == '':
        parts = parts[1:]
    return "/" + "/".join(parts[:levels]) if parts else path


### CACHE START ###
def compute_cache_key(file_path, levels, ex_patterns):
    """Compute unique hash based on input file, levels, and excluded patterns."""
    h = hashlib.sha256()
    h.update(str(file_path).encode())
    h.update(str(levels).encode())
    h.update(json.dumps(ex_patterns, sort_keys=True).encode())
    # optionally include file modification time to invalidate cache if file changed
    h.update(str(file_path.stat().st_mtime).encode())
    return h.hexdigest()

cache_key = compute_cache_key(log_file, levels, ex_patterns)
cache_file = log_file.with_suffix(f".{cache_key[:10]}.cache")

prefix_times = None
if cache_file.exists():
    try:
        with cache_file.open("rb") as cf:
            prefix_times = pickle.load(cf)
        print(f"[cache] Loaded cached data from {cache_file.name}")
    except Exception:
        prefix_times = None
### CACHE END ###


if prefix_times is None:
    print("[info] Parsing log file...")
    prefix_times = defaultdict(list)

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            t = datetime.strptime(m.group("time"), "%Y-%m-%d %H:%M:%S")
            t = t.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
            prefix = get_prefix(m.group("file"), levels)
            if any(fnmatch.fnmatch(prefix, p) for p in ex_patterns):
                continue
            prefix_times[prefix].append(t)

    # save cache
    try:
        with cache_file.open("wb") as cf:
            pickle.dump(prefix_times, cf)
        print(f"[cache] Saved parsed data to {cache_file.name}")
    except Exception as e:
        print(f"[warn] Failed to save cache: {e}")

if not prefix_times:
    print("No matching lines found.")
    sys.exit(0)

# build interactive scatter plot
fig = go.Figure()

# compute min/max for normalization
counts = [len(v) for v in prefix_times.values()]
min_count, max_count = min(counts), max(counts)

def scale_size(count, min_size=6, max_size=30):
    if max_count == min_count:
        return (min_size + max_size) / 2
    return min_size + (count - min_count) / (max_count - min_count) * (max_size - min_size)

for i, (prefix, times) in enumerate(sorted(prefix_times.items())):
    times = sorted(times)
    count = len(times)
    size = scale_size(count)

    fig.add_trace(
        go.Scattergl(
            x=[times[0]] if times else [],
            y=[prefix] if times else [],
            mode="markers",
            marker=dict(size=size, opacity=0.7),
            name=f"{prefix} ({count})",
            hovertemplate="<b>%{y}</b><br>Count: "+str(count)+"<extra></extra>"
        )
    )

# define event times
DATE = "2025-11-10"

def to_dt(t):
    return datetime.strptime(f"{DATE} {t}", "%Y-%m-%d %H:%M:%S")

events = {
    "tracing start": to_dt("09:39:16"),
    "container start": to_dt("09:39:22"),
    "config phase start": to_dt("09:39:22"),
    "config phase end": to_dt("09:39:55"),
    "loading phase start": to_dt("09:39:55"),
    "loading phase end": to_dt("09:42:13"),
    "compile phase start": to_dt("09:42:19"),
    "compile phase end": to_dt("09:43:04"),
    "memory profiling start": to_dt("09:43:05"),
    "memory profiling end": to_dt("09:43:06"),
    "graph capture start": to_dt("09:43:06"),
    "graph capture end": to_dt("09:43:17"),
    "vLLM ready": to_dt("09:43:19"),
    "tracing stopped": to_dt("09:46:40"),
}

# add shaded and vertical events (unchanged)
fig.add_vrect(
    x0=events["config phase start"], x1=events["config phase end"],
    fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0,
    annotation_text="Config Phase", annotation_position="top left"
)
fig.add_vrect(
    x0=events["loading phase start"], x1=events["loading phase end"],
    fillcolor="LightSkyBlue", opacity=0.3, layer="below", line_width=0,
    annotation_text="Loading Phase", annotation_position="top left"
)
fig.add_vrect(
    x0=events["compile phase start"], x1=events["compile phase end"],
    fillcolor="LightGreen", opacity=0.3, layer="below", line_width=0,
    annotation_text="Compile Phase", annotation_position="top left"
)
fig.add_vrect(
    x0=events["memory profiling start"], x1=events["memory profiling end"],
    fillcolor="LightGray", opacity=0.3, layer="below", line_width=0,
    annotation_text="Memory Profiling", annotation_position="top left"
)
fig.add_vrect(
    x0=events["graph capture start"], x1=events["graph capture end"],
    fillcolor="Khaki", opacity=0.3, layer="below", line_width=0,
    annotation_text="Graph Capture", annotation_position="top left"
)

for label in ["tracing start", "container start", "vLLM ready", "tracing stopped"]:
    t = events[label]
    fig.add_vline(x=t, line_dash="dot", line_color="black", opacity=0.8)
    fig.add_annotation(
        x=t,
        y=1.05,
        xref="x",
        yref="paper",
        text=label.replace("_", " ").title(),
        showarrow=False,
        font=dict(size=10, color="black"),
        align="center"
    )

fig.update_layout(
    title="File Access Timeline by Prefix (per-second resolution)",
    xaxis_title="Time (seconds)",
    yaxis_title="Path Prefix (first N levels)",
    height=1000,
    template="plotly_white",
    legend=dict(
        title="Path Prefixes",
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=1.02
    ),
    margin=dict(r=200)
)

fig.update_xaxes(
    tickformat="%H:%M:%S",
    rangeslider_visible=True,
    showgrid=True
)

fig.show()
