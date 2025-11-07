#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from datetime import datetime
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

def get_prefix(path: str, levels: int = 3) -> str:
    parts = path.strip().split('/')
    if parts and parts[0] == '':
        parts = parts[1:]
    return "/" + "/".join(parts[:levels]) if parts else path

# parse file accesses
prefix_times = defaultdict(list)

with log_file.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        t = datetime.strptime(m.group("time"), "%Y-%m-%d %H:%M:%S")
        prefix = get_prefix(m.group("file"), levels)
        prefix_times[prefix].append(t)

if not prefix_times:
    print("No matching lines found.")
    sys.exit(0)

# build interactive scatter plot
fig = go.Figure()

# compute min/max for normalization
counts = [len(v) for v in prefix_times.values()]
min_count, max_count = min(counts), max(counts)

def scale_size(count, min_size=6, max_size=30):
    # linearly scale count to marker size range
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

fig.update_layout(
    title=f"File Access Timeline by Prefix (per-second resolution)",
    xaxis_title="Time (seconds)",
    yaxis_title=f"Path Prefix (first {levels} levels)",
    height=700,
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
