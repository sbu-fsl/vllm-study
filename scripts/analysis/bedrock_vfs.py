import argparse
import os
import re
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt


# regexes for parsing VFS trace logs
# example log line:
# 1234 {pid=0 tid=0 proc=sshd}{vfs_read}{fname=passwd, count=1024, ret=512}
EVENT_RE = re.compile(
	r"^(?P<ts>\d+)\s+\{[^}]*}\{(?P<op>vfs_read|vfs_write)}\{(?P<fields>.*)}$"
)
FIELD_RE = re.compile(r"(fname|count|ret)=([^,}]+)")


def parse_event(line: str) -> Dict[str, any] | None:
	"""Parse a single event line.
	
	Parameters
	----------
	line : str
		A single line from the VFS trace log.

	Returns
	-------
	Dict[str, any] | None
		A dictionary with keys: 'timestamp', 'op', 'fname', 'count', 'ret' if parsing is successful.
		Returns None if the line does not match the expected format or is missing required fields.
	"""

	match = EVENT_RE.match(line.strip())
	if not match:
		return None

	fields_blob = match.group("fields")
	fields = {key: value.strip() for key, value in FIELD_RE.findall(fields_blob)}

	if "fname" not in fields or "count" not in fields or "ret" not in fields:
		return None

	try:
		timestamp = int(match.group("ts"))
		count = int(fields["count"])
		ret = int(fields["ret"])
	except ValueError:
		return None

	op = "read" if match.group("op") == "vfs_read" else "write"
	return {
		"timestamp": timestamp,
		"op": op,
		"fname": fields["fname"],
		"count": count,
		"ret": ret,
	}


def get_extension(fname: str) -> str:
	"""Extract the file extension from a filename.
	
	Parameters
	----------
	fname : str
		The filename from which to extract the extension.

	Returns
	-------
	str
		The file extension in lowercase, including the dot (e.g., '.txt').
		If the filename has no extension, returns the filename itself.
	"""

	_, ext = os.path.splitext(fname)
	return ext.lower() if ext else fname


def load_events(log_path: str) -> List[Dict[str, any]]:
	"""Load and parse events from a VFS trace log file.
	
	Parameters
	----------
	log_path : str
		The path to the VFS trace log file.

	Returns
	-------
	List[Dict[str, any]]
		A list of parsed event dictionaries.
	"""

	events = []
	with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
		for line in handle:
			event = parse_event(line)
			if event is not None:
				events.append(event)
	
	return events


def filter_events_by_percent_range(events: List[Dict[str, any]], start_pct: float, end_pct: float) -> List[Dict[str, any]]:
	"""Filter events to only include those within a specified percentage range of the timeline.
	
	Parameters
	----------
	events : List[Dict[str, any]]
		A list of parsed event dictionaries.
	start_pct : float
		The starting percentage of the timeline.
	end_pct : float
		The ending percentage of the timeline.

	Returns
	-------
	List[Dict[str, any]]
		A list of filtered event dictionaries.
	"""

	if not (0.0 <= start_pct < end_pct <= 100.0):
		raise ValueError("Range must satisfy 0 <= range-start-pct < range-end-pct <= 100")

	min_ts = min(event["timestamp"] for event in events)
	max_ts = max(event["timestamp"] for event in events)
	span = max_ts - min_ts

	if span == 0:
		return list(events)

	start_ts = min_ts + int(span * (start_pct / 100.0))
	end_ts = min_ts + int(span * (end_pct / 100.0))

	if end_pct == 100.0:
		return [event for event in events if start_ts <= event["timestamp"] <= end_ts]

	return [event for event in events if start_ts <= event["timestamp"] < end_ts]


def bucketize(events: List[Dict[str, any]], bucket_seconds: float, base_ts: float | None = None) -> Tuple[List[float], List[int], List[int], List[int], List[int]]:
	"""Aggregate events into time buckets for plotting.
	
	Parameters
	----------
	events : List[Dict[str, any]]
		A list of parsed event dictionaries.
	bucket_seconds : float
		The size of each time bucket in seconds.
	base_ts : float | None
		Optional base timestamp to use as the start of the timeline. If None, the minimum event timestamp will be used.
	
	Returns
	-------
	Tuple[List[float], List[int], List[int], List[int], List[int]]
		A tuple containing:
		- time_axis: List[float] - The relative time for each bucket.
		- read_counts: List[int] - The count of read operations in each bucket.
		- write_counts: List[int] - The count of write operations in each bucket.
		- read_bytes: List[int] - The total bytes read in each bucket.
		- write_bytes: List[int] - The total bytes written in each bucket.
	"""

	first_ts = base_ts if base_ts is not None else min(event["timestamp"] for event in events)
	bucket_ns = int(bucket_seconds * 1e9)
	if bucket_ns <= 0:
		raise ValueError("bucket_seconds must be > 0")

	per_bucket = defaultdict(
		lambda: {
			"read_count": 0,
			"write_count": 0,
			"read_bytes": 0,
			"write_bytes": 0,
		}
	)

	for event in events:
		ret = event["ret"]
		if ret <= 0:
			continue

		bytes_value = min(event["count"], ret)
		bucket_index = (event["timestamp"] - first_ts) // bucket_ns
		bucket = per_bucket[bucket_index]

		if event["op"] == "read":
			bucket["read_count"] += 1
			bucket["read_bytes"] += bytes_value
		else:
			bucket["write_count"] += 1
			bucket["write_bytes"] += bytes_value

	if not per_bucket:
		return [], [], [], [], []

	max_bucket = max(per_bucket)
	time_axis = [index * bucket_seconds for index in range(max_bucket + 1)]
	read_counts = [per_bucket[index]["read_count"] for index in range(max_bucket + 1)]
	write_counts = [per_bucket[index]["write_count"] for index in range(max_bucket + 1)]
	read_bytes = [per_bucket[index]["read_bytes"] for index in range(max_bucket + 1)]
	write_bytes = [per_bucket[index]["write_bytes"] for index in range(max_bucket + 1)]

	return time_axis, read_counts, write_counts, read_bytes, write_bytes


def extension_insights(events: List[Dict[str, any]]) -> List[Tuple[str, int, int, int, int]]:
	"""Aggregate insights by file extension.
	
	Parameters
	----------
	events : List[Dict[str, any]]
		A list of parsed event dictionaries.
	
	Returns
	-------
	List[Tuple[str, int, int, int, int]]
		A list of tuples containing extension insights.
		Each tuple contains:
		- extension: str - The file extension.
		- read_unique_files: int - The count of unique files read with this extension.
		- write_unique_files: int - The count of unique files written with this extension.
		- read_events: int - The total count of read events for this extension.
		- write_events: int - The total count of write events for this extension.
	"""

	by_ext = defaultdict(
		lambda: {
			"read_unique_files": set(),
			"write_unique_files": set(),
			"read_events": 0,
			"write_events": 0,
		}
	)

	for event in events:
		if event["ret"] <= 0:
			continue

		ext = get_extension(event["fname"])
		entry = by_ext[ext]
		if event["op"] == "read":
			entry["read_events"] += 1
			entry["read_unique_files"].add(event["fname"])
		else:
			entry["write_events"] += 1
			entry["write_unique_files"].add(event["fname"])

	rows = []
	for ext, entry in by_ext.items():
		rows.append(
			(
				ext,
				len(entry["read_unique_files"]),
				len(entry["write_unique_files"]),
				entry["read_events"],
				entry["write_events"],
			)
		)

	rows.sort(key=lambda item: (item[1] + item[2], item[3] + item[4]), reverse=True)
	return rows


def print_extension_table(rows, top_n):
	"""Print a table of extension insights."""

	if not rows:
		print("No positive-ret read/write events found for extension insights.")
		return

	display_rows = rows[:top_n] if top_n > 0 else rows

	print("\nExtension insights (positive ret only):")
	print(
		f"{'Extension':<15} {'Read Files':>10} {'Write Files':>11} {'Read Events':>11} {'Write Events':>12}"
	)
	print("-" * 64)
	for ext, read_files, write_files, read_events, write_events in display_rows:
		print(
			f"{ext:<15} {read_files:>10} {write_files:>11} {read_events:>11} {write_events:>12}"
		)


def plot_metrics(time_axis, read_counts, write_counts, read_bytes, write_bytes, output_path, show, use_log_scale=False):
	"""Plot read/write counts and bytes over time.
	
	Parameters
	----------
	time_axis : List[float]
		Time values for x-axis.
	read_counts : List[int]
		Read operation counts.
	write_counts : List[int]
		Write operation counts.
	read_bytes : List[int]
		Read bytes.
	write_bytes : List[int]
		Write bytes.
	output_path : str
		Path to save the output image.
	show : bool
		Whether to display the plot.
	use_log_scale : bool
		Whether to use logarithmic scale for y-axis (default: False).
	"""
	
	plt.style.use("seaborn-v0_8-whitegrid")
	fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

	axes[0].plot(time_axis, read_counts, label="Read Count", linewidth=1.6)
	axes[0].plot(time_axis, write_counts, label="Write Count", linewidth=1.6)
	axes[0].set_title("VFS Read/Write Operation Counts Over Relative Time")
	axes[0].set_ylabel("Operation Count")
	if use_log_scale:
		axes[0].set_yscale("log")
	axes[0].legend()

	axes[1].plot(time_axis, read_bytes, label="Read Bytes", linewidth=1.6)
	axes[1].plot(time_axis, write_bytes, label="Write Bytes", linewidth=1.6)
	axes[1].set_title("VFS Read/Write Bytes Over Relative Time")
	axes[1].set_xlabel("Relative Time (seconds)")
	axes[1].set_ylabel("Bytes")
	if use_log_scale:
		axes[1].set_yscale("log")
	axes[1].legend()

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	print(f"Saved plot: {output_path}")

	if show:
		plt.show()
	else:
		plt.close(fig)


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Parse VFS trace logs and plot read/write counts and bytes over relative time. "
			"Only positive-ret operations are included in metrics."
		)
	)
	parser.add_argument(
		"--logs-dir",
		default="logs",
		help="Path to logs directory (will look for vfs/trace_#.log files inside)",
	)
	parser.add_argument(
		"--bucket-seconds",
		type=float,
		default=1.0,
		help="Time bucket size in seconds for aggregation (default: 1.0)",
	)
	parser.add_argument(
		"--top-ext",
		type=int,
		default=20,
		help="How many extension rows to print (0 for all)",
	)
	parser.add_argument(
		"--output-dir",
		default="logs",
		help="Output directory to store the plot image",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display plot window in addition to saving",
	)
	parser.add_argument(
		"--log-scale",
		action="store_true",
		help="Use logarithmic scale for y-axis values",
	)
	parser.add_argument(
		"--range-start-pct",
		type=float,
		default=0.0,
		help="Start of timeline range as percentage (default: 0)",
	)
	parser.add_argument(
		"--range-end-pct",
		type=float,
		default=100.0,
		help="End of timeline range as percentage (default: 100)",
	)
	args = parser.parse_args()

	# find all trace_#.log files in vfs subdirectory
	vfs_dir = os.path.join(args.logs_dir, "vfs")
	if not os.path.isdir(vfs_dir):
		raise SystemExit(f"VFS directory not found: {vfs_dir}")

	trace_files = sorted(
		[
			os.path.join(vfs_dir, f)
			for f in os.listdir(vfs_dir)
			if re.match(r"trace_\d+\.log$", f)
		],
		key=lambda x: int(re.search(r"trace_(\d+)\.log$", x).group(1))
	)

	if not trace_files:
		raise SystemExit(f"No trace_#.log files found in {vfs_dir}")

	print(f"Found {len(trace_files)} trace files:")
	for f in trace_files:
		print(f"  - {os.path.basename(f)}")

	# load and aggregate events from all trace files
	events = []
	for trace_file in trace_files:
		events.extend(load_events(trace_file))

	if not events:
		raise SystemExit("No parseable VFS read/write events were found in the trace logs.")

	base_ts = min(event["timestamp"] for event in events)
	events = filter_events_by_percent_range(
		events, args.range_start_pct, args.range_end_pct
	)
	if not events:
		raise SystemExit("No events found in the selected percentage range.")

	print(
		f"Using timeline range: {args.range_start_pct:.2f}% to {args.range_end_pct:.2f}%"
	)

	time_axis, read_counts, write_counts, read_bytes, write_bytes = bucketize(
		events, args.bucket_seconds, base_ts=base_ts
	)

	if not time_axis:
		raise SystemExit("No positive-ret read/write events found to plot.")

	os.makedirs(args.output_dir, exist_ok=True)
	output_path = os.path.join(args.output_dir, "vfs_rw_over_time.png")
	plot_metrics(
		time_axis,
		read_counts,
		write_counts,
		read_bytes,
		write_bytes,
		output_path,
		args.show,
		use_log_scale=args.log_scale,
	)

	rows = extension_insights(events)
	print_extension_table(rows, args.top_ext)


if __name__ == "__main__":
	main()
