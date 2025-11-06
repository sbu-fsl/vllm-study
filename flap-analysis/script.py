# file: script.py
from functools import reduce
import re
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# gloal vars
LOGFILE_PATH = [
    "logs/read_bytes.txt",
    "logs/write_bytes.txt",
    "logs/read_times.txt",
    "logs/write_times.txt",
    "logs/read_counts.txt",
    "logs/write_counts.txt",
]
TOPNQ = 50

# regex patterns
RGX_PTRN = "@(\w+)\[(.+?)\]:\s*(\d+)"

def df_from_logfile(logfile_path: str) -> pd.DataFrame:
    """get a logfile and convert it into a 3-dimention pandas dataframe"""
    print(f'loading file {logfile_path}...')

    data = []
    with open(logfile_path, "r") as file:
        for line in file:
            # input line = "@metric[file]: value"
            # output line = [(metric, file, value)]
            split_line = re.findall(RGX_PTRN, line)
            # this pattern exists once per each line
            if len(split_line) > 0:
                index = split_line[0]

                file_name = index[1]
                metric_name = index[0]
                metric_value = int(index[2])

                # skip unnamed files
                if len(file_name) > 0:
                    data.append({
                        "file": file_name,
                        metric_name: metric_value,
                    })

    # convert to pandas dataframe to perform operations
    return pd.DataFrame(data)

def main():
    # convert all log files to pandas dataframe
    dfs = [df_from_logfile(x) for x in LOGFILE_PATH]
    # merge all dataframes based on file key
    df = reduce(lambda left, right: pd.merge(left, right, on='file'), dfs)

    ######## queries ########
    # top files that where used by read (time)
    top_read_called = df.nlargest(TOPNQ, "read_time")
    top_read_called.to_csv("logs/top_read_times.csv", index=False)
    
    # top files that where used by write (time)
    top_write_called = df.nlargest(TOPNQ, "write_time")
    top_write_called.to_csv("logs/top_write_times.csv", index=False)

    # top large files that where used for read
    top_read_bytes = df.nlargest(TOPNQ, "read_bytes")
    top_read_bytes.to_csv("logs/top_read_bytes.csv", index=False)

    # top large files that where used for write
    top_write_bytes = df.nlargest(TOPNQ, "write_bytes")
    top_write_bytes.to_csv("logs/top_write_bytes.csv", index=False)

    # top files that where used by read (count)
    top_read_counts = df.nlargest(TOPNQ, "read_count")
    top_read_counts.to_csv("logs/top_read_counts.csv", index=False)

    # top files that where used by write (count)
    top_write_counts = df.nlargest(TOPNQ, "write_count")
    top_write_counts.to_csv("logs/top_write_counts.csv", index=False)

    ######## plots ########
    # fig = make_subplots(
    #     rows=4, cols=1,
    #     shared_xaxes=False,
    #     subplot_titles=(
    #         "Accessed files for READ",
    #         "Accessed files for WRITE",
    #         "Used bytes for READ",
    #         "Used bytes for WRITE",
    #     )
    # )

    # fig.add_trace(
    #     go.Bar(
    #         x=top_read_called["file"],
    #         y=top_read_called["read_time"],
    #         marker_color='teal',
    #         hovertemplate="<b>%{x}</b><br>Read ops: %{y:,}<extra></extra>"
    #     ),
    #     row=1, col=1
    # )
    # fig.add_trace(
    #     go.Bar(
    #         x=top_write_called["file"],
    #         y=top_write_called["write_time"],
    #         marker_color='teal',
    #         hovertemplate="<b>%{x}</b><br>Write ops: %{y:,}<extra></extra>"
    #     ),
    #     row=2, col=1
    # )
    # fig.add_trace(
    #     go.Bar(
    #         x=top_read_bytes["file"],
    #         y=top_read_bytes["read_bytes"],
    #         marker_color='steelblue',
    #         hovertemplate="<b>%{x}</b><br>Read bytes: %{y:,}<extra></extra>"
    #     ),
    #     row=3, col=1
    # )
    # fig.add_trace(
    #     go.Bar(
    #         x=top_write_bytes["file"],
    #         y=top_write_bytes["write_bytes"],
    #         marker_color='steelblue',
    #         hovertemplate="<b>%{x}</b><br>Write bytes: %{y:,}<extra></extra>"
    #     ),
    #     row=4, col=1
    # )

    # fig.update_layout(
    #     height=5000,
    #     template="seaborn",
    #     title=f"File Access Patterns (Top {TOPNQ})",
    #     showlegend=False,
    #     xaxis_tickangle=-45,
    #     margin=dict(b=120)
    # )

    # fig.update_yaxes(autorange=True)

    # # export to svg
    # pyo.plot(fig, filename="plots.html", image="svg", output_type="file", image_filename='plots_svg', auto_open=False)

if __name__ == "__main__":
    main()
    sys.exit(0)
