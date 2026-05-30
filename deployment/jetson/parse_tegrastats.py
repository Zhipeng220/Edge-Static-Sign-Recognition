import re
import argparse
import pandas as pd
import numpy as np


def parse_tegrastats_line(line):
    row = {}

    # RAM 220/38955MB
    m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if m:
        row["ram_used_mb"] = int(m.group(1))
        row["ram_total_mb"] = int(m.group(2))

    # GR3D_FREQ 12%
    m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if m:
        row["gpu_util_percent"] = int(m.group(1))

    # CPU [1%@1190,off,...]
    cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
    if cpu_match:
        cpu_text = cpu_match.group(1)
        utils = []
        for part in cpu_text.split(","):
            m = re.search(r"(\d+)%@", part)
            if m:
                utils.append(int(m.group(1)))
        if utils:
            row["cpu_util_mean_percent"] = float(np.mean(utils))

    # Power examples:
    # POM_5V_IN 4012/4012
    # VDD_IN 5102mW/5102mW
    power_matches = re.findall(r"([A-Z0-9_]+)\s+(\d+)m?W?/(\d+)m?W?", line)

    for name, now, avg in power_matches:
        if "POM" in name or "VDD" in name or "VIN" in name:
            row[f"{name.lower()}_now_mw"] = int(now)
            row[f"{name.lower()}_avg_mw"] = int(avg)

    return row


def parse_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = parse_tegrastats_line(line)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarize(df):
    summary = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_std"] = df[col].std(ddof=1)
            summary[f"{col}_min"] = df[col].min()
            summary[f"{col}_max"] = df[col].max()

    return pd.DataFrame([summary])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    df = parse_file(args.log)
    summary = summarize(df)

    raw_path = f"{args.out_prefix}_tegrastats_raw.csv"
    summary_path = f"{args.out_prefix}_tegrastats_summary.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(summary.T)


if __name__ == "__main__":
    main()