# utils/log_parser.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Parses experiment log files and summarizes key info like accuracy,
#              errors, and timestamps for quick reviews across runs.

import os
import re
from datetime import datetime
from config import LOGS_DIR

def parse_log_file(filepath):
    """
    Parses a single log file and extracts relevant summary metrics.

    Parameters:
    filepath (str): Full path to the log file

    Returns:
    dict: Summary with timestamp, accuracy (if found), and any error messages
    """
    summary = {"file": os.path.basename(filepath), "timestamp": None, "accuracy": None, "error": None}

    with open(filepath, "r") as f:
        for line in f:
            if not summary["timestamp"]:
                match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if match:
                    summary["timestamp"] = match.group(1)

            if "accuracy" in line.lower():
                acc_match = re.search(r"accuracy[:\s]+([0-9.]+)", line.lower())
                if acc_match:
                    summary["accuracy"] = float(acc_match.group(1))

            if "ERROR" in line or "Exception" in line:
                summary["error"] = line.strip()

    return summary

def summarize_all_logs():
    """
    Summarizes all log files found in the LOGS_DIR.

    Returns:
    list: Sorted summaries by timestamp
    """
    summaries = []
    for fname in os.listdir(LOGS_DIR):
        if fname.endswith(".log"):
            path = os.path.join(LOGS_DIR, fname)
            summaries.append(parse_log_file(path))

    summaries = sorted(summaries, key=lambda x: x["timestamp"] or "")
    return summaries

def print_summary_table(summaries):
    """
    Prints a formatted summary table from parsed logs.

    Parameters:
    summaries (list): Output from summarize_all_logs()
    """
    print("\nExperiment Log Summary")
    print("-" * 60)
    for entry in summaries:
        acc_display = f"{entry['accuracy']:.4f}" if entry["accuracy"] is not None else "N/A"
        error_display = entry["error"] if entry["error"] else ""
        print(f"{entry['timestamp']} | {entry['file']} | Accuracy: {acc_display} {error_display}")
    print("-" * 60)

if __name__ == "__main__":
    summaries = summarize_all_logs()
    print_summary_table(summaries)
