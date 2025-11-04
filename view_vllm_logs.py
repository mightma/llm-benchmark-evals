#!/usr/bin/env python3
"""
VLLM Log Viewer Script

A standalone script for viewing VLLM server logs without running the full monitor.
"""

import os
import argparse
import sys
from datetime import datetime


def find_latest_logs():
    """Find the most recent VLLM log files."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return None, None

    # Find stdout logs
    stdout_logs = sorted([f for f in os.listdir(logs_dir)
                         if f.startswith("vllm_server") and not f.endswith("_error.log")])

    # Find stderr logs
    stderr_logs = sorted([f for f in os.listdir(logs_dir)
                         if f.startswith("vllm_server_error")])

    latest_stdout = os.path.join(logs_dir, stdout_logs[-1]) if stdout_logs else None
    latest_stderr = os.path.join(logs_dir, stderr_logs[-1]) if stderr_logs else None

    return latest_stdout, latest_stderr


def show_logs(log_file, lines=None, follow=False, log_type="STDOUT"):
    """Display log file content."""
    if not log_file or not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return

    print(f"📄 {log_type} Log: {log_file}")
    print("=" * 80)

    try:
        if follow:
            # Follow mode (like tail -f)
            import time
            with open(log_file, 'r', encoding='utf-8') as f:
                # Go to end of file
                f.seek(0, 2)
                print("👀 Following log file... (Press Ctrl+C to stop)")

                try:
                    while True:
                        line = f.readline()
                        if line:
                            print(line.rstrip())
                        else:
                            time.sleep(1)
                except KeyboardInterrupt:
                    print("\n👋 Stopped following log file.")
        else:
            # Show specific number of lines
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

                if lines and lines > 0:
                    # Show last N lines
                    display_lines = all_lines[-lines:]
                    if len(all_lines) > lines:
                        print(f"... (showing last {lines} of {len(all_lines)} lines)")
                else:
                    # Show all lines
                    display_lines = all_lines

                for line in display_lines:
                    print(line.rstrip())

        print(f"\n📊 File info: {len(open(log_file).readlines())} total lines")

    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def list_all_logs():
    """List all available VLLM log files."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print("ℹ️  No logs directory found")
        return

    log_files = [f for f in os.listdir(logs_dir) if f.startswith("vllm_server")]

    if not log_files:
        print("ℹ️  No VLLM log files found")
        return

    print("📁 Available VLLM log files:")
    print("=" * 60)

    for log_file in sorted(log_files):
        path = os.path.join(logs_dir, log_file)
        size = os.path.getsize(path)
        mtime = datetime.fromtimestamp(os.path.getmtime(path))

        log_type = "ERROR" if "_error" in log_file else "STDOUT"
        size_mb = size / (1024 * 1024)

        print(f"📄 {log_file}")
        print(f"   Type: {log_type} | Size: {size_mb:.2f}MB | Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


def search_logs(pattern, case_sensitive=False):
    """Search for pattern in all log files."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print("ℹ️  No logs directory found")
        return

    log_files = [f for f in os.listdir(logs_dir) if f.startswith("vllm_server")]

    if not log_files:
        print("ℹ️  No VLLM log files found")
        return

    found_matches = False
    search_pattern = pattern if case_sensitive else pattern.lower()

    print(f"🔍 Searching for: '{pattern}' (case {'sensitive' if case_sensitive else 'insensitive'})")
    print("=" * 80)

    for log_file in sorted(log_files):
        path = os.path.join(logs_dir, log_file)
        matches = []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    search_line = line if case_sensitive else line.lower()
                    if search_pattern in search_line:
                        matches.append((line_num, line.rstrip()))
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
            continue

        if matches:
            found_matches = True
            print(f"\n📄 {log_file} ({len(matches)} matches):")
            print("-" * 50)
            for line_num, line in matches[-10:]:  # Show last 10 matches
                print(f"{line_num:6d}: {line}")

    if not found_matches:
        print(f"❌ No matches found for: '{pattern}'")


def main():
    parser = argparse.ArgumentParser(description="View VLLM server logs")
    parser.add_argument("--lines", "-n", type=int, default=50,
                       help="Number of lines to show (default: 50, 0 for all)")
    parser.add_argument("--follow", "-f", action="store_true",
                       help="Follow log file (like tail -f)")
    parser.add_argument("--errors", "-e", action="store_true",
                       help="Show error logs instead of stdout")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available log files")
    parser.add_argument("--search", "-s", type=str,
                       help="Search for pattern in all log files")
    parser.add_argument("--case-sensitive", action="store_true",
                       help="Case sensitive search")
    parser.add_argument("--file", type=str,
                       help="Specific log file to view")

    args = parser.parse_args()

    if args.list:
        list_all_logs()
        return

    if args.search:
        search_logs(args.search, args.case_sensitive)
        return

    if args.file:
        log_file = args.file
        if not os.path.isabs(log_file):
            log_file = os.path.join("logs", log_file)
        log_type = "ERROR" if "_error" in log_file else "STDOUT"
        show_logs(log_file, args.lines, args.follow, log_type)
        return

    # Default: show latest logs
    stdout_log, stderr_log = find_latest_logs()

    if args.errors:
        if stderr_log:
            show_logs(stderr_log, args.lines, args.follow, "ERROR")
        else:
            print("❌ No error log files found")
    else:
        if stdout_log:
            show_logs(stdout_log, args.lines, args.follow, "STDOUT")
        else:
            print("❌ No stdout log files found")

        # Also show recent errors if available
        if stderr_log and not args.follow:
            print("\n" + "=" * 80)
            show_logs(stderr_log, 10, False, "RECENT ERRORS")


if __name__ == "__main__":
    main()