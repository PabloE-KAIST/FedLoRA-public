#!/usr/bin/env python3
"""Merge distributed FL server + worker logs into a unified timeline.

Worker containers run in UTC; the server runs in KST (UTC+9).
This script normalises all timestamps to the server's local timezone
and interleaves lines chronologically, prefixed with the source tag.

Usage:
    python3 merge_logs.py <server_log> <worker_log_dir> > unified.log
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
UTC = timezone.utc

# Matches "2026-04-25 03:15:03,025" at the start of a line (with optional
# ANSI escape codes that the FederatedScope logger emits).
TS_RE = re.compile(
    r'^(?:\x1b\[[0-9;]*m)*'           # optional ANSI prefix
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
)


def parse_ts(line: str) -> datetime | None:
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def read_tagged(path: str, tag: str, tz: timezone):
    """Yield (datetime_kst, tagged_line) for every line in *path*."""
    last_ts = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            ts = parse_ts(line)
            if ts is not None:
                ts = ts.replace(tzinfo=tz).astimezone(KST)
                last_ts = ts
            tagged = f"[{tag}] {line}"
            yield (last_ts or datetime.min.replace(tzinfo=KST), tagged)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <server_log> <worker_log_dir>",
              file=sys.stderr)
        sys.exit(1)

    server_log = sys.argv[1]
    worker_dir = sys.argv[2]

    entries: list[tuple[datetime, str]] = []

    if os.path.isfile(server_log):
        entries.extend(read_tagged(server_log, "server", KST))

    for fname in sorted(os.listdir(worker_dir)):
        if not fname.endswith(".log"):
            continue
        tag = fname.removesuffix(".log")
        fpath = os.path.join(worker_dir, fname)
        entries.extend(read_tagged(fpath, tag, UTC))

    entries.sort(key=lambda e: e[0])

    for _, line in entries:
        print(line)


if __name__ == "__main__":
    main()
