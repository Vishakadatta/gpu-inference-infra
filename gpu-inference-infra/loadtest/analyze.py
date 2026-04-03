#!/usr/bin/env python3
"""
analyze.py — Parse load test results and produce summary statistics.

Usage:
  python3 analyze.py results/loadtest_short_1234567890.json
  python3 analyze.py results/*.json    # compare multiple runs
"""

import json
import sys
import os
from collections import defaultdict


def percentile(sorted_data, p):
    """Calculate the p-th percentile of sorted data."""
    if not sorted_data:
        return 0
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def analyze_file(filepath):
    """Analyze a single load test results file."""

    with open(filepath) as f:
        data = json.load(f)

    config = data["config"]
    results = data["results"]

    print("=" * 60)
    print(f"  Load Test Analysis: {os.path.basename(filepath)}")
    print("=" * 60)
    print(f"  Model:        {config['model']}")
    print(f"  Prompt:       {config['prompt_length']}")
    print(f"  Max tokens:   {config['max_tokens']}")
    print(f"  Duration:     {data['duration_seconds']}s")
    print(f"  Total reqs:   {data['total_requests']}")
    print()

    # Group by concurrency level
    by_concurrency = defaultdict(list)
    for r in results:
        by_concurrency[r["concurrency"]].append(r)

    # Print header
    print(f"  {'Conc':>4}  {'OK':>4}  {'Err':>3}  {'Avg(ms)':>8}  {'p50(ms)':>8}  "
          f"{'p95(ms)':>8}  {'p99(ms)':>8}  {'Tok/s':>7}  {'Req/s':>6}")
    print("  " + "-" * 72)

    for conc in sorted(by_concurrency.keys()):
        reqs = by_concurrency[conc]
        successes = [r for r in reqs if r["status"] == "success"]
        errors = [r for r in reqs if r["status"] == "error"]

        if not successes:
            print(f"  {conc:>4}  {0:>4}  {len(errors):>3}  {'N/A':>8}  {'N/A':>8}  "
                  f"{'N/A':>8}  {'N/A':>8}  {'N/A':>7}  {'N/A':>6}")
            continue

        latencies = sorted([r["latency_ms"] for r in successes])
        tps_values = [r["tokens_per_second"] for r in successes]

        avg_lat = sum(latencies) / len(latencies)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        avg_tps = sum(tps_values) / len(tps_values)

        # Throughput: total successful requests / total wall time for this concurrency
        total_time_s = max(latencies) / 1000  # approximate wall time
        rps = len(successes) / total_time_s if total_time_s > 0 else 0

        print(f"  {conc:>4}  {len(successes):>4}  {len(errors):>3}  {avg_lat:>8.0f}  "
              f"{p50:>8.0f}  {p95:>8.0f}  {p99:>8.0f}  {avg_tps:>7.1f}  {rps:>6.1f}")

    # Error summary
    all_errors = [r for r in results if r["status"] == "error"]
    if all_errors:
        print(f"\n  Errors ({len(all_errors)} total):")
        error_types = defaultdict(int)
        for e in all_errors:
            error_types[e["error_message"][:80]] += 1
        for msg, count in error_types.items():
            print(f"    [{count}x] {msg}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <results_file.json> [more_files...]")
        print("       python3 analyze.py results/*.json")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        analyze_file(filepath)
