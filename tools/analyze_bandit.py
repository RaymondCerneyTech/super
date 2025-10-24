from __future__ import annotations

import argparse
import json
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def moving_average(values: Iterable[float], window: int) -> List[float]:
    if window <= 0:
        return []
    dq: deque[float] = deque()
    totals: List[float] = []
    running = 0.0
    for value in values:
        dq.append(value)
        running += value
        if len(dq) > window:
            running -= dq.popleft()
        totals.append(running / len(dq))
    return totals


def sparkline(series: List[float]) -> str:
    if not series:
        return ""
    low = min(series)
    high = max(series)
    if high - low < 1e-9:
        return "-" * len(series)
    levels = [".", "-", "=", "#", "*"]
    span = high - low
    output = []
    for value in series:
        ratio = (value - low) / span
        index = min(len(levels) - 1, int(ratio * len(levels)))
        output.append(levels[index])
    return "".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze bandit JSONL logs.")
    parser.add_argument("--log-file", default="logs/bandit.jsonl", help="Path to JSON Lines log file.")
    parser.add_argument("--window", type=int, default=50, help="Window size for moving average.")
    parser.add_argument("--csv-out", help="Optional CSV output path with idx,overall,cluster rows.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    records = load_records(log_path)
    if not records:
        print(f"No records found at {log_path}")
        return 0

    cluster_totals: Dict[str, List[float]] = defaultdict(list)
    overall_values: List[float] = []
    for record in records:
        rewards = record.get("final_rewards") or {}
        overall = float(rewards.get("overall", 0.0))
        cluster = str(record.get("chosen_cluster", "unknown"))
        cluster_totals[cluster].append(overall)
        overall_values.append(overall)

    print(f"Total runs: {len(records)}")
    for cluster, values in sorted(cluster_totals.items()):
        mean_score = sum(values) / len(values) if values else 0.0
        print(f"  {cluster:>8}: mean overall={mean_score:.3f} (n={len(values)})")

    window = max(1, args.window)
    trend = moving_average(overall_values, window)
    if trend:
        print(f"{window}-event moving average (last {min(len(trend), window)} runs): {trend[-1]:.3f}")
        tail = trend[-min(len(trend), 40):]
        print("Trend:", sparkline(tail))

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("idx,overall,cluster\n")
            for idx, (overall, record) in enumerate(zip(overall_values, records), start=1):
                cluster = record.get("chosen_cluster", "")
                handle.write(f"{idx},{overall:.6f},{cluster}\n")
        print(f"Wrote CSV to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
