from __future__ import annotations

import csv
import io
from statistics import mean
from typing import Any, Dict, Iterable, List

from core.interfaces import Behavior, Context, Result


class ReportFromData(Behavior):
    name = "report_from_data"
    inputs = ["table"]
    outputs = ["report_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        raw_table = data.get("table") or ctx.get("table")
        title = data.get("report_title") or ctx.get("report_title") or "Data Summary"

        records = self._load_records(raw_table)
        if not records:
            report = f"{title}\n\nNo data available."
        else:
            report = self._summarize_records(title, records)

        data["report_text"] = report
        logs = [f"Report generated with {len(records)} records." ]

        return {
            "ok": True,
            "output": {"report_text": report},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _load_records(self, table: Any) -> List[Dict[str, Any]]:
        if table is None:
            return []
        if isinstance(table, list):
            return [row for row in table if isinstance(row, dict)]
        if isinstance(table, str):
            reader = csv.DictReader(io.StringIO(table))
            return [row for row in reader]
        return []

    def _summarize_records(self, title: str, records: List[Dict[str, Any]]) -> str:
        fields = records[0].keys()
        numeric_fields = [f for f in fields if self._is_numeric_column(records, f)]
        categorical_fields = [f for f in fields if f not in numeric_fields]

        lines = [title, "", f"Total records: {len(records)}"]

        for field in numeric_fields:
            values = [float(row[field]) for row in records if self._is_float(row.get(field))]
            if not values:
                continue
            lines.append(f"Average {field}: {mean(values):.2f}")

        for field in categorical_fields:
            buckets = {}
            for row in records:
                value = (row.get(field) or "").strip()
                if not value:
                    continue
                buckets[value] = buckets.get(value, 0) + 1
            if buckets:
                top_value, count = max(buckets.items(), key=lambda item: item[1])
                lines.append(f"Most frequent {field}: {top_value} ({count})")

        return "\n".join(lines)

    def _is_numeric_column(self, records: List[Dict[str, Any]], field: str) -> bool:
        values = [row.get(field) for row in records]
        return all(self._is_float(value) for value in values if value not in (None, ""))

    def _is_float(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False


__all__ = ["ReportFromData"]
