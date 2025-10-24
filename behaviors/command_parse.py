from __future__ import annotations

import re
from typing import Any, Dict, List

from core.interfaces import Behavior, Context, Result


class CommandParse(Behavior):
    name = "command_parse"
    inputs = ["task"]
    outputs = ["steps"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        task = (data.get("task") or ctx.get("task") or "").strip()
        if not task:
            return {
                "ok": False,
                "logs": ["No task provided for command_parse"],
                "reward": 0.0,
                "effects": [],
                "rationale": {"why": "missing task", "evidence": []},
            }

        steps: List[Dict[str, Any]] = []
        lowered = task.lower()

        steps.extend(self._parse_download(lowered, task))
        steps.extend(self._parse_unzip(lowered, task))
        steps.extend(self._parse_zip(lowered, task))
        steps.extend(self._parse_summarize(lowered, task))
        steps.extend(self._parse_list(lowered, task))

        if not steps:
            return {
                "ok": True,
                "logs": ["No actionable command patterns matched."],
                "reward": 0.5,
                "effects": [],
                "rationale": {"why": "no matches", "evidence": [task]},
                "output": {"steps": []},
            }

        return {
            "ok": True,
            "logs": [f"Parsed {len(steps)} step(s) from task."],
            "reward": 1.0,
            "effects": ["have_steps"],
            "rationale": {"why": "matched rules", "evidence": [task]},
            "output": {"steps": steps},
        }

    def _parse_download(self, lowered: str, raw: str) -> List[Dict[str, Any]]:
        pattern = re.compile(r"download\s+(?P<url>\S+)\s+to\s+(?P<dst>\S+)")
        steps: List[Dict[str, Any]] = []
        for match in pattern.finditer(lowered):
            url = self._extract_group(raw, match, "url")
            dst = self._extract_group(raw, match, "dst")
            steps.append(
                {
                    "behavior": "http_download",
                    "args": {"url": url, "dst": dst},
                }
            )
        return steps

    def _parse_unzip(self, lowered: str, raw: str) -> List[Dict[str, Any]]:
        pattern = re.compile(r"unzip\s+(?P<src>\S+)\s+(?:to|into)\s+(?P<dst>\S+)")
        steps: List[Dict[str, Any]] = []
        for match in pattern.finditer(lowered):
            src = self._extract_group(raw, match, "src")
            dst = self._extract_group(raw, match, "dst")
            steps.append(
                {
                    "behavior": "zip_ops",
                    "args": {"op": "unzip", "src": src, "dst": dst},
                }
            )
        return steps

    def _parse_zip(self, lowered: str, raw: str) -> List[Dict[str, Any]]:
        pattern = re.compile(r"zip\s+(?P<src>\S+)\s+(?:to|into)\s+(?P<dst>\S+)")
        steps: List[Dict[str, Any]] = []
        for match in pattern.finditer(lowered):
            src = self._extract_group(raw, match, "src")
            dst = self._extract_group(raw, match, "dst")
            steps.append(
                {
                    "behavior": "zip_ops",
                    "args": {"op": "zip", "src": src, "dst": dst},
                }
            )
        return steps

    def _parse_summarize(self, lowered: str, raw: str) -> List[Dict[str, Any]]:
        pattern = re.compile(
            r"summar(?:ize|ise)\s+(?P<src>\S+)\s+(?:to|into|and\s+save\s+to)\s+(?P<dst>\S+)"
        )
        steps: List[Dict[str, Any]] = []
        for match in pattern.finditer(lowered):
            src = self._extract_group(raw, match, "src")
            dst = self._extract_group(raw, match, "dst")
            steps.extend(
                [
                    {"behavior": "files_read", "args": {"path": src}},
                    {"behavior": "summarize", "args": {}},
                    {
                        "behavior": "files_write",
                        "args": {"path": dst, "content_key": "summary", "overwrite": True},
                    },
                ]
            )
        return steps

    def _parse_list(self, lowered: str, raw: str) -> List[Dict[str, Any]]:
        pattern = re.compile(r"list\s+files\s+(?:under|in)\s+(?P<dir>\S+)(?:\s+with\s+(?P<glob>\S+))?")
        steps: List[Dict[str, Any]] = []
        for match in pattern.finditer(lowered):
            directory = self._extract_group(raw, match, "dir")
            glob = self._extract_group(raw, match, "glob") or "*.txt"
            steps.append(
                {
                    "behavior": "files_list",
                    "args": {"path": directory, "glob": glob, "recursive": True},
                }
            )
        return steps

    @staticmethod
    def _extract_group(raw: str, match: re.Match[str], group: str) -> str:
        start, end = match.span(group)
        return raw[start:end]


__all__ = ["CommandParse"]
