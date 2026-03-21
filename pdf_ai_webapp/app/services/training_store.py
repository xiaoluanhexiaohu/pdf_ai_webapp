from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrainingStore:
    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_rules(self) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        except Exception:
            return []

    def append_rules(self, new_rules: List[Dict[str, Any]]) -> int:
        if not new_rules:
            return 0
        rules = self.load_rules()
        rules.extend(new_rules)
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        return len(new_rules)


def parse_training_rules(raw_text: str) -> List[Dict[str, str]]:
    """Each line: keyword|anchor_text|mode(optional)."""
    results: List[Dict[str, str]] = []
    if not raw_text.strip():
        return results

    for line in raw_text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = [part.strip() for part in clean.split("|")]
        if len(parts) < 2:
            continue
        mode = parts[2] if len(parts) >= 3 else "below"
        if mode not in {"below", "right", "appendix_page"}:
            mode = "below"
        results.append(
            {
                "keyword": parts[0],
                "anchor_text": parts[1],
                "mode": mode,
            }
        )
    return results
