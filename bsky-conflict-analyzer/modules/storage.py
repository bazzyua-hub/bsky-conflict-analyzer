# -*- coding: utf-8 -*-
"""
分析結果をローカルに保存・読み込みするモジュール。
無料枠のAPIを何度も消費しないよう、一度収集・分類した結果を
JSON/CSVとして保存し、次回はそこから読み込めるようにする。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_runs")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def save_run(name: str, payload: dict) -> str:
    """分析結果一式(dict)をJSONとして保存する。ファイルパスを返す。"""
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")) or "run"
    path = os.path.join(OUTPUT_DIR, f"{safe_name}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def list_saved_runs() -> list[str]:
    ensure_output_dir()
    return sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")], reverse=True
    )


def load_run(filename: str) -> dict:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
