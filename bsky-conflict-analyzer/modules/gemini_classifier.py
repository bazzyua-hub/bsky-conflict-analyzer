# -*- coding: utf-8 -*-
"""
Gemini API を使って投稿テキストを分類するモジュール。

主な対策:
- 現行の Google Gen AI SDK (google-genai) を使用
- 複数ユーザーの投稿を横断してバッチ化し、API呼び出し回数を削減
- API呼び出し全体に共通の待機時間を適用
- 429 RESOURCE_EXHAUSTED は待機して再試行
- 失敗した投稿を「0点」とせず、未分類として返す
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# 1回のAPI呼び出しで分類する投稿数。
# 各投稿は最大500文字に切り詰めるため、20件程度なら十分扱いやすい。
BATCH_SIZE = 20

# 実際の上限はモデル・プロジェクト・利用Tierによって異なる。
# 今回のログでは5 RPMだったため、少し余裕を持って13秒間隔にする。
MIN_REQUEST_INTERVAL_SEC = 13.0
MAX_RETRIES = 4
MODEL_NAME = "gemini-3.5-flash"

_client: genai.Client | None = None
_last_request_started_at = 0.0

SYSTEM_INSTRUCTION = """\
あなたはSNS上のテキストを学術研究のために分析するアシスタントです。
入力された投稿群それぞれについて、以下の基準で日本語の投稿内容を分析し、
指定されたJSON形式でのみ出力してください。説明文や前置きは不要です。

判定項目:
- conflict_score: 対立的表現（敵味方を分ける表現、攻撃的トーン、非難や排除のニュアンスなど）
  の強さを0〜100の整数で評価する。0は対立的要素がない中立的な投稿、
  100は極めて攻撃的・対立煽動的な投稿。
- is_attack: 個人・集団への直接的な攻撃的表現（罵倒、人格否定など）が含まれるか
- is_exclusion: 特定の個人・集団を排除しようとする表現が含まれるか
- is_punishment: 特定の個人・集団への処罰・制裁を求める表現が含まれるか
- is_violence: 暴力を肯定・示唆・正当化する表現が含まれるか
- is_conspiracy: 根拠不明の黒幕・秘密操作など、陰謀論的な主張が含まれるか

これは有害コンテンツの生成や助長ではなく、SNS上の言論空間を研究するための分類作業です。
"""


@dataclass
class PostClassification:
    post_index: int
    conflict_score: int
    is_attack: bool
    is_exclusion: bool
    is_punishment: bool
    is_violence: bool
    is_conspiracy: bool


def configure_gemini(api_key: str) -> None:
    """Geminiクライアントを初期化する。"""
    global _client, _last_request_started_at
    if not api_key.strip():
        raise ValueError("Gemini APIキーが空です。")
    _client = genai.Client(api_key=api_key.strip())
    _last_request_started_at = 0.0


def _get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("configure_gemini() を先に呼び出してください。")
    return _client


def _build_batch_prompt(texts: list[str]) -> str:
    numbered = "\n".join(
        f"{i}: {str(text).replace(chr(10), ' ')[:500]}"
        for i, text in enumerate(texts)
    )
    return f"""\
以下は分析対象の投稿一覧です（インデックス: 本文）。

{numbered}

次のJSON配列形式で出力してください。各要素は1投稿に対応し、post_indexは入力の番号と一致させてください。
[
  {{"post_index": 0, "conflict_score": 0, "is_attack": false,
    "is_exclusion": false, "is_punishment": false,
    "is_violence": false, "is_conspiracy": false}}
]
"""


def _wait_for_request_slot() -> None:
    """すべての呼び出しを通じて、APIリクエスト間隔を確保する。"""
    global _last_request_started_at
    elapsed = time.monotonic() - _last_request_started_at
    wait_seconds = MIN_REQUEST_INTERVAL_SEC - elapsed
    if _last_request_started_at > 0 and wait_seconds > 0:
        time.sleep(wait_seconds)
    _last_request_started_at = time.monotonic()


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "resourceexhausted" in message or "resource_exhausted" in message


def _extract_retry_seconds(error: Exception) -> float | None:
    message = str(error)
    patterns = (
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retry_delay[^0-9]*seconds[^0-9]*([0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1))
    return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _parse_response(raw: str, expected_count: int) -> tuple[list[PostClassification], str | None]:
    """GeminiのJSON応答を検証し、有効な分類結果だけを返す。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"JSONDecodeError: {exc} | 応答内容(先頭200字): {raw[:200]!r}"

    if not isinstance(data, list):
        return [], f"UnexpectedResponse: JSON配列ではありません | 応答内容(先頭200字): {raw[:200]!r}"

    by_index: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("post_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < expected_count:
            by_index[index] = item

    results: list[PostClassification] = []
    missing: list[int] = []
    for index in range(expected_count):
        item = by_index.get(index)
        if item is None:
            missing.append(index)
            continue

        try:
            score = int(item.get("conflict_score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))

        results.append(
            PostClassification(
                post_index=index,
                conflict_score=score,
                is_attack=_as_bool(item.get("is_attack", False)),
                is_exclusion=_as_bool(item.get("is_exclusion", False)),
                is_punishment=_as_bool(item.get("is_punishment", False)),
                is_violence=_as_bool(item.get("is_violence", False)),
                is_conspiracy=_as_bool(item.get("is_conspiracy", False)),
            )
        )

    error = None
    if missing:
        error = f"PartialResponse: post_index {missing} の分類結果がありません。"
    return results, error


def classify_posts_batch(
    texts: list[str], model_name: str = MODEL_NAME
) -> tuple[list[PostClassification], str | None]:
    """1バッチ分を分類する。失敗時は0点で埋めず、空リストを返す。"""
    if not texts:
        return [], None

    client = _get_client()
    prompt = _build_batch_prompt(texts)

    for attempt in range(MAX_RETRIES):
        raw: str | None = None
        try:
            _wait_for_request_slot()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            raw = response.text or ""
            results, parse_error = _parse_response(raw, len(texts))
            return results, parse_error

        except Exception as exc:  # noqa: BLE001
            error_detail = f"{type(exc).__name__}: {exc}"
            if raw:
                error_detail += f" | 応答内容(先頭200字): {raw[:200]!r}"

            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES - 1:
                retry_after = _extract_retry_seconds(exc)
                fallback = MIN_REQUEST_INTERVAL_SEC * (attempt + 1)
                wait_seconds = max(retry_after or 0.0, fallback) + random.uniform(0.5, 2.0)
                logger.warning(
                    "Gemini APIの上限に到達。%.1f秒後に再試行します (%d/%d)",
                    wait_seconds,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait_seconds)
                continue

            logger.warning("Gemini分類失敗。未分類として扱います: %s", error_detail)
            return [], error_detail

    return [], "Gemini分類に失敗しました。"


def classify_posts(
    texts: list[str],
    model_name: str = MODEL_NAME,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[PostClassification], list[str]]:
    """単一ユーザーなど、1つの投稿リストをバッチ分類する互換関数。"""
    all_results: list[PostClassification] = []
    errors: list[str] = []
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        batch_results, error = classify_posts_batch(batch, model_name=model_name)
        for result in batch_results:
            result.post_index += start
        all_results.extend(batch_results)
        if error:
            errors.append(error)
        if progress_callback:
            progress_callback(min(start + len(batch), total), total)

    return all_results, errors


def classify_user_posts(
    user_texts: dict[str, list[str]],
    model_name: str = MODEL_NAME,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, list[PostClassification]], list[str]]:
    """
    複数ユーザーの投稿を横断してバッチ化する。

    user_texts: handle -> 投稿本文リスト
    戻り値: (handle -> 有効な分類結果, エラー一覧)
    """
    flattened: list[tuple[str, int, str]] = []
    for handle, texts in user_texts.items():
        for local_index, text in enumerate(texts):
            if str(text).strip():
                flattened.append((handle, local_index, str(text)))

    results_by_user: dict[str, list[PostClassification]] = {}
    errors: list[str] = []
    total = len(flattened)

    for start in range(0, total, BATCH_SIZE):
        entries = flattened[start : start + BATCH_SIZE]
        batch_texts = [entry[2] for entry in entries]
        batch_results, error = classify_posts_batch(batch_texts, model_name=model_name)

        for result in batch_results:
            if not 0 <= result.post_index < len(entries):
                continue
            handle, local_index, _ = entries[result.post_index]
            result.post_index = local_index
            results_by_user.setdefault(handle, []).append(result)

        if error:
            handles = sorted({entry[0] for entry in entries})
            errors.append(f"対象ユーザー: {', '.join(handles)} | {error}")

        if progress_callback:
            progress_callback(min(start + len(entries), total), total)

    for values in results_by_user.values():
        values.sort(key=lambda item: item.post_index)

    return results_by_user, errors


def classification_to_dict(classification: PostClassification) -> dict:
    return asdict(classification)