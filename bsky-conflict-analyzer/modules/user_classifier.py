# -*- coding: utf-8 -*-
"""
ユーザーを行動タイプに分類し、Geminiの分類結果を集計するモジュール。

Gemini分類に失敗したユーザーは0点に置き換えず、「未分類」として集計から除外する。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pandas as pd

BEHAVIOR_TYPES = ["投稿型", "拡散型", "返信型", "閲覧型"]
HARMFUL_LABELS = {
    "is_attack": "攻撃的表現",
    "is_exclusion": "排除要求",
    "is_punishment": "処罰要求",
    "is_violence": "暴力肯定表現",
    "is_conspiracy": "陰謀論的表現",
}

LOW_ACTIVITY_SAMPLE_THRESHOLD = 3


def classify_behavior_type(
    original_rate: float,
    reply_rate: float,
    repost_rate: float,
    sampled: int,
) -> str:
    """活動比率から行動タイプを1つ判定するルールベース分類。"""
    if sampled < LOW_ACTIVITY_SAMPLE_THRESHOLD:
        return "閲覧型"

    scores = {
        "投稿型": original_rate,
        "返信型": reply_rate,
        "拡散型": repost_rate,
    }
    top_type = max(scores, key=scores.get)
    if scores[top_type] <= 0:
        return "閲覧型"
    return top_type


@dataclass
class UserRecord:
    handle: str
    followers_count: int
    behavior_type: str
    mean_conflict_score: float | None
    n_posts_classified: int
    classification_status: str
    conflict_level: str = "未分類"
    harmful_flags: dict = field(default_factory=dict)


def build_user_table(user_activity: dict, user_post_scores: dict) -> pd.DataFrame:
    """ユーザーごとの行動タイプ・分類結果をDataFrameにまとめる。"""
    records: list[UserRecord] = []

    for handle, metrics in user_activity.items():
        behavior_type = classify_behavior_type(
            metrics.original_post_rate,
            metrics.reply_rate,
            metrics.repost_rate,
            metrics.total_posts_sampled,
        )

        classifications = user_post_scores.get(handle, [])
        if classifications:
            mean_score: float | None = statistics.mean(
                classification.conflict_score for classification in classifications
            )
            count = len(classifications)
            flags = {
                label: sum(
                    1 for classification in classifications
                    if getattr(classification, key)
                ) / count
                for key, label in HARMFUL_LABELS.items()
            }
            status = "分類済み"
        else:
            mean_score = None
            count = 0
            flags = {label: None for label in HARMFUL_LABELS.values()}
            status = "未分類"

        records.append(
            UserRecord(
                handle=handle,
                followers_count=metrics.followers_count,
                behavior_type=behavior_type,
                mean_conflict_score=mean_score,
                n_posts_classified=count,
                classification_status=status,
                harmful_flags=flags,
            )
        )

    dataframe = pd.DataFrame(
        [
            {
                "handle": record.handle,
                "followers_count": record.followers_count,
                "behavior_type": record.behavior_type,
                "mean_conflict_score": record.mean_conflict_score,
                "n_posts_classified": record.n_posts_classified,
                "classification_status": record.classification_status,
                **record.harmful_flags,
            }
            for record in records
        ]
    )

    if dataframe.empty:
        return dataframe

    dataframe["conflict_level"] = "未分類"
    classified_mask = dataframe["classification_status"] == "分類済み"

    for behavior_type in dataframe.loc[classified_mask, "behavior_type"].unique():
        group_mask = classified_mask & (dataframe["behavior_type"] == behavior_type)
        median_score = dataframe.loc[group_mask, "mean_conflict_score"].median()
        dataframe.loc[group_mask, "conflict_level"] = dataframe.loc[
            group_mask, "mean_conflict_score"
        ].apply(lambda score: "高い" if score >= median_score else "低い")

    return dataframe


def summarize_harmful_expression_rates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """分類に成功したユーザーだけを、行動タイプ×対立表現レベルで集計する。"""
    if dataframe.empty:
        return pd.DataFrame()

    classified = dataframe[
        (dataframe["classification_status"] == "分類済み")
        & dataframe["conflict_level"].isin(["高い", "低い"])
    ].copy()

    if classified.empty:
        return pd.DataFrame()

    label_columns = list(HARMFUL_LABELS.values())
    grouped = (
        classified.groupby(["behavior_type", "conflict_level"], dropna=False)[label_columns]
        .mean()
        .reset_index()
    )
    counts = (
        classified.groupby(["behavior_type", "conflict_level"], dropna=False)
        .size()
        .reset_index(name="ユーザー数")
    )
    return grouped.merge(counts, on=["behavior_type", "conflict_level"])