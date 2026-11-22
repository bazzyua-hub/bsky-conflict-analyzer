# -*- coding: utf-8 -*-
"""
Bluesky (AT Protocol) からデータを取得するモジュール。

必要なもの:
    - Bluesky のハンドル（例: yourname.bsky.social）
    - アプリパスワード（Bluesky の 設定 > プライバシーとセキュリティ > アプリパスワード で発行）
      ※ 通常のログインパスワードではなく、必ず「アプリパスワード」を使うこと。

主な機能:
    1. login()                    : ログイン
    2. collect_from_target_post() : 特定の投稿へのリプライ・リポストのユーザーを収集（拡散ネットワークの起点）
    3. collect_from_keyword()     : キーワード/ハッシュタグ検索で投稿とユーザーを収集
    4. get_user_recent_posts()    : ユーザーの直近投稿を取得（行動タイプ分類・対立表現分析に使う）
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from atproto import Client

logger = logging.getLogger(__name__)

# Bluesky の一般的なレート制限に配慮し、各呼び出しの間に軽くウェイトを入れる
REQUEST_DELAY_SEC = 0.3


@dataclass
class CollectedPost:
    uri: str
    cid: str
    author_handle: str
    author_did: str
    text: str
    created_at: str
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0


@dataclass
class DiffusionEdge:
    """拡散ネットワークの1本の辺（誰から誰への反応か）"""
    source_handle: str  # 反応した人（リプライ/リポストした人）
    target_handle: str  # 反応された人（元投稿の著者、または返信元の著者）
    edge_type: str       # "reply" or "repost"
    post_uri: str = ""


def login(handle: str, app_password: str) -> Client:
    """Blueskyにログインしてクライアントを返す"""
    client = Client()
    client.login(handle, app_password)
    logger.info("Blueskyにログインしました: %s", handle)
    return client


def _safe_call(fn, *args, retries: int = 3, **kwargs):
    """API呼び出しをリトライ付きで実行する共通ラッパー"""
    last_err = None
    for attempt in range(retries):
        try:
            result = fn(*args, **kwargs)
            time.sleep(REQUEST_DELAY_SEC)
            return result
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            logger.warning("API呼び出し失敗（%s回目）: %s ... %s秒待機", attempt + 1, e, wait)
            time.sleep(wait)
    raise last_err


def parse_post_url(url_or_uri: str) -> str:
    """
    Bluesky の投稿URL（https://bsky.app/profile/xxx/post/yyy）を
    at:// URI 形式に変換する。すでに at:// 形式ならそのまま返す。
    """
    if url_or_uri.startswith("at://"):
        return url_or_uri
    # https://bsky.app/profile/{handle_or_did}/post/{rkey}
    parts = url_or_uri.rstrip("/").split("/")
    try:
        idx = parts.index("profile")
        actor = parts[idx + 1]
        rkey = parts[idx + 3]
    except (ValueError, IndexError):
        raise ValueError(f"投稿URLの形式を解釈できません: {url_or_uri}")
    return f"at://{actor}/app.bsky.feed.post/{rkey}"


def resolve_actor_to_did(client: Client, actor: str) -> str:
    """ハンドルまたはDIDをDIDに解決する（at://URI組み立てに必要な場合がある）"""
    if actor.startswith("did:"):
        return actor
    profile = _safe_call(client.app.bsky.actor.get_profile, {"actor": actor})
    return profile.did


def collect_from_target_post(
    client: Client,
    post_url_or_uri: str,
    max_replies: int = 100,
    max_reposts: int = 100,
    reply_depth: int = 6,
) -> tuple[list[CollectedPost], list[DiffusionEdge]]:
    """
    特定の投稿を起点に、リプライとリポストをたどってユーザーと拡散ネットワークの辺を収集する。

    戻り値:
        posts: 収集した投稿（元投稿＋リプライ本文を含む。対立表現分析の対象になる）
        edges: 拡散ネットワークの辺リスト（reply / repost）
    """
    uri = parse_post_url(post_url_or_uri)
    posts: list[CollectedPost] = []
    edges: list[DiffusionEdge] = []

    # 元投稿の情報を取得
    thread_resp = _safe_call(
        client.app.bsky.feed.get_post_thread, {"uri": uri, "depth": reply_depth}
    )
    root = thread_resp.thread

    root_post_view = getattr(root, "post", None)
    if root_post_view is None:
        raise ValueError("対象の投稿が見つかりませんでした。URLを確認してください。")

    root_author_handle = root_post_view.author.handle
    posts.append(
        CollectedPost(
            uri=root_post_view.uri,
            cid=root_post_view.cid,
            author_handle=root_author_handle,
            author_did=root_post_view.author.did,
            text=getattr(root_post_view.record, "text", ""),
            created_at=getattr(root_post_view.record, "created_at", ""),
            like_count=getattr(root_post_view, "like_count", 0) or 0,
            repost_count=getattr(root_post_view, "repost_count", 0) or 0,
            reply_count=getattr(root_post_view, "reply_count", 0) or 0,
        )
    )

    # --- リプライを再帰的にたどる ---
    def walk_replies(node, parent_handle: str, count_holder: list[int]):
        replies = getattr(node, "replies", None) or []
        for child in replies:
            if count_holder[0] >= max_replies:
                return
            child_post = getattr(child, "post", None)
            if child_post is None:
                continue
            author_handle = child_post.author.handle
            text = getattr(child_post.record, "text", "")
            posts.append(
                CollectedPost(
                    uri=child_post.uri,
                    cid=child_post.cid,
                    author_handle=author_handle,
                    author_did=child_post.author.did,
                    text=text,
                    created_at=getattr(child_post.record, "created_at", ""),
                    like_count=getattr(child_post, "like_count", 0) or 0,
                    repost_count=getattr(child_post, "repost_count", 0) or 0,
                    reply_count=getattr(child_post, "reply_count", 0) or 0,
                )
            )
            edges.append(
                DiffusionEdge(
                    source_handle=author_handle,
                    target_handle=parent_handle,
                    edge_type="reply",
                    post_uri=child_post.uri,
                )
            )
            count_holder[0] += 1
            walk_replies(child, author_handle, count_holder)

    walk_replies(root, root_author_handle, [0])

    # --- リポストしたユーザーを取得 ---
    cursor = None
    reposted_by = []
    while len(reposted_by) < max_reposts:
        params = {"uri": root_post_view.uri, "cid": root_post_view.cid, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        resp = _safe_call(client.app.bsky.feed.get_reposted_by, params)
        reposted_by.extend(resp.reposted_by)
        cursor = getattr(resp, "cursor", None)
        if not cursor or not resp.reposted_by:
            break

    for user in reposted_by[:max_reposts]:
        edges.append(
            DiffusionEdge(
                source_handle=user.handle,
                target_handle=root_author_handle,
                edge_type="repost",
                post_uri=root_post_view.uri,
            )
        )

    logger.info(
        "収集完了: 投稿%d件, 辺%d本（リプライ由来 + リポスト由来）", len(posts), len(edges)
    )
    return posts, edges


def collect_from_keyword(
    client: Client, keyword: str, max_posts: int = 100
) -> list[CollectedPost]:
    """キーワード/ハッシュタグで投稿を検索して収集する"""
    posts: list[CollectedPost] = []
    cursor = None
    while len(posts) < max_posts:
        params = {"q": keyword, "limit": min(100, max_posts - len(posts))}
        if cursor:
            params["cursor"] = cursor
        resp = _safe_call(client.app.bsky.feed.search_posts, params)
        batch = resp.posts
        if not batch:
            break
        for p in batch:
            posts.append(
                CollectedPost(
                    uri=p.uri,
                    cid=p.cid,
                    author_handle=p.author.handle,
                    author_did=p.author.did,
                    text=getattr(p.record, "text", ""),
                    created_at=getattr(p.record, "created_at", ""),
                    like_count=getattr(p, "like_count", 0) or 0,
                    repost_count=getattr(p, "repost_count", 0) or 0,
                    reply_count=getattr(p, "reply_count", 0) or 0,
                )
            )
        cursor = getattr(resp, "cursor", None)
        if not cursor:
            break
    logger.info("キーワード検索収集完了: '%s' -> %d件", keyword, len(posts))
    return posts


@dataclass
class UserActivityMetrics:
    handle: str
    followers_count: int = 0
    follows_count: int = 0
    total_posts_sampled: int = 0
    original_post_count: int = 0
    reply_count: int = 0
    repost_count: int = 0
    original_post_rate: float = 0.0
    reply_rate: float = 0.0
    repost_rate: float = 0.0
    sample_texts: list[str] = field(default_factory=list)


def get_user_recent_posts(
    client: Client, handle: str, sample_size: int = 30
) -> UserActivityMetrics:
    """
    ユーザーの直近の投稿群を取得し、行動タイプ分類に使う統計量を計算する。
    （オリジナル投稿率・リプライ率・リポスト率）
    """
    metrics = UserActivityMetrics(handle=handle)

    try:
        profile = _safe_call(client.app.bsky.actor.get_profile, {"actor": handle})
        metrics.followers_count = getattr(profile, "followers_count", 0) or 0
        metrics.follows_count = getattr(profile, "follows_count", 0) or 0
    except Exception as e:  # noqa: BLE001
        logger.warning("プロフィール取得失敗 (%s): %s", handle, e)

    try:
        resp = _safe_call(
            client.app.bsky.feed.get_author_feed,
            {"actor": handle, "limit": sample_size},
        )
        feed = resp.feed
    except Exception as e:  # noqa: BLE001
        logger.warning("フィード取得失敗 (%s): %s", handle, e)
        return metrics

    for item in feed:
        metrics.total_posts_sampled += 1
        is_repost = getattr(item, "reason", None) is not None
        record = getattr(item.post, "record", None)
        is_reply = bool(getattr(record, "reply", None)) if record else False

        if is_repost:
            metrics.repost_count += 1
        elif is_reply:
            metrics.reply_count += 1
            text = getattr(record, "text", "")
            if text:
                metrics.sample_texts.append(text)
        else:
            metrics.original_post_count += 1
            text = getattr(record, "text", "")
            if text:
                metrics.sample_texts.append(text)

    n = max(metrics.total_posts_sampled, 1)
    metrics.original_post_rate = metrics.original_post_count / n
    metrics.reply_rate = metrics.reply_count / n
    metrics.repost_rate = metrics.repost_count / n

    return metrics
