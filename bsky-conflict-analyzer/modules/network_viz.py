# -*- coding: utf-8 -*-
"""
リプライ・リポストによる拡散ネットワークを pyvis で可視化するモジュール。
ノードの色 = 対立表現レベル（高い=赤系, 低い=青系, 不明=グレー）
ノードの大きさ = フォロワー数（対数スケール）
辺の色 = reply(オレンジ) / repost(緑)
"""

from __future__ import annotations

import math

import networkx as nx
from pyvis.network import Network

COLOR_HIGH = "#e05252"   # 対立表現が高い
COLOR_LOW = "#4a7fd6"    # 対立表現が低い
COLOR_UNKNOWN = "#a0a0a0"

EDGE_COLOR = {"reply": "#e8a13a", "repost": "#4caf7d"}


def build_diffusion_graph(edges: list, user_lookup: dict) -> nx.DiGraph:
    """
    edges: bluesky_client.DiffusionEdge のリスト
    user_lookup: handle -> {"conflict_level": str, "followers_count": int} の辞書
                 （user_classifier.build_user_table の結果から作る）
    """
    g = nx.DiGraph()

    for e in edges:
        for handle in (e.source_handle, e.target_handle):
            if handle not in g:
                info = user_lookup.get(handle, {})
                g.add_node(
                    handle,
                    conflict_level=info.get("conflict_level", ""),
                    followers_count=info.get("followers_count", 0),
                )
        g.add_edge(e.source_handle, e.target_handle, edge_type=e.edge_type)

    return g


def render_pyvis_html(g: nx.DiGraph, height_px: int = 650) -> str:
    """networkx グラフを pyvis でHTML文字列にレンダリングする"""
    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
    )
    net.barnes_hut(gravity=-3000, spring_length=120)

    for node, data in g.nodes(data=True):
        followers = data.get("followers_count", 0) or 0
        size = 10 + min(30, math.log1p(followers) * 4)
        level = data.get("conflict_level", "")
        color = COLOR_HIGH if level == "高い" else COLOR_LOW if level == "低い" else COLOR_UNKNOWN
        title = f"@{node}<br>フォロワー: {followers}<br>対立表現: {level or '不明'}"
        net.add_node(node, label=node, size=size, color=color, title=title)

    for src, dst, data in g.edges(data=True):
        etype = data.get("edge_type", "reply")
        net.add_edge(src, dst, color=EDGE_COLOR.get(etype, "#999999"), title=etype)

    return net.generate_html(notebook=False)
