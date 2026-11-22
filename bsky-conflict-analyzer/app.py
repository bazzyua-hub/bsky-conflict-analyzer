"""
SNS空間の言論空間 ～対立的・有害表現の拡散分析～
Streamlit メインアプリ

実行方法:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(__file__))

from modules import bluesky_client as bsky
from modules import gemini_classifier as gem
from modules import user_classifier as uc
from modules import network_viz as viz
from modules import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SNS対立表現 拡散分析", layout="wide")

# ---------------------------------------------------------------------------
#
DEFAULT_BSKY_HANDLE = ""  # 空のままで構いません（画面左のサイドバーに入力します）
DEFAULT_BSKY_APP_PASSWORD = ""  # 空のままで構いません（画面左のサイドバーに入力します）
DEFAULT_GEMINI_API_KEY = ""  # 空のままで構いません（画面左のサイドバーに入力します）


# ---------------------------------------------------------------------------
# セッション状態の初期化
# ---------------------------------------------------------------------------
for key, default in {
    "bsky_client": None,
    "posts": [],
    "edges": [],
    "user_activity": {},
    "user_post_scores": {},
    "user_table": pd.DataFrame(),
    "summary_table": pd.DataFrame(),
    "log_messages": [],
    "gemini_errors": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def log(msg: str) -> None:
    st.session_state["log_messages"].append(msg)
    logger.info(msg)


# ---------------------------------------------------------------------------
# サイドバー: 認証情報・パラメータ
# ---------------------------------------------------------------------------
st.sidebar.header("① 認証情報")
bsky_handle = st.sidebar.text_input(
    "Blueskyハンドル", value=DEFAULT_BSKY_HANDLE, placeholder="example.bsky.social"
)
bsky_app_password = st.sidebar.text_input(
    "Blueskyアプリパスワード", value=DEFAULT_BSKY_APP_PASSWORD, type="password"
)
gemini_api_key = st.sidebar.text_input(
    "Gemini APIキー", value=DEFAULT_GEMINI_API_KEY, type="password"
)

st.sidebar.markdown("---")
st.sidebar.header("② 収集方法")
collection_mode = st.sidebar.radio(
    "データ収集の方法",
    ["特定の投稿から収集（ネットワーク図あり）", "キーワード検索から収集", "両方"],
)

target_post_url = ""
keyword = ""
if collection_mode in ("特定の投稿から収集（ネットワーク図あり）", "両方"):
    target_post_url = st.sidebar.text_input(
        "対象投稿のURL", placeholder="https://bsky.app/profile/xxx/post/yyy"
    )
if collection_mode in ("キーワード検索から収集", "両方"):
    keyword = st.sidebar.text_input("検索キーワード/ハッシュタグ", placeholder="例: #議論")

st.sidebar.markdown("---")
st.sidebar.header("③ 規模の調整（無料枠に配慮）")
max_replies = st.sidebar.slider("最大リプライ数", 10, 300, 80, step=10)
max_reposts = st.sidebar.slider("最大リポスト数", 10, 300, 80, step=10)
max_keyword_posts = st.sidebar.slider("キーワード検索の最大取得件数", 10, 300, 60, step=10)
max_users_to_analyze = st.sidebar.slider(
    "行動分析・分類の対象ユーザー数上限", 5, 100, 30, step=5,
    help="Gemini API呼び出し回数に直結します。無料枠が心配な場合は小さめに。"
)
texts_per_user = st.sidebar.slider("1ユーザーあたり分析する投稿数", 1, 10, 5)

run_button = st.sidebar.button("🚀 分析を実行", type="primary", use_container_width=True)

st.sidebar.markdown("---")
with st.sidebar.expander("💾 過去の結果を読み込む"):
    saved = storage.list_saved_runs()
    if saved:
        pick = st.selectbox("保存済みの結果", saved)
        if st.button("読み込む"):
            data = storage.load_run(pick)
            st.session_state["user_table"] = pd.DataFrame(data.get("user_table", []))
            st.session_state["summary_table"] = pd.DataFrame(data.get("summary_table", []))
            st.success(f"{pick} を読み込みました")
    else:
        st.caption("保存済みの結果はまだありません")


# ---------------------------------------------------------------------------
# タイトル
# ---------------------------------------------------------------------------
st.title("SNS空間の言論空間 ～対立的・有害表現の拡散分析～")
st.caption("Bluesky APIとGemini APIを用いて、対立的表現の拡散パターンを分析します。")


# ---------------------------------------------------------------------------
# 分析パイプライン
# ---------------------------------------------------------------------------
def run_pipeline():
    if not bsky_handle or not bsky_app_password:
        st.error("Blueskyのハンドルとアプリパスワードを入力してください。")
        return
    if not gemini_api_key:
        st.error("Gemini APIキーを入力してください。")
        return
    if collection_mode != "キーワード検索から収集" and not target_post_url:
        st.error("対象投稿のURLを入力してください。")
        return
    if collection_mode != "特定の投稿から収集（ネットワーク図あり）" and not keyword:
        st.error("検索キーワードを入力してください。")
        return

    status = st.status("分析を実行中...", expanded=True)

    # --- 1. ログイン ---
    status.write("Blueskyにログイン中...")
    try:
        client = bsky.login(bsky_handle, bsky_app_password)
    except Exception as e:  # noqa: BLE001
        status.update(label="ログイン失敗", state="error")
        st.error(f"Blueskyへのログインに失敗しました: {e}")
        return

    try:
        gem.configure_gemini(gemini_api_key)
    except Exception as ex:  # noqa: BLE001
        status.update(label="Gemini初期化失敗", state="error")
        st.error(f"Gemini APIの初期化に失敗しました: {ex}")
        return

    st.session_state["gemini_errors"] = []

    posts: list = []
    edges: list = []

    # --- 2. データ収集 ---
    if collection_mode in ("特定の投稿から収集（ネットワーク図あり）", "両方"):
        status.write(f"対象投稿からリプライ・リポストを収集中... (最大 {max_replies}件 / {max_reposts}件)")
        try:
            p, e = bsky.collect_from_target_post(
                client, target_post_url, max_replies=max_replies, max_reposts=max_reposts
            )
            posts.extend(p)
            edges.extend(e)
            status.write(f"→ 投稿 {len(p)} 件、拡散の辺 {len(e)} 本を収集しました。")
        except Exception as ex:  # noqa: BLE001
            status.write(f"⚠ 対象投稿の収集でエラー: {ex}")

    if collection_mode in ("キーワード検索から収集", "両方"):
        status.write(f"キーワード '{keyword}' で検索中... (最大 {max_keyword_posts}件)")
        try:
            p = bsky.collect_from_keyword(client, keyword, max_posts=max_keyword_posts)
            posts.extend(p)
            status.write(f"→ 投稿 {len(p)} 件を追加収集しました。")
        except Exception as ex:  # noqa: BLE001
            status.write(f"⚠ キーワード検索でエラー: {ex}")

    if not posts:
        status.update(label="収集結果が0件でした", state="error")
        st.error("投稿を収集できませんでした。URL/キーワードを確認してください。")
        return

    # 収集対象ユーザーの一覧（投稿の著者 ＋ 拡散ネットワークの参加者）
    handles = {p.author_handle for p in posts}
    handles |= {e.source_handle for e in edges} | {e.target_handle for e in edges}
    # setの順序は実行ごとに変わるため、研究結果の再現性を保つために並べ替える
    handles = sorted(handles)[:max_users_to_analyze]
    status.write(f"行動分析の対象ユーザー: {len(handles)} 人")

    # --- 3. ユーザーごとの活動指標を取得（行動タイプ分類の材料） ---
    user_activity = {}
    progress_bar = status.progress(0.0, text="ユーザー活動データを取得中...")
    for i, h in enumerate(handles):
        try:
            metrics = bsky.get_user_recent_posts(client, h, sample_size=30)
            user_activity[h] = metrics
        except Exception as ex:  # noqa: BLE001
            status.write(f"⚠ {h} の取得でエラー: {ex}")
        progress_bar.progress((i + 1) / len(handles), text=f"ユーザー活動データを取得中... ({i+1}/{len(handles)})")

    # --- 4. Gemini APIで対立表現・有害表現を分類 ---
    status.write("Gemini APIで投稿内容を分類中...（複数ユーザーをまとめてバッチ処理します）")
    gemini_progress = status.progress(0.0, text="Gemini分類 準備中...")

    user_texts = {
        handle: metrics.sample_texts[:texts_per_user]
        for handle, metrics in user_activity.items()
        if metrics.sample_texts[:texts_per_user]
    }
    total_texts = sum(len(texts) for texts in user_texts.values())

    def update_gemini_progress(done: int, total: int) -> None:
        denominator = max(total, 1)
        gemini_progress.progress(
            done / denominator,
            text=f"Gemini分類中... ({done}/{total}投稿)",
        )

    try:
        user_post_scores, gemini_error_samples = gem.classify_user_posts(
            user_texts, progress_callback=update_gemini_progress
        )
    except Exception as ex:  # noqa: BLE001
        user_post_scores = {}
        gemini_error_samples = [f"{type(ex).__name__}: {ex}"]
        status.write(f"⚠ Gemini分類でエラー: {ex}")

    if total_texts == 0:
        gemini_progress.progress(1.0, text="分類対象の投稿がありませんでした。")

    st.session_state["gemini_errors"] = gemini_error_samples
    if gemini_error_samples:
        status.write(
            f"⚠ Gemini分類で {len(gemini_error_samples)} 件のエラーが発生しました。"
            "結果画面の上部に詳細を表示します。"
        )

    # --- 5. 集計 ---
    status.write("結果を集計中...")
    user_table = uc.build_user_table(user_activity, user_post_scores)
    summary_table = uc.summarize_harmful_expression_rates(user_table)

    st.session_state["posts"] = posts
    st.session_state["edges"] = edges
    st.session_state["user_activity"] = user_activity
    st.session_state["user_post_scores"] = user_post_scores
    st.session_state["user_table"] = user_table
    st.session_state["summary_table"] = summary_table

    status.update(label="分析が完了しました ✅", state="complete")


if run_button:
    run_pipeline()


# ---------------------------------------------------------------------------
# 結果表示
# ---------------------------------------------------------------------------
user_table = st.session_state["user_table"]
summary_table = st.session_state["summary_table"]

gemini_errors = st.session_state.get("gemini_errors", [])
if gemini_errors:
    unique_errors = list(dict.fromkeys(gemini_errors))  # 重複除去（順序維持）
    st.error(
        f"⚠ Gemini APIの呼び出しで {len(gemini_errors)} 件のエラーが発生しました。"
        "失敗した投稿・ユーザーは『未分類』として扱い、対立表現の集計から除外しています。"
        "下記のエラー内容を確認してください。"
    )
    with st.expander("Geminiエラーの詳細を見る", expanded=True):
        for e in unique_errors[:10]:
            st.code(e)

if user_table.empty:
    st.info("左のサイドバーで認証情報とパラメータを入力し、「分析を実行」を押してください。")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 概要", "🕸 拡散ネットワーク図", "📋 ユーザー別テーブル", "💾 保存/出力"])

    with tab1:
        classified_count = int((user_table["classification_status"] == "分類済み").sum())
        unclassified_count = int((user_table["classification_status"] == "未分類").sum())
        col1, col2 = st.columns(2)
        col1.metric("Gemini分類済みユーザー", classified_count)
        col2.metric("未分類ユーザー", unclassified_count)

        st.subheader("行動タイプ × 対立表現レベル 別 有害表現の出現率")
        if not summary_table.empty:
            st.dataframe(summary_table, use_container_width=True)

            melted = summary_table.melt(
                id_vars=["behavior_type", "conflict_level", "ユーザー数"],
                value_vars=list(uc.HARMFUL_LABELS.values()),
                var_name="有害表現の種類",
                value_name="出現率",
            )
            fig = px.bar(
                melted,
                x="behavior_type",
                y="出現率",
                color="有害表現の種類",
                facet_col="conflict_level",
                barmode="group",
                title="行動タイプ・対立表現レベル別の有害表現出現率",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("集計対象のデータがありません。")

        st.subheader("行動タイプの分布")
        type_counts = user_table["behavior_type"].value_counts().reset_index()
        type_counts.columns = ["behavior_type", "人数"]
        fig2 = px.pie(type_counts, names="behavior_type", values="人数", title="行動タイプの内訳")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("リプライ・リポストによる拡散ネットワーク")
        edges = st.session_state["edges"]
        if not edges:
            st.caption("ネットワーク図を表示するには「特定の投稿から収集」モードを使ってください。")
        else:
            user_lookup = {
                row["handle"]: {
                    "conflict_level": row["conflict_level"],
                    "followers_count": row["followers_count"],
                }
                for _, row in user_table.iterrows()
            }
            g = viz.build_diffusion_graph(edges, user_lookup)
            html = viz.render_pyvis_html(g)
            components.html(html, height=680, scrolling=True)
            st.caption("🔴 対立表現が高い　🔵 対立表現が低い　⚪ 未分類　｜　オレンジ辺=リプライ　緑辺=リポスト")

    with tab3:
        st.subheader("ユーザー別の分類結果")
        st.dataframe(user_table, use_container_width=True)

    with tab4:
        st.subheader("結果の保存")
        run_name = st.text_input("保存名", value="bsky_analysis")
        if st.button("この結果をローカルに保存する"):
            payload = {
                "user_table": user_table.to_dict(orient="records"),
                "summary_table": summary_table.to_dict(orient="records"),
            }
            path = storage.save_run(run_name, payload)
            st.success(f"保存しました: {path}")

        csv_bytes = user_table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "ユーザー別テーブルをCSVでダウンロード", csv_bytes, file_name="user_table.csv", mime="text/csv"
        )
        if not summary_table.empty:
            csv_bytes2 = summary_table.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "集計テーブルをCSVでダウンロード", csv_bytes2, file_name="summary_table.csv", mime="text/csv"
            )

if st.session_state["log_messages"]:
    with st.expander("実行ログ"):
        for m in st.session_state["log_messages"][-100:]:
            st.text(m)