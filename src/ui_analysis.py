from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.chunk_labels import infer_chunk_label
from src.cluster_index import load_cluster_index
from src.cluster_report import build_cluster_quality_report
from src.document_cluster_index import load_document_cluster_index
from src.summary_index import load_summary_index
from src.vectordb import get_corpus_rows


def get_latest_retrieval_overlay() -> dict | None:
    """가장 최근 assistant 메시지에서 retrieval 시각화 정보를 추출"""
    for message in reversed(st.session_state.messages):
        if message.get("role") != "assistant":
            continue
        pipeline_info = message.get("pipeline_info") or {}
        retrieval = pipeline_info.get("retrieval") or {}
        if retrieval.get("query_vector") is not None:
            return retrieval
    return None


def _prepare_chart_points(
    df: pd.DataFrame,
    color_field: str,
    query_df: pd.DataFrame,
    query_clusters: list[str],
    focus_values: list[str],
) -> pd.DataFrame:
    chart_df = df.copy()
    chart_df["point_kind"] = "chunk"
    chart_df["select_key"] = chart_df[color_field].astype(str)
    chart_df["is_query_cluster"] = chart_df["cluster_id"].isin(query_clusters)
    chart_df["point_size"] = np.where(chart_df["is_query_cluster"], 130, 70)
    chart_df["is_focus"] = chart_df[color_field].astype(str).isin(focus_values) if focus_values else True

    if query_df.empty:
        return chart_df

    query_points = query_df.copy()
    query_points["point_kind"] = "query"
    query_points["select_key"] = "__query__"
    query_points["is_query_cluster"] = False
    query_points["point_size"] = 260
    query_points["is_focus"] = True
    query_points["cluster_id"] = "질문"
    query_points["chunk_label"] = "질문"
    query_points["file_name"] = ""
    query_points["page_number"] = -1
    query_points["department"] = ""
    query_points["year"] = ""
    query_points["section_title"] = ""
    query_points["content_type"] = "query"
    query_points["preview"] = query_points["label"]
    return pd.concat([chart_df, query_points], ignore_index=True, sort=False)


def _sample_corpus_rows(rows: list[dict], max_points: int) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    idx = np.linspace(0, len(rows) - 1, num=max_points, dtype=int)
    return [rows[i] for i in idx]


def _build_cluster_dataframe(
    rows: list[dict],
    max_points: int,
) -> tuple[pd.DataFrame, dict, PCA | None]:
    valid_rows = []
    for row in rows:
        vector = row.get("vector")
        meta = dict(row.get("metadata") or {})
        text = (row.get("document") or "").strip()
        if vector is None or meta.get("cluster_id") is None or not text:
            continue
        valid_rows.append(
            {
                "vector": vector,
                "cluster_id": int(meta.get("cluster_id")),
                "chunk_label": meta.get("chunk_label") or infer_chunk_label(meta.get("section_title", ""), text),
                "file_name": meta.get("file_name", "Unknown"),
                "page_number": meta.get("page_number", -1),
                "department": meta.get("department", ""),
                "year": meta.get("year", ""),
                "section_title": meta.get("section_title", ""),
                "content_type": meta.get("content_type", "text"),
                "preview": text[:120] + "..." if len(text) > 120 else text,
            }
        )

    sampled_rows = _sample_corpus_rows(valid_rows, max_points=max_points)
    if not sampled_rows:
        return pd.DataFrame(), {"total_points": 0, "plotted_points": 0, "clusters": 0}, None

    matrix = np.asarray([row["vector"] for row in sampled_rows], dtype=np.float32)
    reducer: PCA | None = None
    if len(sampled_rows) == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(matrix)

    records = []
    for idx, row in enumerate(sampled_rows):
        records.append(
            {
                "x": float(coords[idx][0]),
                "y": float(coords[idx][1]),
                "cluster_id": str(row["cluster_id"]),
                "chunk_label": row["chunk_label"],
                "file_name": row["file_name"],
                "page_number": row["page_number"],
                "department": row["department"],
                "year": row["year"],
                "section_title": row["section_title"],
                "content_type": row["content_type"],
                "preview": row["preview"],
            }
        )

    df = pd.DataFrame.from_records(records)
    stats = {
        "total_points": len(valid_rows),
        "plotted_points": len(sampled_rows),
        "clusters": df["cluster_id"].nunique(),
    }
    return df, stats, reducer


def _build_query_overlay(
    retrieval_overlay: dict | None,
    reducer: PCA | None,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    if not retrieval_overlay:
        return pd.DataFrame(), [], []

    query_vector = retrieval_overlay.get("query_vector")
    if query_vector is None:
        return pd.DataFrame(), [], []

    selected_clusters = [
        str(item.get("cluster_id"))
        for item in retrieval_overlay.get("selected_clusters", [])
        if item.get("cluster_id") is not None
    ]
    selected_meta = retrieval_overlay.get("selected_clusters", [])

    if reducer is None:
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        query_matrix = np.asarray([query_vector], dtype=np.float32)
        coords = reducer.transform(query_matrix)

    query_df = pd.DataFrame.from_records(
        [
            {
                "x": float(coords[0][0]),
                "y": float(coords[0][1]),
                "label": retrieval_overlay.get("retrieval_query", "질문"),
                "retriever_mode": retrieval_overlay.get("retriever_mode", ""),
            }
        ]
    )
    return query_df, selected_clusters, selected_meta


def _build_document_cluster_dataframe() -> tuple[pd.DataFrame, dict, PCA | None, dict[str, str]]:
    document_cluster_index = load_document_cluster_index()
    if not document_cluster_index:
        return pd.DataFrame(), {"total_documents": 0, "clusters": 0}, None, {}

    docs = document_cluster_index.get("documents", [])
    valid_docs = []
    file_to_cluster: dict[str, str] = {}
    for doc in docs:
        payload = dict(doc.get("payload") or {})
        vector = doc.get("vector")
        file_name = payload.get("file_name")
        if not vector or not file_name:
            continue
        cluster_id = str(payload.get("document_cluster_id", "0"))
        file_to_cluster[file_name] = cluster_id
        valid_docs.append(
            {
                "vector": vector,
                "document_cluster_id": cluster_id,
                "file_name": file_name,
                "department": payload.get("department", ""),
                "year": payload.get("year", ""),
                "project_name": payload.get("project_name", ""),
                "top_labels": ", ".join(payload.get("top_labels", [])[:4]),
                "top_sections": ", ".join(payload.get("top_sections", [])[:4]),
                "keywords": ", ".join(payload.get("keywords", [])[:6]),
                "chunk_count": int(payload.get("chunk_count", 0)),
                "summary_text": doc.get("text", ""),
            }
        )

    if not valid_docs:
        return pd.DataFrame(), {"total_documents": 0, "clusters": 0}, None, {}

    matrix = np.asarray([row["vector"] for row in valid_docs], dtype=np.float32)
    reducer: PCA | None = None
    if len(valid_docs) == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(matrix)

    records = []
    for idx, row in enumerate(valid_docs):
        records.append(
            {
                "x": float(coords[idx][0]),
                "y": float(coords[idx][1]),
                "document_cluster_id": row["document_cluster_id"],
                "file_name": row["file_name"],
                "department": row["department"],
                "year": row["year"],
                "project_name": row["project_name"],
                "top_labels": row["top_labels"],
                "top_sections": row["top_sections"],
                "keywords": row["keywords"],
                "chunk_count": row["chunk_count"],
                "summary_text": row["summary_text"],
            }
        )

    df = pd.DataFrame.from_records(records)
    stats = {
        "total_documents": len(valid_docs),
        "clusters": df["document_cluster_id"].nunique(),
        "cluster_meta": document_cluster_index,
    }
    return df, stats, reducer, file_to_cluster


def _build_document_query_overlay(
    retrieval_overlay: dict | None,
    reducer: PCA | None,
    file_to_cluster: dict[str, str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not retrieval_overlay:
        return pd.DataFrame(), [], []

    query_vector = retrieval_overlay.get("query_vector")
    if query_vector is None:
        return pd.DataFrame(), [], []

    guidance = retrieval_overlay.get("summary_guidance") or {}
    guided_files = guidance.get("file_names", []) if isinstance(guidance, dict) else []
    guided_clusters = []
    for file_name in guided_files:
        cluster_id = file_to_cluster.get(file_name)
        if cluster_id and cluster_id not in guided_clusters:
            guided_clusters.append(cluster_id)

    if reducer is None:
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        coords = reducer.transform(np.asarray([query_vector], dtype=np.float32))

    query_df = pd.DataFrame.from_records(
        [
            {
                "x": float(coords[0][0]),
                "y": float(coords[0][1]),
                "label": retrieval_overlay.get("retrieval_query", "질문"),
            }
        ]
    )
    return query_df, guided_clusters, guided_files


def render_cluster_map(rag) -> None:
    with st.expander("🗺️ 클러스터 맵", expanded=False):
        cluster_meta = load_cluster_index()
        if not cluster_meta:
            st.info("사전 군집화 메타데이터가 없습니다. `python index_data.py`로 재인덱싱한 뒤 확인하세요.")
            return

        rows = get_corpus_rows(rag.vectorstore)
        if not rows:
            st.info("시각화할 벡터 데이터가 없습니다.")
            return

        col1, col2 = st.columns([1, 1])
        max_points = col1.selectbox(
            "표시할 최대 청크 수",
            options=[300, 500, 1000, 2000],
            index=2,
        )

        retrieval_overlay = get_latest_retrieval_overlay()
        df, stats, reducer = _build_cluster_dataframe(rows, max_points=max_points)
        if df.empty:
            st.info("클러스터가 할당된 청크가 없어 맵을 그릴 수 없습니다.")
            return

        cluster_options = sorted(df["cluster_id"].unique(), key=lambda x: int(x))
        selected_clusters = col2.multiselect(
            "표시할 클러스터",
            options=cluster_options,
            default=[],
            placeholder="전체 클러스터",
        )
        if selected_clusters:
            df = df[df["cluster_id"].isin(selected_clusters)]

        query_df, query_clusters, query_cluster_meta = _build_query_overlay(retrieval_overlay, reducer)
        if selected_clusters and query_clusters:
            query_clusters = [cluster for cluster in query_clusters if cluster in selected_clusters]

        control_col1, control_col2 = st.columns([1, 1])
        color_mode = control_col1.selectbox(
            "색상 기준",
            options=[("chunk_label", "라벨"), ("cluster_id", "클러스터")],
            format_func=lambda item: item[1],
            key="cluster_map_color_mode",
        )
        show_label_text = control_col2.checkbox(
            "라벨 텍스트 표시",
            value=True,
            key="cluster_map_show_labels",
        )
        color_field = color_mode[0]
        color_title = "라벨" if color_field == "chunk_label" else "클러스터"
        focus_options = sorted(df[color_field].astype(str).unique().tolist())
        focus_values = st.multiselect(
            f"상세 보기 {color_title}",
            options=focus_options,
            default=[],
            placeholder=f"특정 {color_title}만 강조하려면 선택하세요.",
            key=f"cluster_map_focus_{color_field}",
        )
        chart_df = _prepare_chart_points(df, color_field, query_df, query_clusters, focus_values)

        meta_cols = st.columns(4)
        meta_cols[0].metric("총 청크", f"{stats['total_points']:,}")
        meta_cols[1].metric("차트 표시", f"{len(df):,}")
        meta_cols[2].metric("클러스터 수", cluster_meta.get("n_clusters", stats["clusters"]))
        meta_cols[3].metric("임베딩 모델", cluster_meta.get("embedding_model", "-"))

        chart = (
            alt.Chart(chart_df)
            .mark_point(filled=True)
            .encode(
                x=alt.X("x:Q", title="PCA-1"),
                y=alt.Y("y:Q", title="PCA-2"),
                color=alt.condition(
                    "datum.point_kind === 'query'",
                    alt.value("#111827"),
                    alt.Color(f"{color_field}:N", title=color_title),
                ),
                shape=alt.Shape(
                    "point_kind:N",
                    title="포인트 유형",
                    scale=alt.Scale(domain=["chunk", "query"], range=["circle", "diamond"]),
                ),
                size=alt.Size("point_size:Q", legend=None),
                opacity=alt.condition("datum.is_focus === true", alt.value(0.95), alt.value(0.15)),
                stroke=alt.condition(
                    "datum.is_query_cluster === true",
                    alt.value("black"),
                    alt.value("white"),
                ),
                strokeWidth=alt.condition(
                    "datum.is_query_cluster === true",
                    alt.value(1.5),
                    alt.value(0.2),
                ),
                tooltip=[
                    alt.Tooltip("cluster_id:N", title="Cluster"),
                    alt.Tooltip("chunk_label:N", title="라벨"),
                    alt.Tooltip("content_type:N", title="타입"),
                    alt.Tooltip("file_name:N", title="파일"),
                    alt.Tooltip("page_number:Q", title="페이지"),
                    alt.Tooltip("department:N", title="부문"),
                    alt.Tooltip("year:N", title="연도"),
                    alt.Tooltip("section_title:N", title="섹션"),
                    alt.Tooltip("preview:N", title="미리보기"),
                ],
            )
            .interactive()
            .properties(height=520)
        )
        st.altair_chart(chart, use_container_width=True, key=f"cluster_map_{color_field}")
        if show_label_text and color_field == "chunk_label":
            label_centers = (
                df.groupby("chunk_label", as_index=False)
                .agg(chunks=("chunk_label", "size"))
                .sort_values("chunks", ascending=False)
                .head(12)
            )
            if not label_centers.empty:
                st.caption(
                    "상위 라벨: " + ", ".join(
                        f"{row.chunk_label}({row.chunks})"
                        for row in label_centers.itertuples(index=False)
                    )
                )
        st.caption("차트 클릭 선택은 현재 Streamlit 제약으로 지원되지 않아, 위 `상세 보기` 선택기로 동일한 필터링을 제공합니다.")

        summary_group = "chunk_label" if color_field == "chunk_label" else "cluster_id"
        summary_source = df[df[color_field].astype(str).isin(focus_values)] if focus_values else df
        summary = (
            summary_source.groupby(summary_group, as_index=False)
            .agg(chunks=(summary_group, "size"), files=("file_name", "nunique"))
            .sort_values(["chunks", summary_group], ascending=[False, True])
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if focus_values:
            selected_text = ", ".join(focus_values)
            st.markdown(f"**선택한 {color_title}:** {selected_text}")
            detail_df = (
                df[df[color_field].astype(str).isin(focus_values)][
                    ["cluster_id", "chunk_label", "content_type", "file_name", "page_number", "section_title", "preview"]
                ]
                .sort_values(["cluster_id", "file_name", "page_number"], ascending=[True, True, True])
                .head(50)
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        if query_cluster_meta:
            selected_text = ", ".join(
                f"{item.get('cluster_id')} (score={item.get('score', 0):.3f}, size={item.get('size', 0)})"
                for item in query_cluster_meta
            )
            st.caption(f"최근 질문이 선택한 클러스터: {selected_text}")
        st.caption(
            f"cluster_index 생성 시각: {cluster_meta.get('created_at', '-')} | "
            f"샘플링 전 청크 수: {stats['total_points']:,}"
        )


def render_document_cluster_map() -> None:
    with st.expander("🧭 문서 요약 클러스터 맵", expanded=True):
        df, stats, reducer, file_to_cluster = _build_document_cluster_dataframe()
        cluster_meta = stats.get("cluster_meta") or {}
        if df.empty:
            st.info("문서 요약 클러스터가 없습니다. `요약 인덱스 재생성`을 먼저 실행하세요.")
            return

        retrieval_overlay = get_latest_retrieval_overlay()
        query_df, guided_clusters, guided_files = _build_document_query_overlay(
            retrieval_overlay,
            reducer,
            file_to_cluster,
        )

        control_col1, control_col2 = st.columns([1, 1])
        color_mode = control_col1.selectbox(
            "문서 맵 색상 기준",
            options=[("document_cluster_id", "문서 클러스터"), ("department", "부문"), ("year", "연도")],
            format_func=lambda item: item[1],
            key="document_cluster_color_mode",
        )
        cluster_options = sorted(df["document_cluster_id"].unique(), key=lambda value: int(value))
        selected_clusters = control_col2.multiselect(
            "문서 클러스터 선택",
            options=cluster_options,
            default=[],
            placeholder="전체 문서 클러스터",
            key="document_cluster_select",
        )
        if selected_clusters:
            df = df[df["document_cluster_id"].isin(selected_clusters)]
            guided_clusters = [cluster for cluster in guided_clusters if cluster in selected_clusters]

        color_field = color_mode[0]
        chart_df = df.copy()
        chart_df["point_kind"] = "document"
        chart_df["point_size"] = np.where(chart_df["document_cluster_id"].isin(guided_clusters), 180, 100)
        chart_df["is_guided_cluster"] = chart_df["document_cluster_id"].isin(guided_clusters)

        if not query_df.empty:
            query_points = query_df.copy()
            query_points["point_kind"] = "query"
            query_points["document_cluster_id"] = "질문"
            query_points["file_name"] = ""
            query_points["department"] = ""
            query_points["year"] = ""
            query_points["project_name"] = ""
            query_points["top_labels"] = ""
            query_points["top_sections"] = ""
            query_points["keywords"] = ""
            query_points["chunk_count"] = 0
            query_points["summary_text"] = query_points["label"]
            query_points["point_size"] = 260
            query_points["is_guided_cluster"] = False
            chart_df = pd.concat([chart_df, query_points], ignore_index=True, sort=False)

        meta_cols = st.columns(4)
        meta_cols[0].metric("문서 수", f"{stats['total_documents']:,}")
        meta_cols[1].metric("문서 클러스터", cluster_meta.get("n_clusters", stats["clusters"]))
        meta_cols[2].metric("색상 기준", color_mode[1])
        meta_cols[3].metric("생성 시각", str(cluster_meta.get("created_at", "-"))[:19])
        st.caption(
            f"알고리즘: {cluster_meta.get('algorithm', 'unknown')} | "
            f"noise 문서: {cluster_meta.get('noise_documents', 0)} | "
            f"min_cluster_size={cluster_meta.get('params', {}).get('min_cluster_size', '-')}"
        )

        chart = (
            alt.Chart(chart_df)
            .mark_point(filled=True)
            .encode(
                x=alt.X("x:Q", title="PCA-1"),
                y=alt.Y("y:Q", title="PCA-2"),
                color=alt.condition(
                    "datum.point_kind === 'query'",
                    alt.value("#111827"),
                    alt.Color(f"{color_field}:N", title=color_mode[1]),
                ),
                shape=alt.Shape(
                    "point_kind:N",
                    scale=alt.Scale(domain=["document", "query"], range=["circle", "diamond"]),
                    title="포인트 유형",
                ),
                size=alt.Size("point_size:Q", legend=None),
                stroke=alt.condition(
                    "datum.is_guided_cluster === true",
                    alt.value("black"),
                    alt.value("white"),
                ),
                strokeWidth=alt.condition(
                    "datum.is_guided_cluster === true",
                    alt.value(1.8),
                    alt.value(0.3),
                ),
                tooltip=[
                    alt.Tooltip("document_cluster_id:N", title="문서 클러스터"),
                    alt.Tooltip("file_name:N", title="파일"),
                    alt.Tooltip("department:N", title="부문"),
                    alt.Tooltip("year:N", title="연도"),
                    alt.Tooltip("project_name:N", title="프로젝트"),
                    alt.Tooltip("chunk_count:Q", title="청크 수"),
                    alt.Tooltip("top_labels:N", title="주요 라벨"),
                    alt.Tooltip("top_sections:N", title="주요 섹션"),
                    alt.Tooltip("keywords:N", title="키워드"),
                ],
            )
            .interactive()
            .properties(height=500)
        )
        st.altair_chart(chart, use_container_width=True, key="document_cluster_map")

        summary = (
            df.groupby("document_cluster_id", as_index=False)
            .agg(
                documents=("file_name", "size"),
                departments=("department", "nunique"),
                years=("year", "nunique"),
            )
            .sort_values(["documents", "document_cluster_id"], ascending=[False, True])
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption("`document_cluster_id = -1`은 HDBSCAN이 어느 군집에도 안정적으로 속하지 않는 문서로 판단한 noise입니다.")

        detail_df = df[
            [
                "document_cluster_id",
                "file_name",
                "department",
                "year",
                "project_name",
                "chunk_count",
                "top_labels",
                "top_sections",
                "keywords",
            ]
        ].sort_values(["document_cluster_id", "file_name"], ascending=[True, True])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        if guided_files:
            st.caption("최근 질문이 요약 인덱스로 우선 선택한 문서: " + ", ".join(guided_files[:6]))


def render_cluster_quality_report(rag) -> None:
    st.subheader("클러스터 품질 리포트")
    cluster_meta = load_cluster_index()
    if not cluster_meta:
        st.info("클러스터 메타데이터가 없습니다. 먼저 인덱싱 또는 클러스터 재계산을 실행하세요.")
        return

    rows = get_corpus_rows(rag.vectorstore)
    if not rows:
        st.info("품질을 계산할 벡터 데이터가 없습니다.")
        return

    sample_size = st.selectbox(
        "지표 계산 샘플 수",
        options=[1000, 2000, 3000, 5000],
        index=1,
        key="cluster_quality_sample_size",
        help="Silhouette/Davies-Bouldin/Calinski-Harabasz는 샘플 기반으로 계산합니다.",
    )

    signature = (cluster_meta.get("created_at"), len(rows), sample_size)
    if st.session_state.cluster_quality_signature != signature:
        st.session_state.cluster_quality_report = None
        st.session_state.cluster_quality_signature = signature

    if st.button("리포트 계산/새로고침", key="cluster_quality_refresh"):
        st.session_state.cluster_quality_report = build_cluster_quality_report(rows, sample_size=sample_size)

    report = st.session_state.cluster_quality_report
    if report is None:
        st.info("리포트를 보려면 `리포트 계산/새로고침`을 누르세요.")
        return

    summary = report.get("summary", {})
    metrics = report.get("metrics", {})
    cluster_rows = report.get("clusters", [])

    metric_cols = st.columns(6)
    metric_cols[0].metric("총 청크", f"{summary.get('total_points', 0):,}")
    metric_cols[1].metric("클러스터", summary.get("clusters", 0))
    metric_cols[2].metric("샘플 수", f"{summary.get('sampled_points', 0):,}")
    metric_cols[3].metric("최소 크기", summary.get("min_cluster_size", 0))
    metric_cols[4].metric("중간 크기", summary.get("median_cluster_size", 0))
    metric_cols[5].metric("최대 크기", summary.get("max_cluster_size", 0))

    score_cols = st.columns(3)
    silhouette = metrics.get("silhouette_cosine")
    dbi = metrics.get("davies_bouldin")
    ch = metrics.get("calinski_harabasz")
    score_cols[0].metric("Silhouette", f"{silhouette:.3f}" if silhouette is not None else "-")
    score_cols[1].metric("Davies-Bouldin", f"{dbi:.3f}" if dbi is not None else "-")
    score_cols[2].metric("Calinski-Harabasz", f"{ch:.1f}" if ch is not None else "-")

    st.caption(
        "해석 가이드: Silhouette는 클수록 좋고, Davies-Bouldin은 작을수록 좋고, "
        "Calinski-Harabasz는 클수록 좋습니다. 이 값만으로 충분하지 않고 실제 검색 품질과 함께 봐야 합니다."
    )

    if cluster_rows:
        cluster_df = pd.DataFrame(cluster_rows).sort_values(["chunks", "cluster_id"], ascending=[False, True])
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)


def render_summary_index_report() -> None:
    st.subheader("요약 인덱스 리포트")
    summary_index = load_summary_index()
    if not summary_index:
        st.info("요약 인덱스가 없습니다. 인덱싱 또는 `요약 인덱스 재생성`을 먼저 실행하세요.")
        return

    documents = summary_index.get("documents", [])
    sections = summary_index.get("sections", [])
    metric_cols = st.columns(4)
    created_at = str(summary_index.get("created_at", "-"))
    metric_cols[0].metric("총 엔트리", f"{summary_index.get('total_entries', 0):,}")
    metric_cols[1].metric("문서 요약", f"{len(documents):,}")
    metric_cols[2].metric("섹션 요약", f"{len(sections):,}")
    metric_cols[3].metric("생성 시각", created_at[:19])

    if documents:
        doc_df = pd.DataFrame(
            [
                {
                    "file_name": entry.get("payload", {}).get("file_name"),
                    "department": entry.get("payload", {}).get("department"),
                    "year": entry.get("payload", {}).get("year"),
                    "chunk_count": entry.get("payload", {}).get("chunk_count"),
                    "top_labels": ", ".join(entry.get("payload", {}).get("top_labels", [])[:4]),
                    "top_sections": ", ".join(entry.get("payload", {}).get("top_sections", [])[:4]),
                    "summary_excerpt": " / ".join(entry.get("payload", {}).get("summary_excerpt", [])[:3]),
                }
                for entry in documents
            ]
        ).sort_values(["chunk_count", "file_name"], ascending=[False, True])
        with st.expander("문서 요약 보기", expanded=False):
            st.dataframe(doc_df, use_container_width=True, hide_index=True)

    if sections:
        section_df = pd.DataFrame(
            [
                {
                    "file_name": entry.get("payload", {}).get("file_name"),
                    "chunk_label": entry.get("payload", {}).get("chunk_label"),
                    "content_type": entry.get("payload", {}).get("content_type"),
                    "section_title": entry.get("payload", {}).get("section_title"),
                    "chunk_count": entry.get("payload", {}).get("chunk_count"),
                    "keywords": ", ".join(entry.get("payload", {}).get("keywords", [])[:6]),
                    "summary_excerpt": " / ".join(entry.get("payload", {}).get("summary_excerpt", [])[:2]),
                }
                for entry in sections
            ]
        ).sort_values(["chunk_count", "file_name"], ascending=[False, True])
        with st.expander("섹션 요약 보기", expanded=False):
            st.dataframe(section_df.head(200), use_container_width=True, hide_index=True)


def render_analysis_tab(rag) -> None:
    render_summary_index_report()
    render_document_cluster_map()
    render_cluster_map(rag)
    render_cluster_quality_report(rag)
