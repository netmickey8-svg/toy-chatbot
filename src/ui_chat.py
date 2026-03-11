from __future__ import annotations

import streamlit as st

from src.index_logs import load_index_log
from src.retrieval_pipeline import HYBRID_RETRIEVER_MODES
from src.ui_indexing import render_index_log


def display_pipeline_info(pipeline_info: dict):
    """RAG 파이프라인 단계별 정보를 expander로 표시"""
    if not pipeline_info:
        return

    with st.expander("🔬 RAG 파이프라인 처리 과정", expanded=False):
        qa = pipeline_info.get("query_analysis", {})
        if qa:
            st.markdown("#### 1️⃣ 쿼리 분석")
            st.metric("페이지 필터", qa.get("page_filter") or "없음")
            st.code(f"원본 쿼리: {qa.get('original_query', '')}", language=None)
            st.divider()

        ret = pipeline_info.get("retrieval", {})
        if ret:
            st.markdown("#### 2️⃣ Retrieval")
            col1, col2, col3 = st.columns(3)
            col1.metric("임베딩 모델", ret.get("model", ""))
            col2.metric("검색된 청크", f"{ret.get('results_count', 0)}개")
            col3.metric("검색 모드", ret.get("retriever_mode", "dense"))
            if ret.get("retriever_mode") in HYBRID_RETRIEVER_MODES:
                st.caption(
                    f"dense={ret.get('dense_weight')} / sparse={ret.get('sparse_weight')} | "
                    f"cluster={ret.get('cluster_enabled')} ({ret.get('clusters')}개)"
                )
            guidance = ret.get("summary_guidance") or {}
            guided_files = guidance.get("file_names", []) if isinstance(guidance, dict) else []
            guided_sections = guidance.get("section_filters", []) if isinstance(guidance, dict) else []
            focus_labels = ret.get("focus_labels", [])
            if guided_files:
                st.caption("요약 인덱스 후보 문서: " + ", ".join(guided_files[:6]))
            if guided_sections:
                section_text = ", ".join(
                    f"{item.get('chunk_label', '기타')} / {item.get('section_title', '본문')} / {item.get('content_type', 'text')}"
                    for item in guided_sections[:4]
                )
                st.caption("요약 인덱스 후보 섹션: " + section_text)
            if focus_labels:
                st.caption("질문 의도 기반 우선 라벨: " + ", ".join(focus_labels))

            chunks = ret.get("chunks", [])
            if chunks:
                st.markdown("**검색된 청크 (유사도순):**")
                for chunk in chunks:
                    sim = chunk.get("similarity", 0)
                    sim_pct = f"{sim * 100:.1f}%"
                    with st.container():
                        line = (
                            f"**#{chunk['rank']}** | 유사도: `{sim_pct}` | "
                            f"📄 {chunk.get('file_name', '?')[:40]} | "
                            f"{chunk.get('department', '')} | "
                            f"{chunk.get('year', '')}년 p{chunk.get('page', '-')}"
                        )
                        if chunk.get("content_type") or chunk.get("chunk_label"):
                            line += (
                                f" | {chunk.get('content_type', 'text')}"
                                f" / {chunk.get('chunk_label', '기타')}"
                            )
                        if chunk.get("hybrid_score") is not None:
                            line += (
                                f" | d={chunk.get('dense_score')} "
                                f"s={chunk.get('sparse_score')} "
                                f"h={chunk.get('hybrid_score')}"
                            )
                        if chunk.get("cluster_id") is not None:
                            line += f" | cluster={chunk.get('cluster_id')}"
                        st.markdown(line)
                        st.text(chunk.get("preview", ""))
                        st.markdown("---")
            st.divider()

        prm = pipeline_info.get("prompt", {})
        if prm:
            st.markdown("#### 3️⃣ 프롬프트 생성")
            col1, col2 = st.columns(2)
            col1.metric("컨텍스트 길이", f"{prm.get('context_length', 0):,}자")
            col2.metric("전체 프롬프트 길이", f"{prm.get('total_prompt_length', 0):,}자")
            with st.expander("프롬프트 미리보기", expanded=False):
                st.code(prm.get("prompt_preview", ""), language=None)
            st.divider()

        gen = pipeline_info.get("generation", {})
        if gen:
            st.markdown("#### 4️⃣ LLM 답변 생성")
            status = gen.get("status", "")
            if status == "성공":
                col1, col2, col3 = st.columns(3)
                col1.metric("모델", gen.get("model", ""))
                col2.metric("Temperature", gen.get("temperature", ""))
                col3.metric("답변 길이", f"{gen.get('answer_length', 0):,}자")
            else:
                st.warning(f"상태: {status}")


def render_sources(docs: list) -> None:
    """참조 문서 카드 렌더링"""
    with st.expander("📚 참조된 제안서"):
        for doc in docs:
            meta = doc.metadata
            st.markdown(
                f"""
            <div class="source-card">
                <strong>📄 {meta.get('file_name', 'Unknown')[:50]}...</strong><br>
                📁 {meta.get('department', '')} | 📅 {meta.get('year', '')}년 | 📄 {meta.get('page_number', '')}p<br>
                🏷️ {meta.get('content_type', 'text')} / {meta.get('chunk_label', '기타')} | 섹션: {meta.get('section_title', '본문')}
            </div>
            """,
                unsafe_allow_html=True,
            )


def display_chat_history():
    """채팅 기록 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "pipeline_info" in message and message["pipeline_info"]:
                display_pipeline_info(message["pipeline_info"])
            if "sources" in message and message["sources"]:
                render_sources(message["sources"])


def render_chat_tab(rag) -> None:
    """챗봇 탭 렌더링"""
    if st.session_state.last_chunk_preview:
        with st.expander("🧩 청킹 결과 미리보기 (최근 업로드)", expanded=False):
            for index, item in enumerate(st.session_state.last_chunk_preview, 1):
                st.markdown(f"**#{index}** | p{item['page_number']} | {item['length']}자")
                st.text(item["text"])
                st.markdown("---")

    display_chat_history()

    if prompt := st.chat_input("제안서에 대해 질문하세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("🤔 답변을 생성하는 중..."):
                answer, docs, pipeline_info = rag.ask(question=prompt)

            st.markdown(answer)
            display_pipeline_info(pipeline_info)

            if docs:
                render_sources(docs)

            if docs:
                file_names = []
                for doc in docs:
                    name = doc.metadata.get("file_name")
                    if name and name not in file_names:
                        file_names.append(name)
                if file_names:
                    with st.expander("🧾 인덱싱 로그"):
                        for name in file_names:
                            log = load_index_log(name)
                            render_index_log(log)
                            st.markdown("---")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": docs,
                "pipeline_info": pipeline_info,
            }
        )
