from __future__ import annotations

import streamlit as st

from src.rag_chain import RAGChain


def initialize_session_state() -> None:
    """앱에서 공통으로 쓰는 세션 상태를 초기화한다."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "last_chunk_preview" not in st.session_state:
        st.session_state.last_chunk_preview = []
    if "rag_init_error" not in st.session_state:
        st.session_state.rag_init_error = None
    if "cluster_quality_report" not in st.session_state:
        st.session_state.cluster_quality_report = None
    if "cluster_quality_signature" not in st.session_state:
        st.session_state.cluster_quality_signature = None


class SafeRAGFallback:
    """초기화 실패 시 UI를 유지하기 위한 폴백 객체"""

    def __init__(self, error: Exception) -> None:
        self.vectorstore = None
        self._error = str(error)

    def is_ready(self) -> bool:
        return False

    def ask(self, *args, **kwargs):
        return (
            f"초기화 실패로 답변할 수 없습니다: {self._error}",
            [],
            {"generation": {"status": f"초기화 실패: {self._error}"}},
        )


def load_rag_chain():
    """세션에 RAG 체인이 없으면 초기화한다."""
    if st.session_state.rag_chain is None:
        with st.spinner("🔄 챗봇을 초기화하는 중..."):
            try:
                st.session_state.rag_chain = RAGChain()
                st.session_state.rag_init_error = None
            except Exception as error:
                st.session_state.rag_init_error = str(error)
                st.session_state.rag_chain = SafeRAGFallback(error)
                st.error(f"초기화 실패: {error}")
    return st.session_state.rag_chain
