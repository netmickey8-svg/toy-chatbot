"""
제안서 챗봇 - Streamlit UI
"""
import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain
from src.vectordb import get_indexed_file_names
from src.index_logs import load_index_log


# 페이지 설정
st.set_page_config(
    page_title="제안서 챗봇",
    page_icon="📄",
    layout="wide"
)

# 커스텀 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .source-card {
        background-color: #FFF3E0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


def load_rag_chain():
    """RAG 체인 로드"""
    if st.session_state.rag_chain is None:
        with st.spinner("🔄 챗봇을 초기화하는 중..."):
            st.session_state.rag_chain = RAGChain()
    return st.session_state.rag_chain


def render_index_log(log: dict) -> None:
    """인덱싱 로그 표시"""
    if not log:
        st.info("인덱싱 로그를 찾을 수 없습니다.")
        return

    st.markdown(
        f"**파일명:** {log.get('file_name', '-')}\n\n"
        f"**부문/연도:** {log.get('department', '-')} / {log.get('year', '-')}\n\n"
        f"**페이지:** {log.get('extracted_pages', 0)}/{log.get('total_pages', 0)}\n\n"
        f"**텍스트/표/OCR 페이지:** "
        f"{log.get('text_pages', 0)}/{log.get('table_pages', 0)}/{log.get('ocr_pages', 0)}\n\n"
        f"**총 문자수:** {log.get('total_chars', 0):,}\n\n"
        f"**총 청크:** {log.get('total_chunks', 0)}\n\n"
        f"**인덱싱 시각(UTC):** {log.get('indexed_at', '-')}"
    )

    chunks_per_page = log.get("chunks_per_page", {})
    if chunks_per_page:
        lines = []
        for page in sorted(chunks_per_page.keys(), key=lambda x: int(x)):
            lines.append(f"p{page}: {chunks_per_page[page]}")
        st.code("\n".join(lines), language=None)


def display_pipeline_info(pipeline_info: dict):
    """RAG 파이프라인 단계별 정보를 expander로 표시"""
    if not pipeline_info:
        return

    with st.expander("🔬 RAG 파이프라인 처리 과정", expanded=False):
        # ── Step 1: 쿼리 분석 ────────────────────
        qa = pipeline_info.get("query_analysis", {})
        if qa:
            st.markdown("#### 1️⃣ 쿼리 분석")
            col1, col2, col3 = st.columns(3)
            col1.metric("부문 필터", qa.get("department_filter", "-"))
            col2.metric("연도 필터", qa.get("year_filter", "-"))
            col3.metric("페이지 필터", qa.get("page_filter") or "없음")
            st.code(f"원본 쿼리: {qa.get('original_query', '')}", language=None)
            st.divider()

        # ── Step 2: 검색 결과 ────────────────────
        ret = pipeline_info.get("retrieval", {})
        if ret:
            st.markdown("#### 2️⃣ Dense Retrieval (벡터 검색)")
            col1, col2 = st.columns(2)
            col1.metric("임베딩 모델", ret.get("model", ""))
            col2.metric("검색된 청크", f"{ret.get('results_count', 0)}개")

            chunks = ret.get("chunks", [])
            if chunks:
                st.markdown("**검색된 청크 (유사도순):**")
                for chunk in chunks:
                    sim = chunk.get("similarity", 0)
                    sim_pct = f"{sim * 100:.1f}%"
                    with st.container():
                        st.markdown(
                            f"**#{chunk['rank']}** | 유사도: `{sim_pct}` | "
                            f"📄 {chunk.get('file_name', '?')[:40]} | "
                            f"{chunk.get('department', '')} | "
                            f"{chunk.get('year', '')}년 p{chunk.get('page', '-')}"
                        )
                        st.text(chunk.get("preview", ""))
                        st.markdown("---")
            st.divider()

        # ── Step 3: 프롬프트 ─────────────────────
        prm = pipeline_info.get("prompt", {})
        if prm:
            st.markdown("#### 3️⃣ 프롬프트 생성")
            col1, col2 = st.columns(2)
            col1.metric("컨텍스트 길이", f"{prm.get('context_length', 0):,}자")
            col2.metric("전체 프롬프트 길이", f"{prm.get('total_prompt_length', 0):,}자")
            with st.expander("프롬프트 미리보기", expanded=False):
                st.code(prm.get("prompt_preview", ""), language=None)
            st.divider()

        # ── Step 4: LLM 생성 ─────────────────────
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


def display_chat_history():
    """채팅 기록 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 파이프라인 정보 표시
            if "pipeline_info" in message and message["pipeline_info"]:
                display_pipeline_info(message["pipeline_info"])

            # 참조 문서 표시
            if "sources" in message and message["sources"]:
                with st.expander("📚 참조된 제안서"):
                    for doc in message["sources"]:
                        meta = doc.metadata
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 {meta.get('file_name', 'Unknown')[:50]}...</strong><br>
                            📁 {meta.get('department', '')} | 📅 {meta.get('year', '')}년 | 📄 {meta.get('page_number', '')}p
                        </div>
                        """, unsafe_allow_html=True)


def main():
    """메인 앱"""
    initialize_session_state()

    # 헤더
    st.markdown('<h1 class="main-header">📄 제안서 챗봇</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # RAG 체인 로드
    rag = load_rag_chain()

    # 사이드바 - 필터 옵션
    with st.sidebar:
        st.header("🔍 검색 필터")

        department = st.selectbox(
            "부문 선택",
            options=["전체", "R&D부문", "SI부문"],
            index=0
        )

        year = st.selectbox(
            "연도 선택",
            options=["전체", "2024", "2025", "2026"],
            index=0
        )

        file_options = ["전체"]
        indexed_files = []
        try:
            indexed_files = get_indexed_file_names(rag.vectorstore)
        except Exception:
            indexed_files = []
        if indexed_files:
            file_options += indexed_files

        file_name = st.selectbox(
            "파일 선택",
            options=file_options,
            index=0
        )

        st.markdown("---")
        st.markdown("""
        ### 💡 사용 예시
        - "블록체인 관련 제안서 알려줘"
        - "2025년 플랫폼 프로젝트는?"
        - "메타버스 프로젝트 내용 요약해줘"
        - "대구시 관련 사업 있어?"
        """)

        st.markdown("---")
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state.messages = []
            st.rerun()

    # 벡터스토어 확인
    if not rag.is_ready():
        st.error("""
        ⚠️ **벡터스토어가 없습니다!**

        먼저 다음 명령어로 PDF를 인덱싱하세요:
        ```bash
        python index_data.py
        ```
        """)
        return

    # 채팅 기록 표시
    display_chat_history()

    # 사용자 입력
    if prompt := st.chat_input("제안서에 대해 질문하세요..."):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # 필터 설정
        dept_filter = None if department == "전체" else department
        year_filter = None if year == "전체" else year
        file_filter = None if file_name == "전체" else file_name

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 답변을 생성하는 중..."):
                answer, docs, pipeline_info = rag.ask(
                    question=prompt,
                    department=dept_filter,
                    year=year_filter,
                    file_name=file_filter,
                )

            st.markdown(answer)

            # 파이프라인 처리 과정 표시
            display_pipeline_info(pipeline_info)

            # 참조 문서 표시
            if docs:
                with st.expander("📚 참조된 제안서"):
                    for doc in docs:
                        meta = doc.metadata
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 {meta.get('file_name', 'Unknown')[:50]}...</strong><br>
                            📁 {meta.get('department', '')} | 📅 {meta.get('year', '')}년 | 📄 {meta.get('page_number', '')}p
                        </div>
                        """, unsafe_allow_html=True)

            # 인덱싱 로그 표시
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

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": docs,
            "pipeline_info": pipeline_info
        })


if __name__ == "__main__":
    main()
