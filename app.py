"""
제안서 챗봇 - Streamlit UI
"""
import streamlit as st
from pathlib import Path
import re
import logging
import traceback
import faulthandler

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain
from src.vectordb import upsert_vectorstore, get_indexed_file_names
from src.pdf_processor import process_pdf
from src.chunker import chunk_document
from src.index_logs import load_index_log, write_index_logs
from config import DATA_DIR

LOG_PATH = Path(__file__).parent / "app_runtime.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
try:
    _fh = open(LOG_PATH, "a", encoding="utf-8")
    faulthandler.enable(_fh)
except Exception:
    pass
logging.info("app.py imported")


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
    if "last_chunk_preview" not in st.session_state:
        st.session_state.last_chunk_preview = []
    if "rag_init_error" not in st.session_state:
        st.session_state.rag_init_error = None


class SafeRAGFallback:
    """초기화 실패 시 UI가 죽지 않도록 하는 폴백 객체"""

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
    """RAG 체인 로드"""
    if st.session_state.rag_chain is None:
        with st.spinner("🔄 챗봇을 초기화하는 중..."):
            try:
                st.session_state.rag_chain = RAGChain()
                st.session_state.rag_init_error = None
            except Exception as e:
                st.session_state.rag_init_error = str(e)
                st.session_state.rag_chain = SafeRAGFallback(e)
                st.error(f"초기화 실패: {e}")
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


def infer_department(file_name: str) -> str:
    """파일명 기반 부문 자동 분류"""
    name = file_name.upper()
    if "SI" in name or "SYSTEM INTEGRATION" in name:
        return "SI부문"
    if "R&D" in name or "RND" in name or "RD" in name:
        return "R&D부문"
    return "R&D부문"


def save_uploaded_pdf(uploaded_file, department: str) -> Path:
    """업로드된 PDF를 data 디렉토리에 저장"""
    target_dir = DATA_DIR / department
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / uploaded_file.name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def display_pipeline_info(pipeline_info: dict):
    """RAG 파이프라인 단계별 정보를 expander로 표시"""
    if not pipeline_info:
        return

    with st.expander("🔬 RAG 파이프라인 처리 과정", expanded=False):
        # ── Step 1: 쿼리 분석 ────────────────────
        qa = pipeline_info.get("query_analysis", {})
        if qa:
            st.markdown("#### 1️⃣ 쿼리 분석")
            st.metric("페이지 필터", qa.get("page_filter") or "없음")
            st.code(f"원본 쿼리: {qa.get('original_query', '')}", language=None)
            st.divider()

        # ── Step 2: 검색 결과 ────────────────────
        ret = pipeline_info.get("retrieval", {})
        if ret:
            st.markdown("#### 2️⃣ Retrieval")
            col1, col2, col3 = st.columns(3)
            col1.metric("임베딩 모델", ret.get("model", ""))
            col2.metric("검색된 청크", f"{ret.get('results_count', 0)}개")
            col3.metric("검색 모드", ret.get("retriever_mode", "dense"))
            if ret.get("retriever_mode") in {"hybrid", "hybrid_cluster"}:
                st.caption(
                    f"dense={ret.get('dense_weight')} / sparse={ret.get('sparse_weight')} | "
                    f"cluster={ret.get('cluster_enabled')} ({ret.get('clusters')}개)"
                )

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
    logging.info("main() start")
    initialize_session_state()

    # 헤더
    st.markdown('<h1 class="main-header">📄 제안서 챗봇</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # RAG 체인 로드
    rag = load_rag_chain()
    logging.info("rag loaded; ready=%s", getattr(rag, "is_ready", lambda: False)())

    # 사이드바
    with st.sidebar:
        st.header("📥 PDF 업로드")
        uploaded_files = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            st.caption(f"선택된 파일: {len(uploaded_files)}개")
            if st.button("선택 파일 인덱싱"):
                try:
                    all_docs = []
                    all_chunks = []
                    overwrite_files = []
                    preview = []

                    for uploaded_file in uploaded_files:
                        auto_department = infer_department(uploaded_file.name)
                        saved_path = save_uploaded_pdf(uploaded_file, auto_department)
                        doc = process_pdf(saved_path, auto_department)
                        if not doc:
                            continue
                        chunks = chunk_document(doc)
                        if not chunks:
                            continue
                        all_docs.append(doc)
                        all_chunks.extend(chunks)
                        overwrite_files.append(doc.metadata.file_name)
                        for chunk in chunks[:3]:
                            meta = chunk.metadata or {}
                            preview.append(
                                {
                                    "page_number": meta.get("page_number", "-"),
                                    "length": len(chunk.page_content),
                                    "text": (
                                        chunk.page_content[:300] + "..."
                                        if len(chunk.page_content) > 300
                                        else chunk.page_content
                                    ),
                                }
                            )

                    if not all_chunks:
                        st.error("인덱싱 가능한 텍스트를 찾지 못했습니다.")
                    else:
                        rag.vectorstore = upsert_vectorstore(
                            all_chunks,
                            rag.vectorstore,
                            overwrite_files=overwrite_files,
                        )
                        write_index_logs(all_docs, all_chunks)
                        rag.refresh_retriever()
                        st.session_state.last_chunk_preview = preview[:10]
                        st.success(f"인덱싱 완료: 파일 {len(all_docs)}개, 청크 {len(all_chunks)}개")
                        st.rerun()
                except Exception as e:
                    st.error(f"업로드/인덱싱 오류: {e}")

        st.markdown("---")
        st.header("📚 인덱싱된 파일")
        try:
            names = get_indexed_file_names(rag.vectorstore)
            st.caption(f"총 {len(names)}개")
            if names:
                with st.expander("파일 목록 보기", expanded=False):
                    for n in names:
                        st.write(f"- {n}")
        except Exception as e:
            st.caption(f"파일 목록 조회 실패: {e}")

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

    # 최근 업로드 청킹 결과 미리보기
    if st.session_state.last_chunk_preview:
        with st.expander("🧩 청킹 결과 미리보기 (최근 업로드)", expanded=False):
            for i, item in enumerate(st.session_state.last_chunk_preview, 1):
                st.markdown(
                    f"**#{i}** | p{item['page_number']} | {item['length']}자"
                )
                st.text(item["text"])
                st.markdown("---")

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

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 답변을 생성하는 중..."):
                answer, docs, pipeline_info = rag.ask(
                    question=prompt,
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
    # python app.py 로 직접 실행하면 서버가 바로 종료되는 것처럼 보일 수 있어 명확히 안내
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            print("이 앱은 직접 실행이 아니라 아래 명령으로 실행해야 합니다:")
            print("python -m streamlit run app.py --server.fileWatcherType none")
            raise SystemExit(0)
    except ImportError:
        pass

    try:
        main()
    except Exception as e:
        logging.error("Unhandled exception: %s", e)
        logging.error(traceback.format_exc())
        raise
