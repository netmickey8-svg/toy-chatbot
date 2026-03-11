from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import CLUSTER_N_CLUSTERS, DATA_DIR
from src.chunker import chunk_all_documents, chunk_document
from src.index_logs import write_index_logs
from src.pdf_processor import process_all_pdfs, process_pdf
from src.vectordb import (
    create_vectorstore,
    get_indexed_file_names,
    recluster_collection,
    rebuild_summary_index,
    upsert_vectorstore,
)


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
    with open(target_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())
    return target_path


def list_data_pdf_entries(indexed_names: set[str]) -> list[dict]:
    """data 디렉토리 내 PDF 파일과 인덱싱 여부를 함께 반환"""
    if not DATA_DIR.exists():
        return []

    entries = []
    for path in sorted(DATA_DIR.rglob("*.pdf")):
        entries.append(
            {
                "label": str(path.relative_to(DATA_DIR)),
                "path": path,
                "file_name": path.name,
                "department": path.parent.name,
                "indexed": path.name in indexed_names,
            }
        )
    return entries


def build_preview_from_chunks(chunks: list, limit: int = 10) -> list[dict]:
    """최근 인덱싱된 청크 미리보기 생성"""
    preview = []
    for chunk in chunks[:limit]:
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
    return preview


def process_selected_paths(paths: list[Path]) -> tuple[list, list]:
    """선택한 PDF 파일들만 처리하여 문서/청크를 생성"""
    documents = []
    chunks = []
    for path in paths:
        department = path.parent.name
        document = process_pdf(path, department)
        if not document:
            continue
        doc_chunks = chunk_document(document)
        if not doc_chunks:
            continue
        documents.append(document)
        chunks.extend(doc_chunks)
    return documents, chunks


def render_sidebar(rag) -> tuple[set[str], list[dict]]:
    """인덱싱/업로드 관련 사이드바를 렌더링한다."""
    indexed_names = set(get_indexed_file_names(getattr(rag, "vectorstore", None)))
    pdf_entries = list_data_pdf_entries(indexed_names)
    entry_by_label = {entry["label"]: entry for entry in pdf_entries}

    with st.sidebar:
        st.header("📥 PDF 업로드")
        uploaded_files = st.file_uploader("PDF 파일 업로드", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            st.caption(f"선택된 파일: {len(uploaded_files)}개")
            if st.button("선택 파일 저장"):
                try:
                    for uploaded_file in uploaded_files:
                        auto_department = infer_department(uploaded_file.name)
                        save_uploaded_pdf(uploaded_file, auto_department)
                    st.success("파일 저장 완료. 아래 인덱싱 관리에서 전체 인덱싱 또는 재인덱싱을 실행하세요.")
                    st.rerun()
                except Exception as error:
                    st.error(f"업로드 오류: {error}")

        st.markdown("---")
        st.header("🛠️ 인덱싱 관리")
        st.caption(f"data 폴더 PDF: {len(pdf_entries)}개 | 인덱싱된 파일: {len(indexed_names)}개")

        if st.button("전체 인덱싱 실행", disabled=not pdf_entries):
            try:
                with st.spinner("전체 PDF를 인덱싱하는 중..."):
                    all_docs = process_all_pdfs(DATA_DIR)
                    all_chunks = chunk_all_documents(all_docs)
                    if not all_chunks:
                        st.error("인덱싱 가능한 텍스트를 찾지 못했습니다.")
                    else:
                        rag.vectorstore = create_vectorstore(all_chunks)
                        write_index_logs(all_docs, all_chunks)
                        rag.refresh_retriever()
                        st.session_state.last_chunk_preview = build_preview_from_chunks(all_chunks)
                        st.success(f"전체 인덱싱 완료: 파일 {len(all_docs)}개, 청크 {len(all_chunks)}개")
                        st.rerun()
            except Exception as error:
                st.error(f"전체 인덱싱 오류: {error}")

        if st.button("클러스터만 재계산", disabled=not indexed_names):
            try:
                with st.spinner("저장된 벡터 기준으로 클러스터를 다시 계산하는 중..."):
                    cluster_meta = recluster_collection(getattr(rag, "vectorstore", None))
                    rag.refresh_retriever()
                    if cluster_meta:
                        st.success(
                            "클러스터 재계산 완료: "
                            f"{cluster_meta.get('n_clusters', CLUSTER_N_CLUSTERS)}개 클러스터"
                        )
                    else:
                        st.warning("클러스터 재계산 결과가 비어 있습니다.")
                    st.rerun()
            except Exception as error:
                st.error(f"클러스터 재계산 오류: {error}")

        if st.button("요약 인덱스 재생성", disabled=not indexed_names):
            try:
                with st.spinner("저장된 벡터 기준으로 요약 인덱스를 다시 생성하는 중..."):
                    summary_index = rebuild_summary_index(getattr(rag, "vectorstore", None))
                    rag.refresh_retriever()
                    if summary_index:
                        st.success(
                            "요약 인덱스 재생성 완료: "
                            f"문서 {len(summary_index.get('documents', []))}개 / "
                            f"섹션 {len(summary_index.get('sections', []))}개"
                        )
                    else:
                        st.warning("요약 인덱스 재생성 결과가 비어 있습니다.")
                    st.rerun()
            except Exception as error:
                st.error(f"요약 인덱스 재생성 오류: {error}")

        selectable_labels = [entry["label"] for entry in pdf_entries]
        selected_labels = st.multiselect(
            "인덱싱할 파일 선택",
            options=selectable_labels,
            placeholder="새 파일과 기존 파일을 모두 선택할 수 있습니다.",
        )
        st.caption(
            "선택 파일 인덱싱은 새 파일/기존 파일을 모두 처리합니다. "
            "기존 파일을 다시 선택하면 해당 파일만 덮어써서 재인덱싱합니다."
        )
        st.caption(
            "클러스터 수만 바꾸고 싶으면 `.env`의 "
            "`CLUSTER_N_CLUSTERS`를 수정한 뒤 `클러스터만 재계산`을 실행하면 됩니다."
        )
        if st.button("선택 파일 인덱싱", disabled=not selected_labels):
            try:
                selected_paths = [entry_by_label[label]["path"] for label in selected_labels]
                with st.spinner("선택 파일을 인덱싱하는 중..."):
                    selected_docs, selected_chunks = process_selected_paths(selected_paths)
                    if not selected_chunks:
                        st.error("선택한 파일에서 인덱싱 가능한 텍스트를 찾지 못했습니다.")
                    else:
                        overwrite_files = [doc.metadata.file_name for doc in selected_docs]
                        rag.vectorstore = upsert_vectorstore(
                            selected_chunks,
                            collection=getattr(rag, "vectorstore", None),
                            overwrite_files=overwrite_files,
                        )
                        write_index_logs(selected_docs, selected_chunks)
                        rebuild_summary_index(rag.vectorstore)
                        recluster_collection(rag.vectorstore)
                        rag.refresh_retriever()
                        st.session_state.last_chunk_preview = build_preview_from_chunks(selected_chunks)
                        st.success(
                            f"선택 파일 인덱싱 완료: 파일 {len(selected_docs)}개, 청크 {len(selected_chunks)}개"
                        )
                        st.rerun()
            except Exception as error:
                st.error(f"선택 파일 인덱싱 오류: {error}")

        st.markdown("---")
        st.header("📚 인덱싱된 파일")
        st.caption(f"총 {len(indexed_names)}개")
        if indexed_names:
            with st.expander("파일 목록 보기", expanded=False):
                for name in sorted(indexed_names):
                    st.write(f"- {name}")

        st.markdown("---")
        st.header("🗂️ data 폴더 파일")
        if pdf_entries:
            with st.expander("PDF 목록 보기", expanded=False):
                file_df = pd.DataFrame(
                    [
                        {
                            "경로": entry["label"],
                            "부문": entry["department"],
                            "인덱싱됨": "Y" if entry["indexed"] else "N",
                        }
                        for entry in pdf_entries
                    ]
                )
                st.dataframe(file_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            """
        ### 💡 사용 예시
        - "블록체인 관련 제안서 알려줘"
        - "2025년 플랫폼 프로젝트는?"
        - "메타버스 프로젝트 내용 요약해줘"
        - "대구시 관련 사업 있어?"
        """
        )

        st.markdown("---")
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state.messages = []
            st.rerun()

    return indexed_names, pdf_entries
