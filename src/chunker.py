"""
문서 청킹(분할) 모듈
====================
역할:
    - pdf_processor.py에서 추출된 전체 페이지 텍스트를 LLM이 처리하기 적합한
      크기의 청크(chunk)로 분할
    - 각 청크에 출처(파일명/부문/연도/페이지) 메타데이터를 유지

왜 청킹이 필요한가:
    - LLM 컨텍스트 한계: 전체 PDF를 한번에 LLM에 전달 불가
    - 검색 정확도: 작은 단위로 나눌수록 의미 있는 청크 단위로 검색 가능
    - bge-m3 최적 입력 크기: 512토큰 이내 (한국어 약 800자 기준)

청킹 전략:
    RecursiveCharacterTextSplitter:
        문단(\\n\\n) → 줄(\\n) → 문장(.) → 문자 순서로 재귀 분할
        → 의미 단위를 최대한 보존하면서 CHUNK_SIZE 이하로 분할

설정값 (config.py):
    CHUNK_SIZE    = 800  (청크 최대 문자 수)
    CHUNK_OVERLAP = 100  (청크 간 겹침 문자 수 → 문맥 연속성 유지)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from src.pdf_processor import ProcessedDocument


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    텍스트 분할기(RecursiveCharacterTextSplitter) 생성

    분할 순서 (구분자 우선순위):
        1. 문단 구분 (\\n\\n)   ← 가장 먼저 시도 (의미 단위 최대 보존)
        2. 줄 바꿈 (\\n)
        3. 문장 끝 (.)
        4. 공백 ( )
        5. 문자 단위            ← 최후 수단 (의미 손실 발생)

    Returns:
        설정된 RecursiveCharacterTextSplitter 인스턴스
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 한국어 문장 구분자 추가 (마침표 포함)
        separators=["\n\n", "\n", ".", "。", "!", "?", " ", ""],
        length_function=len,
    )


def _clean_text(text: str) -> str:
    """
    청킹 전 텍스트 전처리

    처리 내용:
        - 연속 공백/탭→단일 공백 (벡터 검색 노이즈 감소)
        - 3줄 이상 연속 빈줄 → 2줄로 정규화
        - 양쪽 공백 제거

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    text = re.sub(r"[ \t]+", " ", text)           # 연속 공백 정규화
    text = re.sub(r"\n{3,}", "\n\n", text)         # 연속 빈줄 정규화
    return text.strip()


def chunk_document(doc: ProcessedDocument) -> list[Document]:
    """
    단일 문서를 청크 리스트로 분할

    처리 흐름:
        ProcessedDocument (전체 텍스트)
          → 페이지별 텍스트 정제
          → RecursiveCharacterTextSplitter로 청크 분할
          → 각 청크에 메타데이터 추가 (file_name, department, year, page_number)
          → LangChain Document 리스트 반환

    Args:
        doc: pdf_processor.process_pdf()에서 반환된 ProcessedDocument

    Returns:
        청크 단위로 분할된 LangChain Document 리스트
        (각각 page_content + metadata 포함)
    """
    splitter = create_text_splitter()
    chunks: list[Document] = []

    for page in doc.pages:
        clean_content = _clean_text(page.content)
        if not clean_content:
            continue  # 빈 페이지 스킵

        # 현재 페이지를 CHUNK_SIZE 이하 단위로 분할
        text_chunks = splitter.split_text(clean_content)

        for chunk_text in text_chunks:
            if not chunk_text.strip():
                continue

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        # 출처 추적: 어느 파일의 몇 페이지에서 왔는지
                        "file_path":     doc.metadata.file_path,
                        "file_name":     doc.metadata.file_name,
                        # 필터링용: 부문/연도별 검색 필터에 사용
                        "department":    doc.metadata.department,
                        "year":          doc.metadata.year,
                        "project_name":  doc.metadata.project_name,
                        "page_number":   page.page_number,
                        "total_pages":   doc.metadata.total_pages,
                    },
                )
            )

    return chunks


def chunk_all_documents(docs: list[ProcessedDocument]) -> list[Document]:
    """
    모든 문서를 청크로 분할 (index_data.py에서 호출하는 메인 함수)

    Args:
        docs: process_all_pdfs()에서 반환된 ProcessedDocument 리스트

    Returns:
        모든 문서의 청크를 합친 LangChain Document 리스트
    """
    all_chunks: list[Document] = []

    for doc in docs:
        doc_chunks = chunk_document(doc)
        print(
            f"  [OK] {doc.metadata.file_name[:40]:40s} → {len(doc_chunks):3d}개 청크"
        )
        all_chunks.extend(doc_chunks)

    print(f"\n[INFO] 총 {len(all_chunks)}개 청크 생성 ({len(docs)}개 문서)")
    return all_chunks
