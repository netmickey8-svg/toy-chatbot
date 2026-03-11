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
from dataclasses import dataclass
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from src.chunk_labels import infer_chunk_label
from src.pdf_processor import ContentBlock, ProcessedDocument


@dataclass
class SectionBlock:
    """페이지 내부에서 감지한 섹션 단위 블록"""

    title: str
    content: str
    content_type: str = "text"


def _split_into_units(text: str) -> list[str]:
    """
    텍스트를 의미 단위(문단/줄/문장)로 1차 분해

    - 외부 라이브러리 의존 없이 동작하도록 단순 규칙 기반 분해 사용
    - 문단(빈 줄) → 줄 → 문장 끝 구두점 기준 분해
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    units: list[str] = []
    paragraphs = re.split(r"\n{2,}", text)
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        for line in lines:
            # 문장 끝 구두점 기준 분리 (한국어 마침표 포함)
            sentences = re.split(r"(?<=[\.\!\?\u3002])\s+", line)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    units.append(sent)
    return units


def _pack_units_to_chunks(units: list[str]) -> list[str]:
    """
    분해된 단위를 CHUNK_SIZE 기준으로 병합하여 청크 생성
    """
    if not units:
        return []

    chunks: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        if len(current) + 1 + len(unit) <= CHUNK_SIZE:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            if CHUNK_OVERLAP > 0:
                overlap = current[-CHUNK_OVERLAP:]
                current = f"{overlap} {unit}"
            else:
                current = unit

    if current:
        chunks.append(current)

    return chunks


_SECTION_PATTERNS = [
    re.compile(r"^\s*[0-9]{1,2}[\.\)]\s*.+$"),                 # 1. 제목 / 1) 제목
    re.compile(r"^\s*[가-힣A-Za-z]+\.\s*.+$"),                  # 가. 제목 / A. Title
    re.compile(r"^\s*[①-⑳]\s*.+$"),                             # ① 제목
    re.compile(r"^\s*\[[^\]]+\]\s*.+$"),                        # [섹션] 제목
    re.compile(r"^\s*(사업|과제|기술|평가|참여|제출|일정).{0,30}$"),  # 도메인형 짧은 제목
]


def _is_section_heading(line: str) -> bool:
    text = (line or "").strip()
    if not text:
        return False
    # 지나치게 긴 문장은 제목으로 취급하지 않음
    if len(text) > 80:
        return False
    return any(p.match(text) for p in _SECTION_PATTERNS)


def _split_by_sections(text: str) -> list[SectionBlock]:
    """
    텍스트를 섹션 제목 기준으로 분할.
    제목이 없으면 기본 섹션 1개로 반환.
    """
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    sections: list[SectionBlock] = []

    current_title = "본문"
    current_lines: list[str] = []

    for line in lines:
        if not line:
            continue
        if _is_section_heading(line):
            if current_lines:
                sections.append(
                    SectionBlock(title=current_title, content="\n".join(current_lines).strip())
                )
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(SectionBlock(title=current_title, content="\n".join(current_lines).strip()))

    if not sections:
        return [SectionBlock(title="본문", content=text.strip())]
    return sections


def _build_blocks_for_chunking(page_content: str, blocks: list[ContentBlock]) -> list[SectionBlock]:
    """
    페이지 내부 블록을 청킹용 섹션 블록으로 변환한다.

    - text/ocr: 제목 감지 기반 섹션 분리
    - table: 표 자체를 하나의 섹션으로 유지
    """
    if not blocks:
        return _split_by_sections(page_content)

    section_blocks: list[SectionBlock] = []
    for block in blocks:
        clean_block = _clean_text(block.content)
        if not clean_block:
            continue

        if block.content_type == "table":
            section_blocks.append(
                SectionBlock(
                    title="TABLE",
                    content=clean_block,
                    content_type="table",
                )
            )
            continue

        if block.content_type == "ocr":
            ocr_sections = _split_by_sections(clean_block)
            for section in ocr_sections:
                section.content_type = "ocr"
            section_blocks.extend(ocr_sections)
            continue

        text_sections = _split_by_sections(clean_block)
        for section in text_sections:
            section.content_type = "text"
        section_blocks.extend(text_sections)

    return section_blocks


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
    chunks: list[Document] = []

    for page in doc.pages:
        clean_content = _clean_text(page.content)
        if not clean_content:
            continue  # 빈 페이지 스킵

        # 1차: 블록/섹션 단위 분리
        section_blocks = _build_blocks_for_chunking(clean_content, getattr(page, "blocks", []))

        # 2차: 각 섹션을 길이 기준으로 재청킹
        for section_idx, block in enumerate(section_blocks, 1):
            units = (
                [line.strip() for line in block.content.splitlines() if line.strip()]
                if block.content_type == "table"
                else _split_into_units(block.content)
            )
            text_chunks = _pack_units_to_chunks(units) or [block.content[:CHUNK_SIZE]]

            for chunk_idx, chunk_text in enumerate(text_chunks, 1):
                if not chunk_text.strip():
                    continue

                chunk_label = infer_chunk_label(block.title, chunk_text)

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            # 출처 추적: 어느 파일의 몇 페이지에서 왔는지
                            "file_path": doc.metadata.file_path,
                            "file_name": doc.metadata.file_name,
                            "department": doc.metadata.department,
                            "year": doc.metadata.year,
                            "project_name": doc.metadata.project_name,
                            "page_number": page.page_number,
                            "total_pages": doc.metadata.total_pages,
                            # 섹션 인지형 청킹 메타데이터
                            "section_title": block.title,
                            "section_index": section_idx,
                            "chunk_index": chunk_idx,
                            "content_type": block.content_type,
                            "chunk_label": chunk_label,
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
