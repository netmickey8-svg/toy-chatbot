"""
RAG(검색 증강 생성) 체인 모듈
==============================
역할:
    - 사용자 질문을 받아 관련 제안서를 검색하고 LLM으로 답변을 생성하는
      핵심 파이프라인(RAG Chain)을 구현

RAG 파이프라인 흐름 (발표자료 참고):
    ┌──────────────────────────────────────────────────────────┐
    │  [Query 단계]  사용자 질문 → 벡터 검색 → 관련 청크 K개   │
    │  [LLM 단계]   관련 청크 + 질문 → Prompt 생성 → Gemini  │
    │               → 자연어 답변 생성                         │
    └──────────────────────────────────────────────────────────┘

의존 모듈:
    - src/vectordb.py  : 트랜스포머 임베딩 기반 문서 검색
    - config.py        : Gemini API 키, 모델명 설정

사용:
    rag = RAGChain()
    answer, docs = rag.ask("블록체인 관련 제안서 알려줘")
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai
from langchain_core.documents import Document

from config import GOOGLE_API_KEY, GEMINI_MODEL
from src.vectordb import load_vectorstore, search_documents

# ──────────────────────────────────────────────
# Gemini API 초기화
# ──────────────────────────────────────────────
# 모듈 로드 시점에 API 키 설정 (RAGChain 인스턴스마다 중복 호출 방지)
genai.configure(api_key=GOOGLE_API_KEY)

# ──────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# ──────────────────────────────────────────────
# {context}: 검색된 제안서 내용을 포맷팅한 텍스트 (format_documents() 결과)
# {question}: 사용자 원본 질문
SYSTEM_PROMPT = """당신은 제안서 추천 전문가입니다.
사용자의 질문에 맞는 제안서를 검색하고 관련 내용을 설명해주세요.

## 지침
1. 검색된 제안서 내용을 기반으로 정확하게 답변하세요.
2. 관련 제안서의 이름, 연도, 부문(R&D/SI)을 명시하세요.
3. 검색 결과가 없거나 관련 없으면 솔직히 알려주세요.
4. 답변은 친절하고 이해하기 쉽게 작성하세요.

## 검색된 제안서 정보:
{context}

## 사용자 질문:
{question}"""


def format_documents(docs: list[Document]) -> str:
    """
    검색된 문서 청크들을 LLM 프롬프트용 텍스트로 포맷팅

    역할:
        - 각 Document의 메타데이터(파일명/부문/연도/페이지)와 본문을
          구조화된 텍스트로 변환하여 LLM이 참고할 컨텍스트를 구성

    Args:
        docs: search_documents()에서 반환된 Document 리스트

    Returns:
        LLM 프롬프트에 삽입할 포맷팅된 문자열
        (검색 결과 없으면 안내 메시지 반환)
    """
    if not docs:
        return "검색된 제안서가 없습니다."

    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        formatted.append(
            f"### 제안서 {i}\n"
            f"- **파일명**: {meta.get('file_name', 'Unknown')}\n"
            f"- **부문**: {meta.get('department', 'Unknown')}\n"
            f"- **연도**: {meta.get('year', 'Unknown')}년\n"
            f"- **프로젝트명**: {meta.get('project_name', 'Unknown')}\n"
            f"- **페이지**: {meta.get('page_number', '-')}\n"
            f"- **내용**:\n{doc.page_content}\n"
        )

    return "\n---\n".join(formatted)


class RAGChain:
    """
    RAG(Retrieval-Augmented Generation) 기반 질의응답 체인

    구조:
        __init__: 벡터스토어 로드 (bge-m3 임베딩 인덱스)
        ask():    질문 → 검색 → LLM 답변 생성의 전체 파이프라인 실행

    사용 예:
        rag = RAGChain()
        if rag.is_ready():
            answer, docs = rag.ask("2025년 AI 관련 제안서 알려줘")
    """

    def __init__(self) -> None:
        """
        벡터스토어 로드

        - load_vectorstore()를 호출하여 ChromaDB에서 bge-m3 임베딩 인덱스 로드
        - 인덱스 없으면 self.vectorstore = None → is_ready() == False
        """
        self.vectorstore = load_vectorstore()

    def ask(
        self,
        question: str,
        department: str | None = None,
        year: str | None = None,
    ) -> tuple[str, list[Document]]:
        """
        사용자 질문에 대한 RAG 기반 답변 생성

        처리 단계:
            1. [Query]  질문에서 페이지 번호 파싱 (예: "3페이지" → page_number=3)
            2. [Query]  bge-m3 임베딩으로 관련 청크 검색 (Dense Retrieval)
            3. [LLM]    검색 결과를 컨텍스트로 프롬프트 생성
            4. [LLM]    Gemini API로 자연어 답변 생성

        Args:
            question:   사용자 질문 텍스트
            department: 부문 필터 ("R&D부문" / "SI부문" / None=전체)
            year:       연도 필터 ("2025" 등 / None=전체)

        Returns:
            tuple: (답변 텍스트, 참조된 Document 리스트)
        """
        # ── Step 1: 페이지 번호 파싱 ──────────────────────
        # 예: "3페이지 내용 알려줘" → page_number = 3
        # 버그 수정: r"(\d+)\s*페이지" (이전: 이중 이스케이프 r"(\\d+)\\s*페이지")
        page_match = re.search(r"(\d+)\s*페이지", question)
        page_number = int(page_match.group(1)) if page_match else None

        # ── Step 2: 트랜스포머 임베딩 기반 문서 검색 ──────
        docs = search_documents(
            query=question,
            collection=self.vectorstore,
            department=department,
            year=year,
            page_number=page_number,
        )

        if not docs:
            return "관련된 제안서를 찾지 못했습니다. 다른 키워드로 검색해보세요.", []

        # ── Step 3: 프롬프트 생성 ─────────────────────────
        context = format_documents(docs)
        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        # ── Step 4: Gemini LLM 답변 생성 ──────────────────
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,       # 낮을수록 일관된 답변
                    "max_output_tokens": 1024,
                },
            )
            answer = (
                response.text.strip()
                if getattr(response, "text", None)
                else "답변 생성에 실패했습니다."
            )
        except Exception as e:
            # API 오류(네트워크 장애, 키 만료 등) 발생 시 사용자 친화적 메시지 반환
            print(f"[ERROR] Gemini API 오류: {e}")
            answer = f"⚠️ 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n(상세: {e})"

        return answer, docs

    def is_ready(self) -> bool:
        """
        벡터스토어 준비 여부 확인

        Returns:
            True면 검색 가능, False면 인덱싱 필요 (python index_data.py 실행)
        """
        return self.vectorstore is not None


# ──────────────────────────────────────────────
# 단독 실행 테스트 (python src/rag_chain.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGChain()

    if not rag.is_ready():
        print("[WARN] 먼저 벡터스토어를 생성하세요: python index_data.py")
    else:
        test_questions = [
            "블록체인 관련 제안서 알려줘",
            "2025년에 진행된 프로젝트는?",
            "메타버스 관련 내용이 있어?",
        ]

        for q in test_questions:
            print(f"\n{'='*60}")
            print(f"[Q] {q}")
            answer, docs = rag.ask(q)
            print(f"\n[A] {answer}")
            print(f"\n[SOURCES] 참조 문서: {len(docs)}개")
