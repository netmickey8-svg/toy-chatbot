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

import os
import re
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from src.vectordb import (
    load_vectorstore,
    search_documents,
    get_corpus_rows,
    get_embedding_function,
)
from src.hybrid_retriever import HybridRetriever, HybridConfig

# ──────────────────────────────────────────────
# 시스템 프롬프트 템플릿
# ──────────────────────────────────────────────
# {context}: 검색된 제안서 내용을 포맷팅한 텍스트 (format_documents() 결과)
# {question}: 사용자 원본 질문
SYSTEM_PROMPT = """당신은 제안서 전문가입니다.
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
        self.embedding_model = EMBEDDING_MODEL
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.llm_model = LLM_MODEL
        self.retriever_mode = os.getenv("RETRIEVER_MODE", "hybrid").lower()
        self.retrieval_trace_path = Path("vectorstore") / "retrieval_trace.jsonl"
        self.hybrid = None
        self._build_hybrid_retriever()

    def _build_hybrid_retriever(self) -> None:
        self.hybrid = None
        if self.retriever_mode not in {"hybrid", "hybrid_cluster"}:
            return
        if self.vectorstore is None:
            return
        cfg = HybridConfig(
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
            use_clustering=self.retriever_mode == "hybrid_cluster",
            n_clusters=int(os.getenv("CLUSTER_N_CLUSTERS", "6")),
            cluster_top_n=int(os.getenv("CLUSTER_TOP_N", "2")),
        )
        rows = get_corpus_rows(self.vectorstore)
        self.hybrid = HybridRetriever(rows, get_embedding_function(), cfg)

    def refresh_retriever(self) -> None:
        """
        벡터스토어 변경(웹 업로드 인덱싱) 직후 retriever 캐시를 갱신
        """
        self.vectorstore = load_vectorstore()
        self._build_hybrid_retriever()

    def _write_retrieval_trace(self, query: str, retrieval_info: dict) -> None:
        try:
            self.retrieval_trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "retriever_mode": self.retriever_mode,
                "query": query,
                "info": retrieval_info,
            }
            with self.retrieval_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _normalize_retrieval_query(question: str) -> str:
        """
        표현이 다른 동의 질의를 retrieval 친화 키워드로 확장
        - 생성용 질문(question)은 원문 유지
        - 검색용 질의만 확장하여 일관된 섹션을 찾도록 유도
        """
        q = question.strip()
        lowered = q.lower()

        people_patterns = [
            "참여하는 사람",
            "누가 참여",
            "참여자",
            "참여 인원",
            "참여 인력",
            "참여인력",
            "투입 인력",
            "투입인력",
        ]
        if any(p in q for p in people_patterns) or "participant" in lowered:
            return q + " 참여인력 투입인력 연구원 인력구성 참여율"

        return q

    def _resolve_available_model(self) -> str | None:
        """LLM 서버에서 사용 가능한 모델명을 조회"""
        try:
            models = self.client.models.list()
            if not models or not getattr(models, "data", None):
                return None
            names = [m.id for m in models.data if getattr(m, "id", None)]
            if self.llm_model in names:
                return self.llm_model
            return names[0] if names else None
        except Exception:
            return None

    @staticmethod
    def _extract_max_tokens_budget(error_text: str) -> int | None:
        """
        에러 메시지에서 허용 가능한 max_tokens 상한을 파싱
        예: (1024 > 2048 - 1377) -> 671
        """
        m = re.search(r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)", error_text)
        if not m:
            return None
        try:
            budget = int(m.group(2)) - int(m.group(3))
            return max(1, budget)
        except Exception:
            return None

    @staticmethod
    def _extract_input_token_limits(error_text: str) -> tuple[int, int] | None:
        """
        에러 메시지에서 (최대 입력 토큰, 실제 입력 토큰) 파싱
        예: maximum context length is 2048 ... has 3368 input tokens
        """
        m = re.search(
            r"maximum context length is (\d+).*?has (\d+) input tokens",
            error_text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None

    @staticmethod
    def _shrink_text(text: str, target_len: int) -> str:
        if target_len <= 0:
            return ""
        if len(text) <= target_len:
            return text
        return text[:target_len] + "\n...(생략)..."

    def _chat_with_auto_max_tokens(self, prompt: str, max_tokens: int = 1024):
        """
        max_tokens 초과 에러가 나면 가능한 범위로 자동 축소 후 재시도
        """
        try:
            return self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "너는 제안서 전문가다. 근거 기반으로 간결하게 답한다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
        except Exception as e:
            err = str(e)
            if "max_tokens" not in err and "max_completion_tokens" not in err:
                raise
            budget = self._extract_max_tokens_budget(err)
            if not budget:
                raise
            retry_tokens = max(64, min(512, budget - 16))
            return self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "너는 제안서 전문가다. 근거 기반으로 간결하게 답한다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=retry_tokens,
            )

    def ask(
        self,
        question: str,
    ) -> tuple[str, list[Document], dict]:
        """
        사용자 질문에 대한 RAG 기반 답변 생성

        Returns:
            tuple: (답변 텍스트, 참조된 Document 리스트, 파이프라인 단계별 정보)
        """
        # 파이프라인 단계별 정보를 수집하는 딕셔너리
        pipeline_info: dict = {}

        # ── Step 1: 쿼리 분석 ─────────────────────────────
        page_match = re.search(r"(\d+)\s*페이지", question)
        page_number = int(page_match.group(1)) if page_match else None

        pipeline_info["query_analysis"] = {
            "original_query": question,
            "page_filter": page_number,
        }

        # ── Step 2: 트랜스포머 임베딩 기반 문서 검색 ──────
        retrieval_query = self._normalize_retrieval_query(question)
        retrieval_info = {}
        if self.retriever_mode in {"hybrid", "hybrid_cluster"} and self.hybrid is not None:
            docs, retrieval_info = self.hybrid.retrieve(retrieval_query, k=5)
        else:
            docs = search_documents(query=retrieval_query, collection=self.vectorstore)
            retrieval_info = {
                "total_candidates": None,
                "cluster_enabled": False,
                "clusters": 0,
                "dense_weight": 1.0,
                "sparse_weight": 0.0,
                "top_docs": [],
            }

        retrieval_info["retrieval_query"] = retrieval_query
        self._write_retrieval_trace(question, retrieval_info)

        pipeline_info["retrieval"] = {
            "model": self.embedding_model,
            "method": (
                "Hybrid Retrieval (Dense+Sparse)"
                if self.retriever_mode in {"hybrid", "hybrid_cluster"}
                else "Dense Retrieval (코사인 유사도)"
            ),
            "retriever_mode": self.retriever_mode,
            "cluster_enabled": retrieval_info.get("cluster_enabled", False),
            "clusters": retrieval_info.get("clusters", 0),
            "dense_weight": retrieval_info.get("dense_weight"),
            "sparse_weight": retrieval_info.get("sparse_weight"),
            "results_count": len(docs),
            "chunks": [
                {
                    "rank": i + 1,
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "department": doc.metadata.get("department", ""),
                    "year": doc.metadata.get("year", ""),
                    "page": doc.metadata.get("page_number", "-"),
                    "similarity": (
                        doc.metadata.get("hybrid_score")
                        if doc.metadata.get("hybrid_score") is not None
                        else doc.metadata.get("similarity", 0)
                    ),
                    "dense_score": doc.metadata.get("dense_score"),
                    "sparse_score": doc.metadata.get("sparse_score"),
                    "hybrid_score": doc.metadata.get("hybrid_score"),
                    "cluster_id": doc.metadata.get("cluster_id"),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for i, doc in enumerate(docs)
            ],
        }

        if not docs:
            pipeline_info["generation"] = {"status": "검색 결과 없음 - LLM 호출 생략"}
            return "관련된 제안서를 찾지 못했습니다. 다른 키워드로 검색해보세요.", [], pipeline_info

        # ── Step 3: 프롬프트 생성 ─────────────────────────
        context = format_documents(docs)
        # 작은 컨텍스트 LLM(예: 2k) 대응 기본 길이 제한
        initial_context_chars = int(os.getenv("PROMPT_CONTEXT_CHARS", "2400"))
        context = self._shrink_text(context, initial_context_chars)
        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        pipeline_info["prompt"] = {
            "template": "시스템 프롬프트 + 검색 컨텍스트 + 사용자 질문",
            "context_length": len(context),
            "total_prompt_length": len(prompt),
            "prompt_preview": prompt[:500] + "...(이하 생략)" if len(prompt) > 500 else prompt,
        }

        # ── Step 4: 로컬 LLM 답변 생성 ───────────────────
        try:
            answer = None
            # 입력 토큰 초과 시 컨텍스트를 점진 축소하며 재시도
            for _ in range(4):
                try:
                    response = self._chat_with_auto_max_tokens(prompt, max_tokens=1024)
                    answer = response.choices[0].message.content.strip()
                    pipeline_info["generation"] = {
                        "model": self.llm_model,
                        "temperature": 0.3,
                        "status": "성공",
                        "answer_length": len(answer),
                    }
                    break
                except Exception as e:
                    err = str(e)
                    limits = self._extract_input_token_limits(err)
                    if not limits:
                        raise
                    max_ctx, input_tokens = limits
                    # 여유 토큰(시스템/응답) 300을 확보하고 비율 축소
                    target_input = max(300, max_ctx - 300)
                    ratio = max(0.25, min(0.9, target_input / max(input_tokens, 1)))
                    new_len = max(500, int(len(context) * ratio))
                    if new_len >= len(context):
                        new_len = len(context) - 200
                    if new_len <= 300:
                        raise
                    context = self._shrink_text(context, new_len)
                    prompt = SYSTEM_PROMPT.format(context=context, question=question)
                    pipeline_info["prompt"]["context_length"] = len(context)
                    pipeline_info["prompt"]["total_prompt_length"] = len(prompt)
                    pipeline_info["prompt"]["prompt_preview"] = (
                        prompt[:500] + "...(이하 생략)" if len(prompt) > 500 else prompt
                    )

            if answer is None:
                raise RuntimeError("입력 길이 축소 재시도 후에도 답변 생성 실패")
        except Exception as e:
            err = str(e)
            fallback_used = None
            if "does not exist" in err.lower() or "404" in err:
                resolved = self._resolve_available_model()
                if resolved and resolved != self.llm_model:
                    self.llm_model = resolved
                    fallback_used = resolved
                    try:
                        response = self._chat_with_auto_max_tokens(prompt, max_tokens=1024)
                        answer = response.choices[0].message.content.strip()
                        pipeline_info["generation"] = {
                            "model": self.llm_model,
                            "temperature": 0.3,
                            "status": "성공(자동 모델 전환)",
                            "answer_length": len(answer),
                            "fallback_model": fallback_used,
                        }
                        return answer, docs, pipeline_info
                    except Exception as retry_e:
                        err = f"{e} | fallback({self.llm_model}) 실패: {retry_e}"

            print(f"[ERROR] LLM API 오류: {err}")
            answer = f"⚠️ 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n(상세: {err})"
            pipeline_info["generation"] = {
                "model": self.llm_model,
                "status": f"오류: {err}",
                "fallback_model": fallback_used,
            }

        return answer, docs, pipeline_info

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
