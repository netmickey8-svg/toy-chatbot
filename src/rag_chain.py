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
)
from src.query_intent import (
    is_index_inventory_question,
    is_people_question,
    is_people_overlap_question,
    normalize_retrieval_query,
)
from src.retrieval_pipeline import HYBRID_RETRIEVER_MODES, RetrievalPipeline

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

PEOPLE_QUERY_PROMPT = """당신은 제안서 참여인력 정리 도우미입니다.
사용자의 질문은 제안서별 참여인력이나 이름 목록을 묻는 질문입니다.

## 매우 중요한 규칙
1. 사람 이름으로 명시된 표현만 이름으로 적으세요.
2. 문장 전체, 설명 문구, 절차 문구, 긴 표 행 전체를 사람 이름으로 쓰면 안 됩니다.
3. 이름이 불명확하면 억지로 추정하지 말고 "이름 명시 없음" 또는 "확인 불가"라고 답하세요.
4. 같은 파일명 아래의 여러 청크/페이지는 모두 동일한 하나의 제안서입니다.
5. 파일별로 구분해서 정리하고, 파일명과 페이지를 근거로 제시하세요.

## 출력 형식
- 파일명
  - 페이지
  - 참여인력: 이름1, 이름2 ...
  - 비고: 이름이 명확하지 않으면 그 사유

## 검색된 제안서 정보:
{context}

## 사용자 질문:
{question}"""

PEOPLE_OVERLAP_PROMPT = """당신은 제안서 참여인력 비교 분석가입니다.
사용자의 질문은 여러 제안서 사이에 같은 인물이 겹치는지 확인하는 것입니다.

## 매우 중요한 규칙
1. 검색된 제안서 내용에 동일한 사람 이름이 서로 다른 제안서에서 명시적으로 확인될 때만 "겹친다"고 답하세요.
2. 역할, 직급, 표현이 비슷하다는 이유만으로 같은 사람이라고 추론하지 마세요.
3. 한 제안서 안에서 같은 이름이 여러 번 반복되어도 그것만으로 "제안서 간 중복"이라고 판단하지 마세요.
4. 같은 파일명 아래의 여러 청크/페이지는 모두 동일한 하나의 제안서입니다. 서로 다른 제안서로 세면 안 됩니다.
5. 이름이 일부만 보이거나 표가 잘려 있어 동일 인물 여부를 확정할 수 없으면 "확인 불가"라고 답하세요.
6. 답변은 반드시 제안서 파일명과 페이지를 근거로 제시하세요.
7. 가능하면 아래 형식을 따르세요.

## 답변 형식
- 확인 결과: 겹침 있음 / 겹침 확인 불가 / 겹침 없음
- 근거:
  - 파일명, 페이지, 확인된 이름
- 주의:
  - OCR 누락, 표 잘림, 약칭 등으로 확정이 어려운 경우 명시

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
            f"- **타입/라벨**: {meta.get('content_type', 'text')} / {meta.get('chunk_label', '기타')}\n"
            f"- **내용**:\n{doc.page_content}\n"
        )

    return "\n---\n".join(formatted)


def format_documents_grouped_by_file(docs: list[Document]) -> str:
    """
    같은 파일의 여러 청크를 하나의 제안서로 묶어 프롬프트에 전달한다.
    참여인력 중복 확인처럼 파일 단위 판정이 중요한 질문에 사용한다.
    """
    if not docs:
        return "검색된 제안서가 없습니다."

    grouped: dict[str, list[Document]] = {}
    order: list[str] = []
    for doc in docs:
        file_name = (doc.metadata or {}).get("file_name", "Unknown")
        if file_name not in grouped:
            grouped[file_name] = []
            order.append(file_name)
        grouped[file_name].append(doc)

    formatted = []
    for index, file_name in enumerate(order, 1):
        file_docs = grouped[file_name]
        first_meta = file_docs[0].metadata or {}
        page_numbers = []
        for doc in file_docs:
            page = (doc.metadata or {}).get("page_number")
            if page is not None and page not in page_numbers:
                page_numbers.append(page)

        chunk_lines = []
        for chunk_index, doc in enumerate(file_docs, 1):
            meta = doc.metadata or {}
            chunk_lines.append(
                f"  - 청크 {chunk_index} | p{meta.get('page_number', '-')} | "
                f"{meta.get('content_type', 'text')} / {meta.get('chunk_label', '기타')}\n"
                f"    {doc.page_content}"
            )

        formatted.append(
            f"### 파일 {index}\n"
            f"- **파일명**: {file_name}\n"
            f"- **부문**: {first_meta.get('department', 'Unknown')}\n"
            f"- **연도**: {first_meta.get('year', 'Unknown')}년\n"
            f"- **프로젝트명**: {first_meta.get('project_name', 'Unknown')}\n"
            f"- **참조 페이지**: {', '.join(str(page) for page in page_numbers) if page_numbers else '-'}\n"
            f"- **같은 파일 내 청크 수**: {len(file_docs)}\n"
            f"- **내용 묶음**:\n" + "\n".join(chunk_lines)
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
        self.retrieval = RetrievalPipeline(self.vectorstore, self.retriever_mode)

    def refresh_retriever(self) -> None:
        """
        벡터스토어 변경(웹 업로드 인덱싱) 직후 retriever 캐시를 갱신
        """
        self.vectorstore = load_vectorstore()
        self.retrieval.refresh(self.vectorstore)

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

        # 인덱싱된 문서 목록/개수 질문은 LLM 대신 메타데이터로 직접 응답
        if is_index_inventory_question(question):
            names = self.retrieval.list_indexed_files()
            if not names:
                answer = "현재 인덱싱된 제안서가 없습니다."
            else:
                lines = [f"- {n}" for n in names[:20]]
                answer = (
                    f"현재 인덱싱된 제안서는 총 {len(names)}개입니다.\n\n"
                    f"파일 목록:\n" + "\n".join(lines)
                )
                if len(names) > 20:
                    answer += f"\n... 외 {len(names) - 20}개"
            pipeline_info["retrieval"] = {
                "method": "Inventory Lookup",
                "retriever_mode": self.retriever_mode,
                "results_count": len(names),
                "chunks": [],
            }
            pipeline_info["generation"] = {
                "model": "metadata",
                "status": "성공(메타데이터 직접 응답)",
                "answer_length": len(answer),
            }
            return answer, [], pipeline_info

        # ── Step 2: 트랜스포머 임베딩 기반 문서 검색 ──────
        retrieval_query = normalize_retrieval_query(question)
        docs, retrieval_info = self.retrieval.retrieve(retrieval_query, k=5)

        retrieval_info["retrieval_query"] = retrieval_query
        self._write_retrieval_trace(question, retrieval_info)

        pipeline_info["retrieval"] = {
            "model": self.embedding_model,
            "method": (
                "Hybrid Retrieval (Dense+Sparse)"
                if self.retriever_mode in HYBRID_RETRIEVER_MODES
                else "Dense Retrieval (코사인 유사도)"
            ),
            "retriever_mode": self.retriever_mode,
            "retrieval_query": retrieval_query,
            "query_vector": retrieval_info.get("query_vector"),
            "cluster_enabled": retrieval_info.get("cluster_enabled", False),
            "clusters": retrieval_info.get("clusters", 0),
            "selected_clusters": retrieval_info.get("selected_clusters", []),
            "summary_guidance": retrieval_info.get("summary_guidance", {}),
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
                    "content_type": doc.metadata.get("content_type"),
                    "chunk_label": doc.metadata.get("chunk_label"),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for i, doc in enumerate(docs)
            ],
        }

        if not docs:
            pipeline_info["generation"] = {"status": "검색 결과 없음 - LLM 호출 생략"}
            return "관련된 제안서를 찾지 못했습니다. 다른 키워드로 검색해보세요.", [], pipeline_info

        # ── Step 3: 프롬프트 생성 ─────────────────────────
        is_people_query = is_people_question(question)
        is_overlap_query = is_people_overlap_question(question)
        context = (
            format_documents_grouped_by_file(docs)
            if is_people_query
            else format_documents(docs)
        )
        # 작은 컨텍스트 LLM(예: 2k) 대응 기본 길이 제한
        initial_context_chars = int(os.getenv("PROMPT_CONTEXT_CHARS", "2400"))
        context = self._shrink_text(context, initial_context_chars)
        if is_overlap_query:
            prompt_template = PEOPLE_OVERLAP_PROMPT
        elif is_people_query:
            prompt_template = PEOPLE_QUERY_PROMPT
        else:
            prompt_template = SYSTEM_PROMPT
        prompt = prompt_template.format(context=context, question=question)

        pipeline_info["prompt"] = {
            "template": (
                "참여인력 중복 확인 전용 프롬프트 + 검색 컨텍스트 + 사용자 질문"
                if is_overlap_query
                else "참여인력 정리 전용 프롬프트 + 검색 컨텍스트 + 사용자 질문"
                if is_people_query
                else "시스템 프롬프트 + 검색 컨텍스트 + 사용자 질문"
            ),
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
                    prompt = prompt_template.format(context=context, question=question)
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
