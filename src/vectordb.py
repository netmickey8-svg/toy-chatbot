"""
벡터스토어 관리 모듈
====================
역할:
    - PDF에서 추출된 텍스트 청크를 트랜스포머 임베딩으로 변환하여 벡터DB에 저장
    - 사용자 쿼리를 임베딩하여 유사 청크를 검색(Dense Retrieval)하는 기능 제공

핵심 기술 (발표자료 참고):
    ┌──────────────────────────────────────────────────────────┐
    │  [Query 단계]                                             │
    │   쿼리 → Embedding → ChromaDB 벡터 검색 → 상위 K개 반환  │
    │                                                          │
    │  [임베딩 모델]                                            │
    │   BAAI/bge-m3 (Transformer Encoder-Only, Dense 방식)     │
    │   - 한국어 포함 100+ 언어 지원                            │
    │   - Mean Pooling으로 문장 임베딩 생성                     │
    │   - 의미 기반 검색 (TF-IDF 키워드 매칭보다 정확)          │
    └──────────────────────────────────────────────────────────┘

파일 구조:
    vectorstore/
    └── chroma_db/    ← ChromaDB 영구 저장소 (git 제외)

사용법:
    # 인덱싱 (index_data.py에서 호출)
    store = create_vectorstore(chunks)

    # 검색 (rag_chain.py에서 호출)
    docs = search_documents(query="블록체인", department="R&D부문")
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가 (어디서 실행해도 동작하도록)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
import chromadb

# chromadb 1.5.x에서 임베딩 함수 import 경로 변경 대응
try:
    # chromadb >= 1.5.x: 각 임베딩 함수가 별도 모듈로 분리됨
    from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
        SentenceTransformerEmbeddingFunction,
    )
except ImportError:
    # chromadb < 1.5.x 하위 호환
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, TOP_K_RESULTS


# ──────────────────────────────────────────────
# ChromaDB 컬렉션 설정
# ──────────────────────────────────────────────
# 컬렉션명: 벡터스토어 내 문서 그룹 식별자
COLLECTION_NAME = "proposals"

# ChromaDB 저장 경로
CHROMA_DIR = VECTORSTORE_DIR / "chroma_db"


def _get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """
    트랜스포머 기반 임베딩 함수 반환

    [ 임베딩 모델: BAAI/bge-m3 ]
    - 구조: Transformer Encoder-Only (BERT 계열)
    - 방식: Dense Embedding (Mean Pooling)
    - 특징: 한국어+영어 동시 처리, 의미 기반 유사도 검색 가능
    - 최초 실행 시 Hugging Face에서 자동 다운로드 (~2GB)
    - 대안(경량): "jhgan/ko-sroberta-multitask" (~400MB)

    Returns:
        ChromaDB 호환 임베딩 함수 (SentenceTransformerEmbeddingFunction)
    """
    print(f"[EMB] 임베딩 모델 로드: {EMBEDDING_MODEL}")
    print(f"      (최초 실행 시 모델 다운로드가 필요합니다 - ~2GB)")
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
        # normalize_embeddings: 코사인 유사도 계산을 위해 L2 정규화 적용
        normalize_embeddings=True,
    )


def _get_chroma_client() -> chromadb.PersistentClient:
    """
    영구 저장 ChromaDB 클라이언트 반환

    역할:
        - CHROMA_DIR에 데이터를 파일로 저장하여 재시작 후에도 유지
        - 기존 벡터스토어가 있으면 자동으로 불러옴

    Returns:
        chromadb.PersistentClient
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def create_vectorstore(chunks: list[Document]) -> chromadb.Collection:
    """
    문서 청크를 트랜스포머로 임베딩하여 ChromaDB에 저장

    처리 흐름:
        청크 리스트
          → 각 청크 텍스트를 bge-m3 모델로 임베딩 (Dense Vector)
          → ChromaDB 컬렉션에 upsert (기존 데이터 대체)
          → CHROMA_DIR에 영구 저장

    Args:
        chunks: chunker.py에서 생성된 LangChain Document 리스트
                각 Document는 page_content(텍스트)와 metadata를 포함

    Returns:
        생성된 chromadb.Collection 객체
    """
    print(f"[INFO] {len(chunks)}개 청크를 임베딩하여 저장합니다...")
    print(f"       저장 위치: {CHROMA_DIR}")

    embedding_fn = _get_embedding_function()
    client = _get_chroma_client()

    # 기존 컬렉션 삭제 후 재생성 (재인덱싱 시 중복 방지)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] 기존 컬렉션 삭제 후 재생성")
    except Exception:
        pass  # 처음 실행 시 컬렉션이 없으면 무시

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        # cosine: 벡터 방향(의미)만 비교, 크기는 무시 → 의미 유사도 검색에 적합
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB는 문자열 ID가 필요 → 인덱스를 문자열로 변환
    ids = [str(i) for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    # ChromaDB metadata는 str/int/float/bool만 허용 → None 값 제거
    metadatas = [
        {k: v for k, v in chunk.metadata.items() if v is not None}
        for chunk in chunks
    ]

    # 배치 단위로 upsert (메모리 효율)
    batch = EMBEDDING_BATCH_SIZE
    for start in range(0, len(chunks), batch):
        end = min(start + batch, len(chunks))
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  [OK] {end}/{len(chunks)}개 처리 완료")

    print(f"\n[OK] 벡터스토어 생성 완료!")
    return collection


def load_vectorstore() -> chromadb.Collection | None:
    """
    저장된 ChromaDB 컬렉션 로드

    역할:
        - 앱 시작 시 호출하여 기존 인덱스를 메모리에 불러옴
        - 벡터스토어가 없으면 None 반환 (인덱싱 필요 안내용)

    Returns:
        chromadb.Collection 또는 None (벡터스토어 없을 경우)
    """
    if not CHROMA_DIR.exists():
        print("[WARN] 벡터스토어 없음. 먼저 'python index_data.py'를 실행하세요.")
        return None

    try:
        embedding_fn = _get_embedding_function()
        client = _get_chroma_client()
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
        count = collection.count()
        if count == 0:
            print("[WARN] 벡터스토어가 비어 있습니다. 재인덱싱을 실행하세요.")
            return None
        print(f"[OK] 벡터스토어 로드 완료 ({count}개 청크)")
        return collection
    except Exception as e:
        print(f"[WARN] 벡터스토어 로드 실패: {e}")
        return None


def search_documents(
    query: str,
    collection: chromadb.Collection | None = None,
    department: str | None = None,
    year: str | None = None,
    page_number: int | None = None,
    k: int = TOP_K_RESULTS,
) -> list[Document]:
    """
    쿼리와 의미적으로 유사한 문서 청크 검색 (Dense Retrieval)

    검색 흐름 (발표자료 Query 단계):
        쿼리 텍스트
          → bge-m3로 임베딩 (Dense Vector)
          → ChromaDB에서 코사인 유사도 기반 상위 K개 검색
          → 필터 조건(부문/연도/페이지) 적용 후 반환

    Args:
        query:       사용자 검색 쿼리 (예: "블록체인 관련 프로젝트")
        collection:  ChromaDB 컬렉션 (None이면 자동 로드)
        department:  부문 필터 (예: "R&D부문"), None이면 전체 검색
        year:        연도 필터 (예: "2025"), None이면 전체 검색
        page_number: 특정 페이지 필터 (None이면 전체 페이지)
        k:           반환할 최대 문서 수

    Returns:
        관련도 높은 순으로 정렬된 LangChain Document 리스트
    """
    if collection is None:
        collection = load_vectorstore()
        if collection is None:
            return []

    # ChromaDB where 필터 조건 구성 (메타데이터 기반 필터링)
    where: dict | None = None
    conditions = []
    if department:
        conditions.append({"department": {"$eq": department}})
    if year:
        conditions.append({"year": {"$eq": str(year)}})
    if page_number is not None:
        conditions.append({"page_number": {"$eq": page_number}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[WARN] 검색 오류: {e}")
        return []

    # ChromaDB 결과를 LangChain Document 형식으로 변환
    docs = []
    if results and results["documents"] and results["documents"][0]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=text, metadata=meta or {}))

    return docs


# ──────────────────────────────────────────────
# 단독 실행 테스트 (python src/vectordb.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import DATA_DIR
    from src.pdf_processor import process_all_pdfs
    from src.chunker import chunk_all_documents

    print("=" * 50)
    print("1단계: PDF 처리")
    docs = process_all_pdfs(DATA_DIR)

    print("\n" + "=" * 50)
    print("2단계: 청킹")
    chunks = chunk_all_documents(docs)

    print("\n" + "=" * 50)
    print("3단계: 벡터스토어 생성 (bge-m3 임베딩)")
    store = create_vectorstore(chunks)

    print("\n" + "=" * 50)
    print("4단계: 검색 테스트")
    test_query = "블록체인 관련 프로젝트"
    results = search_documents(test_query, store)
    print(f"\n[QUERY] '{test_query}'")
    for i, doc in enumerate(results, 1):
        print(f"\n결과 {i}:")
        print(f"  파일: {doc.metadata.get('file_name', 'Unknown')[:50]}")
        print(f"  부문: {doc.metadata.get('department', 'Unknown')}")
        print(f"  연도: {doc.metadata.get('year', 'Unknown')}년")
        print(f"  내용: {doc.page_content[:150]}...")
