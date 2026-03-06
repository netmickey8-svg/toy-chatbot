"""
벡터스토어 관리 모듈
====================
기본 백엔드: Qdrant(원격)
보조 백엔드: simple(로컬 JSON, 디버그용)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from openai import OpenAI
from qdrant_client import QdrantClient, models

from config import (
    VECTORSTORE_DIR,
    VECTOR_BACKEND,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_HASH_DIM,
    EMBEDDING_BATCH_SIZE,
    TOP_K_RESULTS,
)


SIMPLE_STORE_FILE = VECTORSTORE_DIR / "simple_store.json"


def _use_simple_backend() -> bool:
    return VECTOR_BACKEND == "simple"


def _meta_match(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_meta_match(meta, cond) for cond in where["$and"])
    for key, cond in where.items():
        if isinstance(cond, dict) and "$eq" in cond:
            if str(meta.get(key)) != str(cond["$eq"]):
                return False
        elif meta.get(key) != cond:
            return False
    return True


class SimpleCollection:
    """Qdrant 대체용 최소 로컬 컬렉션"""

    def __init__(self, file_path: Path, embedding_fn) -> None:
        self.file_path = file_path
        self.embedding_fn = embedding_fn
        self._rows: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._rows = []
            return
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._rows = data if isinstance(data, list) else []
        except Exception:
            self._rows = []

    def _save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file_path)

    def count(self) -> int:
        return len(self._rows)

    def get_corpus(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self._rows:
            rows.append(
                {
                    "id": r.get("id"),
                    "document": r.get("document", ""),
                    "metadata": dict(r.get("metadata", {})),
                    "vector": r.get("vector"),
                }
            )
        return rows

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        vectors = self.embedding_fn(documents)
        by_id = {row["id"]: row for row in self._rows if "id" in row}
        for i, rid in enumerate(ids):
            by_id[rid] = {
                "id": rid,
                "document": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "vector": vectors[i],
            }
        self._rows = list(by_id.values())
        self._save()

    def delete(self, where: dict | None = None) -> None:
        if not where:
            self._rows = []
            self._save()
            return
        self._rows = [r for r in self._rows if not _meta_match(r.get("metadata", {}), where)]
        self._save()

    def query(self, query_text: str, n_results: int) -> list[Document]:
        if not self._rows:
            return []
        emb = np.array([r["vector"] for r in self._rows], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        q = np.array(self.embedding_fn([query_text])[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sim = emb @ q
        top_idx = np.argsort(-sim)[: max(1, min(n_results, len(self._rows)))]
        docs: list[Document] = []
        for i in top_idx:
            row = self._rows[i]
            meta = dict(row.get("metadata", {}))
            meta["distance"] = round(float(1 - sim[i]), 4)
            meta["similarity"] = round(float(sim[i]), 4)
            docs.append(Document(page_content=row.get("document", ""), metadata=meta))
        return docs

    def get_indexed_file_names(self) -> list[str]:
        names = {r.get("metadata", {}).get("file_name") for r in self._rows}
        return sorted([n for n in names if n])


class QdrantCollection:
    """Qdrant 컬렉션 래퍼 (기존 호출 인터페이스 유지)"""

    def __init__(self, client: QdrantClient, name: str, embedding_fn) -> None:
        self.client = client
        self.name = name
        self.embedding_fn = embedding_fn

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.name in existing:
            return
        self.client.create_collection(
            collection_name=self.name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        vectors = self.embedding_fn(documents)
        if not vectors:
            return
        self._ensure_collection(len(vectors[0]))
        points = []
        for i, pid in enumerate(ids):
            payload = dict(metadatas[i] if i < len(metadatas) else {})
            payload["document"] = documents[i]
            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vectors[i],
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.name, points=points, wait=True)

    def reset(self, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.name in existing:
            self.client.delete_collection(collection_name=self.name)
        self.client.create_collection(
            collection_name=self.name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def delete(self, where: dict | None = None) -> None:
        if not where:
            return
        must: list[models.FieldCondition] = []
        for key, cond in where.items():
            if isinstance(cond, dict) and "$eq" in cond:
                must.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=cond["$eq"]),
                    )
                )
        if not must:
            return
        self.client.delete(
            collection_name=self.name,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=True,
        )

    def query(self, query_text: str, n_results: int) -> list[Document]:
        vector = self.embedding_fn([query_text])[0]
        points = self.client.query_points(
            collection_name=self.name,
            query=vector,
            limit=n_results,
            with_payload=True,
        ).points
        docs: list[Document] = []
        for p in points:
            payload = dict(p.payload or {})
            text = payload.pop("document", "")
            score = float(p.score) if p.score is not None else 0.0
            payload["similarity"] = round(score, 4)
            payload["distance"] = round(1 - score, 4)
            docs.append(Document(page_content=text, metadata=payload))
        return docs

    def get_indexed_file_names(self) -> list[str]:
        names: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.name,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            if not points:
                break
            for p in points:
                name = (p.payload or {}).get("file_name")
                if name:
                    names.add(name)
            if offset is None:
                break
        return sorted(names)

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.name, exact=False).count
        except Exception:
            return 0

    def get_corpus(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.name,
                limit=256,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
            if not points:
                break
            for p in points:
                payload = dict(p.payload or {})
                text = payload.pop("document", "")
                vector = p.vector
                if isinstance(vector, dict):
                    # named vector 케이스 대응
                    vector = next(iter(vector.values())) if vector else None
                rows.append(
                    {
                        "id": str(p.id),
                        "document": text,
                        "metadata": payload,
                        "vector": vector,
                    }
                )
            if offset is None:
                break
        return rows


def _get_embedding_function():
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        print(f"[EMB] OpenAI 임베딩 사용: {EMBEDDING_MODEL}")
        return OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)

    if EMBEDDING_PROVIDER == "remote_openai":
        print(f"[EMB] 원격 임베딩 사용: {EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}")
        client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)

        class RemoteOpenAIEmbeddingFunction:
            def name(self) -> str:
                return f"remote-openai-{EMBEDDING_MODEL}"

            def __call__(self, input: list[str]) -> list[list[float]]:
                texts = [input] if isinstance(input, str) else list(input)
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
                return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

        return RemoteOpenAIEmbeddingFunction()

    if EMBEDDING_PROVIDER == "hashing":
        from sklearn.feature_extraction.text import HashingVectorizer

        print(f"[EMB] 해시 임베딩 사용: n_features={EMBEDDING_HASH_DIM}")

        class HashingEmbeddingFunction:
            def __init__(self) -> None:
                self.vectorizer = HashingVectorizer(
                    n_features=EMBEDDING_HASH_DIM,
                    alternate_sign=False,
                    norm="l2",
                )

            def __call__(self, input: list[str]) -> list[list[float]]:
                texts = [input] if isinstance(input, str) else list(input)
                return self.vectorizer.transform(texts).toarray().tolist()

        return HashingEmbeddingFunction()

    print(f"[EMB] 로컬 임베딩 모델 로드: {EMBEDDING_MODEL}")
    try:
        from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        normalize_embeddings=True,
    )


def get_embedding_function():
    """외부 모듈(hybrid retriever)에서 재사용할 임베딩 함수"""
    return _get_embedding_function()


def _chunk_id(meta: dict, chunk_text: str, idx: int) -> str:
    base = f"{meta.get('file_path','')}|{meta.get('page_number','')}|{idx}|{chunk_text[:80]}"
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()


def _ensure_collection(embedding_fn):
    if _use_simple_backend():
        return SimpleCollection(SIMPLE_STORE_FILE, embedding_fn)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
    return QdrantCollection(client, QDRANT_COLLECTION, embedding_fn)


def create_vectorstore(chunks: list[Document]):
    print(f"[INFO] {len(chunks)}개 청크를 임베딩하여 저장합니다...")
    # 빈 청크는 검색 품질을 크게 떨어뜨리므로 제외
    chunks = [c for c in chunks if (c.page_content or "").strip()]
    if not chunks:
        raise ValueError("유효한 청크가 없습니다. (빈 텍스트만 존재)")

    embedding_fn = _get_embedding_function()
    collection = _ensure_collection(embedding_fn)
    # 전체 재인덱싱: 기존 파일들 모두 삭제 후 업서트
    if _use_simple_backend():
        collection.delete(where={})
    else:
        # 기존 컬렉션의 잔존/테스트 포인트까지 포함해 완전 초기화
        sample_vec = embedding_fn([chunks[0].page_content])[0]
        collection.reset(vector_size=len(sample_vec))
    return upsert_vectorstore(chunks, collection=collection)


def upsert_vectorstore(
    chunks: list[Document],
    collection: Any | None = None,
    overwrite_files: list[str] | None = None,
) -> Any:
    if not chunks:
        raise ValueError("업서트할 청크가 없습니다.")
    chunks = [c for c in chunks if (c.page_content or "").strip()]
    if not chunks:
        raise ValueError("업서트할 유효 청크가 없습니다. (빈 텍스트)")

    embedding_fn = _get_embedding_function()
    if collection is None:
        collection = _ensure_collection(embedding_fn)

    if overwrite_files:
        for name in overwrite_files:
            try:
                collection.delete(where={"file_name": {"$eq": name}})
            except Exception:
                continue

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{k: v for k, v in (chunk.metadata or {}).items() if v is not None} for chunk in chunks]
    ids = [_chunk_id(metadatas[i], texts[i], i) for i in range(len(chunks))]

    batch = EMBEDDING_BATCH_SIZE
    for start in range(0, len(chunks), batch):
        end = min(start + batch, len(chunks))
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  [OK] {end}/{len(chunks)}개 처리 완료")
    return collection


def load_vectorstore() -> Any | None:
    try:
        embedding_fn = _get_embedding_function()
        collection = _ensure_collection(embedding_fn)
        if collection.count() == 0:
            print("[WARN] 벡터스토어가 비어 있습니다. 먼저 인덱싱하세요.")
            return None
        print("[OK] 벡터스토어 로드 완료")
        return collection
    except Exception as e:
        print(f"[WARN] 벡터스토어 로드 실패: {e}")
        return None


def search_documents(
    query: str,
    collection: Any | None = None,
    k: int = TOP_K_RESULTS,
) -> list[Document]:
    if collection is None:
        collection = load_vectorstore()
        if collection is None:
            return []
    try:
        docs = collection.query(query_text=query, n_results=k)
        # 빈 본문/핵심 메타 누락 문서는 제외
        filtered = []
        for d in docs:
            text = (d.page_content or "").strip()
            meta = d.metadata or {}
            if not text:
                continue
            if not meta.get("file_name"):
                continue
            filtered.append(d)
        return filtered
    except Exception as e:
        print(f"[WARN] 검색 오류: {e}")
        return []


def get_indexed_file_names(collection: Any | None = None) -> list[str]:
    if collection is None:
        collection = load_vectorstore()
        if collection is None:
            return []
    try:
        return collection.get_indexed_file_names()
    except Exception:
        return []


def get_corpus_rows(collection: Any | None = None) -> list[dict[str, Any]]:
    if collection is None:
        collection = load_vectorstore()
        if collection is None:
            return []
    try:
        return collection.get_corpus()
    except Exception:
        return []
