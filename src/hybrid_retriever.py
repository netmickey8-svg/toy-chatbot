from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class HybridConfig:
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    use_clustering: bool = False
    n_clusters: int = 6
    cluster_top_n: int = 2


class HybridRetriever:
    """
    Dense + Sparse 하이브리드 검색기
    - Dense: 코사인(sim) (임베딩 벡터)
    - Sparse: TF-IDF 코사인(sim)
    - Hybrid: 가중합
    - Optional: KMeans 클러스터 prefilter
    """

    def __init__(self, corpus_rows: list[dict[str, Any]], embedding_fn, config: HybridConfig) -> None:
        self.embedding_fn = embedding_fn
        self.config = config

        rows = []
        for r in corpus_rows:
            text = (r.get("document") or "").strip()
            meta = dict(r.get("metadata") or {})
            vec = r.get("vector")
            if not text or not meta.get("file_name") or vec is None:
                continue
            rows.append(
                {
                    "id": r.get("id"),
                    "document": text,
                    "metadata": meta,
                    "vector": vec,
                }
            )
        self.rows = rows
        self.texts = [r["document"] for r in rows]

        if not self.rows:
            self.vectors = np.zeros((0, 1), dtype=np.float32)
            self.labels = None
            self.centers = None
            self.tfidf = None
            self.sparse_matrix = None
            return

        dense = np.asarray([r["vector"] for r in rows], dtype=np.float32)
        dense = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
        self.vectors = dense

        self.tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        self.sparse_matrix = self.tfidf.fit_transform(self.texts)

        self.labels = None
        self.centers = None
        if config.use_clustering and len(rows) >= 8:
            k = max(2, min(config.n_clusters, len(rows) // 2))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.labels = km.fit_predict(self.vectors)
            centers = km.cluster_centers_.astype(np.float32)
            centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
            self.centers = centers

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def _cluster_mask(self, qvec: np.ndarray) -> np.ndarray:
        if self.labels is None or self.centers is None:
            return np.ones((len(self.rows),), dtype=bool)
        sim = self.centers @ qvec
        top = np.argsort(-sim)[: max(1, min(self.config.cluster_top_n, len(sim)))]
        selected = set(top.tolist())
        return np.array([lbl in selected for lbl in self.labels], dtype=bool)

    def retrieve(self, query: str, k: int) -> tuple[list[Document], dict]:
        if not self.rows:
            return [], {"total_candidates": 0}

        qvec = np.asarray(self.embedding_fn([query])[0], dtype=np.float32)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

        dense_scores = self.vectors @ qvec
        q_sparse = self.tfidf.transform([query])
        sparse_scores = (self.sparse_matrix @ q_sparse.T).toarray().reshape(-1)

        dense_n = self._minmax(dense_scores)
        sparse_n = self._minmax(sparse_scores)

        mask = self._cluster_mask(qvec) if self.config.use_clustering else np.ones_like(dense_n, dtype=bool)
        combined = self.config.dense_weight * dense_n + self.config.sparse_weight * sparse_n
        combined_masked = np.where(mask, combined, -1e9)

        top_idx = np.argsort(-combined_masked)[: max(1, min(k, len(self.rows)))]
        docs: list[Document] = []
        trace = []
        for rank, idx in enumerate(top_idx, 1):
            if combined_masked[idx] < -1e8:
                continue
            row = self.rows[idx]
            meta = dict(row["metadata"])
            meta["dense_score"] = round(float(dense_scores[idx]), 4)
            meta["sparse_score"] = round(float(sparse_scores[idx]), 4)
            meta["hybrid_score"] = round(float(combined[idx]), 4)
            if self.labels is not None:
                meta["cluster_id"] = int(self.labels[idx])
            docs.append(Document(page_content=row["document"], metadata=meta))
            trace.append(
                {
                    "rank": rank,
                    "file_name": meta.get("file_name"),
                    "page_number": meta.get("page_number"),
                    "dense_score": meta["dense_score"],
                    "sparse_score": meta["sparse_score"],
                    "hybrid_score": meta["hybrid_score"],
                    "cluster_id": meta.get("cluster_id"),
                }
            )

        info = {
            "total_candidates": len(self.rows),
            "cluster_enabled": bool(self.config.use_clustering and self.labels is not None),
            "clusters": int(len(self.centers)) if self.centers is not None else 0,
            "dense_weight": self.config.dense_weight,
            "sparse_weight": self.config.sparse_weight,
            "top_docs": trace,
        }
        return docs, info
