from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document

from config import CLUSTER_TOP_N
from src.cluster_index import load_cluster_index, select_top_clusters
from src.hybrid_retriever import HybridConfig, HybridRetriever
from src.vectordb import (
    get_corpus_rows,
    get_embedding_function,
    get_indexed_file_names,
    search_documents,
)


HYBRID_RETRIEVER_MODES = {"hybrid", "hybrid_cluster"}


class RetrievalPipeline:
    """검색기 생성과 검색 모드 분기를 담당한다."""

    def __init__(self, vectorstore: Any, retriever_mode: str) -> None:
        self.vectorstore = vectorstore
        self.retriever_mode = retriever_mode
        self.hybrid: HybridRetriever | None = None
        self.cluster_index: dict[str, Any] | None = None
        self.embedding_fn = get_embedding_function()
        self._build_hybrid_retriever()

    def refresh(self, vectorstore: Any) -> None:
        self.vectorstore = vectorstore
        self._build_hybrid_retriever()

    def list_indexed_files(self) -> list[str]:
        return get_indexed_file_names(self.vectorstore)

    def retrieve(self, query: str, k: int = 5) -> tuple[list[Document], dict[str, Any]]:
        if self.retriever_mode == "hybrid_cluster":
            return self._retrieve_with_cluster_prefilter(query, k=k)

        if self.retriever_mode in HYBRID_RETRIEVER_MODES and self.hybrid is not None:
            return self.hybrid.retrieve(query, k=k)

        docs = search_documents(query=query, collection=self.vectorstore, k=k)
        return docs, {
            "total_candidates": None,
            "cluster_enabled": False,
            "clusters": 0,
            "dense_weight": 1.0,
            "sparse_weight": 0.0,
            "top_docs": [],
        }

    def _build_hybrid_retriever(self) -> None:
        self.hybrid = None
        self.cluster_index = load_cluster_index()

        if self.retriever_mode == "hybrid_cluster":
            return

        if self.retriever_mode not in HYBRID_RETRIEVER_MODES:
            return
        if self.vectorstore is None:
            return

        cfg = HybridConfig(
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
            use_clustering=False,
            n_clusters=0,
            cluster_top_n=0,
        )
        rows = get_corpus_rows(self.vectorstore)
        self.hybrid = HybridRetriever(rows, self.embedding_fn, cfg)

    def _retrieve_with_cluster_prefilter(
        self,
        query: str,
        k: int,
    ) -> tuple[list[Document], dict[str, Any]]:
        if self.vectorstore is None:
            return [], {
                "total_candidates": 0,
                "cluster_enabled": False,
                "clusters": 0,
                "selected_clusters": [],
                "dense_weight": 0.7,
                "sparse_weight": 0.3,
                "top_docs": [],
            }

        cfg = HybridConfig(
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
            use_clustering=False,
            n_clusters=0,
            cluster_top_n=0,
        )
        query_vector = self.embedding_fn([query])[0]
        cluster_ids, cluster_info = select_top_clusters(
            query_vector,
            self.cluster_index,
            top_n=CLUSTER_TOP_N,
        )

        where = {"cluster_id": {"$in": cluster_ids}} if cluster_ids else None
        rows = get_corpus_rows(self.vectorstore, where=where)
        reranker = HybridRetriever(rows, self.embedding_fn, cfg)
        docs, info = reranker.retrieve(query, k=k)
        info["cluster_enabled"] = bool(cluster_ids)
        info["clusters"] = cluster_info.get("clusters", 0)
        info["selected_clusters"] = cluster_info.get("selected_clusters", [])
        return docs, info
