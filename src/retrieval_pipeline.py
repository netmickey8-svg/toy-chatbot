from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document

from config import (
    CLUSTER_TOP_N,
    MAX_CHUNKS_PER_FILE,
    OCR_FOCUS_EXTRA_PENALTY,
    OCR_SCORE_PENALTY,
    RETRIEVAL_FETCH_MULTIPLIER,
    SUMMARY_TOP_DOCS,
    SUMMARY_TOP_SECTIONS,
)
from src.cluster_index import load_cluster_index, select_top_clusters
from src.hybrid_retriever import HybridConfig, HybridRetriever
from src.query_intent import detect_query_focus, is_structured_focus_question
from src.summary_index import load_summary_index, select_summary_guidance
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
        self.summary_index: dict[str, Any] | None = None
        self.embedding_fn = get_embedding_function()
        self._build_hybrid_retriever()

    def refresh(self, vectorstore: Any) -> None:
        self.vectorstore = vectorstore
        self._build_hybrid_retriever()

    def list_indexed_files(self) -> list[str]:
        return get_indexed_file_names(self.vectorstore)

    @staticmethod
    def _base_score(doc: Document) -> float:
        meta = doc.metadata or {}
        if meta.get("hybrid_score") is not None:
            return float(meta.get("hybrid_score", 0.0))
        return float(meta.get("similarity", 0.0))

    @staticmethod
    def _boost_by_focus(
        docs: list[Document],
        focus: dict[str, list[str]],
        query: str,
    ) -> list[Document]:
        preferred_labels = set(focus.get("preferred_labels", []))
        preferred_content_types = set(focus.get("preferred_content_types", []))
        structured_focus = is_structured_focus_question(query)

        def score(doc: Document) -> float:
            meta = doc.metadata or {}
            boosted = RetrievalPipeline._base_score(doc)
            content_type = str(meta.get("content_type", "text"))
            if content_type == "ocr":
                boosted -= OCR_SCORE_PENALTY
                if structured_focus:
                    boosted -= OCR_FOCUS_EXTRA_PENALTY
            elif content_type == "table" and structured_focus:
                boosted += 0.03
            if preferred_labels and meta.get("chunk_label") in preferred_labels:
                boosted += 0.08
            if preferred_content_types and meta.get("content_type") in preferred_content_types:
                boosted += 0.03
            elif preferred_content_types and meta.get("content_type") not in preferred_content_types:
                boosted -= 0.02
            return boosted

        return sorted(docs, key=score, reverse=True)

    @staticmethod
    def _diversify_docs(docs: list[Document], k: int) -> list[Document]:
        if len(docs) <= k:
            return docs

        selected: list[Document] = []
        overflow: list[Document] = []
        per_file_counts: dict[str, int] = {}

        for doc in docs:
            file_name = str((doc.metadata or {}).get("file_name", ""))
            if per_file_counts.get(file_name, 0) < MAX_CHUNKS_PER_FILE:
                selected.append(doc)
                per_file_counts[file_name] = per_file_counts.get(file_name, 0) + 1
                if len(selected) >= k:
                    return selected
            else:
                overflow.append(doc)

        for doc in overflow:
            selected.append(doc)
            if len(selected) >= k:
                break
        return selected[:k]

    def _finalize_docs(
        self,
        docs: list[Document],
        info: dict[str, Any],
        k: int,
        focus: dict[str, list[str]],
        query: str,
    ) -> tuple[list[Document], dict[str, Any]]:
        reranked = self._boost_by_focus(docs, focus, query=query)
        final_docs = self._diversify_docs(reranked, k)
        info["focus_labels"] = focus.get("preferred_labels", [])
        info["focus_content_types"] = focus.get("preferred_content_types", [])
        return final_docs, info

    def _build_hybrid_retriever(self) -> None:
        self.hybrid = None
        self.cluster_index = load_cluster_index()
        self.summary_index = load_summary_index()

        if self.retriever_mode == "hybrid_cluster":
            return
        if self.retriever_mode not in HYBRID_RETRIEVER_MODES:
            return
        if self.vectorstore is None:
            return

        cfg = HybridConfig(
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.85")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.15")),
            use_clustering=False,
            n_clusters=0,
            cluster_top_n=0,
        )
        rows = get_corpus_rows(self.vectorstore)
        self.hybrid = HybridRetriever(rows, self.embedding_fn, cfg)

    def retrieve(self, query: str, k: int = 5) -> tuple[list[Document], dict[str, Any]]:
        if self.retriever_mode == "hybrid_cluster":
            return self._retrieve_with_cluster_prefilter(query, k=k)

        fetch_k = max(k, k * RETRIEVAL_FETCH_MULTIPLIER)
        query_vector = self.embedding_fn([query])[0]
        focus = detect_query_focus(query)
        summary_guidance = select_summary_guidance(
            query_vector,
            self.summary_index,
            top_docs=SUMMARY_TOP_DOCS,
            top_sections=SUMMARY_TOP_SECTIONS,
        )

        if self.retriever_mode in HYBRID_RETRIEVER_MODES and self.hybrid is not None:
            if summary_guidance.get("file_names"):
                where = {"file_name": {"$in": summary_guidance["file_names"]}}
                rows = get_corpus_rows(self.vectorstore, where=where)
                cfg = HybridConfig(
                    dense_weight=float(os.getenv("DENSE_WEIGHT", "0.85")),
                    sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.15")),
                    use_clustering=False,
                    n_clusters=0,
                    cluster_top_n=0,
                )
                hybrid = HybridRetriever(rows, self.embedding_fn, cfg)
                docs, info = hybrid.retrieve(query, k=fetch_k)
            else:
                docs, info = self.hybrid.retrieve(query, k=fetch_k)

            docs, info = self._finalize_docs(docs, info, k=k, focus=focus, query=query)
            info["query_vector"] = query_vector
            info["selected_clusters"] = []
            info["summary_guidance"] = summary_guidance
            return docs, info

        docs = search_documents(query=query, collection=self.vectorstore, k=fetch_k)
        docs, info = self._finalize_docs(
            docs,
            {
                "total_candidates": None,
                "cluster_enabled": False,
                "clusters": 0,
                "dense_weight": 1.0,
                "sparse_weight": 0.0,
                "selected_clusters": [],
                "query_vector": query_vector,
                "summary_guidance": summary_guidance,
                "top_docs": [],
            },
            k=k,
            focus=focus,
            query=query,
        )
        return docs, info

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
                "dense_weight": 0.85,
                "sparse_weight": 0.15,
                "top_docs": [],
            }

        fetch_k = max(k, k * RETRIEVAL_FETCH_MULTIPLIER)
        cfg = HybridConfig(
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.85")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.15")),
            use_clustering=False,
            n_clusters=0,
            cluster_top_n=0,
        )
        query_vector = self.embedding_fn([query])[0]
        focus = detect_query_focus(query)
        summary_guidance = select_summary_guidance(
            query_vector,
            self.summary_index,
            top_docs=SUMMARY_TOP_DOCS,
            top_sections=SUMMARY_TOP_SECTIONS,
        )
        cluster_ids, cluster_info = select_top_clusters(
            query_vector,
            self.cluster_index,
            top_n=CLUSTER_TOP_N,
        )

        where: dict[str, Any] | None = None
        if cluster_ids or summary_guidance.get("file_names"):
            and_filters: list[dict[str, Any]] = []
            if cluster_ids:
                and_filters.append({"cluster_id": {"$in": cluster_ids}})
            if summary_guidance.get("file_names"):
                and_filters.append({"file_name": {"$in": summary_guidance["file_names"]}})
            where = {"$and": and_filters} if len(and_filters) > 1 else and_filters[0]

        rows = get_corpus_rows(self.vectorstore, where=where)
        reranker = HybridRetriever(rows, self.embedding_fn, cfg)
        docs, info = reranker.retrieve(query, k=fetch_k)
        docs, info = self._finalize_docs(docs, info, k=k, focus=focus, query=query)
        info["cluster_enabled"] = bool(cluster_ids)
        info["clusters"] = cluster_info.get("clusters", 0)
        info["selected_clusters"] = cluster_info.get("selected_clusters", [])
        info["query_vector"] = query_vector
        info["summary_guidance"] = summary_guidance
        return docs, info
