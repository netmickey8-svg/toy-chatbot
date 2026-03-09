from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from langchain_core.documents import Document
from sklearn.cluster import KMeans

from config import (
    CLUSTER_INDEX_PATH,
    CLUSTER_N_CLUSTERS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)


def _embed_texts(texts: list[str], embedding_fn) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, len(texts))
        vectors.extend(embedding_fn(texts[start:end]))
    dense = np.asarray(vectors, dtype=np.float32)
    return dense


def _build_cluster_metadata(
    vectors: np.ndarray,
    n_clusters: int | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if vectors.size == 0:
        return np.array([], dtype=int), None

    normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    target_clusters = max(1, min(n_clusters or CLUSTER_N_CLUSTERS, len(normalized)))

    if target_clusters == 1:
        labels = np.zeros((len(normalized),), dtype=int)
        centers = np.mean(normalized, axis=0, keepdims=True)
    else:
        km = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(normalized)
        centers = km.cluster_centers_.astype(np.float32)

    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    sizes = Counter(int(label) for label in labels.tolist())
    cluster_meta = {
        "version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_clusters": int(len(centers)),
        "total_chunks": int(len(normalized)),
        "centroids": [
            {
                "cluster_id": int(cluster_id),
                "size": int(sizes.get(cluster_id, 0)),
                "vector": center.tolist(),
            }
            for cluster_id, center in enumerate(centers)
        ],
    }
    return labels, cluster_meta


def build_cluster_index(
    chunks: list[Document],
    embedding_fn,
    n_clusters: int | None = None,
) -> tuple[list[Document], dict[str, Any] | None]:
    texts = [(chunk.page_content or "").strip() for chunk in chunks]
    valid_pairs = [(idx, text) for idx, text in enumerate(texts) if text]
    if not valid_pairs:
        return chunks, None

    chunk_indexes = [idx for idx, _ in valid_pairs]
    valid_texts = [text for _, text in valid_pairs]
    vectors = _embed_texts(valid_texts, embedding_fn)
    if vectors.size == 0:
        return chunks, None

    labels, cluster_meta = _build_cluster_metadata(vectors, n_clusters=n_clusters)
    if cluster_meta is None:
        return chunks, None

    for local_idx, chunk_idx in enumerate(chunk_indexes):
        chunk = chunks[chunk_idx]
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["cluster_id"] = int(labels[local_idx])

    return chunks, cluster_meta


def build_cluster_index_from_rows(
    rows: list[dict[str, Any]],
    n_clusters: int | None = None,
) -> tuple[dict[str, int], dict[str, Any] | None]:
    valid_rows = []
    for row in rows:
        row_id = row.get("id")
        vector = row.get("vector")
        if row_id is None or vector is None:
            continue
        valid_rows.append((str(row_id), vector))

    if not valid_rows:
        return {}, None

    row_ids = [row_id for row_id, _ in valid_rows]
    vectors = np.asarray([vector for _, vector in valid_rows], dtype=np.float32)
    labels, cluster_meta = _build_cluster_metadata(vectors, n_clusters=n_clusters)
    if cluster_meta is None:
        return {}, None

    assignments = {
        row_ids[idx]: int(labels[idx])
        for idx in range(len(row_ids))
    }
    return assignments, cluster_meta


def save_cluster_index(cluster_meta: dict[str, Any] | None) -> None:
    if not cluster_meta:
        return
    CLUSTER_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTER_INDEX_PATH.write_text(
        json.dumps(cluster_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_cluster_index() -> None:
    if CLUSTER_INDEX_PATH.exists():
        CLUSTER_INDEX_PATH.unlink()


def load_cluster_index() -> dict[str, Any] | None:
    if not CLUSTER_INDEX_PATH.exists():
        return None
    try:
        return json.loads(CLUSTER_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def select_top_clusters(
    query_vector: list[float] | np.ndarray,
    cluster_meta: dict[str, Any] | None,
    top_n: int,
) -> tuple[list[int], dict[str, Any]]:
    if not cluster_meta:
        return [], {"cluster_enabled": False, "clusters": 0, "selected_clusters": []}

    centroids = cluster_meta.get("centroids") or []
    if not centroids:
        return [], {"cluster_enabled": False, "clusters": 0, "selected_clusters": []}

    qvec = np.asarray(query_vector, dtype=np.float32)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    rows = []
    for item in centroids:
        vec = np.asarray(item.get("vector") or [], dtype=np.float32)
        if vec.size == 0:
            continue
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        rows.append(
            {
                "cluster_id": int(item["cluster_id"]),
                "size": int(item.get("size", 0)),
                "score": float(vec @ qvec),
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    selected = rows[: max(1, min(top_n, len(rows)))]
    return [item["cluster_id"] for item in selected], {
        "cluster_enabled": True,
        "clusters": int(cluster_meta.get("n_clusters", len(rows))),
        "selected_clusters": selected,
    }
