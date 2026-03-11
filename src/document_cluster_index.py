from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN

from config import (
    DOCUMENT_CLUSTER_INDEX_PATH,
    DOCUMENT_CLUSTER_MIN_CLUSTER_SIZE,
    DOCUMENT_CLUSTER_MIN_SAMPLES,
    EMBEDDING_MODEL,
)


def _mean_center(vectors: np.ndarray) -> list[float]:
    center = np.mean(vectors, axis=0, keepdims=False)
    center = center / (np.linalg.norm(center) + 1e-12)
    return center.astype(np.float32).tolist()


def build_document_cluster_index(
    summary_index: dict[str, Any] | None,
    n_clusters: int | None = None,
) -> dict[str, Any] | None:
    if not summary_index:
        return None

    documents = summary_index.get("documents", [])
    valid_docs = []
    for entry in documents:
        vector = entry.get("vector")
        payload = dict(entry.get("payload") or {})
        if not vector or not payload.get("file_name"):
            continue
        valid_docs.append(
            {
                "id": entry.get("id"),
                "text": entry.get("text", ""),
                "vector": vector,
                "payload": payload,
            }
        )

    if not valid_docs:
        return None

    vectors = np.asarray([doc["vector"] for doc in valid_docs], dtype=np.float32)
    normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    if len(normalized) == 1:
        labels = np.zeros((1,), dtype=int)
    else:
        clusterer = HDBSCAN(
            min_cluster_size=max(2, min(DOCUMENT_CLUSTER_MIN_CLUSTER_SIZE, len(normalized))),
            min_samples=max(1, min(DOCUMENT_CLUSTER_MIN_SAMPLES, len(normalized) - 1)),
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(normalized)

    sizes = Counter(int(label) for label in labels.tolist())
    unique_clusters = sorted(cluster_id for cluster_id in sizes.keys() if cluster_id >= 0)
    centroids = []
    for cluster_id in unique_clusters:
        cluster_vectors = normalized[labels == cluster_id]
        centroids.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(sizes.get(cluster_id, 0)),
                "vector": _mean_center(cluster_vectors),
            }
        )

    cluster_rows = []
    for idx, doc in enumerate(valid_docs):
        payload = dict(doc["payload"])
        payload["document_cluster_id"] = int(labels[idx])
        cluster_rows.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "vector": doc["vector"],
                "payload": payload,
            }
        )

    return {
        "version": 2,
        "algorithm": "hdbscan",
        "embedding_model": EMBEDDING_MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_clusters": int(len(unique_clusters)),
        "noise_documents": int(sizes.get(-1, 0)),
        "total_documents": int(len(valid_docs)),
        "documents": cluster_rows,
        "centroids": centroids,
        "params": {
            "min_cluster_size": int(max(2, min(DOCUMENT_CLUSTER_MIN_CLUSTER_SIZE, len(normalized)))),
            "min_samples": int(max(1, min(DOCUMENT_CLUSTER_MIN_SAMPLES, max(1, len(normalized) - 1)))),
            "requested_n_clusters": n_clusters,
        },
    }


def save_document_cluster_index(cluster_index: dict[str, Any] | None) -> None:
    if not cluster_index:
        return
    DOCUMENT_CLUSTER_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_CLUSTER_INDEX_PATH.write_text(
        json.dumps(cluster_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_document_cluster_index() -> dict[str, Any] | None:
    if not DOCUMENT_CLUSTER_INDEX_PATH.exists():
        return None
    try:
        return json.loads(DOCUMENT_CLUSTER_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_document_cluster_index() -> None:
    if DOCUMENT_CLUSTER_INDEX_PATH.exists():
        DOCUMENT_CLUSTER_INDEX_PATH.unlink()
