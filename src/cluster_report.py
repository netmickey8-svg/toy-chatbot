from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def build_cluster_quality_report(
    rows: list[dict[str, Any]],
    sample_size: int = 2000,
) -> dict[str, Any]:
    valid_rows = []
    for row in rows:
        meta = dict(row.get("metadata") or {})
        vector = row.get("vector")
        if vector is None or meta.get("cluster_id") is None:
            continue
        valid_rows.append(
            {
                "id": row.get("id"),
                "cluster_id": int(meta["cluster_id"]),
                "file_name": meta.get("file_name", "Unknown"),
                "vector": vector,
            }
        )

    if not valid_rows:
        return {
            "summary": {
                "total_points": 0,
                "clusters": 0,
                "sampled_points": 0,
            },
            "metrics": {},
            "clusters": [],
        }

    cluster_counts = Counter(item["cluster_id"] for item in valid_rows)
    cluster_files: dict[int, Counter] = defaultdict(Counter)
    for item in valid_rows:
        cluster_files[item["cluster_id"]][item["file_name"]] += 1

    cluster_records = []
    for cluster_id, count in sorted(cluster_counts.items(), key=lambda item: item[0]):
        file_counter = cluster_files[cluster_id]
        dominant_file, dominant_count = file_counter.most_common(1)[0]
        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "chunks": count,
                "files": len(file_counter),
                "dominant_file": dominant_file,
                "dominant_share": round(dominant_count / max(count, 1), 3),
            }
        )

    sample_n = min(sample_size, len(valid_rows))
    rng = np.random.default_rng(42)
    if sample_n < len(valid_rows):
        sample_idx = rng.choice(len(valid_rows), size=sample_n, replace=False)
        sampled_rows = [valid_rows[int(idx)] for idx in sample_idx]
    else:
        sampled_rows = valid_rows

    labels = np.asarray([item["cluster_id"] for item in sampled_rows], dtype=np.int32)
    vectors = np.asarray([item["vector"] for item in sampled_rows], dtype=np.float32)
    vectors = _normalize_vectors(vectors)

    metrics: dict[str, float | None] = {
        "silhouette_cosine": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2 and len(vectors) > len(unique_labels):
        metrics["silhouette_cosine"] = float(silhouette_score(vectors, labels, metric="cosine"))
        metrics["davies_bouldin"] = float(davies_bouldin_score(vectors, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(vectors, labels))

    summary = {
        "total_points": len(valid_rows),
        "clusters": len(cluster_counts),
        "sampled_points": len(sampled_rows),
        "min_cluster_size": min(cluster_counts.values()),
        "median_cluster_size": int(np.median(list(cluster_counts.values()))),
        "max_cluster_size": max(cluster_counts.values()),
    }
    return {
        "summary": summary,
        "metrics": metrics,
        "clusters": cluster_records,
    }
