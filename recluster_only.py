from __future__ import annotations

from config import CLUSTER_N_CLUSTERS
from src.vectordb import load_vectorstore, recluster_collection


def main() -> int:
    collection = load_vectorstore()
    if collection is None:
        print("[ERROR] 벡터스토어가 없습니다. 먼저 인덱싱을 실행하세요.")
        return 1

    cluster_meta = recluster_collection(collection)
    if not cluster_meta:
        print("[ERROR] 클러스터 재계산에 실패했습니다.")
        return 1

    print("[OK] 클러스터 재계산 완료")
    print(f"  - cluster count: {cluster_meta.get('n_clusters', CLUSTER_N_CLUSTERS)}")
    print(f"  - total chunks:  {cluster_meta.get('total_chunks', 0)}")
    print("  - metadata:      vectorstore/cluster_index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
