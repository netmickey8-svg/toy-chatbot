from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.rag_chain import RAGChain


QUERIES = [
    "문서의 핵심 목적을 요약해줘",
    "지원 대상과 자격 요건을 알려줘",
    "평가 항목과 배점 기준을 알려줘",
    "제출 서류와 제출 방법을 정리해줘",
    "일정(공고/접수/선정) 관련 내용을 알려줘",
]


def run_mode(mode: str) -> dict:
    os.environ["RETRIEVER_MODE"] = mode
    rag = RAGChain()
    records = []
    for q in QUERIES:
        answer, docs, info = rag.ask(q)
        ret = info.get("retrieval", {})
        gen = info.get("generation", {})
        records.append(
            {
                "query": q,
                "results_count": ret.get("results_count"),
                "method": ret.get("method"),
                "cluster_enabled": ret.get("cluster_enabled"),
                "top_files": [d.metadata.get("file_name") for d in docs[:3]],
                "top_pages": [d.metadata.get("page_number") for d in docs[:3]],
                "answer_len": len(answer or ""),
                "gen_status": gen.get("status"),
            }
        )
    return {"mode": mode, "records": records}


def main() -> None:
    results = []
    for mode in ["dense", "hybrid", "hybrid_cluster"]:
        results.append(run_mode(mode))

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open("vectorstore/retrieval_mode_comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[OK] comparison saved: vectorstore/retrieval_mode_comparison.json")


if __name__ == "__main__":
    main()
