"""
인덱싱 로그 저장/조회 모듈
==========================
역할:
    - 인덱싱 시 PDF 파싱/청킹 로그를 파일로 저장
    - 질문 시 파일별 인덱싱 로그를 조회
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from config import INDEX_LOG_DIR
from src.pdf_processor import ProcessedDocument


LOG_MAP_NAME = "index_log_map.json"


def _file_hash(file_path: str) -> str:
    return hashlib.md5(file_path.encode("utf-8", errors="ignore")).hexdigest()


def _load_log_map() -> dict[str, str]:
    if not INDEX_LOG_DIR.exists():
        return {}
    map_path = INDEX_LOG_DIR / LOG_MAP_NAME
    if not map_path.exists():
        return {}
    try:
        return json.loads(map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_log_map(data: dict[str, str]) -> None:
    INDEX_LOG_DIR.mkdir(parents=True, exist_ok=True)
    map_path = INDEX_LOG_DIR / LOG_MAP_NAME
    map_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_index_logs(docs: list[ProcessedDocument], chunks: list[Document]) -> None:
    """
    인덱싱 로그를 파일별로 저장
    """
    if not docs:
        return

    INDEX_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 청크 통계 집계
    chunk_stats: dict[str, dict[str, dict[int, int] | int]] = {}
    for chunk in chunks:
        meta = chunk.metadata or {}
        file_name = meta.get("file_name", "Unknown")
        page_number = int(meta.get("page_number", 0) or 0)
        if file_name not in chunk_stats:
            chunk_stats[file_name] = {
                "total_chunks": 0,
                "chunks_per_page": {},
            }
        chunk_stats[file_name]["total_chunks"] += 1
        if page_number > 0:
            chunks_per_page = chunk_stats[file_name]["chunks_per_page"]
            chunks_per_page[page_number] = chunks_per_page.get(page_number, 0) + 1

    log_map = _load_log_map()
    timestamp = datetime.now(timezone.utc).isoformat()

    for doc in docs:
        meta = doc.metadata
        stats = doc.stats or {}
        file_hash = _file_hash(meta.file_path)
        log_path = INDEX_LOG_DIR / f"{file_hash}.json"

        file_chunk_stats = chunk_stats.get(meta.file_name, {})
        data = {
            "file_name": meta.file_name,
            "file_path": meta.file_path,
            "department": meta.department,
            "year": meta.year,
            "total_pages": meta.total_pages,
            "extracted_pages": stats.get("extracted_pages", 0),
            "text_pages": stats.get("text_pages", 0),
            "table_pages": stats.get("table_pages", 0),
            "ocr_pages": stats.get("ocr_pages", 0),
            "total_chars": stats.get("total_chars", 0),
            "total_chunks": file_chunk_stats.get("total_chunks", 0),
            "chunks_per_page": file_chunk_stats.get("chunks_per_page", {}),
            "indexed_at": timestamp,
        }

        log_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log_map[meta.file_name] = file_hash

    _save_log_map(log_map)


def load_index_log(file_name: str) -> dict | None:
    """
    파일명 기준 인덱싱 로그 조회
    """
    if not file_name:
        return None
    log_map = _load_log_map()
    file_hash = log_map.get(file_name)
    if not file_hash:
        return None
    log_path = INDEX_LOG_DIR / f"{file_hash}.json"
    if not log_path.exists():
        return None
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None
