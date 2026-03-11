from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config import EMBEDDING_BATCH_SIZE, SUMMARY_INDEX_PATH


_STOPWORDS = {
    "그리고", "또한", "대한", "위한", "기반", "사업", "시스템", "플랫폼", "제안",
    "구축", "운영", "유지", "관리", "기능", "개선", "용역", "사업화", "지원",
    "서비스", "내용", "관련", "통한", "기존", "분야", "추진", "대한민국",
    "있습니다", "합니다", "합니다.", "위해", "통해", "수행", "적용", "제공",
}


def _embed_texts(texts: list[str], embedding_fn) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, len(texts))
        vectors.extend(embedding_fn(texts[start:end]))
    return vectors


def _extract_keywords(texts: list[str], top_n: int = 10) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9\-]{1,}", text)
        for token in tokens:
            lowered = token.lower()
            if lowered in _STOPWORDS or len(lowered) < 2:
                continue
            counter[lowered] += 1
    return [token for token, _ in counter.most_common(top_n)]


def _split_candidates(text: str, content_type: str | None = None) -> list[str]:
    clean = (text or "").replace("\r\n", "\n").strip()
    if not clean:
        return []

    if content_type == "table":
        return [line.strip() for line in clean.splitlines() if len(line.strip()) >= 4]

    parts = re.split(r"(?<=[\.\!\?\u3002])\s+|\n{1,}", clean)
    return [part.strip() for part in parts if len(part.strip()) >= 10]


def _score_candidate(candidate: str, keywords: list[str]) -> float:
    lowered = candidate.lower()
    score = 0.0

    for keyword in keywords:
        if keyword in lowered:
            score += 2.0

    score += min(len(candidate) / 120.0, 2.0)

    if re.search(r"\d{4}", candidate):
        score += 0.8
    if re.search(r"[A-Za-z]{2,}", candidate):
        score += 0.3
    if re.search(r"[0-9]+", candidate):
        score += 0.4
    if any(token in candidate for token in ["구축", "개발", "운영", "플랫폼", "시스템", "서비스"]):
        score += 0.8
    return score


def _select_representative_lines(
    group_rows: list[dict[str, Any]],
    keywords: list[str],
    max_lines: int,
) -> list[str]:
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()

    for row in group_rows:
        meta = dict(row.get("metadata") or {})
        content_type = meta.get("content_type", "text")
        for candidate in _split_candidates(row.get("document", ""), content_type=content_type):
            normalized = re.sub(r"\s+", " ", candidate).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            score = _score_candidate(normalized, keywords)
            candidates.append((score, normalized))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for _, candidate in candidates:
        if any(candidate in existing or existing in candidate for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_lines:
            break
    return selected


def _build_document_summary(group_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    meta = dict(group_rows[0].get("metadata") or {})
    labels = Counter((row.get("metadata") or {}).get("chunk_label", "기타") for row in group_rows)
    content_types = Counter((row.get("metadata") or {}).get("content_type", "text") for row in group_rows)
    section_titles = Counter((row.get("metadata") or {}).get("section_title", "본문") for row in group_rows)
    texts = [row.get("document", "") for row in group_rows]
    keywords = _extract_keywords(texts)
    top_labels = [label for label, _ in labels.most_common(4)]
    top_sections = [title for title, _ in section_titles.most_common(4)]
    representative_lines = _select_representative_lines(group_rows, keywords, max_lines=3)
    summary_excerpt = " / ".join(representative_lines)

    summary_text = (
        f"문서 요약 | 파일: {meta.get('file_name', 'Unknown')} | "
        f"프로젝트명: {meta.get('project_name', '')} | "
        f"부문: {meta.get('department', '')} | 연도: {meta.get('year', '')} | "
        f"주요 라벨: {', '.join(top_labels)} | "
        f"주요 섹션: {', '.join(top_sections)} | "
        f"핵심 키워드: {', '.join(keywords[:6])} | "
        f"핵심 내용: {summary_excerpt}"
    )
    payload = {
        "summary_type": "document",
        "file_name": meta.get("file_name"),
        "department": meta.get("department"),
        "year": meta.get("year"),
        "project_name": meta.get("project_name"),
        "top_labels": top_labels,
        "top_sections": top_sections,
        "content_types": dict(content_types),
        "keywords": keywords,
        "chunk_count": len(group_rows),
        "summary_excerpt": representative_lines,
    }
    return summary_text, payload


def _build_section_summary(group_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    meta = dict(group_rows[0].get("metadata") or {})
    texts = [row.get("document", "") for row in group_rows]
    keywords = _extract_keywords(texts)
    label = meta.get("chunk_label", "기타")
    section_title = meta.get("section_title", "본문")
    content_type = meta.get("content_type", "text")
    representative_lines = _select_representative_lines(group_rows, keywords, max_lines=2)
    summary_excerpt = " / ".join(representative_lines)

    summary_text = (
        f"섹션 요약 | 파일: {meta.get('file_name', 'Unknown')} | "
        f"라벨: {label} | 섹션: {section_title} | 타입: {content_type} | "
        f"키워드: {', '.join(keywords[:6])} | "
        f"핵심 내용: {summary_excerpt}"
    )
    payload = {
        "summary_type": "section",
        "file_name": meta.get("file_name"),
        "department": meta.get("department"),
        "year": meta.get("year"),
        "project_name": meta.get("project_name"),
        "section_title": section_title,
        "content_type": content_type,
        "chunk_label": label,
        "keywords": keywords,
        "chunk_count": len(group_rows),
        "summary_excerpt": representative_lines,
    }
    return summary_text, payload


def build_summary_index_from_rows(rows: list[dict[str, Any]], embedding_fn) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_section: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        meta = dict(row.get("metadata") or {})
        file_name = meta.get("file_name")
        text = (row.get("document") or "").strip()
        if not file_name or not text:
            continue
        by_file[file_name].append(row)
        key = (
            file_name,
            str(meta.get("chunk_label", "기타")),
            str(meta.get("section_title", "본문")),
            str(meta.get("content_type", "text")),
        )
        by_section[key].append(row)

    entries: list[dict[str, Any]] = []
    summary_texts: list[str] = []

    for file_name, group_rows in by_file.items():
        summary_text, payload = _build_document_summary(group_rows)
        entries.append(
            {
                "id": f"doc::{file_name}",
                "text": summary_text,
                "payload": payload,
            }
        )
        summary_texts.append(summary_text)

    for key, group_rows in by_section.items():
        file_name, chunk_label, section_title, content_type = key
        summary_text, payload = _build_section_summary(group_rows)
        entries.append(
            {
                "id": f"section::{file_name}::{chunk_label}::{section_title}::{content_type}",
                "text": summary_text,
                "payload": payload,
            }
        )
        summary_texts.append(summary_text)

    vectors = _embed_texts(summary_texts, embedding_fn) if summary_texts else []
    for idx, entry in enumerate(entries):
        vector = vectors[idx] if idx < len(vectors) else []
        entry["vector"] = vector

    return {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "documents": [entry for entry in entries if entry["payload"]["summary_type"] == "document"],
        "sections": [entry for entry in entries if entry["payload"]["summary_type"] == "section"],
    }


def save_summary_index(summary_index: dict[str, Any] | None) -> None:
    if not summary_index:
        return
    SUMMARY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_INDEX_PATH.write_text(
        json.dumps(summary_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_summary_index() -> dict[str, Any] | None:
    if not SUMMARY_INDEX_PATH.exists():
        return None
    try:
        return json.loads(SUMMARY_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_summary_index() -> None:
    if SUMMARY_INDEX_PATH.exists():
        SUMMARY_INDEX_PATH.unlink()


def _score_entries(entries: list[dict[str, Any]], query_vector: np.ndarray) -> list[dict[str, Any]]:
    scored = []
    qvec = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    for entry in entries:
        vector = np.asarray(entry.get("vector") or [], dtype=np.float32)
        if vector.size == 0:
            continue
        vector = vector / (np.linalg.norm(vector) + 1e-12)
        score = float(vector @ qvec)
        scored.append(
            {
                "id": entry.get("id"),
                "text": entry.get("text"),
                "payload": entry.get("payload", {}),
                "score": score,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def select_summary_guidance(
    query_vector: list[float] | np.ndarray,
    summary_index: dict[str, Any] | None,
    top_docs: int,
    top_sections: int,
) -> dict[str, Any]:
    if not summary_index:
        return {
            "documents": [],
            "sections": [],
            "file_names": [],
            "section_filters": [],
        }

    qvec = np.asarray(query_vector, dtype=np.float32)
    doc_rows = _score_entries(summary_index.get("documents", []), qvec)[:top_docs]
    section_rows = _score_entries(summary_index.get("sections", []), qvec)[:top_sections]
    file_names = []
    for row in doc_rows:
        name = row.get("payload", {}).get("file_name")
        if name and name not in file_names:
            file_names.append(name)

    section_filters = []
    for row in section_rows:
        payload = row.get("payload", {})
        section_filters.append(
            {
                "file_name": payload.get("file_name"),
                "chunk_label": payload.get("chunk_label"),
                "section_title": payload.get("section_title"),
                "content_type": payload.get("content_type"),
                "score": row.get("score"),
            }
        )

    return {
        "documents": doc_rows,
        "sections": section_rows,
        "file_names": file_names,
        "section_filters": section_filters,
    }
