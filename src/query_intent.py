from __future__ import annotations

import re


_INDEX_INVENTORY_PATTERNS = [
    r"제안서.*(몇|개수|갯수|수)",
    r"문서.*(몇|개수|갯수|수)",
    r"하나만\s*있",
    r"파일.*(목록|리스트|몇)",
    r"인덱싱.*(파일|문서).*(몇|목록|리스트)",
]

_PEOPLE_PATTERNS = [
    "참여하는 사람",
    "누가 참여",
    "참여자",
    "참여 인원",
    "참여 인력",
    "참여인력",
    "투입 인력",
    "투입인력",
]

_OVERLAP_PATTERNS = [
    "겹쳐",
    "겹치",
    "중복",
    "동일 인물",
    "같은 사람",
    "겹치는 사람",
    "중복 참여",
]

_BUDGET_PATTERNS = [
    "예산",
    "사업비",
    "금액",
    "비용",
    "견적",
]

_SCHEDULE_PATTERNS = [
    "일정",
    "로드맵",
    "기간",
    "언제",
    "착수",
    "완료",
]

_PERFORMANCE_PATTERNS = [
    "실적",
    "레퍼런스",
    "경험",
    "유사사업",
]

_TECH_PATTERNS = [
    "기술",
    "아키텍처",
    "구성도",
    "플랫폼",
    "시스템 구성",
    "api",
    "db",
]

_OPERATION_PATTERNS = [
    "운영",
    "유지보수",
    "유지관리",
    "장애",
    "모니터링",
]

_FOCUS_RULES = [
    (_PEOPLE_PATTERNS, ["참여인력"], ["text", "table"]),
    (_BUDGET_PATTERNS, ["예산/비용"], ["text", "table"]),
    (_SCHEDULE_PATTERNS, ["일정/계획"], ["text", "table"]),
    (_PERFORMANCE_PATTERNS, ["수행실적"], ["text"]),
    (_TECH_PATTERNS, ["기술/아키텍처", "기능/서비스"], ["text", "ocr"]),
    (_OPERATION_PATTERNS, ["운영/유지보수"], ["text"]),
]


def is_index_inventory_question(question: str) -> bool:
    q = (question or "").strip()
    return any(re.search(pattern, q) for pattern in _INDEX_INVENTORY_PATTERNS)


def normalize_retrieval_query(question: str) -> str:
    """
    표현이 다른 동의 질의를 retrieval 친화 키워드로 확장한다.
    생성용 질문(question)은 원문 유지하고 검색용 질의만 확장한다.
    """
    q = (question or "").strip()
    lowered = q.lower()

    if any(pattern in q for pattern in _PEOPLE_PATTERNS) or "participant" in lowered:
        return q + " 참여인력 투입인력 연구원 인력구성 참여율"

    return q


def detect_query_focus(question: str) -> dict[str, list[str]]:
    """
    질문 유형에 따라 우선적으로 볼 라벨/타입 힌트를 반환한다.
    """
    q = (question or "").strip()
    lowered = q.lower()

    preferred_labels: list[str] = []
    preferred_content_types: list[str] = []
    for patterns, labels, content_types in _FOCUS_RULES:
        if any(pattern in q for pattern in patterns) or any(pattern in lowered for pattern in patterns):
            for label in labels:
                if label not in preferred_labels:
                    preferred_labels.append(label)
            for content_type in content_types:
                if content_type not in preferred_content_types:
                    preferred_content_types.append(content_type)

    return {
        "preferred_labels": preferred_labels,
        "preferred_content_types": preferred_content_types,
    }


def is_people_question(question: str) -> bool:
    q = (question or "").strip()
    lowered = q.lower()
    return any(pattern in q for pattern in _PEOPLE_PATTERNS) or "participant" in lowered


def is_people_overlap_question(question: str) -> bool:
    q = (question or "").strip()
    if not is_people_question(q):
        return False
    return any(pattern in q for pattern in _OVERLAP_PATTERNS)


def is_structured_focus_question(question: str) -> bool:
    q = (question or "").strip()
    lowered = q.lower()
    pattern_groups = [
        _PEOPLE_PATTERNS,
        _BUDGET_PATTERNS,
        _SCHEDULE_PATTERNS,
    ]
    return any(any(pattern in q for pattern in group) for group in pattern_groups) or any(
        token in lowered for token in ["participant", "budget", "schedule"]
    )
