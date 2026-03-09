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
