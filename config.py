"""
프로젝트 전역 설정 관리 모듈
============================
역할:
    - 환경 변수(.env)를 로드하여 각 모듈에서 공통으로 사용할 설정값 제공
    - 경로, API 키, 임베딩 모델, 청킹, 검색 설정을 한 곳에서 관리
사용:
    from config import GOOGLE_API_KEY, EMBEDDING_MODEL, DATA_DIR
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# .env 파일 자동 로드 (프로젝트 루트 기준)
# ──────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
# 프로젝트 루트 디렉토리 (config.py 위치 기준)
PROJECT_ROOT: Path = Path(__file__).parent

# 제안서 PDF가 저장된 디렉토리
# 환경변수 DATA_DIR 미설정 시 기본값 사용 (다른 PC에서 실행 시 .env에 DATA_DIR 지정 권장)
DATA_DIR: Path = Path(os.getenv("DATA_DIR", r"C:\Users\User\Downloads\제안서"))

# 벡터스토어(임베딩 인덱스) 저장 디렉토리
VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"

# ──────────────────────────────────────────────
# Google Gemini API 설정
# ──────────────────────────────────────────────
# API 키: .env 파일의 GOOGLE_API_KEY 항목에서 로드
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

# 사용할 Gemini 모델명
# 참고: https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ──────────────────────────────────────────────
# 임베딩 설정 (API 또는 로컬)
# ──────────────────────────────────────────────
# EMBEDDING_PROVIDER:
#   - "openai": OpenAI Embeddings API 사용 (권장)
#   - "local" : 로컬 임베딩 모델 사용 (bge-m3 등)
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

# OpenAI API 키 (EMBEDDING_PROVIDER=openai일 때 필요)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# 임베딩 모델명
# - OpenAI 권장: text-embedding-3-small / text-embedding-3-large
# - 로컬 권장: BAAI/bge-m3
_DEFAULT_EMBEDDING_MODEL = (
    "text-embedding-3-small"
    if EMBEDDING_PROVIDER == "openai"
    else "BAAI/bge-m3"
)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)

# 임베딩 벡터 배치 처리 크기 (메모리 부족 시 줄임)
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ──────────────────────────────────────────────
# OCR 설정 (이미지 기반 PDF 텍스트 추출)
# ──────────────────────────────────────────────
# OCR 활성화 여부 (텍스트가 없는 스캔 PDF 처리용)
OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"

# Tesseract OCR 언어 설정 (한글+영어 동시 인식)
OCR_LANG: str = os.getenv("OCR_LANG", "kor+eng")

# OCR 적용 기준: 페이지 텍스트가 이 글자 수 미만이면 OCR 수행
OCR_MIN_TEXT_CHARS: int = int(os.getenv("OCR_MIN_TEXT_CHARS", "30"))

# Tesseract 실행 파일 경로 (설치 경로가 기본값과 다를 경우 .env에 지정)
# 예: TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD")

# ──────────────────────────────────────────────
# 청킹(Chunking) 설정
# ──────────────────────────────────────────────
# 각 청크의 최대 문자 수 (너무 작으면 문맥 손실, 너무 크면 검색 정확도 저하)
CHUNK_SIZE: int = 800

# 청크 간 겹치는 문자 수 (문맥 연속성 유지)
CHUNK_OVERLAP: int = 100

# ──────────────────────────────────────────────
# 검색(Retrieval) 설정
# ──────────────────────────────────────────────
# 한 번 검색 시 반환할 최대 문서(청크) 수
TOP_K_RESULTS: int = 5
