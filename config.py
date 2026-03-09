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
# .env 값을 현재 프로세스 환경변수보다 우선 적용해 실행 일관성 보장
load_dotenv(override=True)

# Chroma telemetry 스레드 비활성화 (Windows 환경 안정성)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
# 프로젝트 루트 디렉토리 (config.py 위치 기준)
PROJECT_ROOT: Path = Path(__file__).parent

# 제안서 PDF가 저장된 디렉토리
# 환경변수 DATA_DIR 미설정 시 프로젝트 내부 data/ 사용
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

# 벡터스토어(임베딩 인덱스) 저장 디렉토리
VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"

# 인덱싱 로그 저장 디렉토리
INDEX_LOG_DIR: Path = VECTORSTORE_DIR / "index_logs"

# 사전 군집화 메타데이터 저장 경로
CLUSTER_INDEX_PATH: Path = VECTORSTORE_DIR / "cluster_index.json"

# ──────────────────────────────────────────────
# LLM API 설정 (OpenAI 호환 로컬 서버)
# ──────────────────────────────────────────────
# 문서 기준 기본값: Qwen3-vl-2b vLLM 서버
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://192.168.5.94:8001/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3-vl-2b-instruct")

# ──────────────────────────────────────────────
# 임베딩 설정 (API 또는 로컬)
# ──────────────────────────────────────────────
# EMBEDDING_PROVIDER:
#   - "openai" : OpenAI Embeddings API 사용
#   - "local"  : sentence-transformers 기반 로컬 임베딩 (torch 필요)
#   - "hashing": torch 없이 동작하는 CPU 해시 임베딩
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

# 원격 OpenAI 호환 임베딩 서버 설정 (EMBEDDING_PROVIDER=remote_openai)
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://192.168.5.95:8080/v1")
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "EMPTY")

# hashing 임베딩 차원수 (EMBEDDING_PROVIDER=hashing 일 때 사용)
EMBEDDING_HASH_DIM: int = int(os.getenv("EMBEDDING_HASH_DIM", "1024"))

# 임베딩 벡터 배치 처리 크기 (메모리 부족 시 줄임)
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# 벡터 백엔드 설정: qdrant / simple
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "qdrant").lower()

# Qdrant 설정
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://192.168.5.95:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "proposals")

# 사전 군집화 설정
CLUSTERING_ENABLED: bool = os.getenv("CLUSTERING_ENABLED", "true").lower() == "true"
CLUSTER_N_CLUSTERS: int = int(os.getenv("CLUSTER_N_CLUSTERS", "12"))
CLUSTER_TOP_N: int = int(os.getenv("CLUSTER_TOP_N", "2"))

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
