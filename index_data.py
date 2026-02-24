"""
데이터 인덱싱 스크립트
======================
역할:
    - 제안서 폴더(DATA_DIR)의 PDF 파일을 처리하여 벡터스토어를 생성
    - 새 PDF가 추가되거나 업데이트된 경우 이 스크립트를 다시 실행

실행 방법:
    python index_data.py

파이프라인:
    1. PDF 처리 (pdf_processor.py)
       └→ 텍스트 + 표 추출, OCR 적용 (스캔 PDF)
    2. 청킹 (chunker.py)
       └→ 각 페이지를 800자 단위 청크로 분할 (겹침 100자)
    3. 임베딩 & 벡터스토어 저장 (vectordb.py)
       └→ bge-m3 모델로 Dense 임베딩 생성 → ChromaDB에 저장

출력:
    vectorstore/chroma_db/  ← 생성된 벡터스토어 (앱 재시작 시 자동 로드)

주의:
    - 첫 실행 시 BAAI/bge-m3 모델 자동 다운로드 (~2GB)
    - 재실행 시 기존 벡터스토어 삭제 후 재생성 (전체 재색인)
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, VECTORSTORE_DIR
from src.pdf_processor import process_all_pdfs
from src.chunker import chunk_all_documents
from src.vectordb import create_vectorstore


def main() -> None:
    """인덱싱 파이프라인 실행"""
    start_time = time.time()

    print("=" * 60)
    print("  제안서 RAG 인덱싱 시작")
    print(f"  데이터 경로: {DATA_DIR}")
    print(f"  저장 경로:   {VECTORSTORE_DIR}")
    print("=" * 60)

    # ── Step 1: PDF 처리 ─────────────────────────────
    print("\n[Step 1/3] PDF 텍스트 추출 (텍스트 레이어 + 표 + OCR)")
    documents = process_all_pdfs(DATA_DIR)
    if not documents:
        print("[ERROR] 처리된 문서가 없습니다. DATA_DIR 경로를 확인하세요.")
        print(f"        현재 설정: {DATA_DIR}")
        return

    # ── Step 2: 청킹 ─────────────────────────────────
    print(f"\n[Step 2/3] 청킹 ({len(documents)}개 문서 → 800자 단위 분할)")
    chunks = chunk_all_documents(documents)
    if not chunks:
        print("[ERROR] 청크가 생성되지 않았습니다.")
        return

    # ── Step 3: 임베딩 & 저장 ─────────────────────────
    print(f"\n[Step 3/3] bge-m3 임베딩 생성 및 ChromaDB 저장 ({len(chunks)}개 청크)")
    create_vectorstore(chunks)

    # ── 완료 ──────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ✅ 인덱싱 완료!")
    print(f"  - 처리 문서: {len(documents)}개")
    print(f"  - 생성 청크: {len(chunks)}개")
    print(f"  - 소요 시간: {elapsed:.1f}초")
    print(f"  - 저장 위치: {VECTORSTORE_DIR / 'chroma_db'}")
    print("\n  이제 'streamlit run app.py'로 챗봇을 실행하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
