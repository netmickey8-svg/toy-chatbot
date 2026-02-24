"""
PDF 파일 텍스트 추출 모듈
==========================
역할:
    - 제안서 PDF에서 실제 텍스트(텍스트 레이어), 표(Table), 이미지(OCR) 세 방식으로
      텍스트를 추출하여 구조화된 PageContent/ProcessedDocument 객체로 반환
    - chunker.py → vectordb.py의 임베딩 파이프라인에서 첫 번째 단계

추출 방식 (우선순위):
    1. 텍스트 레이어 (PyMuPDF): 텍스트 기반 PDF에서 직접 추출 (빠르고 정확)
    2. 표 추출 (pdfplumber):    표 구조를 탭/줄 단위 텍스트로 변환
    3. OCR (pytesseract):       텍스트 레이어가 부족한 스캔 PDF에 적용

데이터 구조:
    PDF 파일
      └── ProcessedDocument
            ├── metadata: DocumentMetadata (파일명, 부문, 연도, 프로젝트명, 총 페이지)
            └── pages: list[PageContent]   (페이지별 추출 텍스트)
                  ├── page_number: int
                  └── content: str

폴더 구조 (data_dir):
    제안서/
    ├── R&D부문/  ← department = "R&D부문"
    │   └── *.pdf
    └── SI부문/   ← department = "SI부문"
        └── *.pdf
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz          # PyMuPDF: 텍스트 레이어 추출 + 이미지 렌더링(OCR용)
import pdfplumber    # 표 구조 추출에 특화
import pytesseract   # OCR 엔진 (Tesseract 래퍼)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# 데이터 클래스 정의
# ──────────────────────────────────────────────

@dataclass
class DocumentMetadata:
    """
    PDF 문서 메타데이터

    파일명 파싱으로 부문/연도/프로젝트명을 자동 추출
    (예: "[정성제안서]2025년 대구산업경제동향.pdf" → year="2025")
    """
    file_path:    str    # 절대 경로 (출처 추적용)
    file_name:    str    # 파일명만 (화면 표시용)
    department:   str    # "R&D부문" 또는 "SI부문" (폴더명 기반)
    year:         str    # 4자리 연도 ("2025" 등, 미포함 시 "Unknown")
    project_name: str    # 파일명에서 파싱한 프로젝트명
    total_pages:  int    # 전체 페이지 수 (처리 완료 후 설정)


@dataclass
class PageContent:
    """
    단일 페이지 추출 결과

    Attributes:
        page_number: 1-based 페이지 번호
        content:     페이지에서 추출된 전체 텍스트
                     (텍스트 레이어 + 표 + OCR 결과 결합)
    """
    page_number: int
    content:     str


@dataclass
class ProcessedDocument:
    """
    처리 완료된 PDF 문서

    Attributes:
        metadata: DocumentMetadata (파일 정보)
        pages:    PageContent 리스트 (페이지별 텍스트)
    """
    metadata: DocumentMetadata
    pages:    list[PageContent] = field(default_factory=list)


# ──────────────────────────────────────────────
# 메타데이터 파싱
# ──────────────────────────────────────────────

def extract_metadata_from_filename(file_path: Path, department: str) -> DocumentMetadata:
    """
    파일명에서 연도/프로젝트명 자동 파싱

    파싱 예시:
        "[정성제안서]2025년 대구산업경제동향.pdf"
            → year="2025", project_name="대구산업경제동향"
        "2024년 메타버스 특화 콘텐츠 제작.pdf"
            → year="2024", project_name="메타버스 특화 콘텐츠 제작"

    Args:
        file_path:  PDF 파일 경로
        department: 상위 폴더명 ("R&D부문" 또는 "SI부문")

    Returns:
        DocumentMetadata (total_pages=0, 처리 후 업데이트)
    """
    file_name = file_path.name

    # 연도 추출: 4자리 숫자 + "년"
    year_match = re.search(r"(\d{4})년", file_name)
    year = year_match.group(1) if year_match else "Unknown"

    # 프로젝트명 추출
    bracket_match = re.search(r"\(([^)]+)\)", file_name)
    if bracket_match:
        project_name = bracket_match.group(1)
    else:
        project_name = file_name
        project_name = re.sub(r"^\[.*?\]", "", project_name)    # [정성제안서] 제거
        project_name = re.sub(r"\d{4}년\s*", "", project_name)  # 연도 제거
        project_name = re.sub(r"\.pdf$", "", project_name, flags=re.IGNORECASE)

    return DocumentMetadata(
        file_path=str(file_path),
        file_name=file_name,
        department=department,
        year=year,
        project_name=project_name.strip(),
        total_pages=0,
    )


# ──────────────────────────────────────────────
# 텍스트 추출 (3가지 방식 결합)
# ──────────────────────────────────────────────

def _extract_tables_by_page(file_path: Path) -> dict[int, str]:
    """
    pdfplumber로 페이지별 표(Table) 추출

    방식:
        각 셀을 탭 구분(\t), 행을 줄바꿈(\n)으로 변환하여
        LLM이 읽기 쉬운 형식으로 반환

    Args:
        file_path: PDF 파일 경로

    Returns:
        {page_number(1-based): 표_텍스트} 딕셔너리
    """
    tables_by_page: dict[int, str] = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if not tables:
                    continue
                table_texts = []
                for table in tables:
                    rows = [
                        "\t".join(cell or "" for cell in row)
                        for row in table
                    ]
                    table_texts.append("\n".join(rows))
                tables_by_page[page_idx] = "\n\n".join(table_texts)
    except Exception as e:
        print(f"[WARN] 표 추출 실패 ({file_path.name}): {e}")
    return tables_by_page


def extract_pages_from_pdf(file_path: Path) -> tuple[list[PageContent], int]:
    """
    PDF의 모든 페이지에서 텍스트를 추출하여 PageContent 리스트로 반환

    각 페이지마다 3가지 방식을 결합:
        1. PyMuPDF get_text()   → 텍스트 레이어 (기본)
        2. pdfplumber tables   → 표 내용 (있는 경우)
        3. pytesseract OCR     → 스캔 페이지 처리
           (텍스트 레이어가 OCR_MIN_TEXT_CHARS 미만일 경우에만 수행)

    Args:
        file_path: PDF 파일 경로

    Returns:
        tuple: (PageContent 리스트, 총 페이지 수)
    """
    # OCR 설정 로드 (임포트를 함수 내부로 배치해 순환 참조 방지)
    from config import OCR_ENABLED, OCR_LANG, OCR_MIN_TEXT_CHARS, TESSERACT_CMD
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # Step 1: 표 사전 추출 (pdfplumber는 문서 전체를 한번에 처리하는 게 효율적)
    tables_by_page = _extract_tables_by_page(file_path)

    pages: list[PageContent] = []
    try:
        fitz_doc = fitz.open(file_path)
        total_pages = len(fitz_doc)

        for page_num, fitz_page in enumerate(fitz_doc, 1):
            # Step 2: 텍스트 레이어 추출
            page_text = fitz_page.get_text()

            # Step 3: 표 텍스트 (pdfplumber 결과 병합)
            table_text = tables_by_page.get(page_num, "")

            # Step 4: OCR (스캔 PDF 대응)
            ocr_text = ""
            if OCR_ENABLED and len(page_text.strip()) < OCR_MIN_TEXT_CHARS:
                try:
                    # 200 DPI로 렌더링 후 PIL Image로 변환
                    pix = fitz_page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang=OCR_LANG)
                except Exception as e:
                    print(f"[WARN] OCR 실패 ({file_path.name} p{page_num}): {e}")

            # 3가지 결과 결합 (섹션 구분자로 LLM 인식 용이)
            parts = []
            if page_text.strip():
                parts.append(page_text)
            if table_text.strip():
                parts.append("[TABLE]\n" + table_text)
            if ocr_text.strip():
                parts.append("[OCR]\n" + ocr_text)

            combined = "\n\n".join(parts).strip()
            if combined:
                pages.append(PageContent(page_number=page_num, content=combined))

        fitz_doc.close()
        return pages, total_pages

    except Exception as e:
        print(f"[ERROR] PDF 처리 오류 ({file_path.name}): {e}")
        return [], 0


# ──────────────────────────────────────────────
# 문서 처리 메인 함수
# ──────────────────────────────────────────────

def process_pdf(file_path: Path, department: str) -> ProcessedDocument | None:
    """
    단일 PDF 파일을 처리하여 ProcessedDocument 반환

    처리 흐름:
        파일명 파싱 → 텍스트/표/OCR 추출 → ProcessedDocument 조립

    Args:
        file_path:  PDF 파일 경로
        department: 부문명 ("R&D부문" 또는 "SI부문")

    Returns:
        ProcessedDocument 또는 추출 실패 시 None
    """
    metadata = extract_metadata_from_filename(file_path, department)
    pages, total_pages = extract_pages_from_pdf(file_path)

    if not pages:
        return None

    metadata.total_pages = total_pages
    return ProcessedDocument(metadata=metadata, pages=pages)


def process_all_pdfs(data_dir: Path) -> list[ProcessedDocument]:
    """
    제안서 폴더 하위의 모든 PDF를 처리 (index_data.py에서 호출)

    폴더 구조 가정:
        data_dir/
        ├── R&D부문/*.pdf
        └── SI부문/*.pdf

    Args:
        data_dir: 제안서 최상위 디렉토리 (config.py의 DATA_DIR)

    Returns:
        처리 완료된 ProcessedDocument 리스트
    """
    if not data_dir.exists():
        print(f"[ERROR] 데이터 디렉토리 없음: {data_dir}")
        print("        .env 파일에 DATA_DIR을 설정하거나 경로를 확인하세요.")
        return []

    documents: list[ProcessedDocument] = []

    for dept_folder in sorted(data_dir.iterdir()):
        if not dept_folder.is_dir():
            continue

        department = dept_folder.name
        pdf_files = list(dept_folder.rglob("*.pdf"))
        print(f"\n[DIR] {department} ({len(pdf_files)}개 PDF)")

        for pdf_file in sorted(pdf_files):
            print(f"  [FILE] {pdf_file.name[:55]}")
            doc = process_pdf(pdf_file, department)
            if doc:
                documents.append(doc)
                print(f"     [OK] {doc.metadata.total_pages}페이지, {len(doc.pages)}개 추출 완료")
            else:
                print(f"     [FAIL] 텍스트 추출 실패")

    print(f"\n[OK] 총 {len(documents)}개 문서 처리 완료")
    return documents


# ──────────────────────────────────────────────
# 단독 실행 테스트 (python src/pdf_processor.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import DATA_DIR

    docs = process_all_pdfs(DATA_DIR)
    for doc in docs[:2]:
        print(f"\n{'='*50}")
        print(f"파일: {doc.metadata.file_name}")
        print(f"부문: {doc.metadata.department} | 연도: {doc.metadata.year}")
        print(f"페이지 수: {doc.metadata.total_pages}")
        if doc.pages:
            print(f"1페이지 미리보기:\n{doc.pages[0].content[:200]}...")
