# 📄 제안서 챗봇

PDF 제안서를 기반으로 질문에 답변하는 **RAG(Retrieval-Augmented Generation) 챗봇**입니다.  
발표자료 권장 스택: **bge-m3 Dense 임베딩 + ChromaDB + Gemini LLM**

---

## 🏗️ RAG 파이프라인 구조

```
[인덱싱]  PDF → 텍스트/표/OCR 추출 → 청킹(800자) → bge-m3 임베딩 → ChromaDB 저장
[검색]    쿼리 → bge-m3 임베딩 → ChromaDB 유사도 검색 → 상위 5개 청크
[생성]    검색 결과 + 쿼리 → Gemini Prompt → 자연어 답변
```

## 🚀 기능

| 기능 | 설명 |
|------|------|
| PDF 파싱 | 텍스트 레이어(PyMuPDF) + 표(pdfplumber) + OCR(Tesseract) 병합 |
| 트랜스포머 임베딩 | BAAI/bge-m3 Dense 임베딩 (한국어+영어, 의미 기반 검색) |
| 벡터 검색 | ChromaDB 코사인 유사도 검색 |
| LLM 답변 | Google Gemini 기반 자연어 답변 |
| 필터링 | 부문별(R&D/SI), 연도별 검색 필터 |
| 채팅 UI | Streamlit 기반 채팅 인터페이스 |

## 📁 프로젝트 구조

```
chatbot/
├── app.py              # Streamlit 챗봇 UI (진입점)
├── index_data.py       # PDF 인덱싱 스크립트 (처음/PDF 추가 시 실행)
├── config.py           # 전역 설정 관리 (API키, 경로, 모델 설정)
├── requirements.txt    
├── .env                # API 키 (git 제외)
├── .env.example        # 환경변수 예시
└── src/
    ├── pdf_processor.py  # PDF → PageContent 추출 (텍스트+표+OCR)
    ├── chunker.py        # 페이지 텍스트 → 청크 분할 (RecursiveCharacterTextSplitter)
    ├── vectordb.py       # bge-m3 임베딩 + ChromaDB 저장/검색
    └── rag_chain.py      # RAG 파이프라인 (검색 → Gemini 답변 생성)
```

## ⚙️ 설치 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

> **⚠️ bge-m3 모델 다운로드**: 최초 인덱싱 시 BAAI/bge-m3 모델이 자동 다운로드됩니다 (~2GB).  
> 경량 대안: `.env`에 `EMBEDDING_MODEL=jhgan/ko-sroberta-multitask` 추가 (~400MB)

### 2. OCR 설치 (선택, 스캔 PDF 처리용)

```bash
# Windows: https://github.com/UB-Mannheim/tesseract/wiki 에서 설치
# 설치 후 .env에 경로 설정 (기본 경로와 다를 경우):
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 3. 환경변수 설정

`.env.example`을 복사하여 `.env` 파일 생성 후 API 키 설정:

```bash
copy .env.example .env
# .env 파일에서 GOOGLE_API_KEY 설정
```

### 4. PDF 인덱싱 (처음 실행 시 / PDF 추가 시)

```bash
python index_data.py
```

### 5. 챗봇 실행

```bash
streamlit run app.py
```

---

## 🔧 주요 설정 (`.env`)

| 설정 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `GOOGLE_API_KEY` | (필수) | Google AI Studio API 키 |
| `DATA_DIR` | `C:\Users\User\Downloads\제안서` | PDF 폴더 경로 |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini LLM 모델명 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 임베딩 모델명 |
| `OCR_ENABLED` | `true` | OCR 활성화 여부 |

## 🔬 기술 스택 (발표자료 기반)

| 구분 | 사용 기술 | 발표자료 분류 |
|------|-----------|-------------|
| 임베딩 | BAAI/bge-m3 | Transformer Encoder-Only (Dense) |
| 벡터DB | ChromaDB | 코사인 유사도 검색 |
| LLM | Google Gemini | Decoder-Only |
| PDF 파싱 | PyMuPDF + pdfplumber | - |
| OCR | Tesseract | - |
| UI | Streamlit | - |
