# Chatbot 발표자료 요약

## 목적/목표
- 제안서의 리스크를 확인하고 점수를 제공하는 RAG 기반 서비스
- MVP 수준의 제안서 챗봇 구현
- PDF 근거 미리보기로 답변 검증 가능하도록 설계

## 핵심 개념 (RAG)
- LLM이 답변 생성 전에 관련 문서를 검색해 컨텍스트를 주입
- 저장(인덱싱)과 질의(검색) 모두 임베딩 필요
- 데이터 변경 시 파인튜닝보다 비용 효율적 (DB만 갱신)
- 파이프라인: 사용자 질의 → Query Embedding → VectorDB 검색 → 관련 chunk → 프롬프트 주입 → LLM 응답

## 기술 요소
- Embedding: 비정형 데이터를 의미 기반 벡터로 변환, 유사도 검색 가능
- Chunk 크기와 문맥 유지가 품질에 영향
- 권장 임베딩 모델: bge-m3, E5, multilingual 계열
- LangChain: RAG 파이프라인 구성/오케스트레이션, 유지보수성 향상
- VectorDB: 고차원 벡터 저장 및 Top-k 유사도 검색, 메타데이터 필터링

## 문제 인식
- PDF 텍스트 추출 시 이미지 내부 글 비중이 큼
- 비정형/특성 나열형 텍스트가 많아 노이즈 제거 필요
- 그림/표에 핵심 정보가 많아 OCR 또는 요약이 필요
- 동일 목차 내 맥락 유지가 중요

## 처리 방안
- PDF 정제/Chunking:
  - 1안: 목차 기준 chunk
  - 2안: 키워드 기반 페이지 단위 chunk
- 그림/표 처리:
  - Deepseek OCR
  - Google Vision
  - Qwen3-vl 이미지/표 요약 생성
- 임베딩 포맷(맥락 유지):
  - [CHAPTER], [CONTENTS], [SUMMARY], [CAPTION], [TYPE], [KEYWORD], [PAGE], [PATH] 등 구조화 텍스트

## 시나리오/구성도
- PDF 분석 및 저장 → 질의 요청 → 검색/컨텍스트 주입 → 응답
- Embedding 파이프라인:
  - Tokenizer → Token Embedding → Transformer Encoder → Pooling → Sentence Embedding → Normalization → VectorDB

## 품질/성능 개선
- 토크나이저 제어
- 임베딩 데이터 검토
- 다중 파일 처리
- 하이브리드 임베딩 서빙 구성
- 프롬프트 고도화
- OpenAI 제거 후 로컬 LLM 사용
- 코드 리팩터링

## 참고 자료
- llama.cpp
- 로컬 임베딩
- 최적화 설정
