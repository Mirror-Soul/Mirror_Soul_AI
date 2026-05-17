# AI API Contract

Mirror Soul 백엔드 서버와 AI 서버 사이의 연동 규칙입니다.

## 1. 전체 연동 원칙

Mirror Soul에서 백엔드와 AI 서버의 역할은 다릅니다.

백엔드는 원본 데이터를 관리합니다.

- userId
- aiProfileId 또는 cloneId
- MBTI
- 자기소개
- 인터뷰 질문/답변 원문
- S3 음성 파일 URL
- 회원 삭제/권한/동의 상태

AI 서버는 백엔드가 전달한 원본 데이터를 기반으로 AI 응답 생성을 위한 파생 데이터를 만듭니다.

- RAG 저장용 텍스트
- OpenAI embedding
- ChromaDB vector
- 사용자별 memory 검색
- persona 기반 LLM 응답
- TTS 음성 파일 생성

즉, 백엔드가 데이터를 저장한 뒤 AI 서버에 학습 요청을 보내면, AI 서버는 해당 데이터를 RAG DB에 저장합니다.

---

## 2. Audio File Policy

Mirror Soul 프로젝트의 음성 파일은 운영체제 호환성을 위해 `.m4a`로 통일합니다.

### 백엔드에서 AI 서버로 넘기는 사용자 음성

- 확장자: `.m4a`
- 예: `https://storage.example.com/interviews/user_123/question_1.m4a`

### AI 서버가 생성해서 반환하는 음성

- 확장자: `.m4a`
- 예: `/assets/user_123/result_audio.m4a`

### 주의

AI 서버 내부에서는 외부 TTS API 응답을 일시적으로 다른 포맷으로 받은 뒤 `.m4a`로 변환할 수 있습니다.
하지만 백엔드/프론트와 주고받는 최종 음성 파일은 `.m4a`여야 합니다.

---

## 3. Environment

### AI Server

기본 로컬 주소:

```text
http://localhost:8000