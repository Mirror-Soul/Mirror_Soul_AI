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
```

---

## 4. Training API

### POST /api/v1/training/profiles

백엔드가 회원 기본 정보와 인터뷰 요약 재료를 전달하면, AI 서버는 원문 전체를 저장하지 않고 핵심 프로필 문서로 가공해 RAG DB에 저장합니다.

Request:

```json
{
  "userId": "user_123",
  "aiProfileId": "profile_user_123",
  "age": 24,
  "gender": "female",
  "mbti": "INFP",
  "description": "자기소개 원문",
  "interests": ["음악", "여행"],
  "interviewTopics": ["가족", "진로"],
  "interviewSamples": [
    {
      "questionId": 1,
      "questionCategory": "가치관",
      "questionText": "가장 중요하게 생각하는 가치는 무엇인가요?",
      "transcript": "인터뷰 답변 STT 텍스트"
    }
  ],
  "keywordLimit": 12
}
```

AI 서버가 실제 RAG 문서에 저장하는 내용은 나이, 성별, MBTI, 핵심 키워드 목록입니다. `description`, `questionText`, `transcript`는 키워드 추출 재료로만 사용하고 원문 전체를 그대로 저장하지 않습니다.

Response:

```json
{
  "success": true,
  "documentId": "member_profile_user_123_profile_user_123",
  "status": "stored",
  "keywords": ["음악", "여행", "가족", "진로"],
  "profileSummary": "[회원 핵심 프로필]\\n..."
}
```

### Chat personalization

`/api/v1/chat`, `/api/v1/chat-voice`, `/api/v1/call`은 답변 생성 전에 회원의 MBTI base profile과 RAG memory를 함께 참고합니다.

답변 생성 우선순위:

1. `userId`로 검색된 RAG memory
2. 회원 persona, Big5 성격, 말투 profile
3. MBTI base profile

`/api/v1/chat`과 `/api/v1/chat-voice` 요청에는 선택적으로 `mbti`를 포함할 수 있습니다. `mbti`가 없으면 `user_persona.mbti`, `user_persona.MBTI`, RAG metadata의 `mbti` 순서로 MBTI를 찾습니다.
