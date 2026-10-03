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

온보딩 RAG의 대표 저장 경로입니다. 백엔드가 회원 기본 정보와 인터뷰 답변을 전달하면 AI 서버는 이를 회원 RAG 문서(`profile_snapshot` 1개 + `interview_memory` 여러 개)로 가공해 upsert합니다. 문서 구조와 규칙은 [rag-memory-structure.md](rag-memory-structure.md)를 참고하세요.

Request (`*` 표시는 이번 RAG 단일화에서 추가된 선택 필드이며, 보내지 않아도 기존처럼 동작합니다):

```jsonc
{
  "userId": "user_123",
  "cloneId": 123,
  "aiProfileId": "profile_user_123",
  "name": "홍길동",            // * 선택
  "nickname": "길동",          // * 선택
  "age": 24,
  "gender": "female",
  "mbti": "INFP",
  "job": "학생",               // * 선택
  "description": "자기소개 원문",
  "interests": ["음악", "여행"],
  "values": ["정직", "가족"],   // * 선택
  "interviewTopics": ["가족", "진로"],
  "interviewSamples": [
    {
      "sourceId": "interview-123",  // * 선택, 답변의 안정적인 원본 ID
      "questionId": 1,
      "questionCategory": "가치관",
      "questionText": "가장 중요하게 생각하는 가치는 무엇인가요?",
      "transcript": "인터뷰 답변 STT 텍스트"
    }
  ],
  "keywordLimit": 12
}
```

인터뷰 문서 ID는 `{userId}:interview_memory:{sourceId}`입니다. `sourceId`가 없으면 `question-{questionId}`, 둘 다 없으면 질문 문장(없으면 답변) 해시로 생성하므로 인터뷰 순서가 바뀌어도 ID가 유지됩니다.

RAG 프로필 저장이 완료되면 AI 서버는 다음 내부 콜백을 호출한다.

```text
POST {CLONE_TRAINING_CALLBACK_BASE_URL}/internal/clone-training/{cloneId}/personality/complete
X-Clone-Training-Callback-Secret: {CLONE_TRAINING_CALLBACK_SECRET}
```

콜백은 RAG 문서 upsert와 이전 문서 정리가 모두 성공한 뒤에만 호출됩니다. 저장이나 콜백이
실패하면 프로필 학습 요청도 실패로 응답하며, 모든 문서는 결정적 ID로 upsert되므로 재시도해도
중복 문서가 생기지 않는다.

`profile_snapshot`에는 요청에 실제로 들어온 값(이름·닉네임, 나이, 성별, MBTI, 직업, 자기소개(최대 500자),
관심사, 가치관, 핵심 키워드)만 기록하고 비어 있는 항목은 추측하거나 `미입력`으로 채우지 않습니다.
인터뷰 질문·답변은 `interview_memory` 문서로 각각 저장합니다.

재학습 시 같은 `userId + profileKey(aiProfileId, 없으면 default)` 범위에서 이 경로가 이전에 저장했던
인터뷰 중 현재 요청에 없는 문서와, 같은 범위의 구버전(`member_profile_summary`, `member_profile_interview`)
문서만 삭제합니다. `/samples`로만 저장된 인터뷰, 다른 profileKey, 다른 회원, 대화·선호 기억 문서는 삭제하지 않습니다.

Response:

```json
{
  "success": true,
  "documentId": "user_123:profile_snapshot:profile_user_123",
  "status": "stored",
  "keywords": ["음악", "여행", "가족", "진로"],
  "profileSummary": "[회원 핵심 프로필]\\n..."
}
```

### POST /api/v1/training/samples

개별 인터뷰 답변 저장 경로입니다(호환 유지). 내부적으로 `/profiles`와 같은 문서 규격·ID 규칙·upsert 서비스를 사용하므로 같은 인터뷰를 `/samples`와 `/profiles`로 모두 보내도 문서는 하나만 남습니다.

Request (기존 필드 그대로, `sourceId`·`cloneId`는 선택):

```json
{
  "userId": "user_123",
  "aiProfileId": "profile_user_123",
  "cloneId": 123,
  "sourceId": "interview-123",
  "questionId": 1,
  "questionCategory": "가치관",
  "questionText": "가장 중요하게 생각하는 가치는 무엇인가요?",
  "transcript": "인터뷰 답변 STT 텍스트",
  "audioUrl": "s3://..."
}
```

Response:

```json
{
  "success": true,
  "documentId": "user_123:interview_memory:interview-123",
  "sampleId": "interview-123",
  "status": "stored"
}
```

`sampleId`는 이제 임의 값이 아니라 문서의 `sourceId`입니다. `mbti`, `description`은 호환을 위해 받지만 인터뷰 문서에는 넣지 않고 `profile_snapshot`에서만 관리합니다.

### POST /api/v1/training/search

회원 RAG 검색(디버그·운영 확인용). 응답 `memories`의 첫 항목은 해당 회원의 프로필 문서(있을 때 정확히 1개, `distance: null`)이고, 이어서 질문과 관련 있는 기억 문서가 최대 `topK`개 옵니다. 프로필은 `topK` 자리를 차지하지 않습니다.

### Chat personalization

`/api/v1/chat`, `/api/v1/chat-voice`, `/api/v1/call`은 답변 생성 전에 회원의 MBTI base profile과 RAG memory를 함께 참고합니다.

답변 생성 우선순위:

1. `userId`로 검색된 RAG memory
2. 회원 persona, Big5 성격, 말투 profile
3. MBTI base profile

`/api/v1/chat`과 `/api/v1/chat-voice` 요청에는 선택적으로 `mbti`를 포함할 수 있습니다. `mbti`가 없으면 `user_persona.mbti`, `user_persona.MBTI`, RAG metadata의 `mbti` 순서로 MBTI를 찾습니다.
