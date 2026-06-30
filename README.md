# Mirror Soul AI

Mirror Soul 프로젝트의 AI 응답 생성, 회원별 개인화, 음성 합성 및 실시간 통화를 담당하는 FastAPI 서버입니다.

## 주요 기능

### 개인화 AI 응답

- OpenAI GPT 기반 한국어 답변 생성
- MBTI 16개 유형별 Base Profile 적용
- 회원별 MBTI, 기본 정보, 성격 및 말투 반영
- 회원별 RAG 기억 검색 결과를 답변에 우선 반영
- RAG 정보가 부족한 경우 MBTI Base Profile로 보완

### RAG 학습 데이터

- 회원 기본 정보와 인터뷰 데이터를 핵심 키워드로 가공
- 나이, 성별, MBTI, 관심사 및 주요 인터뷰 주제 저장
- ChromaDB 기반 회원별 데이터 저장 및 검색
- 사용자 ID를 기준으로 다른 회원의 데이터와 분리

### 음성 처리

- Whisper 기반 음성 인식(STT)
- ElevenLabs 기반 음성 복제 및 음성 합성(TTS)
- 회원별 음성 ID 또는 기본 음성을 이용한 답변 생성
- Big Five 성격 수치를 반영한 TTS 설정 조정

### 실시간 AI 음성통화

- 백엔드 WebSocket 시그널링 서버에 `ai-server`로 접속
- `CALL_INVITE`, `CALL_ACCEPT`, `OFFER`, `ANSWER`, `ICE`, `CALL_END` 처리
- aiortc 기반 WebRTC 연결 및 프론트 마이크 오디오 수신
- 침묵 감지를 이용한 사용자 발화 구간 분리
- 다음 실시간 응답 파이프라인 연결

```text
프론트 마이크
→ WebRTC 오디오 수신
→ Whisper STT
→ MBTI + RAG + GPT 답변 생성
→ ElevenLabs TTS
→ WebRTC 오디오 송출
→ 프론트 스피커
```

실시간 파이프라인은 코드 구현 및 로컬 WebRTC 검증을 완료했으며, 실제 앱·통화 서버·외부 API를 함께 사용하는 통합 테스트가 필요합니다.

## 폴더 구조

```text
model_calling/
  routers/             채팅, 음성 생성 및 회원 등록 API
  signaling/           백엔드 WebSocket 시그널링 처리
  webrtc/              WebRTC 연결과 통화 세션 관리
  realtime/            발화 감지, STT, LLM, TTS 실시간 파이프라인
  repository/          RDS 회원 및 Clone 정보 조회
  services.py          OpenAI, Whisper, ElevenLabs 처리

model_training/
  base_profiles/       MBTI 16개 Base Profile JSON
  routers/             RAG 저장 및 검색 API
  services.py          ChromaDB 저장, 임베딩 및 검색

shared/
  공통 환경설정

docs/
  프론트·백엔드 연동 API 계약

data/
  로컬 회원 Persona 데이터, Git 제외

rag_store/
  로컬 ChromaDB 데이터, Git 제외
```

## 실행 방법

### 1. 가상환경 활성화 및 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 참고하여 `.env`에 필요한 값을 설정합니다.

```env
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=

DB_HOST=
DB_PORT=3306
DB_NAME=
DB_USERNAME=
DB_PASSWORD=

WEBRTC_STUN_URL=stun:stun.l.google.com:19302
WEBRTC_TURN_URL=
WEBRTC_TURN_USERNAME=
WEBRTC_TURN_CREDENTIAL=
```

API Key와 DB 비밀번호가 포함된 `.env` 파일은 GitHub에 커밋하지 않습니다.

### 3. 서버 실행

```bash
python main.py
```

실행 후 Swagger 문서는 다음 주소에서 확인할 수 있습니다.

```text
http://localhost:8000/docs
```

## 실시간 통화 로그

WebRTC 연결이 성공하면 다음 로그가 출력됩니다.

```text
[SIGNALING] CALL_ACCEPT sent: callId=...
[SIGNALING] ANSWER sent: callId=...
[WEBRTC] ICE connection: connected
[WEBRTC] connection: connected
[WEBRTC] track received: kind=audio
```

AI 응답 파이프라인이 정상적으로 처리되면 다음 로그가 추가됩니다.

```text
[REALTIME] STT user=...: 사용자 발화
[REALTIME] LLM user=...: AI 답변
[REALTIME] reply audio queued
```

## 현재 확인이 필요한 항목

- 실제 React Native 앱과 통화 서버를 이용한 통합 테스트
- 서버 환경의 TURN 연결 및 외부 네트워크 통화 검증
- 회원별 ElevenLabs Voice ID 저장·조회 계약
- 통화 지연시간, STT 정확도 및 음성 자연스러움 개선
