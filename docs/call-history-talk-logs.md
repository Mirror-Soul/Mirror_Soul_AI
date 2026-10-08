# 통화 기록 저장 (talk logs)

클론과 통화하면 사용자가 한 말과 클론의 답변이 백엔드 통화 기록에 저장됩니다. 앱은 이 기록을 `GET /history/calls/{callId}/talk-logs`로 불러와 일반 전화의 통화 기록처럼 보여 줍니다.

## 1. 흐름

```text
사용자 발화 → VAD → STT → LLM → TTS → (영상) → 답변 재생 시작
                                                    │
                                                    └─ 턴 완료 시 USER, CLONE 두 건을 기록 대기열에 넣음
                                                          │
                                       백그라운드 작업이 순서대로 POST
                                       /internal/ai/calls/{callId}/talk-logs
```

- 대화를 막지 않습니다. 기록 전송은 별도 작업이 처리하고, 백엔드가 느리거나 실패해도 통화는 그대로 이어집니다.
- 한 턴의 사용자 발화와 클론 답변은 항상 사용자 → 클론 순서로 보냅니다.
- 통화가 끝나면 남은 기록을 최대 `BACKEND_TALK_LOG_FLUSH_TIMEOUT_SECONDS`(기본 10초) 동안 마저 보냅니다.

## 2. 백엔드 계약

백엔드 PR [#188](https://github.com/Mirror-Soul/mirror-soul-backend/pull/188)의 API를 사용합니다. 인증은 통화 컨텍스트 API와 같은 `X-Internal-Api-Key`(`AI_INTERNAL_API_KEY`)입니다.

```http
POST /internal/ai/calls/{callId}/talk-logs
X-Internal-Api-Key: <AI_INTERNAL_API_KEY>
Content-Type: application/json

{
  "eventId": "8d3f1c52-5a8e-4f6e-9c1d-0d6a2f3c7b11",
  "speaker": "USER",
  "message": "오늘 뭐 했어?",
  "startedAt": "2026-10-08T03:00:00.000Z",
  "endedAt": "2026-10-08T03:00:01.800Z"
}
```

| 필드 | 값 |
| --- | --- |
| `eventId` | 발화마다 새 UUID. 재전송해도 같은 값이라 백엔드가 중복 저장하지 않음(`duplicated=true`) |
| `speaker` | `USER`(전화를 건 사람) 또는 `CLONE`(클론). 앱에 보이는 `ME`/`PARTNER`/`MY_TWIN`/`PARTNER_TWIN`은 백엔드가 보는 사람 기준으로 바꿔 줌 |
| `message` | 앞뒤 공백 제거, 최대 2000자 |
| `startedAt`, `endedAt` | UTC ISO-8601. 백엔드가 한국 시간으로 저장 |

시간 계산:

| 화자 | `startedAt` | `endedAt` |
| --- | --- | --- |
| USER | 발화가 끝난 시각 − 녹음 길이 | VAD가 발화 끝을 확정한 시각 |
| CLONE | 답변 음성이 재생되기 시작하는 시각(앞에 재생 중인 음성이 있으면 그만큼 뒤) | 시작 + 답변 음성 길이 |

## 3. 저장하는 턴과 저장하지 않는 턴

| 상황 | 기록 |
| --- | --- |
| 답변까지 재생된 턴 | 사용자, 클론 모두 저장 |
| STT 결과가 비어 있음 / 잡음으로 버려진 발화 | 저장 안 함 |
| 오래 기다려 건너뛴 발화(`UTTERANCE_STALE`) | 저장 안 함 |
| LLM·TTS·영상 실패로 답변이 나가지 않은 턴 | 저장 안 함 (대화 맥락에도 넣지 않는 것과 같은 기준) |
| 통화 종료로 처리 중 취소된 턴 | 저장 안 함 |

## 4. 재시도와 실패

| 응답 | 처리 |
| --- | --- |
| 200, `duplicated=false` | 저장 성공 |
| 200, `duplicated=true` | 이미 저장됨. 성공으로 처리 |
| 네트워크 오류, 타임아웃, 5xx | 같은 `eventId`로 최대 `BACKEND_TALK_LOG_MAX_ATTEMPTS`번 시도 |
| 404 `CALL_4040`, 409 `TALK_LOG_4090`, 400, 401, 503 `INTERNAL_5030` | 재시도 없이 실패 로그만 남김 |

`TALK_LOG_4090`은 통화 상태가 `IN_PROGRESS`/`COMPLETED`가 아닐 때(예: `FAILED`, `CANCELLED`) 나옵니다.

## 5. 설정

통화 서버 `.env`(이미 통화 컨텍스트 API에 쓰는 `BACKEND_API_BASE_URL`, `AI_INTERNAL_API_KEY`를 그대로 사용):

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `BACKEND_TALK_LOG_ENABLED` | `true` | `false`면 기록을 보내지 않음 |
| `BACKEND_TALK_LOG_TIMEOUT_SECONDS` | `5` | 요청 하나의 제한 시간 |
| `BACKEND_TALK_LOG_MAX_ATTEMPTS` | `3` | 일시 오류 때 최대 시도 횟수 |
| `BACKEND_TALK_LOG_RETRY_BACKOFF_SECONDS` | `0.5` | 재시도 간격(시도마다 배수로 증가) |
| `BACKEND_TALK_LOG_FLUSH_TIMEOUT_SECONDS` | `10` | 통화 종료 후 남은 기록을 보내는 최대 시간 |

새로 넣어야 하는 비밀 값은 없습니다. `BACKEND_API_BASE_URL`이나 `AI_INTERNAL_API_KEY`가 비어 있으면 통화는 정상 진행하고 `[TALK_LOG] saving disabled` 로그만 한 번 남깁니다.

## 6. 배포

1. 백엔드: PR #188(`V40__add_talk_log_event_id.sql` 포함)이 운영 API 서버에 배포되어 있어야 합니다.
2. 통화 서버: `main`에 머지하면 자동 배포됩니다. 새 `.env` 값은 기본값으로 동작하므로 따로 넣지 않아도 됩니다.

## 7. 로그로 확인

통화 서버에 접속한 뒤(`docs/aws-server-access.md`의 Call server ssh 명령):

```bash
sudo journalctl -u mirror-soul-call.service -f | grep TALK_LOG
```

```text
[TALK_LOG] saved: callId=120 turn=1 speaker=USER talkLogId=501 duplicated=no chars=9
[TALK_LOG] saved: callId=120 turn=1 speaker=CLONE talkLogId=502 duplicated=no chars=23
[TALK_LOG] call summary: callId=120 saved=8 duplicated=0 failed=0 dropped=0
```

대화 내용 자체는 `[TALK_LOG]` 줄에 다시 찍지 않습니다(글자 수만 표시). 통화 모니터(`tools\monitor-call-events.cmd`)에서는 `[기록]` 태그로 "통화 기록 저장: 사용자 발화 9자" 처럼 보입니다.

| 로그 | 의미 | 조치 |
| --- | --- | --- |
| `save failed ... error_code=INTERNAL_4010` | 내부 API 키 불일치 | 통화 서버와 API 서버의 `AI_INTERNAL_API_KEY` 비교 |
| `save failed ... error_code=TALK_LOG_4090` | 통화가 이미 실패/취소 상태 | 백엔드 통화 상태 전이 확인 |
| `save failed ... error_code=BACKEND_UNAVAILABLE` | API 서버 연결 실패 | API 서버 상태, 보안 그룹 확인 |
| `call summary ... dropped=N` | 종료 후 제한 시간 안에 못 보냄 | `BACKEND_TALK_LOG_FLUSH_TIMEOUT_SECONDS` 늘리기, API 서버 응답 속도 확인 |
