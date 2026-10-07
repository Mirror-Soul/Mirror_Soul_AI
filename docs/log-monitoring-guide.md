# 통합 테스트 로그 확인 가이드

회원가입(RAG·음성·얼굴 학습)과 실시간 통화를 테스트할 때 사용하는 로그 모니터 사용법입니다.
모니터는 **내 Windows PC에서** 실행하며, SSH로 각 서버의 로그를 읽어 한 화면에 정리합니다.
서버에 직접 접속하지 않아도 됩니다.

## 1. 처음 한 번 준비

### 1.1 로컬 저장소를 최신 `main`으로

모니터 코드는 로컬 저장소(`E:\Mirror_Soul_AI\tools`)에서 실행되므로, 로컬도 최신 `main`이어야
최신 로그 형식을 읽습니다.

```bat
cd /d E:\Mirror_Soul_AI
git status --short
git branch --show-current
```

다른 브랜치에 미커밋 변경이 있으면 먼저 커밋하거나 보관(stash)한 뒤 전환합니다.

```bat
git switch main
git pull --ff-only origin main
```

### 1.2 필요한 파일

| 파일 | 위치 | 용도 |
| --- | --- | --- |
| `mirrorsoul-ai-key.pem` | `E:\Mirror_Soul_AI\` | AWS AI 서버 (RAG·음성 워커 로그) |
| `mirrorsoul-call-key.pem` | `E:\Mirror_Soul_AI\` | AWS 통화 서버 (통화 로그) |
| `mirrorsoul_gpu_vscode_ed25519` | `%USERPROFILE%\.ssh\` | GPU 컨테이너 (얼굴 워커 로그) |

키 파일은 Git에 올리지 않습니다. GPU 로그를 보려면 GPU 예약과 컨테이너 시작
(`sudo docker start C084003`)이 먼저 되어 있어야 합니다.

## 2. VS Code에서 모니터 실행

1. 로컬 VS Code 새 창에서 `E:\Mirror_Soul_AI` 폴더를 엽니다. GPU Remote-SSH 창이 아닙니다.
2. 터미널을 열고(`` Ctrl+` ``) 오른쪽 위 분할 버튼으로 두 칸으로 나눕니다.
3. 각 칸에서 아래 명령을 실행합니다.

VS Code 기본 터미널은 **PowerShell**이므로 앞에 `.\`를 붙입니다.

| 칸 | 용도 | PowerShell 명령 |
| --- | --- | --- |
| 왼쪽 | 회원가입·RAG·음성·얼굴 학습 | `.\tools\monitor-ai-pipeline.cmd --color always --gpu-port 40053` |
| 오른쪽 | 실시간 통화 | `.\tools\monitor-realtime-call.cmd --color always` |

CMD 터미널이라면 `cd /d E:\Mirror_Soul_AI` 후 `.\` 없이 `tools\monitor-ai-pipeline.cmd ...`로 실행합니다.

자주 쓰는 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--user-uuid <회원UUID>` | 왼쪽 모니터를 특정 회원에 고정 (생략하면 가장 최근 회원) |
| `--gpu-port <포트>` | GPU 예약 포트가 `40053`이 아닐 때 |
| `--refresh 3` | 갱신 주기(초), 기본 5초 |
| `--since-minutes 60` | 몇 분 전 로그부터 볼지, 기본 180분 |
| `--once` | 한 번만 출력하고 종료 (상태 스냅샷 공유용) |

종료는 `Ctrl+C`, `Terminate batch job (Y/N)?`가 나오면 `Y`.

### 요약 화면과 전체 로그 화면

위 두 모니터는 **현재 상태를 한 화면에 요약**하는 화면입니다. 5초마다 다시 그리지만, 내용이 바뀌었을 때만 새로 그리므로 읽는 동안 화면이 흔들리지 않습니다. 긴 질문과 답변도 잘리지 않고 다음 줄로 이어서 표시합니다. 다만 최근 이벤트만 보여 줍니다.

지난 대화까지 **처음부터 전부** 보려면 전체 로그 화면을 같이 켭니다. 이 화면은 다시 그리지 않고 아래로 계속 이어 붙이기 때문에, 위로 스크롤해서 지난 턴을 읽을 수 있습니다.

| 용도 | PowerShell 명령 |
| --- | --- |
| 통화 전체 로그 (모든 통화·질문·답변·소요 시간) | `.\tools\monitor-call-events.cmd --color always` |
| 학습 전체 로그 (모든 회원의 RAG·음성·얼굴) | `.\tools\monitor-ai-events.cmd --color always --gpu-port 40053` |

통화 전체 로그는 다음처럼 보입니다. 알려진 로그는 한국어 문장으로 바꿔 보여 주고, Ditto·WebRTC·ICE·RAG·STT·LLM·TTS 같은 기술 용어는 그대로 둡니다. 처음 보는 로그는 원문 그대로 나옵니다.

```text
== 통화 107 요청  17:51:33 ================================================
17:51:33 [SIGNAL] 통화 요청 받음
17:51:33 [SIGNAL] 통화 수락: 영상 통화, 클론 회원 524a1300… (clone 29)
17:51:33 [VIDEO ] Ditto 대기 영상 준비 완료 (저장된 영상 재사용, 0.0초)
17:51:34 [WEBRTC] WebRTC 연결: 연결됨

-- 통화 107 · 2번째 대화  17:52:30 ----------------------------------------
17:52:32 [사용자] 어 너 그 팀 프로젝트를 진행할 때 유독 비협조적인 팀원이 있으면 어떻게 해?
                  (음성 인식 1.8초)
17:52:32 [RAG   ] 기억 검색 2건: 인터뷰 1, 프로필 1 (0.2초, AI 서버 저장소, 가장 가까운 거리 0.61)
17:52:33 [클론  ] 그런 상황이라면 그 팀원을 프로젝트에서 제외하고 진행할 것 같아. ...
                  (답변 생성 1.2초)
17:52:46 [VIDEO ] Ditto 렌더 완료 (10.2초 걸림)
17:52:46 [턴    ] 대화 완료: 총 16.4초 | 음성 인식 1.8 | 기억 검색 0.2 | 답변 생성 1.2 | 음성 합성 1.4 | 영상 11.7
```

- 시작하면 최근 60분 기록을 먼저 보여 주고, `여기부터 실시간 로그` 줄 아래로 새 로그를 이어 붙입니다. `--history-minutes 180`으로 늘리거나 `--history-minutes 0`으로 실시간만 볼 수 있습니다.
- 시간은 한국 시간으로 표시하고, 날짜가 바뀌면 `~~ 2026-10-06 (이전 기록) ~~`처럼 날짜 줄이 들어갑니다.
- 시그널링 재연결처럼 같은 내용이 반복되면 한 번만 보여 주고 `(같은 내용 N줄 더 반복)`으로 줄입니다. SDP·ICE 원문처럼 긴 연결 정보는 요약해서 보여 줍니다.
- 화면에 나온 내용은 색 없이 `tmp\monitor-logs\call-events-날짜.log`, `tmp\monitor-logs\ai-events-날짜.log`에 저장됩니다. 테스트가 끝나면 이 파일을 그대로 공유하면 됩니다. 저장하지 않으려면 `--no-save`를 붙입니다.
- VS Code 터미널은 기본으로 1,000줄까지만 위로 스크롤됩니다. `Ctrl+,` → `terminal.integrated.scrollback`을 `10000`으로 바꾸면 긴 테스트도 처음부터 볼 수 있습니다.

## 3. 화면 읽는 법

모든 이벤트 줄 앞에 단계 태그가 붙고, 긴 UUID와 파일 경로는 줄여서 한 줄에 보이게 표시합니다.
(`9fbb1dd7…`는 회원 UUID 앞 8자리입니다.)

| 색 | 의미 |
| --- | --- |
| 초록 | 완료 (`COMPLETED`, `OK`, `READY`, `CONNECTED`) |
| 청록 | 진행 중 (`PROCESSING`, 시작) |
| 노랑 | 주의 (`WARNING`, 건너뜀, 재시도, 서비스 재시작 중) |
| 빨강 | 실패 (`FAILED`, `ERROR`, 거절) |
| 회색 | 대기 (`WAITING`) |

### 3.1 왼쪽: AI PIPELINE MONITOR (회원가입)

| 영역 | 내용 |
| --- | --- |
| `CONNECTIONS / PROCESSES` | AI 서버·AI API·음성 워커·GPU·얼굴 워커 상태. `WARNING (restarting ...)`이면 서비스가 시작 직후 계속 죽는 중입니다 (5절 참고). |
| `PIPELINE (this member)` | 회원 한 명의 RAG·VOICE·FACE 단계, 작업 ID, 점수. `└` 줄은 세부 정보입니다. |
| `RAG └ docs= removed=` | RAG에 저장된 문서 수와 재학습으로 정리된 이전 문서 수 |
| `VOICE └ voice_score=` | 음성 학습 완료 시 계산된 음성 점수 |
| `OVERALL` | AI 로그만으로 계산한 **예상** 종합점수. 공식 점수는 백엔드가 계산·저장합니다. |
| `RECENT AI EVENTS` | `[RAG ]`, `[VOICE ]`, `[FACE ]` 태그가 붙은 최근 이벤트 (위가 오래된 것) |

정상 흐름 예시 (처리 순서는 비동기라 섞여서 나올 수 있습니다):

```
[RAG   ] processing → completed (profile_score, docs)
[VOICE ] processing → batch status=PASSED → completed (voice_score) → status published status=COMPLETED
[FACE  ] preprocessing → video preprocessed → completed (face_score)
```

### 3.2 오른쪽: REALTIME CALL MONITOR (통화)

| 영역 | 내용 |
| --- | --- |
| `CONNECTIONS / SERVICES` | 통화 서버, 시그널링, Ditto 터널·GPU 상태 |
| `LATEST CALL` | 최근 통화의 `callId`, 클론 회원, 통화 종류(`VOICE`/`VIDEO`) |
| `CALL PIPELINE` | SIGNAL → WEBRTC → STT → RAG → LLM → TTS → VIDEO 단계 상태 |
| `LATEST TURN SUMMARY` | 마지막 대화 턴의 단계별 소요 시간(ms) |
| `RECENT CLOSED CALLS` | 종료된 통화 요약 |
| `RECENT CALL EVENTS` | `[SIGNAL]`, `[WEBRTC]`, `[STT ]`, `[RAG ]`, `[LLM ]`, `[TTS ]`, `[VIDEO ]`, `[TRACE ]` 태그가 붙은 이벤트 |

- 통화 초대가 오면 AI 서버가 백엔드 내부 API에서 통화 정보를 한 번 받아 옵니다. 성공하면
  `SIGNAL [COMPLETED] Call accepted (context loaded from backend API)`가 표시됩니다.
- 거절되면 `SIGNAL [FAILED] Call rejected: <사유>`가 표시됩니다.

| 거절 사유 | 의미 |
| --- | --- |
| `CALL_CONTEXT_NOT_FOUND` | 백엔드에 해당 통화가 없음 |
| `INVALID_CALL_STATUS` | 통화 상태가 READY/IN_PROGRESS가 아님 |
| `CLONE_NOT_READY` | 클론 또는 음성이 준비되지 않음 |
| `AI_SERVER_CONFIG_ERROR` | 내부 API 주소·키 설정 문제 (AI 서버 `.env` 또는 백엔드 키 불일치) |
| `CALL_CONTEXT_UNAVAILABLE` | 백엔드 응답 없음·타임아웃 (재시도 후에도 실패) |
| `INVALID_CALL_CONTEXT` | 백엔드 응답 형식 오류 |
| `CALL_CONTEXT_MISMATCH` | 초대 메시지와 통화 정보의 callId/roomId 불일치 |
| `INVALID_CALL_INVITE` | 초대 메시지에 올바른 callId가 없음 (프론트·백엔드 시그널링 확인) |

- `RAG [COMPLETED]` 줄의 `sources=profile_snapshot:1,interview_memory:2`는 통화 답변에 사용된
  회원 기억 종류와 개수입니다. 기존 회원은 `member_profile_summary`처럼 이전 형식으로 보일 수 있으며 정상입니다.
- 음성 통화(`VOICE`)는 `VIDEO [SKIPPED] Voice-only call`이 정상입니다.

## 4. 원본 로그 직접 보기 (모니터에서 빨간색일 때)

모니터는 요약만 보여 줍니다. 오류 원인(Traceback 등)은 원본 로그에서 확인합니다.

### AWS AI 서버 (RAG API, 음성 워커)

```bat
ssh -i "E:\Mirror_Soul_AI\mirrorsoul-ai-key.pem" ec2-user@13.209.220.154
```

```bash
journalctl -u mirrorsoul-ai.service -f -o cat              # RAG/학습 API 실시간
journalctl -u mirrorsoul-voice-worker.service -f -o cat    # 음성 워커 실시간
journalctl -u mirrorsoul-voice-worker.service -n 100 --no-pager   # 최근 100줄
journalctl -u mirrorsoul-ai.service -f -o cat | grep --line-buffered "user_uuid=회원UUID"
```

### AWS 통화 서버

```bat
ssh -i "E:\Mirror_Soul_AI\mirrorsoul-call-key.pem" ec2-user@43.202.181.134
```

```bash
journalctl -u mirror-soul-call.service -f -o cat                 # 통화 실시간
journalctl -u mirror-soul-call.service -f -o cat | grep --line-buffered "callId=91"
journalctl -u mirror-soul-ditto-tunnel.service -n 50 --no-pager  # Ditto 터널
```

### GPU 컨테이너 (얼굴 워커, Ditto)

```bat
ssh -i "%USERPROFILE%\.ssh\mirrorsoul_gpu_vscode_ed25519" -p 40053 mirrorsoul@203.249.75.55
```

```bash
tail -f /shareHost/C084003-ai/logs/face-worker.log
tail -f /shareHost/C084003-ai/logs/ditto-service-1.log
tail -f /shareHost/C084003-ai/logs/ditto-service-2.log
tmux ls          # face-worker, ditto-service-1, ditto-service-2 세션 확인
```

`tail -f`, `journalctl -f`는 `Ctrl+C`로 종료합니다.

## 5. 자주 보는 문제

| 화면 | 확인할 것 |
| --- | --- |
| `Voice worker: WARNING (restarting ...)` | 음성 워커가 시작 직후 종료 반복. `journalctl -u mirrorsoul-voice-worker.service -n 30`에서 마지막 오류 확인 (예: `AWS_SQS_VOICE_TRAINING_RESULT_QUEUE_URL is not configured`) |
| `AI server: ERROR (...)` / `GPU server: ERROR (...)` | SSH 접속 실패. 키 파일 위치, 인스턴스 실행 여부, GPU 예약·포트 확인 |
| `STALE (...)` | 이번 조회가 실패해서 직전 데이터를 보여 주는 중. 잠시 뒤 자동 회복되는지 확인 |
| `Face worker: INACTIVE` | GPU에서 `tmux ls`, 필요하면 `start-ai-services.sh` 실행 |
| `VOICE [FAILED] ... status publish failed` | 음성 결과 큐 전송 실패. SQS 큐 존재 여부와 AI 서버 IAM 권한 확인 |
| `RAG [WARNING] callback=False` | RAG 저장은 됐지만 백엔드 완료 콜백 설정이 없음 |
| `SIGNAL [FAILED] Call rejected: ...` | 3.2절 거절 사유 표 참고 |
| `Ditto GPU: ERROR` | GPU의 Ditto 세션과 통화 서버 터널 재시작 |
