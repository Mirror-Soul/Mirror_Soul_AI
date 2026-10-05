# 영상 통화의 대기 동작과 전환

영상 통화에서 답변과 답변 사이를 자연스럽게 잇는 구조를 설명합니다. 대기 중에 얼굴이 움직이게 하고, 대기 영상과 답변 영상이 바뀔 때 얼굴이 튀지 않게 하는 것이 목표입니다.

## 1. 기존 방식의 문제

기존에는 아래 두 종류의 화면을 번갈아 보냈습니다.

| 상태 | 화면 |
| --- | --- |
| 대기 | 정지 초상화를 아주 조금씩 확대하고 이동(Ken Burns) |
| 답변 | Ditto가 만든 말하는 영상 |

이 둘이 바뀌는 순간 얼굴이 튀어 보였습니다. 원인은 세 가지입니다.

1. **확대·이동 때문에 위치가 어긋남.** 대기 화면은 최대 1.2% 확대되고 위아래로 움직입니다. Ditto 영상의 첫 프레임은 확대가 없는 원래 초상화라서, 전환하는 순간 얼굴 크기와 위치가 한 번에 바뀝니다.
2. **답변 끝 자세가 제각각임.** Ditto 답변은 원래 초상화 자세로 시작하지만, 끝날 때는 고개를 돌리거나 입을 벌린 상태일 수 있습니다. 바로 정지 초상화로 돌아가면 그 차이가 한 프레임 만에 사라집니다.
3. **대기 중에는 진짜 움직임이 없음.** 정지 사진을 확대·이동만 하므로 눈 깜빡임이나 미세한 고개 움직임이 없습니다. 그래서 답변이 시작될 때 "사진이 갑자기 살아나는" 느낌이 납니다.

## 2. 바뀐 구조

핵심 원칙은 **모든 영상이 같은 중립 프레임(원래 초상화 자세)에서 시작하고 같은 프레임으로 끝나게 하는 것**입니다. 그러면 어느 영상에서 어느 영상으로 넘어가도 이어지는 지점의 얼굴이 같습니다.

```text
통화 시작
  └─ 초상화 로드 → (즉시) 정지 초상화 송출
  └─ 무음 오디오로 Ditto 대기 클립 6초 렌더 → 준비되면 대기 루프로 교체

대기 루프:   [중립 → 눈 깜빡임·미세한 움직임 → 중립] 반복
답변 도착:   대기 루프 ──(0.24초 크로스페이드)──▶ 답변 영상 [중립 → 말하기 → 중립]
답변 끝:     답변 마지막 프레임(중립) ──(크로스페이드)──▶ 대기 루프 첫 프레임(중립)
```

### 2.1 답변 끝을 원래 자세로 되돌리기 (GPU)

Ditto에는 지정한 프레임 동안 움직임을 원래 초상화 자세로 섞어 주는 기능(`fade_type="s"`)이 있습니다. 렌더 요청에 아래 값을 함께 보냅니다.

| 필드 | 기본값 | 의미 |
| --- | --- | --- |
| `fade_in_frames` | 2 | 시작 2프레임을 초상화 자세에서 출발 |
| `fade_out_frames` | 8 | 마지막 8프레임(0.32초) 동안 초상화 자세로 복귀 |
| `fade_keys` | `exp,pitch,yaw,roll,t` | 표정, 고개 각도 3개, 위치를 모두 복귀 |

GPU 서비스가 이 필드를 받지 못하는 구버전이어도 오류는 나지 않습니다. 필드를 무시하고 예전처럼 렌더합니다.

### 2.2 Ditto 대기 루프 (통화 서버)

- 통화가 연결되면 16kHz 무음 오디오 6초로 Ditto를 한 번 렌더합니다. 아주 작은 잡음을 섞어 입은 다문 채로 두고, 눈 깜빡임과 미세한 고개 움직임만 나오게 합니다.
- 이 클립은 앞뒤 12프레임을 초상화 자세로 맞춰 렌더합니다. 그래서 끝 프레임과 첫 프레임이 같고, 반복해도 이음매가 보이지 않습니다.
- 같은 클론과 같은 초상화로 다시 통화하면 서버 메모리에 저장된 클립을 재사용합니다(최대 8개). 이때는 GPU를 쓰지 않습니다.
- 답변이 끝나면 대기 루프를 첫 프레임(중립)부터 다시 재생합니다.
- 렌더가 실패해도 통화에는 영향이 없습니다. 정지 초상화로 계속 진행합니다.

### 2.3 크로스페이드 (통화 서버)

대기 루프 중간에 답변이 시작되면 대기 자세와 답변 첫 프레임이 조금 다를 수 있습니다. 이 차이는 6프레임(0.24초) 동안 이전 화면과 새 화면을 섞어서 가립니다.

- 답변 영상의 프레임 순서와 시간은 바뀌지 않습니다. 따라서 음성과 입 모양의 싱크도 그대로입니다.
- 크로스페이드는 화면을 섞기만 하고, 영상을 늦추지 않습니다.

### 2.4 정지 초상화 확대·이동 기본값 끄기

`REALTIME_IDLE_MOTION_ENABLED`의 기본값을 `false`로 바꿨습니다. 대기 루프가 있으면 어차피 쓰이지 않고, 대기 루프가 없을 때(GPU 실패 등)도 튀는 원인이 되기 때문입니다.

## 3. 설정

통화 서버 `.env`:

| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `REALTIME_VIDEO_TRANSITION_FRAMES` | `6` | 전환 시 섞는 프레임 수. `0`이면 예전처럼 바로 전환 |
| `REALTIME_IDLE_MOTION_ENABLED` | `false` | 정지 초상화 확대·이동 |
| `DITTO_CALL_REPLY_FADE_IN_FRAMES` | `2` | 답변 시작 복귀 프레임 |
| `DITTO_CALL_REPLY_FADE_OUT_FRAMES` | `8` | 답변 끝 복귀 프레임 |
| `DITTO_CALL_FADE_KEYS` | `exp,pitch,yaw,roll,t` | 복귀시킬 움직임 종류 |
| `DITTO_CALL_IDLE_LOOP_ENABLED` | `true` | Ditto 대기 루프 사용 |
| `DITTO_CALL_IDLE_LOOP_SECONDS` | `6` | 대기 클립 길이(2~20초) |
| `DITTO_CALL_IDLE_LOOP_FADE_FRAMES` | `12` | 대기 클립 앞뒤 복귀 프레임 |

> 운영 통화 서버 `.env`에 예전 예시값 `REALTIME_IDLE_MOTION_ENABLED=true`가 들어 있다면 `false`로 바꾸거나 그 줄을 지웁니다.

GPU `.env.ditto-service`에는 새로 추가할 값이 없습니다.

## 4. 배포 순서

순서는 상관없습니다. 어느 쪽을 먼저 배포해도 통화는 깨지지 않습니다. 다만 두 서버를 모두 배포해야 완전히 적용됩니다.

1. **통화 서버**: `main`에 머지하면 자동 배포됩니다. 필요하면 위 3절의 `.env`를 확인하고 `sudo systemctl restart mirror-soul-call.service`를 실행합니다.
2. **GPU Ditto 서비스**: `docs/integration-test-preparation.md` 7.1절대로 코드를 `git pull` 합니다. 그다음 `ditto-service` tmux 세션을 종료하고 8절의 일괄 시작 스크립트로 다시 시작합니다. 얼굴 학습 작업이 진행 중이 아닐 때 합니다.
3. GPU의 Ditto 저장소가 fade 기능을 지원하는지 한 번 확인합니다.

   ```bash
   grep -n "fade_type" /shareHost/C084003-ditto/ditto-talkinghead/stream_pipeline_offline.py
   ```

   `self.fade_type = kwargs.get("fade_type", "")` 줄이 보이면 지원합니다.

| 상태 | 결과 |
| --- | --- |
| 통화 서버만 배포 | 크로스페이드와 대기 루프는 동작. 단, 답변 끝과 대기 루프 이음매가 중립으로 맞춰지지 않아 크로스페이드에만 의존 |
| GPU만 배포 | 예전과 동일(통화 서버가 fade 값을 보내지 않음) |
| 둘 다 배포 | 설계대로 동작 |

## 5. 확인할 로그

통화 서버(`journalctl -u mirror-soul-call.service -f`)에서 확인합니다.

```text
[DITTO_CALL] idle loop render started: callId=... seconds=6.0 fade_frames=12
[DITTO_CALL] render slot acquired: ... fade_in=12 fade_out=12
[DITTO_CALL] idle loop ready: callId=... source=render bytes=... elapsed_ms=...
[VIDEO_OUT] idle loop video ready: encoded_bytes=...
[DITTO_CALL] render slot acquired: callId=... turn=1 ... fade_in=2 fade_out=8
[VIDEO_OUT] crossfade started: from=idle-loop to=reply frames=6
[VIDEO_OUT] crossfade started: from=reply to=idle-loop frames=6
```

- 같은 클론으로 다시 통화하면 `source=cache`가 나옵니다.
- 대기 루프가 실패하면 `idle loop unavailable; keeping still portrait`가 나옵니다. 통화는 정지 초상화로 계속됩니다.
- 실시간 통화 모니터에는 `IDLE` 줄이 추가돼 `READY`(대기 루프 재생 중) 또는 `WARNING`(정지 초상화)으로 표시됩니다.

## 6. 테스트 때 볼 것

녹화본에서 아래 구간을 확인합니다.

1. 통화 연결 직후: 처음 몇 초 정지 초상화 → 대기 루프로 바뀔 때 튀지 않는지
2. 대기 중: 눈 깜빡임과 미세한 움직임이 자연스러운지, 입이 움직이지 않는지, 6초마다 반복되는 티가 나는지
3. 답변 시작: 대기 → 답변 전환이 부드러운지, 첫 음절과 입 모양이 맞는지
4. 답변 끝: 고개와 표정이 원래 자세로 돌아오는 속도가 어색하지 않은지
5. 연속 질문: 답변이 끝나고 다음 답변이 올 때까지 어색하지 않은지

조정 방법:

| 증상 | 조정 |
| --- | --- |
| 대기 중 입이 움직임 | `DITTO_CALL_IDLE_LOOP_SECONDS`를 줄이거나, GPU 쪽 무음 처리 검토 |
| 반복이 티 남 | `DITTO_CALL_IDLE_LOOP_SECONDS=10` |
| 답변 끝 복귀가 너무 급함 | `DITTO_CALL_REPLY_FADE_OUT_FRAMES=12` |
| 답변 마지막 말끝 입 모양이 약함 | `DITTO_CALL_REPLY_FADE_OUT_FRAMES=5` |
| 전환 순간이 흐릿함 | `REALTIME_VIDEO_TRANSITION_FRAMES=4` |
| 전환이 여전히 튐 | `REALTIME_VIDEO_TRANSITION_FRAMES=10` |

값을 바꾼 뒤에는 `sudo systemctl restart mirror-soul-call.service`를 실행합니다. 대기 클립 길이나 복귀 프레임을 바꾸면 캐시가 자동으로 새로 만들어집니다.

## 7. 한계와 다음 단계

- **통화 시작 직후 GPU 사용**: 대기 클립 렌더에 GPU 워커 1개가 몇 초 동안 쓰입니다. 워커가 2개라 첫 답변은 다른 워커에서 렌더됩니다. 다만 8초를 넘는 긴 답변은 워커 2개를 모두 쓰므로, 대기 클립 렌더가 끝날 때까지 기다릴 수 있습니다.
- **무음 입력 결과는 실측 필요**: Ditto가 무음에서 만드는 움직임의 양은 실제 렌더로 확인해야 합니다. 너무 정적이거나 입이 움직이면 6절 표대로 조정합니다.
- **답변 대기 시간 자체는 그대로**: 이 변경은 화면의 자연스러움을 개선하는 것이고, 답변을 받기까지 걸리는 시간(STT → LLM → TTS → Ditto)은 줄이지 않습니다. 대기 중에 듣는 모습이 자연스러워져 기다림이 덜 어색해지는 효과만 있습니다.
- **다음 후보**: Ditto `online_mode` 청크 렌더로 답변 영상을 생성되는 대로 송출하면 첫 반응 시간을 줄일 수 있습니다. 구조 변경이 커서 이번 테스트 결과를 보고 결정합니다.
