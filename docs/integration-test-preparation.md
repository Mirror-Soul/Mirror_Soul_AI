# 신규 회원 통합 테스트 준비 및 실행 체크리스트

최종 확인 기준: 2026-09-28 (Asia/Seoul)

이 문서는 신규 회원가입부터 RAG 프로필, 음성 복제, 얼굴 프로필, 클론 준비 완료,
영상 통화까지 검증하기 전에 필요한 준비 작업을 실행 순서대로 정리한 문서입니다.
백엔드 내부 로그 분석은 백엔드 담당자가 진행하고, 이 문서에서는 AI 파트가 직접
실행하거나 정상 여부를 확인해야 하는 항목을 중심으로 다룹니다.

## 현재 준비 상태

2026-09-28 확인 결과입니다.

| 항목 | 현재 상태 | 테스트 전 조치 |
| --- | --- | --- |
| AWS AI API | `active` | 최신 코드 배포 여부 확인 |
| AWS 음성 워커 | `active` | 음성 코드 변경 시 재시작 |
| GPU 연결·메모리 | 접속 정상, 약 24GB 중 1MB 사용 | 예약 시간과 SSH 포트 재확인 |
| GPU Ditto 서비스 | tmux 세션 없음 | 8절에 따라 실행 |
| GPU 얼굴 워커 | tmux 세션 없음 | 9절에 따라 실행 |
| 얼굴 요청·결과 SQS URL | GPU 환경에 설정됨 | 실제 값을 출력하지 말고 유지 |
| `ffmpeg`·`ffprobe` | GPU 환경에 절대경로 설정됨 | 기존 설정 유지 |
| 얼굴 유사도 설정 | GPU 환경에서 확인되지 않음 | 7.3절 설정 추가 후 워커 실행 |
| AWS 통화 서버 | `active` | 유지 |
| AWS Ditto 터널 | `active` | Ditto 실행 후 재시작·헬스체크 |
| 통화 서버 → Ditto 헬스체크 | 현재 실패 | GPU Ditto가 꺼져 있으므로 정상적인 현재 결과 |
| 저장소 배포 버전 | 로컬·AWS·GPU가 서로 다름 | 최신 `main`으로 통일 |
| 구조화 AI 로그·모니터 변경 | 로컬 미커밋 상태 | 커밋·병합·배포 후 사용 |

따라서 현재 상태 그대로는 통합 테스트를 시작하면 안 됩니다. 최소한 최신 코드 배포,
GPU Ditto 실행, 얼굴 워커 실행, 얼굴 유사도 환경변수 적용, 통화 터널 헬스체크까지
완료해야 합니다.

## 1. 전체 구성과 정상 기준

| 구성요소 | 실행 위치 | 정상 기준 |
| --- | --- | --- |
| 백엔드 API·결과 소비자 | AWS API 서버 | 회원가입 요청 발행, 얼굴 결과 소비, 클론 상태 갱신 |
| AI API·RAG | AWS AI 서버 | `mirrorsoul-ai.service`가 `active` |
| 음성 학습 워커 | AWS AI 서버 | `mirrorsoul-voice-worker.service`가 `active` |
| Ditto 렌더 서비스 | 학교 GPU 컨테이너 | `ditto-service` tmux 세션과 `/ready` 정상 |
| 얼굴 학습 워커 | 학교 GPU 컨테이너 | `face-worker` tmux 세션이 실행 중 |
| 통화 서버 | AWS Call 서버 | `mirror-soul-call.service`가 `active` |
| Ditto SSH 터널 | AWS Call 서버 | `mirror-soul-ditto-tunnel.service`가 `active` |
| AI 파이프라인 모니터 | 로컬 Windows | 모든 AI 연결과 워커 상태가 `OK` 또는 `COMPLETED` |

통합 테스트를 시작하기 전에 위 항목이 모두 준비되어야 합니다. 얼굴 워커만 실행하고
Ditto 서비스를 실행하지 않으면 얼굴 점수용 미리보기와 영상 통화를 정상적으로 검증할
수 없습니다.

## 2. 테스트 전 팀 확인 사항

### 2.1 백엔드 담당자에게 확인

다음 항목은 AI 파트에서 백엔드 로그를 직접 확인하지 않고 담당자에게 정상 여부만
확인합니다.

- AWS API 인스턴스와 백엔드 서비스가 실행 중인지
- `FACE_RESULT_CONSUMER_ENABLED=true`인지
- `RAG_PROFILE_ENABLED=true`인지
- 백엔드가 인터뷰 저장 후 `POST /api/v1/training/profiles`를 호출하는지
- 위 요청에 올바른 `cloneId`와 회원 UUID가 포함되는지
- 얼굴 결과 큐 소비 IAM 권한이 적용되어 있는지
- 클론 유사도 구성요소 컬럼과 `clone-similarity-v1` 계산 로직이 배포되어 있는지
- 기존 무본문 RAG 완료 콜백과 점수 JSON이 포함된 콜백을 모두 받을 수 있는지
- 음성 `ACTIVE` + 얼굴 `READY` + 성격 학습 완료 시 클론이 `READY`가 되는지

백엔드 확인이 끝나지 않은 상태에서도 AI 작업 자체는 관찰할 수 있지만, 최종 클론
`READY`와 앱 표시 상태까지는 검증할 수 없습니다.

### 2.2 프론트엔드 담당자에게 확인

- 테스트용 신규 계정을 사용할 것
- 얼굴 영상, 음성 샘플, 인터뷰 답변을 모두 제출할 것
- 가입 시작 시각과 완료 시각을 알려줄 것
- 가입 완료 후 회원 UUID와 `cloneId`를 전달할 것
- 클론이 준비되면 다른 회원 계정으로 실제 통화를 시도할 것

### 2.3 외부 서비스 확인

- ElevenLabs의 커스텀 음성 슬롯이 최소 1개 이상 남아 있어야 합니다.
- 음성 슬롯이 `10 / 10`이면 신규 음성 학습은 실패하므로 사용하지 않는 음성을 먼저
  삭제합니다.
- 테스트 중 사용할 AWS 키, API 키, 콜백 비밀값은 채팅이나 문서에 붙여 넣지 않습니다.

## 3. 최신 코드 배포 상태 맞추기

통합 테스트에서는 로컬, AWS AI 서버, AWS Call 서버, GPU 서버가 서로 다른 커밋을
사용하면 원인 분석이 어려워집니다. 먼저 변경 사항을 커밋·푸시·병합하고 GitHub Actions
배포가 완료되었는지 확인합니다.

현재 작업 트리에 미커밋 변경이 있으면 무조건 `pull`하지 않습니다. 먼저 다음 명령으로
상태를 확인합니다.

```bat
cd /d E:\Mirror_Soul_AI
git status --short
git branch --show-current
git log -1 --oneline
```

배포할 변경이 `main`에 병합된 뒤 각 서버에서 아래 값을 비교합니다.

```bash
git fetch origin
git rev-parse --short HEAD
git rev-parse --short origin/main
```

작업 트리가 깨끗하고 서버가 과거 커밋일 때만 다음을 실행합니다.

```bash
git switch main
git pull --ff-only origin main
```

주의 사항:

- GPU 저장소는 `/shareHost/C084003-ai/Mirror_Soul_AI`입니다.
- AWS AI·Call 저장소는 `/home/ec2-user/Mirror_Soul_AI`입니다.
- GitHub Actions의 AI 배포는 AI API를 재시작하지만 음성 워커 변경까지 항상 재시작하는
  것은 아니므로, 음성 관련 코드가 바뀌었다면 음성 워커도 직접 재시작합니다.
- 현재 추가한 구조화 RAG·얼굴 로그와 AI 모니터 변경은 커밋·병합·서버 배포 후 신규
  작업부터 완전히 반영됩니다.

## 4. AWS 서버 시작

AWS 콘솔 리전은 `아시아 태평양(서울) ap-northeast-2`입니다. 다음 인스턴스가
`실행 중`이고 상태 검사가 `3/3 통과`인지 확인합니다.

| 서버 | 인스턴스 이름 | 주소 |
| --- | --- | --- |
| 백엔드 | `mirrorsoul-api-server` | `43.200.195.143` |
| AI | `mirrorsoul-ai-server` | `13.209.220.154` |
| 통화 | `mirrorsoul-call-server` | `43.202.181.134` |

중지되어 있으면 AWS 콘솔에서 인스턴스를 시작하고 상태 검사가 끝날 때까지 기다립니다.
백엔드 서버의 내부 동작은 백엔드 담당자가 확인합니다.

## 5. AWS AI 서버 준비

### 5.1 접속

Windows CMD에서 실행합니다.

```bat
ssh -i "E:\Mirror_Soul_AI\mirrorsoul-ai-key.pem" ec2-user@13.209.220.154
```

### 5.2 코드와 환경변수 확인

```bash
cd /home/ec2-user/Mirror_Soul_AI
git status --short
git log -1 --oneline
```

비밀값을 출력하지 않고 필수 키가 설정되어 있는지만 확인합니다.

```bash
for key in OPENAI_API_KEY ELEVENLABS_API_KEY AWS_SQS_VOICE_TRAINING_QUEUE_URL CLONE_TRAINING_CALLBACK_BASE_URL CLONE_TRAINING_CALLBACK_SECRET; do
  if grep -q "^${key}=.\+" .env; then echo "${key}=SET"; else echo "${key}=MISSING"; fi
done
```

하나라도 `MISSING`이면 테스트를 시작하지 않습니다.

### 5.3 AI API와 음성 워커 실행

상태를 확인합니다.

```bash
systemctl is-active mirrorsoul-ai.service
systemctl is-active mirrorsoul-voice-worker.service
```

둘 다 `active`가 정상입니다. 꺼져 있으면 다음을 실행합니다.

```bash
sudo systemctl start mirrorsoul-ai.service
sudo systemctl start mirrorsoul-voice-worker.service
```

코드 또는 `.env`를 변경했다면 시작 대신 재시작합니다.

```bash
sudo systemctl restart mirrorsoul-ai.service
sudo systemctl restart mirrorsoul-voice-worker.service
```

AI API 응답을 확인합니다.

```bash
curl -fsS -o /dev/null http://127.0.0.1:8000/openapi.json && echo "AI API OK"
```

문제가 있을 때만 최근 AI 로그를 확인합니다.

```bash
journalctl -u mirrorsoul-ai.service -n 100 --no-pager
journalctl -u mirrorsoul-voice-worker.service -n 100 --no-pager
```

확인이 끝나면 `exit`로 Windows CMD로 돌아옵니다.

## 6. 학교 GPU 서버 예약과 컨테이너 시작

### 6.1 GPU 서버 예약

학교 GPU 예약 페이지에서 Mirror Soul 팀의 RTX 4090 서버 사용 시간을 예약합니다.
예약 화면에서 다음 값을 확인합니다.

- GPU 호스트: `203.249.75.55`
- 호스트 SSH 포트: 현재 기본 `20405`
- 컨테이너 SSH 포트: 현재 기본 `40053`
- 팀 컨테이너: `C084003`

예약이 새로 생성되면서 포트가 변경되었다면 이 문서의 `20405`, `40053` 대신 실제
포트를 사용합니다.

### 6.2 Windows CMD에서 컨테이너 시작

```bat
ssh -p 20405 C084003@203.249.75.55
```

GPU 호스트에서 컨테이너를 시작하고 빠져나옵니다.

```bash
sudo docker start C084003
sudo docker ps
exit
```

`docker ps`에서 `C084003`의 상태가 `Up`이면 정상입니다. 이 단계에서는
`docker exec`로 개발 환경에 들어가지 않습니다.

### 6.3 VS Code Remote-SSH 접속

VS Code에서 `Ctrl+Shift+P`를 누른 뒤 다음을 선택합니다.

```text
Remote-SSH: Connect to Host...
mirrorsoul-gpu-container
```

다음 폴더를 엽니다.

```text
/shareHost/C084003-ditto/ditto-talkinghead
```

VS Code 터미널에서 환경을 활성화합니다.

```bash
conda activate /shareHost/C084003-ditto/conda-env
cd /shareHost/C084003-ditto/ditto-talkinghead
```

정상 프롬프트는 다음과 같습니다.

```text
(/shareHost/C084003-ditto/conda-env) mirrorsoul@TeamC084003:/shareHost/C084003-ditto/ditto-talkinghead$
```

## 7. GPU 코드·환경·자원 사전 점검

### 7.1 AI 워커 저장소 확인

VS Code의 GPU 터미널에서 실행합니다.

```bash
cd /shareHost/C084003-ai/Mirror_Soul_AI
git status --short
git log -1 --oneline
```

최신 `main`이 아니고 작업 트리가 깨끗한 경우에만 코드를 갱신합니다.

```bash
git switch main
git pull --ff-only origin main
```

의존성 파일이 변경된 경우에만 다음을 실행합니다.

```bash
/shareHost/C084003-ditto/conda-env/bin/python -m pip install -r requirements-face.txt
/shareHost/C084003-ditto/conda-env/bin/python -m pip install -r requirements-face-similarity.txt
/shareHost/C084003-ditto/conda-env/bin/python -m pip install -r requirements-ditto-service.txt
```

### 7.2 GPU 메모리 확인

```bash
nvidia-smi
```

다른 작업이 GPU 메모리 대부분을 사용하거나 사용률이 계속 `100%`라면 렌더 서비스를
시작하지 않습니다. 현재 환경은 RTX 4090, 총 메모리 약 24GB가 정상 기준입니다.

### 7.3 얼굴 워커 환경 확인

환경 파일은 다음 위치에 있으며 비밀값을 화면에 그대로 출력하지 않습니다.

```text
/shareHost/C084003-ai/.env.face-worker
/shareHost/C084003-ai/.env.ditto-service
```

필수 얼굴 워커 키의 설정 여부만 확인합니다.

```bash
cd /shareHost/C084003-ai
for key in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SQS_FACE_TRAINING_QUEUE_URL AWS_SQS_FACE_TRAINING_RESULT_QUEUE_URL AWS_S3_BUCKET FFMPEG_BINARY FFPROBE_BINARY; do
  if grep -q "^${key}=.\+" .env.face-worker; then echo "${key}=SET"; else echo "${key}=MISSING"; fi
done
```

얼굴 종합점수까지 검증하려면 다음 설정도 필요합니다.

```env
FACE_SIMILARITY_ENABLE=true
FACE_SIMILARITY_REQUIRED=false
FACE_SIMILARITY_RENDER_ENGINE=ditto
FACE_SIMILARITY_DITTO_AUDIO_PATH=/shareHost/C084003-ditto/ditto-talkinghead/example/audio.wav
```

`FACE_SIMILARITY_REQUIRED=false`는 점수 평가 실패 때문에 얼굴 프로필 전체가 실패하는
것을 막습니다. 현재 얼굴 점수는 상용 제한이 있는 InsightFace 모델이 아니라 OpenCV
기반 렌더 품질 평가를 사용합니다.

현재 서버에서는 `ffmpeg`와 `ffprobe`를 시스템 PATH로 찾지 않고 다음 절대경로를
사용합니다. 이미 `.env.face-worker`에 들어 있으므로 임의로 삭제하지 않습니다.

```env
FFMPEG_BINARY=/shareHost/C084003-ditto/conda-env/bin/ffmpeg
FFPROBE_BINARY=/shareHost/C084003-ditto/conda-env/bin/ffprobe
```

환경 파일 권한도 확인합니다.

```bash
chmod 600 /shareHost/C084003-ai/.env.face-worker
chmod 600 /shareHost/C084003-ai/.env.ditto-service
```

## 8. GPU Ditto 렌더 서비스 실행

기존 세션을 먼저 확인합니다.

```bash
tmux ls
```

`ditto-service` 세션이 없을 때만 다음 명령으로 시작합니다.

```bash
mkdir -p /shareHost/C084003-ai/logs
tmux new-session -d -s ditto-service "bash /shareHost/C084003-ai/run-ditto-service.sh >> /shareHost/C084003-ai/logs/ditto-service.log 2>&1"
```

모델 로드에 RTX 4090 기준 약 25~30초가 걸립니다. 로그를 확인합니다.

```bash
tail -f /shareHost/C084003-ai/logs/ditto-service.log
```

`Ctrl+C`로 로그 보기만 종료한 뒤 헬스체크를 실행합니다. GPU 컨테이너에는 `curl`이
없으므로 `wget`을 사용합니다.

```bash
wget -qO- http://127.0.0.1:8080/health
wget -qO- http://127.0.0.1:8080/ready
```

`/ready`가 성공하기 전에는 얼굴 워커의 점수 미리보기나 영상 통화를 시작하지 않습니다.

## 9. GPU 얼굴 워커 실행

`face-worker` 세션이 이미 있으면 중복 실행하지 않습니다.

```bash
tmux ls
```

세션이 없을 때 다음을 실행합니다.

```bash
tmux new-session -d -s face-worker "bash /shareHost/C084003-ai/run-face-worker.sh >> /shareHost/C084003-ai/logs/face-worker.log 2>&1"
```

정상 실행 여부를 확인합니다.

```bash
tmux ls
tail -n 50 /shareHost/C084003-ai/logs/face-worker.log
```

로그에 다음 문구가 보이면 정상입니다.

```text
[FACE_TRAINING] worker started: mode=production
```

`no message available`은 오류가 아니라 현재 대기 중인 얼굴 작업이 없다는 뜻입니다.

## 10. AWS 통화 서버와 GPU 터널 준비

Ditto `/ready` 확인이 끝난 다음 Windows CMD에서 통화 서버에 접속합니다.

```bat
ssh -i "E:\Mirror_Soul_AI\mirrorsoul-call-key.pem" ec2-user@43.202.181.134
```

서비스 상태를 확인합니다. 정확한 서비스 이름은 `mirror-soul-*`입니다.

```bash
systemctl is-active mirror-soul-call.service
systemctl is-active mirror-soul-ditto-tunnel.service
```

꺼져 있으면 시작하고, GPU 컨테이너를 새로 시작했다면 터널을 재시작합니다.

```bash
sudo systemctl start mirror-soul-call.service
sudo systemctl restart mirror-soul-ditto-tunnel.service
```

통화 서버와 GPU Ditto 연결을 확인합니다.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/ready
```

`18080` 헬스체크가 실패하면 다음 순서로 확인합니다.

1. GPU의 `ditto-service` tmux 세션이 실행 중인지 확인
2. GPU 로컬 `127.0.0.1:8080/ready` 확인
3. 컨테이너 SSH 포트가 현재도 `40053`인지 확인
4. `mirror-soul-ditto-tunnel.service`의 SSH 포트 설정 확인 및 재시작

확인이 끝나면 `exit`로 나옵니다.

## 11. 로컬 AI 파이프라인 모니터 실행

프론트엔드에서 회원가입을 시작하기 전에 별도의 Windows CMD를 열고 실행합니다.

```bat
cd /d E:\Mirror_Soul_AI
tools\monitor-ai-pipeline.cmd
```

GPU 컨테이너 SSH 포트가 바뀌었다면 다음처럼 지정합니다.

```bat
tools\monitor-ai-pipeline.cmd --gpu-port 새포트
```

회원 UUID를 받은 뒤 한 회원만 고정해서 보려면 다음을 사용합니다.

```bat
tools\monitor-ai-pipeline.cmd --user-uuid 회원UUID --gpu-port 새포트
```

테스트 시작 전 화면의 정상 기준:

```text
AI server   : OK
AI API      : OK
Voice worker: OK
GPU server  : OK
Face worker : OK
```

`Face worker : INACTIVE`이면 회원가입을 시작하지 말고 9절의 얼굴 워커를 실행합니다.

## 12. 신규 회원가입 통합 테스트 실행

준비가 끝나면 프론트엔드 담당자에게 신규 회원가입을 시작해 달라고 요청합니다.
재사용 계정보다 새로운 이메일과 신규 회원을 사용하는 것이 작업 ID와 과거 실패 상태를
구분하기 쉽습니다.

다음 정보를 기록합니다.

| 항목 | 기록값 |
| --- | --- |
| 가입 시작 시각 | |
| 회원 UUID | |
| `cloneId` | |
| 음성 `jobId` | |
| 얼굴 `jobId` | |
| 테스트 계정 | |

AI 모니터에서 예상되는 순서는 다음과 같습니다.

1. `RAG [PROCESSING]` → `RAG [COMPLETED]`
2. `VOICE [PROCESSING]` → `VOICE [COMPLETED]`
3. `FACE [PROCESSING]` → `FACE [COMPLETED]`
4. 얼굴·음성·프로필·데이터 신뢰도 구성 점수 표시
5. 구성요소가 모두 모이면 예상 종합점수 표시

처리 순서는 비동기이므로 RAG, 음성, 얼굴의 실제 시작 순서는 달라질 수 있습니다.
한 단계가 먼저 끝났다고 다른 단계가 실패한 것은 아닙니다.

백엔드 담당자에게 다음 최종 상태를 확인받습니다.

- 음성 프로필 `ACTIVE`
- 얼굴 프로필 `READY`
- 성격·개인정보 학습 완료
- 전체 클론 `READY`
- 얼굴·음성·프로필·데이터 신뢰도·감점 점수 저장
- `sync_rate`가 소수점 첫째 자리의 `clone-similarity-v1` 점수로 갱신

## 13. 영상 통화 검증

클론이 `READY`가 된 뒤 다른 회원 계정으로 테스트 클론에게 전화를 겁니다.

확인 항목:

- 통화 연결과 양방향 음성
- 복제 음성 사용 여부
- 원격 비디오 트랙 표시 여부
- S3에 저장된 회원 얼굴 프로필 사용 여부
- 발화와 입 모양의 시간 일치
- 눈 깜빡임과 머리 움직임의 자연스러움
- 긴 답변에서도 영상 생성이 중단되지 않는지
- 한 요청 처리 중 추가 요청이 들어올 때 재시도 또는 대기하는지

통화 중 GPU에서 다음 명령으로 자원을 확인할 수 있습니다.

```bash
watch -n 1 nvidia-smi
```

Ditto는 한 번에 한 렌더 요청을 처리하며 이미 사용 중이면 HTTP `429`가 발생할 수
있습니다. 이 경우 GPU 장애로 단정하지 말고 호출 측 재시도 로그를 함께 확인합니다.

## 14. 주요 실패와 확인 위치

| 증상 | 먼저 확인할 항목 |
| --- | --- |
| 모니터에서 AI API 오류 | `mirrorsoul-ai.service`, AWS AI 인스턴스 |
| 음성 작업이 시작되지 않음 | `mirrorsoul-voice-worker.service`, 음성 SQS URL |
| `voice_limit_reached` | ElevenLabs 커스텀 음성 슬롯 정리 |
| 얼굴 작업이 시작되지 않음 | `face-worker` tmux, 요청 SQS URL과 GPU IAM |
| `ffprobe executable not found` | `.env.face-worker`의 `FFPROBE_BINARY` 절대경로 |
| 얼굴 점수가 `-` | `FACE_SIMILARITY_ENABLE=true`, 최신 GPU 코드, Ditto `/ready` |
| GPU 메모리 부족 | 다른 GPU 작업 종료 후 Ditto·얼굴 워커 재시작 |
| RAG가 `WARNING` | 완료 콜백 환경변수와 백엔드 콜백 응답 확인 |
| 얼굴 완료 후 클론이 READY가 아님 | 백엔드 결과 소비자와 세 가지 준비 조건 확인 |
| 통화 서버의 `18080` 실패 | Ditto 서비스, GPU SSH 포트, 터널 서비스 확인 |
| 앱에 영상이 표시되지 않음 | 프론트 원격 비디오 트랙 처리와 통화 서버 상태 확인 |

## 15. 테스트 종료와 서버 정리

테스트 증거로 회원 UUID, 작업 ID, 실패 메시지, 결과 영상 경로를 먼저 기록합니다.

GPU에서 세션을 종료해야 할 때만 실행합니다.

```bash
tmux kill-session -t face-worker
tmux kill-session -t ditto-service
```

VS Code 연결을 종료한 뒤 Windows CMD에서 GPU 호스트에 다시 접속합니다.

```bat
ssh -p 20405 C084003@203.249.75.55
```

예약 종료 시 팀 운영 규칙에 따라 컨테이너를 중지합니다.

```bash
sudo docker stop C084003
exit
```

AWS 서버는 다른 팀원이 사용할 수 있으므로 별도 합의 없이 중지하지 않습니다.

## 16. 통합 테스트 직전 최종 체크표

- [ ] 최신 AI 코드가 `main`에 병합됨
- [ ] AWS AI·Call 서버와 GPU 저장소의 배포 버전 확인
- [ ] AWS API·AI·Call 인스턴스 실행 및 상태 검사 통과
- [ ] 백엔드 결과 소비자와 RAG 플래그 활성화 확인
- [ ] 백엔드 점수 DB 마이그레이션 및 집계 코드 배포 확인
- [ ] ElevenLabs 커스텀 음성 슬롯 확보
- [ ] GPU 예약 완료 및 `C084003` 컨테이너 실행
- [ ] GPU 메모리 여유 확인
- [ ] GPU 얼굴 워커 환경변수와 IAM 자격증명 설정 확인
- [ ] 얼굴 유사도 설정 활성화 확인
- [ ] `ditto-service` tmux 실행 및 `/ready` 성공
- [ ] `face-worker` tmux 실행 및 시작 로그 확인
- [ ] AWS AI API와 음성 워커 `active`
- [ ] AWS Call 서버와 Ditto 터널 `active`
- [ ] Call 서버에서 `127.0.0.1:18080/ready` 성공
- [ ] 로컬 AI 파이프라인 모니터에서 모든 연결 `OK`
- [ ] 프론트엔드 신규 테스트 계정과 가입 시간 공유
- [ ] 백엔드 담당자가 최종 `READY` 상태 확인 대기

위 체크표가 모두 완료되면 신규 회원가입 통합 테스트를 시작할 수 있습니다.
