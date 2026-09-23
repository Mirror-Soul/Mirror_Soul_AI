
# Model Training

회원별 AI 학습 파이프라인을 담당

## 예정 기능

- 학습 세션 생성 API
- 학습 샘플 저장 API
- OpenAI Embedding 생성
- Vector DB 저장
- Persona profile 생성
- 학습 상태 조회 API

## 얼굴 프로필 워커

백엔드가 온보딩 얼굴 영상을 S3에 저장한 뒤 SQS에 발행하는
`FACE_PROFILE_BUILD` 작업을 소비한다.

운영 모드는 얼굴 영상을 전처리해 회원별 Ditto 얼굴 프로필을 S3에 저장하고,
처리 상태를 결과 SQS로 발행한다. 회원가입 버튼을 누른 시점이 아니라 얼굴 영상의
S3 업로드가 완료된 시점에 백엔드가 작업을 발행해야 한다.

상시 워커:

```bash
python -m model_training.face_training.worker
```

한 건만 운영 처리:

```bash
python -m model_training.face_training.worker --once
```

S3 업로드와 요청 삭제 없이 한 건을 검증:

```bash
python -m model_training.face_training.worker --once --dry-run
```

처리 범위:

- SQS 메시지 계약 검증
- S3 얼굴 영상 다운로드
- FFprobe 영상 메타데이터 검사
- FFmpeg 프레임 추출
- 선명도, 밝기, 대비, 얼굴 크기 및 중앙 정렬 품질 검사
- 정면 및 좌우 프로필 대표 프레임 자동 선택
- 작업별 전처리 manifest 저장
- 선택된 대표 얼굴과 manifest를 회원별 S3 경로에 업로드
- Ditto 엔진 버전과 검증된 렌더 설정을 `face-profile.json`에 저장
- 결과 SQS에 `PROCESSING`, `COMPLETED`, `FAILED` 상태 발행
- `COMPLETED` 발행이 성공한 뒤에만 요청 SQS 메시지 삭제

얼굴 프로필은 고정 문장을 말하는 완성 영상이 아니다. 회원의 대표 얼굴, 품질 정보,
Ditto 렌더 설정을 묶은 통화용 프로필이며 실제 통화 음성이 들어올 때 Ditto가 얼굴
움직임을 생성한다. 가입 시 프리뷰 영상 생성은 선택 기능으로 유지한다.

### Ditto 독립 렌더 검증

백엔드, SQS 또는 S3 없이 얼굴 이미지와 WAV 파일로 Ditto 실행 경계를 검증할 수 있다.
기본값은 v7 비교에서 선택한 `crop_scale=2.3`, `smo_k_d=5`,
`sampling_timesteps=50`, seed `1024`다.

```bash
python -m model_training.face_training.ditto_preview \
  --source /shareHost/C084003-ditto/test-inputs/namseonghyeon-front.jpg \
  --audio /shareHost/C084003-ditto/test-inputs/namseonghyeon-preview.wav \
  --output /shareHost/C084003-ditto/test-results/runner-smoke-test.mp4
```

S3에서 내려받은 `face-profile.json`의 회원별 설정을 사용하려면 `--profile`을
추가한다. 명령줄의 `--crop-scale`, `--smo-k-d`, `--sampling-timesteps`는 프로필
값보다 우선한다. 영상 소스 움직임은 `--smo-k-s`로 평활화하고, 정지 이미지의
눈 깜빡임 간격은 `--blink-open-frames`로 제어할 수 있다. 영상 소스에도 생성된
눈 깜빡임을 적용하려면 `--drive-eye`를 함께 사용한다. `--blink-strength`는 눈을
감는 깊이를 `0`보다 크고 `1` 이하의 배율로 조절한다. 생략하면 Ditto 기본 동작과
같은 `smo_k_s=13`, 무작위 눈 깜빡임 간격, 강도 `1.0`을 사용한다.

```bash
python -m model_training.face_training.ditto_preview \
  --source /path/to/portrait.jpg \
  --audio /path/to/member.wav \
  --profile /path/to/face-profile.json \
  --output /path/to/member-preview.mp4
```

실행 결과와 오류는 출력 영상 옆의 `*.ditto.log`에 저장된다. 이 실행 모듈은 향후
GPU 렌더링 API와 WebRTC 비디오 트랙이 공통으로 사용한다.

GPU 서버에 LivePortrait와 가중치가 준비되어 있으면 검증 모드에서 결과 영상까지
자동 생성할 수 있다. `.env`에서 다음 설정을 활성화한다.

```env
FACE_TRAINING_RUN_LIVEPORTRAIT=true
FACE_TRAINING_LIVEPORTRAIT_REPO_DIR=/workspace/mirror-soul-face/liveportrait
FACE_TRAINING_LIVEPORTRAIT_PYTHON=/opt/conda/envs/mirrorsoul-face/bin/python
```

활성화하면 품질 게이트를 통과한 영상 중 가장 점수가 높은 정면 프레임을 source로,
해당 원본 영상을 driving video로 사용한다. 생성 영상과 실행 로그 경로는
`preprocess-manifest.json`의 `livePortrait` 항목에 기록된다.

SQS 작업이나 S3 결과 업로드 없이 기존 S3 영상으로 얼굴 품질을 반복 검증하려면
preview 명령을 사용한다.

```bash
python -m model_training.face_training.preview \
  --object-key face-videos/00000000-0000-0000-0000-000000000000/face-scan.mov
```

이 명령은 object key에서 사용자 UUID를 추론하고 로컬 작업 디렉터리에 결과를
생성한다. SQS 메시지를 받거나 삭제하지 않으며 결과도 S3에 업로드하지 않는다.
여러 영상을 함께 비교하려면 같은 회원의 `--object-key`를 반복해서 지정한다.

### 회원 음성 얼굴 프리뷰

얼굴과 음성을 같은 회원 기준으로 검증하려면 `--member-voice-preview`를 사용한다.
이 모드는 SQS 메시지의 `userUuid`와 `cloneId`에 해당하는 활성 음성 프로필을
RDS에서 조회하고, 그 회원의 ElevenLabs 복제 음성으로 고정 문장을 생성한 뒤
선택된 얼굴 프레임을 MuseTalk로 애니메이션한다.

```bash
python -m model_training.face_training.preview \
  --object-key face-videos/00000000-0000-0000-0000-000000000000/face-scan.mov \
  --clone-id 1 \
  --member-voice-preview
```

얼굴의 `cloneId`와 활성 음성 프로필의 `clone_id`가 다르거나 활성 Voice ID가
없으면 작업은 즉시 실패한다. 공통 `ELEVENLABS_VOICE_ID`로 다른 회원 목소리를
대체하지 않는다. `ELEVENLABS_API_KEY`와 DB 연결 정보는 필요하지만 Voice ID는
환경변수에 저장하지 않는다.

로컬 GPU 서버에서는 다음 MuseTalk 설정을 사용한다.

```env
FACE_TRAINING_MEMBER_VOICE_PREVIEW_TEXT=안녕하세요. 처음 뵙겠습니다.
FACE_TRAINING_MEMBER_VOICE_PREVIEW_ENABLE=false
FACE_TRAINING_MUSETALK_REPO_DIR=/shareHost/C084003-musetalk/MuseTalk
FACE_TRAINING_MUSETALK_PYTHON=/shareHost/C084003-musetalk/conda-env/bin/python
FACE_TRAINING_MUSETALK_TIMEOUT_SECONDS=900
FACE_TRAINING_MUSETALK_BBOX_SHIFT=0
```

생성된 음성과 MuseTalk 영상 경로는 manifest의 `memberVoicePreview`에 기록한다.
Voice ID 원문은 manifest나 로그에 기록하지 않는다.

기본적으로 정지 프레임 대신 원본 얼굴 영상에서 음성 길이에 맞는 정면 구간을
자동 선택한다. 이 방식은 회원의 실제 눈 깜빡임, 미세한 고개 움직임과 배경 움직임을
유지하고 MuseTalk는 입 모양만 변경한다. 정면 구간의 품질과 비율은 manifest의
`memberVoicePreview.naturalMotion`에 기록된다.

이미 생성한 전처리 manifest와 회원 음성이 있으면 DB나 S3를 다시 조회하지 않고
다음 명령으로 자연스러운 움직임 프리뷰만 만들 수 있다.

```bash
python -m model_training.face_training.natural_motion_preview \
  --manifest /workspace/Mirror_Soul_AI/tmp/face_training/<user>/job-<id>/<run>/preprocess-manifest.json \
  --audio /shareHost/C084003-musetalk/test-inputs/member-preview.mp3 \
  --output-dir /shareHost/C084003-musetalk/test-results/member-natural-motion
```

긴 통화에서 사용할 듣기 상태의 움직임을 검증하려면 짧은 정면 구간을 정방향과
역방향으로 부드럽게 순환한 무음 idle 영상을 만든다. 큰 몸짓을 생성하지 않고 실제
영상의 눈 깜빡임과 미세한 움직임만 유지한다.

```bash
python -m model_training.face_training.idle_motion_preview \
  --manifest /workspace/Mirror_Soul_AI/tmp/face_training/<user>/job-<id>/<run>/preprocess-manifest.json \
  --output /shareHost/C084003-musetalk/test-results/member-idle-30s.mp4 \
  --duration 30 \
  --segment-duration 2.5
```

백엔드 메시지 계약:

```json
{
  "schemaVersion": 1,
  "jobType": "FACE_PROFILE_BUILD",
  "jobId": 1,
  "source": "ONBOARDING_FACE",
  "userUuid": "00000000-0000-0000-0000-000000000000",
  "cloneId": 1,
  "bucket": "bucket-name",
  "objectKeys": ["face-videos/user/video.mp4"]
}
```

성공 시 S3에 다음 객체가 생성된다.

```text
face-results/<userUuid>/job-<jobId>/portrait.jpg
face-results/<userUuid>/job-<jobId>/preprocess-manifest.json
face-results/<userUuid>/job-<jobId>/face-profile.json
face-results/<userUuid>/job-<jobId>/preview.mp4  # 프리뷰가 있을 때만
```

결과 SQS 메시지 계약:

```json
{
  "schemaVersion": 1,
  "eventType": "FACE_PROFILE_BUILD_STATUS",
  "jobId": 1,
  "userUuid": "00000000-0000-0000-0000-000000000000",
  "cloneId": 1,
  "attemptNumber": 1,
  "occurredAt": "2026-09-09T00:00:00+00:00",
  "status": "COMPLETED",
  "result": {
    "profileStatus": "READY_FOR_RENDERING",
    "artifacts": {
      "bucket": "bucket-name",
      "profileKey": "face-results/<userUuid>/job-1/face-profile.json",
      "portraitKey": "face-results/<userUuid>/job-1/portrait.jpg",
      "manifestKey": "face-results/<userUuid>/job-1/preprocess-manifest.json"
    },
    "qualityGatePassed": true
  }
}
```

실패 메시지는 `error.code`, `error.message`, `error.retryable=true`를 포함한다.
기본값에서는 실패한 요청을 삭제하지 않으므로 재시도 후 DLQ로 이동하도록 요청
큐에 redrive policy를 설정해야 한다. 백엔드는 동일 `jobId` 이벤트를 멱등하게
처리하고, `retryable=true` 실패를 최종 실패로 확정하지 않아야 한다.

운영에 필요한 환경 변수:

```env
AWS_SQS_FACE_TRAINING_QUEUE_URL=
AWS_SQS_FACE_TRAINING_RESULT_QUEUE_URL=
FACE_TRAINING_RESULT_PREFIX=face-results
FACE_TRAINING_VISIBILITY_TIMEOUT=900
FACE_TRAINING_VISIBILITY_HEARTBEAT_SECONDS=60
FACE_TRAINING_DELETE_FAILED_MESSAGES=false
FACE_TRAINING_ENGINE=ditto
FACE_TRAINING_ENGINE_VERSION=v0.4-hubert-pytorch
FACE_TRAINING_DITTO_CROP_SCALE=2.3
FACE_TRAINING_DITTO_SMO_K_D=5
FACE_TRAINING_DITTO_SAMPLING_TIMESTEPS=50
FACE_TRAINING_BEST_EFFORT_ENABLE=true
FACE_TRAINING_BEST_EFFORT_MIN_SHARPNESS=30
```

엄격한 품질 게이트가 실패하더라도 정면의 단일 얼굴이 검출되고 다른 거부 사유 없이
선명도만 최선 생성 기준을 넘으면 `LOW/BEST_EFFORT` 프로필을 생성한다. 얼굴 미검출,
다중 얼굴, 비정상적인 얼굴 크기나 위치, 밝기 및 대비 문제는 계속 실패 처리한다.

GPU IAM에는 요청 큐의 receive/delete/change-visibility 권한, 결과 큐의
`sqs:SendMessage`, 입력 및 결과 prefix의 S3 get/put 권한이 필요하다. 백엔드는
결과 큐의 receive/delete 권한과 DB 작업 상태 갱신 로직이 필요하다.

회원 프로필 RAG 저장 요청인 `POST /api/v1/training/profiles`에는 백엔드의
`cloneId`를 반드시 포함한다. 저장이 끝나면 AI 서버가 전체 성격·개인정보 학습 완료를
다음 콜백으로 알린다.

```text
POST {CLONE_TRAINING_CALLBACK_BASE_URL}/internal/clone-training/{cloneId}/personality/complete
X-Clone-Training-Callback-Secret: {CLONE_TRAINING_CALLBACK_SECRET}
```

운영 AI 서버에는 다음 값을 설정한다.

```env
CLONE_TRAINING_CALLBACK_BASE_URL=http://10.0.1.49:8080
CLONE_TRAINING_CALLBACK_SECRET=
```

비밀값은 `.env`에만 저장하고 로그, 문서, Git에는 기록하지 않는다. 콜백 실패 시
프로필 API도 실패로 응답한다. RAG 프로필은 동일 문서 ID로 upsert되므로 요청을
재시도해도 중복 문서가 생성되지 않는다.

GPU 얼굴 워커 전용 의존성은 다음과 같이 설치한다.

```bash
python -m pip install -r requirements-face.txt
```

### 얼굴 유사도 평가

LivePortrait 결과와 원본 회원 얼굴을 비교하려면 별도 의존성을 설치하고 설정을
활성화한다.

```bash
python -m pip install -r requirements-face-similarity.txt
```

```env
FACE_SIMILARITY_ENABLE=true
FACE_SIMILARITY_REQUIRED=false
FACE_SIMILARITY_ACCEPT_INSIGHTFACE_NON_COMMERCIAL_LICENSE=true
FACE_SIMILARITY_CALIBRATION_VERSION=provisional-v1
FACE_SIMILARITY_CALIBRATED=false
```

InsightFace 공개 모델 가중치는 비상업 연구 용도로만 사용한다. 라이선스를 확인하고
동의한 개발 환경에서만 승인 설정을 `true`로 바꾼다.

평가는 생성 영상에서 균등 추출한 프레임을 회원의 품질 통과 원본 프레임 및 같은
시점의 driving video 프레임과 비교한다. 정체성 점수 65%, 렌더링 품질 35%를
기본으로 합산하고 시간적 일관성이 낮은 결과에는 추가 감점을 적용한다. 결과는
manifest의 `faceSimilarity`에 기록된다.

- `score`: UI와 향후 종합 유사도 계산에 사용할 0~95 얼굴 점수
- `identityScore`: 얼굴 임베딩 기반 정체성 보존 점수
- `renderQualityScore`: 얼굴 검출률, 시간적 안정성, 선명도 보존 점수
- `confidence`: 평가 표본 충분성을 나타내는 `low`, `medium`, `high`
- `cosineSimilarity`: 재보정에 사용할 원시 코사인 유사도
- `alignedFrameCount`: 생성 영상과 원본 영상을 같은 순서로 비교한 프레임 수
- `stabilityFactor`: 시간적 일관성으로 최종 점수에 적용된 배율
- `calibrationVersion`, `calibrated`: 점수 보정 버전과 검증 완료 여부

코사인 유사도는 확률이나 퍼센트가 아니다. 현재 기본 임계값은 파이프라인 검증용
`provisional-v1`이므로 `calibrated=false`로 유지한다. 여러 회원의 동일인 결과와
다른 사람 결과를 모아 임계값을 보정한 뒤에만 UI의 공식 얼굴 유사도로 사용한다.
얼굴 임베딩 자체는 생체정보이므로 manifest나 DB에 저장하지 않고 집계 지표만
보관한다.

생성 결과 하나를 수동 평가할 수도 있다.

```bash
python -m model_training.face_training.face_similarity \
  --reference /path/to/front.jpg \
  --reference /path/to/left-profile.jpg \
  --reference /path/to/right-profile.jpg \
  --generated /path/to/generated.mp4 \
  --driving /path/to/original-driving.mp4
```

여러 LivePortrait 설정을 한 번에 생성하고 얼굴 점수로 순위를 매기려면 기존
전처리 manifest를 사용한다. InsightFace 모델은 한 번만 GPU에 로드되며 각 후보의
점수와 경로는 `variant-sweep.json`에 저장된다. 최고점 결과는 `best.mp4`, 원본
비교 영상은 `best-concat.mp4`로 복사된다.

```bash
python -m model_training.face_training.variant_sweep \
  --manifest /path/to/preprocess-manifest.json \
  --multipliers 0.65 0.75 0.85 \
  --crop-scales 2.5 2.7
```

기본 조합은 6개다. 자동 점수는 후보를 줄이기 위한 기준이며, 보정 완료 전에는
`best-concat.mp4`를 사람이 확인한 뒤 최종 결과를 확정한다.

PyTorch 2.3 CUDA 12.1 환경에서는 InsightFace 설치 후 CPU ONNX Runtime이
선택될 수 있다. 이 경우 CUDA 12 및 cuDNN 8과 호환되는 GPU 빌드를 마지막에
설치한다.

```bash
python -m pip install --force-reinstall --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.18.0
```

학교 GPU 서버처럼 AWS IAM Role이 없는 외부 서버에서는 얼굴 작업 큐 수신과
S3 입출력만 허용한 제한적 AWS 자격 증명을 환경변수로 주입해야 한다. 실제 키는
`.env` 또는 Git에 커밋하지 않는다.
