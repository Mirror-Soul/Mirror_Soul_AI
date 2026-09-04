
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

현재 버전은 운영 메시지 유실을 막기 위해 검증 모드만 지원한다.

```bash
python -m model_training.face_training.worker --once --dry-run
```

1차 처리 범위:

- SQS 메시지 계약 검증
- S3 얼굴 영상 다운로드
- FFprobe 영상 메타데이터 검사
- FFmpeg 프레임 추출
- 선명도, 밝기, 대비, 얼굴 크기 및 중앙 정렬 품질 검사
- 정면 및 좌우 프로필 대표 프레임 자동 선택
- 작업별 전처리 manifest 저장
- SQS 메시지 유지

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

결과 S3 업로드와 백엔드 완료 처리가 연결되기 전에는 상시 워커로 실행하지 않는다.

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
