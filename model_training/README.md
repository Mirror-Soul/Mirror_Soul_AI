
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

현재 1차 버전은 운영 메시지 유실을 막기 위해 검증 모드만 지원한다.

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

LivePortrait 캐시 생성, 결과 S3 업로드, 백엔드 완료 처리가 연결되기 전에는
상시 워커로 실행하지 않는다.

GPU 얼굴 워커 전용 의존성은 다음과 같이 설치한다.

```bash
python -m pip install -r requirements-face.txt
```

학교 GPU 서버처럼 AWS IAM Role이 없는 외부 서버에서는 얼굴 작업 큐 수신과
S3 입출력만 허용한 제한적 AWS 자격 증명을 환경변수로 주입해야 한다. 실제 키는
`.env` 또는 Git에 커밋하지 않는다.
