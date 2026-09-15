# 얼굴 모델 최신 작업 상태

마지막 확인 기준: 2026-09-09 작업 세션

이 문서는 얼굴 모델 작업을 재개할 때 가장 먼저 확인하는 기준 문서다.
`gpu-server-access.md`는 접속 절차용이고, 이 문서는 구현 및 실험 진행 상태용이다.

## 1. Ditto 환경

| 구분 | 값 |
| --- | --- |
| 서버 저장소 | `/shareHost/C084003-ditto/ditto-talkinghead` |
| Ditto 커밋 | `c3e47ee` |
| Conda 환경 | `/shareHost/C084003-ditto/conda-env` |
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | `2.5.1+cu121` |
| ONNX Runtime | `1.20.2` |
| ONNX provider | `CUDAExecutionProvider`, `CPUExecutionProvider` |
| OpenCV | `4.11.0` |
| MediaPipe | `0.10.21` |
| Einops | `0.8.1` |
| 체크포인트 | `checkpoints/`, 약 2.2GB |

정상 VS Code 터미널 프롬프트:

```text
(/shareHost/C084003-ditto/conda-env) mirrorsoul@TeamC084003:/shareHost/C084003-ditto/ditto-talkinghead$
```

## 2. Ditto 품질 실험

남성현 정면 얼굴과 실제 ElevenLabs 복제 음성으로 Ditto 생성에 성공했다.

- 최초 결과: `얼굴 모델 작업/v6/namseonghyeon-hello-ditto.mp4`
- 서버 입력 얼굴: `/shareHost/C084003-ditto/test-inputs/namseonghyeon-front.jpg`
- 서버 입력 음성: `/shareHost/C084003-ditto/test-inputs/namseonghyeon-preview.wav`
- 서버 최초 결과: `/shareHost/C084003-ditto/test-results/namseonghyeon-hello-ditto.mp4`

v7에서는 같은 입력과 seed `1024`로 다음 네 후보를 생성했다.

| 결과 | `smo_k_d` | `sampling_timesteps` | 평가 |
| --- | ---: | ---: | --- |
| `01-balanced.mp4` | 3 | 50 | 입 움직임은 크지만 미세 떨림이 가장 많음 |
| `02-smooth.mp4` | 5 | 50 | 떨림 감소와 발음 움직임 유지의 균형이 가장 좋음 |
| `03-very-smooth.mp4` | 7 | 50 | 안정적이지만 발음 움직임이 눌림 |
| `04-high-quality.mp4` | 5 | 75 | 생성 시간만 늘고 입술 및 치아 품질 개선 없음 |

최종 선택은 `얼굴 모델 작업/v7/02-smooth.mp4`다. 기본 Ditto 설정은
`smo_k_d=5`, `sampling_timesteps=50`, `cropScale=2.3`으로 좁혔다.

분석 당시 `02-smooth`는 `01-balanced`보다 입 주변 흔들림이 약 15%, 눈 주변
흔들림이 약 10% 감소했다. 네 후보의 얼굴 영역 유사도는 약 `0.978~0.982`였다.

남은 품질 문제:

- 치아가 개별 형태보다 회색빛 띠처럼 표현됨
- 입꼬리와 볼보다 턱의 상하 움직임이 두드러짐
- 큰 입 모양에서 입술 경계와 인중이 뭉개짐
- 눈과 시선 변화가 적어 얼굴 전체가 정적으로 보임
- 닫힌 입 정면 사진 한 장에서 치아와 구강 내부를 추정해야 하는 한계

다음 품질 실험으로 v8을 제안했지만 아직 실행하지 않았다.

- 얼굴 영상 프레임 추출을 초당 2장에서 5장으로 증가
- 전체 화면 대신 얼굴, 눈, 입술 영역 선명도 평가
- 정면 및 입술 품질이 좋은 상위 원본 프레임 3개 선택
- `smo_k_d=5`, `sampling_timesteps=50` 고정
- 각 프레임에 `crop_scale=2.1`, `2.3`을 적용해 6개 후보 생성
- 최종 후보를 10~15초 음성으로 장시간 안정성 및 립싱크 평가

사용자가 품질 개선보다 회원가입 자동화를 먼저 진행하기로 결정하여 v8은 보류했다.

## 3. 얼굴 프로필 자동화

AI/GPU 워커 자동화가 구현되어 `main`에 병합됐다.

| 구분 | 값 |
| --- | --- |
| 기능 브랜치 | `codex/automate-face-training` |
| 구현 커밋 | `4c54ae8` |
| main 병합 커밋 | `206069b` |
| 검증 | 얼굴 관련 테스트 53개 통과 |

구현 완료 범위:

- `FACE_PROFILE_BUILD` 요청 SQS 상시 수신
- S3 얼굴 영상 다운로드
- 영상 검사, 프레임 추출, 품질 게이트 및 대표 얼굴 선택
- `portrait.jpg`, `preprocess-manifest.json`, `face-profile.json` 생성
- `face-results/<userUuid>/job-<jobId>/`에 산출물 업로드
- 결과 SQS에 `PROCESSING`, `COMPLETED`, `FAILED` 발행
- Ditto 기본 설정을 `face-profile.json`에 저장
- 긴 처리 중 SQS visibility timeout 연장
- `COMPLETED` 결과 발행 성공 후에만 요청 메시지 삭제
- 실패 요청 보존, `retryable=true`, 재시도 및 DLQ 지원
- FIFO 결과 큐 중복 방지 및 `jobId` 멱등 처리 계약

`face-profile.json`은 완성된 고정 영상이나 별도 학습 가중치가 아니다. 회원 대표
얼굴, 품질 정보, Ditto 렌더 설정을 묶은 통화용 얼굴 프로필이며, 실제 통화 음성이
들어올 때 Ditto가 이 프로필로 얼굴 움직임을 생성해야 한다.

## 4. 전체 클론 자동화 상태

AI 얼굴 워커까지는 완료됐지만, 신규 회원가입부터 실제 영상통화까지의 전체 자동화는
아직 완성되지 않았다.

완료된 부분:

- 음성 SQS 작업에서 ElevenLabs 음성 복제 및 음성 프로필 저장
- 얼굴 SQS 작업에서 얼굴 프로필 생성 및 결과 SQS 발행
- 음성과 얼굴을 같은 회원 및 `cloneId` 기준으로 연결하는 AI 코드

남은 부분:

- 백엔드 얼굴 결과 SQS 및 DLQ 생성
- GPU IAM에 결과 큐 `sqs:SendMessage` 권한 추가
- GPU 워커에 `AWS_SQS_FACE_TRAINING_RESULT_QUEUE_URL` 적용
- 백엔드 결과 큐 소비자 구현
- 완료 산출물 S3 key를 DB에 저장하고 얼굴 프로필 `READY/ACTIVE` 처리
- `jobId` 중복 메시지 멱등 처리와 retryable 실패 처리
- 음성 ACTIVE + 얼굴 READY + 성격/개인정보 완료를 합친 전체 클론 `READY` 판정
- GPU 워커 배포 및 상시 실행
- Ditto 실시간 렌더링 API
- 통화 서버의 WebRTC 비디오 트랙 송출
- 신규 테스트 회원의 가입부터 DB READY까지 E2E 검증

현재 통화 서버는 오디오 입출력만 지원하므로 음성통화는 가능하지만 Ditto 얼굴이
포함된 실제 영상통화는 아직 불가능하다.

## 5. 정확한 마지막 중단점

백엔드 팀에 다음 계약을 전달하는 단계까지 완료했다.

- 얼굴 결과 큐와 DLQ 생성
- 결과 큐 URL 및 GPU IAM 권한 제공
- `FACE_PROFILE_BUILD_STATUS` 소비자 구현
- `PROCESSING`, `COMPLETED`, `FAILED` DB 반영
- S3 artifact key 저장 및 얼굴 프로필 READY 처리
- 음성, 얼굴, 성격이 모두 준비됐을 때만 전체 클론 READY 처리

다음 작업은 백엔드 구현 결과와 결과 큐 URL/IAM 권한을 확인한 뒤 GPU 워커에 적용하고,
테스트 회원 한 명으로 `가입 -> 음성/얼굴 생성 -> DB READY`를 통합 검증하는 것이다.
그 이후 Ditto 렌더링 API와 WebRTC 비디오 트랙을 연결해야 실제 클론 영상통화가 된다.

## 6. 백엔드 대기 중 선행 작업

2026-09-15에 `codex/ditto-render-runner` 브랜치에서 백엔드 없이 검증 가능한 Ditto
실행 계층을 추가했다.

- Ditto `c3e47ee`의 `StreamSDK` 및 `inference.run` 호출 래퍼
- `face-profile.json`의 Ditto 렌더 설정 로드 및 검증
- 기본값 `cropScale=2.3`, `smo_k_d=5`, `samplingTimesteps=50`, seed `1024`
- 얼굴 이미지와 WAV만으로 실행하는 독립 CLI
- stdout 및 stderr를 `*.ditto.log`로 보존
- 임시 MP4 생성 후 성공한 결과만 최종 경로로 원자적 교체
- 실패 시 기존 결과 영상 보존
- 전체 로컬 테스트 63개 통과

RTX 4090 실GPU에서도 기존 남성현 얼굴과 음성으로 실행해 다음 결과를 확인했다.

```text
/shareHost/C084003-ditto/test-results/runner-smoke-test.mp4
1080x1920, 25fps, 2.12초, H.264 비디오 + AAC 오디오
```

이 모듈은 향후 GPU Ditto 렌더링 API와 WebRTC 비디오 트랙에서 공통으로 사용한다.
