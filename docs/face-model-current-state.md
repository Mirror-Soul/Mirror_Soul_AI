# 얼굴 모델 최신 작업 상태

마지막 확인 기준: 2026-09-17 작업 세션

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

2026-09-16에 v8 움직임 비교를 실행했다. 모든 결과는 같은 복제 음성, seed `1024`,
`cropScale=2.3`, `smo_k_d=5`, `sampling_timesteps=50`을 사용했다.

| 결과 | 소스 및 제어 | 얼굴 프레임 차이 평균 | 얼굴 선명도 중앙값 | 1차 평가 |
| --- | --- | ---: | ---: | --- |
| `01-static-v7-smooth.mp4` | v7 정지 사진 기준 | 0.590 | 252.89 | 가장 안정적이나 눈이 정적임 |
| `02-blink-focused.mp4` | 정지 사진 + 25프레임 후 제어 눈 깜빡임 | 0.649 | 248.29 | 선명도와 안정성을 유지하면서 눈 깜빡임 추가 |
| `03-motion-natural.mp4` | 2.12초 원본 정면 영상, `smo_k_s=5` | 5.860 | 227.79 | 실제 움직임은 풍부하지만 초반 촬영 흔들림과 흐림이 들어옴 |
| `04-motion-trimmed-blink.mp4` | 첫 0.36초 제거 + `smo_k_s=13` + 제어 눈 깜빡임 | 4.121 | 238.97 | 초반 흔들림을 줄이고 실제 움직임과 눈 깜빡임을 결합 |
| `05-motion-trimmed-soft-blink.mp4` | 4번 + 눈 깜빡임 강도 `0.7` | 4.115 | 240.19 | 눈을 과하게 감는 느낌을 줄인 현재 최종 후보 |

결과와 분석 파일은 `얼굴 모델 작업/v8`에 있다. `02-blink-focused`는 약 1.4초에
눈을 한 번 감으며 v7 대비 얼굴 선명도 감소가 약 1.8%다. `03-motion-natural`은
시간 변화량이 약 10배 커서 정적인 느낌은 줄지만, 현재 원본 클립의 초반 흔들림과
작은 얼굴 크기 때문에 선명도가 약 9.9% 낮다. 3번의 흔들리는 첫 9프레임을 제거한
4번은 프레임 변화량이 약 29.7% 감소하고 선명도가 약 4.9% 개선됐다. 4번에는
`drive_eye=true`, `blink_open_frames=25`를 적용해 약 1.4초에 눈을 한 번 감게 했다.

사용자 검토에서 4번의 눈을 너무 세게 감는 느낌이 확인되어, 기본 눈 변형 배열을
`0.7`배로 줄인 5번을 추가했다. 5번은 4번의 고개 움직임과 깜빡임 시점을 유지하면서
눈꺼풀 이동만 줄였으며 현재 가장 균형이 좋은 후보로 평가한다.

이번 실험을 위해 독립 렌더 실행기에 `smo_k_s`와 눈 깜빡임 간격 제어를 추가했다.
다음 우선순위는 4번과 5번을 직접 비교해 5번을 최종 확정한 뒤,
회원가입 촬영 가이드와 흔들림 없는 `motion-template.mp4` 선별을 구현하는 것이다.

### v8 장문 테스트 준비 상태

5번 설정의 장시간 안정성을 확인하기 위한 복제 음성 입력을 생성했다.

- 문장 파일: `얼굴 모델 작업/v8/long-test-speech.txt`
- 음성 파일: `얼굴 모델 작업/v8/long-test-speech.mp3`
- 실제 길이: `10.031초`
- 서버 입력: `/shareHost/C084003-ditto/test-inputs/namseonghyeon-long-test.mp3`
- 예정 결과: `/shareHost/C084003-ditto/test-results/v8/06-long-motion-soft-blink.mp4`
- 소스: `/shareHost/C084003-ditto/test-inputs/namseonghyeon-motion-trimmed.mp4`
- 설정: `cropScale=2.3`, `smo_k_d=5`, `smo_k_s=13`,
  `blink_strength=0.7`, `drive_eye=true`, seed `1024`
- 장문 눈 깜빡임: `blink_open_frames=0`으로 약 3~5초 무작위 간격 사용

2026-09-16 마지막 실행 시 다른 컨테이너의 작업이 RTX 4090 메모리 `16.5GB`와
연산률 `100%`를 사용하고 있어 렌더링을 시작하지 않았다. 해당 작업이 끝나 GPU
메모리가 확보되면 위 설정 그대로 렌더링하고, 1.76초 모션 템플릿의 반복 경계,
눈 깜빡임 빈도, 입술·치아·턱의 장시간 안정성을 확인한다.

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
- React Native 통화 화면의 원격 비디오 트랙 표시
- 신규 테스트 회원의 가입부터 DB READY까지 E2E 검증

통화 서버의 Ditto WebRTC 비디오 송출 트랙은 구현됐지만 React Native가 아직 원격
비디오 트랙을 수신하고 화면에 표시하지 않으므로 앱에서 얼굴 영상통화는 아직 볼 수
없다.

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
GPU 렌더링 API와 통화 서버 WebRTC 비디오 트랙은 백엔드 대기 중 선행 구현을 완료했다.
다음 작업은 React Native의 수신 비디오 트랙 표시이며, 이후 백엔드 결과 큐와 전체 가입
흐름이 준비되면 실제 클론 영상통화를 E2E로 연결한다.

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

## 7. GPU Ditto 렌더링 서비스

2026-09-15에 `codex/ditto-render-service` 브랜치에서 지속 실행 가능한 내부 API를
구현했다.

- 시작 시 Ditto 모델을 GPU에 한 번만 로드하고 요청 사이에 같은 `StreamSDK` 재사용
- `POST /api/v1/render`에서 얼굴, 음성, 선택적 `face-profile.json`을 받아 MP4 반환
- `GET /health`, `GET /ready`에서 GPU, 모델 로드 시간, 렌더 횟수 및 최근 오류 제공
- `X-Ditto-Api-Key` 인증, 업로드 확장자 및 크기 제한
- v7 `smooth` 설정을 기본값으로 사용하고 요청별 설정 덮어쓰기 지원
- 동시 GPU 렌더 요청은 HTTP `429`로 거절
- 기본 `127.0.0.1` 바인딩, 외부 연결은 TLS 또는 암호화된 사설망 사용
- FFmpeg 경로를 명시적으로 등록하고 최종 MP4 생성 여부 검증
- 모델 로드와 렌더를 같은 전용 OS 스레드에 고정해 CUDA/ONNX 컨텍스트 중복 방지

RTX 4090에서 실제 HTTP 요청 두 건을 연속 실행한 결과:

| 항목 | 결과 |
| --- | ---: |
| 모델 로드 | 30.4초 |
| 첫 렌더 | 16.9초 |
| 같은 모델 재사용 렌더 | 2.4초 |
| 결과 | 1080x1920, 25fps, 2.12초, H.264 + AAC |
| 서비스 테스트 | 10개 통과 |

첫 구현 검증에서 모델 로드와 렌더를 일반 비동기 작업 풀의 서로 다른 스레드에서
실행했을 때 GPU 메모리가 약 22GB까지 중복 점유되어 ONNX Runtime OOM이 발생했다.
전용 단일 스레드로 고정한 뒤 두 요청 모두 성공했고 종료 후 GPU 메모리도 해제됐다.

이 서비스는 현재 완성 MP4를 반환하는 렌더 단계이며, 그 자체로 실시간 통화 영상은
아니다. 실제 통화에는 Ditto 생성 프레임을 통화 서버의 WebRTC 비디오 트랙으로 보내는
연결과 React Native 화면 표시가 추가로 필요하다.

## 8. WebRTC 비디오 트랙

2026-09-15에 `codex/ditto-webrtc-video-track` 브랜치에서 통화 서버의 Ditto 영상
연결을 구현했다.

- `CALL_INVITE.data.mediaType=VIDEO`일 때만 오디오와 비디오 송출 트랙 생성
- Ditto MP4를 메모리에 압축 상태로 보관하고 재생 시점에 프레임 단위로 디코딩
- 90kHz RTP 타임베이스와 25fps 타임스탬프 생성
- 통화 대기 및 답변 종료 후 회원 얼굴 정지 프레임 송출
- TTS MP3를 Ditto API로 렌더한 뒤 영상 큐와 오디오 큐를 순서대로 시작
- GPU가 사용 중이면 HTTP `429`에 지수 백오프 재시도
- Ditto 렌더 실패 시 영상통화를 끊지 않고 음성 답변 계속 재생
- 통화 종료 시 수신, 응답, 얼굴 준비 작업과 비디오 디코더 정리
- S3에서 회원의 최신 `face-profile.json`과 portrait를 통화당 한 번 로드
- 백엔드 준비 전에는 로컬 portrait 및 profile 경로로 통합 테스트 가능

실제 v7 `02-smooth.mp4`를 새 트랙에 넣어 55프레임을 540x960으로 디코딩하고,
프레임 종료 후 정지 화면으로 복귀하는 것까지 확인했다.

같은 날 GPU 서비스와 새 통화 클라이언트를 실제로 연결한 통합 테스트도 통과했다.

| 항목 | 결과 |
| --- | ---: |
| 통화 클라이언트 GPU 렌더 | 성공, 13.911초 |
| 반환 MP4 | 595,171바이트, 1080x1920, 25fps, 2.12초 |
| 코덱 | H.264 비디오 + AAC 오디오 |
| WebRTC 트랙 소비 | 55프레임, 540x960, RTP PTS 0~4860 |
| 영상 종료 상태 | `playing=False`, 대기 화면 복귀 |
| 전체 로컬 테스트 | 85개 통과 |

테스트용 GPU 서비스 종료 후 GPU 프로세스와 메모리가 해제됐고, 서버에 전송한 임시
코드와 결과 파일도 삭제했다.

현재는 완성된 TTS 음성 전체를 Ditto 서비스가 MP4로 만든 뒤 재생하는 턴 기반
영상통화다. 다음 통합 단계는 React Native가 `VIDEO` offer에 수신용 비디오
transceiver를 포함하고 원격 트랙을 화면에 표시하도록 연결하는 것이다. 이후 지연시간
개선 단계에서 Ditto의 `online_mode`를 이용한 생성 중 프레임 스트리밍을 검토한다.

## 9. v9 다중 회원 일반화 테스트

2026-09-16에 S3와 RDS를 읽기 전용으로 조회해 회원들의 얼굴 스캔과
`userUuid + cloneId` 연결을 검증했다. 이전 로컬 산출물 일부는 회원 이름과 clone ID가
다르게 붙어 있었으므로 v9부터는 회원 이름을 실행 식별자로 사용하지 않는다.

- `clone-2`: DB 이름 `admin0001`, 사용자 확인 이름 `강신욱`, 활성 음성 있음
- `clone-4`: DB 이름 `남성현`, 활성 음성 있음
- `clone-5`: DB 이름 `김동빈`, 활성 음성 있음

사용자 확인 결과 `clone-2`와 `clone-6`은 같은 강신욱 회원이다. 일반화 테스트는
서로 다른 3명을 비교해야 하므로 `clone-6`은 제외하고 `clone-2`만 사용한다.
`clone-2`는 DB 활성 복제 음성이 연결되어 있고, 분석 프레임 42개 중 38개가 정면이며
선택한 움직임 구간의 정면 비율도 100%라 대표 데이터로 더 적합하다.

이정준으로 추정되는 `clone-3`은 얼굴 원본과 음성 학습 파일 5개는 있지만 음성 학습이
`FAILED`이고 활성 Voice ID도 없어 얼굴+음성 일반화 비교에서는 제외했다. DB 기준으로
얼굴 파일과 활성 음성이 모두 있는 계정은 `clone-2`, `clone-4`, `clone-5`다.

원본은 회전 메타데이터 적용 후 1080x1920 세로 영상이다. 로컬 CPU로 2fps 프레임
품질 분석을 실행한 결과 `clone-2`, `clone-6`, `clone-4`는 기본 게이트를 통과했다.
`clone-5`는 촬영 영상이 부드러워 기본 선명도 기준에서는 실패했지만 실험용 기준
`min_sharpness=25`에서 정면 대표 프레임 4장이 통과했다.

회원마다 초기 흔들림을 제외한 2초 및 3.5초 자연 움직임 클립을 생성했고, 같은 문장을
각 회원의 복제 음성으로 만들었다. GPU 서버에 전송한 파일 중 `clone-2`, `clone-4`,
`clone-5`의 얼굴 클립 6개와 음성 3개만 사용한다. 3명 x 6설정, 총 18개를 순차
렌더링한다. 배치 실행기는 GPU 여유 메모리
18GB 이상, 사용률 10% 이하를 확인하고 이미 생성된 결과는 건너뛴다.

2026-09-17 GPU가 비워진 뒤 3명 x 6설정의 결과 18개를 모두 렌더링했다. 첫 실행은
비대화형 SSH의 `PATH`에 `ffmpeg`가 없어 강신욱 baseline의 영상 프레임만 생성됐지만,
Conda 환경의 `ffmpeg`로 음성을 합쳐 복구하고 이후 배치는 Conda `bin`을 `PATH`에
포함해 완료했다. 18개 모두 H.264 영상과 AAC 음성 스트림 검증을 통과했다.

서버 결과는 `/shareHost/C084003-ditto/test-results/v9/`, 검토용 최종 파일은
`얼굴 모델 작업/v9/review/`에 있다. `analyze_results.py`로 회원별 6분할 비교 영상과
접촉 시트, `comparison-metrics.json`을 생성했다. 정적 지표에서는 `natural-eye`가
세 회원 모두 baseline보다 평균 선명도가 약 0.5~7.2% 높았다. `longer-motion`은
회원별 움직임 편차가 크고 남성현 결과에서 선명도가 약 16.8% 낮아 공통 기본값으로
사용하기 어렵다. 다음 단계는 원본 비교 영상을 직접 검토해 회원별 상위 1~2개를
선택하고 10초 이상 장문 안정성 테스트를 실행하는 것이다.

## 10. v10 입 모양 및 립싱크 개선

2026-09-17 사용자 검토에서 `clone-2` 강신욱은 연결된 음성이 본인 음성이 아니어서
품질 비교에서 제외했다. 이는 얼굴 모델 튜닝 문제가 아니라 회원 음성 매핑 또는 활성
음성 데이터 문제로 별도 수정해야 한다. v10 입 품질 실험은 데이터가 정상인
`clone-4` 남성현과 `clone-5` 김동빈만 사용했다.

MP3 인코더 지연 영향을 없애기 위해 두 회원의 기준 음성을 16kHz mono WAV로 변환하고,
다음 요소를 7개 후보로 비교했다.

- 입 표정 강도 `1.00`, `0.85`, `0.70`
- 구동 모션 평활화 `smo_k_d=5`와 `3`
- 음성 에너지 기반 VAD로 무음 구간 입을 중립 표정에 가깝게 복귀
- 기존 2초 움직임 소스와 0.80초의 선명한 닫힌 입 중립 프레임
- 상대 구동과 절대 구동

두 회원 x 7개, 총 14개 결과를 렌더링했으며 모든 후보와 음성 포함 비교 영상 2개가
비디오·오디오 스트림 검사를 통과했다. 서버 결과는
`/shareHost/C084003-ditto/test-results/v10/`, 로컬 결과는
`얼굴 모델 작업/v10/results/`, 검토 파일은 `얼굴 모델 작업/v10/review/`에 있다.

초기 추천값은 다음과 같다.

| 회원 | 추천 후보 | 근거 |
| --- | --- | --- |
| 남성현 (`clone-4`) | `motion-exp085` | 기존 자연 움직임 유지, P95 입 벌림 약 8.7% 감소, 추정 지연 0ms |
| 김동빈 (`clone-5`) | `neutral-vad070` | P95 입 벌림 약 19.1% 감소, 음성-입 상관도 `0.419 -> 0.496`, 추정 지연 40ms |

`neutral-absolute085`는 무음 구간의 입 벌림은 더 줄지만 김동빈 음성-입 상관도가
`0.429`로 낮아 기본 후보에서 제외했다. 자동 수치는 과도한 입 벌림과 큰 지연을
거르는 보조 지표이며, 한국어 발음별 입술 모양의 자연스러움은 음성이 포함된 회원별
7분할 비교 영상으로 최종 확인해야 한다.

정확한 다음 단계는 비교 영상에서 위 추천 후보를 직접 확인한 뒤, 선택된 회원별
설정으로 10초 이상 장문을 렌더링해 문장 사이 입 닫힘, 치아·입술 경계, 반복 구간의
안정성을 검증하는 것이다. 이후 회원별 튜닝값을 공통 기본값으로 합칠지 또는 얼굴
프로필별 보정값으로 저장할지 결정한다.

## 11. v11 립싱크 시점 보정 및 장문 검증

2026-09-17 v10의 김동빈 추천값 `neutral-vad070`을 기준으로, 영상 생성용 음성과
최종 MP4에 합치는 원본 음성을 분리했다. 생성용 음성만 앞당겨 Ditto의 입 반응 지연을
보정하고 최종 음성은 원본을 그대로 유지한다.

김동빈은 `0/20/40/80ms`, `smo_k_d=3/1`, 입 표정 강도 `0.70/0.65`를 조합한
8개 후보를 비교했다. 최종 추천은 `c02-advance20`이다.

- 입 지연: `40ms -> 0ms`
- 음성-입 상관도: `0.37382 -> 0.37699`
- 입 움직임 속도 P95: `0.040383 -> 0.037665`
- 설정: 생성용 음성 20ms 앞당김, `smo_k_d=3`, 입 표정 강도 `0.70`

`c06-smo1-advance20`도 상관도는 `0.38022`로 조금 높았지만, 장기적으로 검증된
평활화 강도 3을 유지하고 무음 입 닫힘과 발화 대비가 더 나은 c02를 선택했다.
80ms 후보는 입이 오히려 약 40ms 먼저 움직이고 상관도가 `0.25319`로 떨어져
과보정으로 탈락했다.

남성현의 기존 약 10초 복제 음성으로 6개 장문 후보를 추가 검증했다. 수치상
`l03-advance40`이 지연을 `40ms -> 0ms`, 상관도를 `0.20075 -> 0.21951`로
개선했다. 회원별 최적 보정값이 다르므로 현재는 전 회원 공통 상수보다 얼굴 프로필별
`lipSyncAdvanceMs` 값으로 저장하는 방향이 안전하다.

로컬 결과와 비교 영상은 `얼굴 모델 작업/v11/results/` 및
`얼굴 모델 작업/v11/review/`, 서버 결과는
`/shareHost/C084003-ditto/test-results/v11/`에 있다. 다음 단계는 사용자가
`recommended-clone-5-c02-advance20.mp4`를 직접 확인한 뒤, 승인되면 렌더 서비스와
`face-profile.json`에 회원별 입 강도와 `lipSyncAdvanceMs`를 연결하는 것이다.

## 12. v12 새 장문 및 실제 회원 음성 재검증

2026-09-17 ElevenLabs 계정의 최신 온보딩 클론을 회원 UUID로 정확히 식별해
남성현(`clone-4`)과 김동빈(`clone-5`)의 새 장문 음성을 각각 생성했다. 두 회원은
같은 문장을 사용했고, v10에서 선택한 얼굴 설정을 유지한 채 생성용 음성 선행값만
`0/20/40ms`로 바꿔 총 6개를 렌더링했다.

| 회원 | 추천 후보 | 상관도 | 측정 지연 | 흔들림 평균 | 결론 |
| --- | --- | ---: | ---: | ---: | --- |
| 남성현 | `l02-advance20` | 0.14881 | 0ms | 0.024784 | 20ms가 종합적으로 가장 안정적 |
| 김동빈 | `l01-advance00` | 0.27765 | 40ms | 0.026100 | 상관도와 발화/무음 대비가 가장 좋음 |

남성현 20ms는 0ms의 측정 지연 240ms를 없애면서 상관도와 발화/무음 대비를 높였고,
40ms보다 흔들림도 적었다. 김동빈은 20ms와 40ms에서 측정 지연이 0ms가 됐지만
상관도와 발화/무음 대비가 낮아져, 이번 장문에서는 0ms가 더 자연스러운 후보로
판단했다. v11 짧은 문장에서는 김동빈 20ms가 우세했으므로 특정 선행값을 모든 문장에
고정하기보다 0~20ms 범위를 추가 검증하거나 오디오 특성 기반 보정을 검토해야 한다.

후보 6개와 비교 영상 2개 모두 H.264 비디오와 AAC 오디오 스트림 검사를 통과했다.
로컬 결과는 `얼굴 모델 작업/v12/`, 서버 결과는
`/shareHost/C084003-ditto/test-results/v12/`에 있다. 사용자가 우선 확인할 결과는
`review/recommended-clone-5-advance00.mp4`이며, 비교 영상은 왼쪽부터
`0ms / 20ms / 40ms` 순서다.

## 13. 백엔드 얼굴 결과 소비 및 성격 완료 콜백 연결

2026-09-17 백엔드 팀에서 얼굴 결과 큐 소비자 구현 완료를 전달받았다. 백엔드는
`FACE_PROFILE_BUILD_STATUS`의 `PROCESSING`, `COMPLETED`, `FAILED`를 처리하고,
완료 artifact 경로 검증과 얼굴 프로필 활성화, 중복 메시지 멱등 처리, 전체 클론
READY 판정을 구현했다. 얼굴 결과 SQS URL도 전달받아 로컬 비공개 `.env`에 적용했다.

AI 얼굴 워커의 기존 결과 메시지는 이 계약과 일치한다. `FAILED`에는
`error.retryable`이 포함되고, S3 artifact 업로드 및 결과 메시지 발행이 성공한 뒤에만
요청 메시지를 삭제한다.

성격·개인정보 학습 완료는 `POST /api/v1/training/profiles`가 회원 RAG 프로필을
저장한 직후 백엔드의 다음 내부 콜백을 호출하도록 구현했다.

```text
POST /internal/clone-training/{cloneId}/personality/complete
X-Clone-Training-Callback-Secret: <환경변수 값>
```

이를 위해 프로필 학습 요청에 양의 정수 `cloneId`를 필수로 추가했다. 콜백 관련
환경변수는 AWS AI 서버의 비공개 `.env`에 적용했고, AWS AI 서버에서 백엔드 내부
주소 `10.0.1.49:8080` 연결을 확인했다. 새 콜백과 기존 얼굴 워커 계약 테스트는
총 18개가 통과했다.

아직 완료되지 않은 운영 작업은 다음과 같다.

- 콜백 코드를 커밋·병합하고 AWS AI 서버에 배포한 뒤 `mirrorsoul-ai.service` 재시작
- 백엔드가 `/api/v1/training/profiles` 요청에 실제 `cloneId`를 보내는지 확인
- 학교 GPU에 Mirror Soul AI 얼굴 워커 코드와 실행 환경 배포
- GPU IAM의 결과 큐 `sqs:SendMessage` 및 S3 권한 확인
- GPU 워커 환경에 결과 큐 URL을 적용하고 상시 실행
- 테스트 회원 한 명으로 얼굴·음성·성격 완료 후 DB 전체 클론 READY 확인

## 14. 2026-09-17 최종 중단점과 재개 순서

사용자 요청에 따라 커밋, 푸시, 배포 전 상태에서 작업을 중단했다. 다음 세션에서는
새로운 품질 실험부터 시작하지 말고 아래 상태를 먼저 복구한다.

### 로컬 작업 환경

```powershell
cd E:\Mirror_Soul_AI
git branch --show-current
git status --short
```

현재 브랜치는 `codex/ditto-webrtc-video-track`이다. 작업 트리에는 v12 얼굴 품질 개선
변경과 이번 백엔드 콜백 변경이 아직 커밋되지 않은 상태로 함께 남아 있다. 기존 변경을
되돌리거나 전체 파일을 무조건 커밋하지 말고, 변경 목적별로 파일과 diff를 확인한 뒤
선별 커밋한다.

이번 세션에서 추가한 콜백 관련 핵심 파일:

- `model_training/clone_training_callback.py`
- `model_training/routers/training.py`
- `model_training/schemas.py`
- `tests/model_training/test_clone_training_callback.py`
- `docs/api_contract.md`
- `.env.example`

로컬 비공개 `.env`에는 다음 키를 적용했다. 실제 값은 문서와 Git에 기록하지 않는다.

```text
AWS_SQS_FACE_TRAINING_RESULT_QUEUE_URL
CLONE_TRAINING_CALLBACK_BASE_URL
CLONE_TRAINING_CALLBACK_SECRET
```

### GPU 얼굴·Ditto 환경 복구

CMD 또는 PowerShell에서 GPU 호스트에 접속해 컨테이너만 시작한다.

```powershell
ssh -p 20405 C084003@203.249.75.55
```

GPU 호스트에서:

```bash
sudo docker start C084003
exit
```

이후 VS Code에서 `Remote-SSH: Connect to Host...`를 실행해
`mirrorsoul-gpu-container`를 선택하고 다음 폴더를 연다.

```text
/shareHost/C084003-ditto/ditto-talkinghead
```

VS Code 터미널에서:

```bash
conda activate /shareHost/C084003-ditto/conda-env
cd /shareHost/C084003-ditto/ditto-talkinghead
```

정상 프롬프트:

```text
(/shareHost/C084003-ditto/conda-env) mirrorsoul@TeamC084003:/shareHost/C084003-ditto/ditto-talkinghead$
```

v12 결과는 로컬 `얼굴 모델 작업/v12/`, 서버
`/shareHost/C084003-ditto/test-results/v12/`에 있다. 김동빈 우선 확인 결과는
`review/recommended-clone-5-advance00.mp4`, 남성현 결과는
`review/recommended-clone-4-advance20.mp4`다.

### AWS 환경 상태

- AWS AI 서버: `13.209.220.154`
- AI 저장소: `/home/ec2-user/Mirror_Soul_AI`
- 배포 브랜치: `main`
- 배포 커밋: `e4e6b76`
- `mirrorsoul-ai.service`: 실행 중
- `mirrorsoul-voice-worker.service`: 실행 중
- AI 서버 비공개 `.env`: 콜백 주소와 비밀값 키 적용 완료
- AI 서버에서 백엔드 내부 주소 `10.0.1.49:8080` 연결 확인 완료

콜백 코드는 아직 AWS AI 서버에 배포하지 않았으므로 서비스를 재시작하지 않았다.
학교 GPU에는 Mirror Soul 얼굴 워커 저장소가 아직 배포되지 않았고, 현재 Ditto conda
환경에는 `boto3`도 설치되어 있지 않다. 결과 큐 URL은 로컬 배포 원본에만 적용된
상태다.

### 마지막 검증 결과

```text
콜백 + 얼굴 결과 메시지 + 얼굴 워커 테스트: 18개 통과
git diff --check: 오류 없음, Windows CRLF 경고만 존재
```

### 다음 작업 순서

1. 백엔드가 `POST /api/v1/training/profiles` 요청에 실제 `cloneId`를 포함하는지 확인한다.
2. 현재 diff를 목적별로 검토하고 콜백 변경을 선별 커밋·푸시·병합한다.
3. AWS AI 서버에서 `main`을 갱신하고 테스트 후 `mirrorsoul-ai.service`를 재시작한다.
4. 학교 GPU에 Mirror Soul AI 얼굴 워커와 `boto3` 등 의존성을 배포한다.
5. GPU 워커 비공개 환경에 얼굴 요청·결과 SQS URL과 AWS 자격증명을 적용한다.
6. GPU의 결과 큐 `sqs:SendMessage` 및 S3 get/put 권한을 검증한다.
7. 테스트 회원 한 명으로 얼굴, 음성, 성격 완료와 DB 전체 클론 READY를 확인한다.
8. 이후 React Native 원격 비디오 표시와 실제 영상통화 E2E 테스트로 진행한다.

사용자가 제공한 콜백 비밀값은 대화에 노출됐으므로 통합 테스트 완료 후 교체한다.
