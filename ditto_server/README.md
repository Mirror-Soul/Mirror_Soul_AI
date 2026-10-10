# Ditto GPU Rendering Service

Ditto 모델을 GPU 메모리에 한 번 로드한 뒤 얼굴 이미지와 음성 요청을 순차 처리하는
내부 렌더링 서비스다. 회원가입 자동화가 생성한 `face-profile.json`의 설정을 받아
MP4를 반환한다. 선택적으로 온라인 파이프라인을 함께 로드해 생성되는 프레임을 통화
서버에 즉시 전송할 수 있다.

## 설치

GPU 서버의 AI 저장소에서 Ditto 전용 Python으로 서비스 의존성을 설치한다.
버전은 Python 3.10과 함께 검증한 조합으로 고정되어 있다.

```bash
/shareHost/C084003-ditto/conda-env/bin/python -m pip install \
  -r requirements-ditto-service.txt
```

## 실행

API 키는 충분히 긴 임의 문자열을 사용하고 저장소에 커밋하지 않는다.

```bash
export DITTO_SERVICE_API_KEY='<secret>'
export DITTO_SERVICE_BACKEND=pytorch
export DITTO_SERVICE_HOST=127.0.0.1
export DITTO_SERVICE_PORT=8080
export DITTO_SERVICE_STREAMING_ENABLED=false

/shareHost/C084003-ditto/conda-env/bin/python -m ditto_server.main
```

모델 로드에는 현재 RTX 4090 기준 약 25초가 걸린다. `/ready`가 HTTP 200을 반환한
뒤 렌더 요청을 보낸다.

```bash
curl http://127.0.0.1:8080/ready
```

## PyTorch와 TensorRT 선택

기본 백엔드는 기존과 같은 `pytorch`다. RTX 4090 서버에 준비된 TensorRT 엔진을
사용하려면 `.env.ditto-service`에 다음 값을 넣고 워커를 재시작한다.

```env
DITTO_SERVICE_BACKEND=tensorrt
DITTO_SERVICE_TENSORRT_PYTHON_PATH=/shareHost/C084003-ditto/trt-pkgs
DITTO_SERVICE_TENSORRT_LIBRARY_PATH=/shareHost/C084003-ditto/trt-pkgs/tensorrt_libs
```

`DITTO_SERVICE_DATA_ROOT`와 `DITTO_SERVICE_CONFIG_PATH`를 비워 두면 백엔드에 따라
PyTorch 또는 TensorRT 기본 경로를 자동으로 선택한다. 특정 모델을 검증할 때만 두
경로를 명시한다. `/health`와 `/ready`의 `engine.backend`에서 실제 선택값을 확인할 수
있다. 문제가 생기면 `DITTO_SERVICE_BACKEND=pytorch`로 되돌린 뒤 워커를 재시작한다.

2026-10-10 RTX 4090 격리 시험에서 TensorRT 온라인 경로는 약 47 fps(25 fps 실시간의
1.88배)를 기록했고, 현재 서비스와 같은 오프라인 렌더 경로도 MP4 생성에 성공했다.
실제 회원 얼굴과 긴 답변 품질은 운영 전 별도 확인한다.

## 온라인 프레임 스트리밍

운영 검증 전에는 꺼져 있다. GPU 서비스와 통화 서버 코드를 모두 배포한 뒤 다음 값을
설정하고 GPU 워커를 재시작한다.

```env
DITTO_SERVICE_STREAMING_ENABLED=true
DITTO_SERVICE_ONLINE_CONFIG_PATH=
DITTO_SERVICE_STREAM_JPEG_QUALITY=85
```

TensorRT에서는 온라인 전용 `v0.4_hubert_cfg_trt_online.pkl`을 자동 선택한다. `/ready`의
`engine.streamingAvailable=true`를 확인한 뒤 통화 서버의
`DITTO_CALL_STREAMING_ENABLED=true`를 적용한다. 스트리밍 API는
`POST /api/v1/render/stream`이며 외부 공개 API가 아니라 기존 SSH 터널 내부에서만
사용한다. 응답은 `MSDS1` 헤더와 길이 구분 JPEG 프레임으로 구성된다.

통화 서버는 기본 8프레임을 모은 뒤 음성과 영상을 시작하며, 스트림 초기화나 전송에
실패하면 기존 `/api/v1/render` MP4 경로로 한 번 복구한다. 즉시 롤백하려면 양쪽의
스트리밍 플래그를 `false`로 바꾸고 서비스를 재시작한다.

## 렌더 요청

```bash
curl -X POST http://127.0.0.1:8080/api/v1/render \
  -H "X-Ditto-Api-Key: ${DITTO_SERVICE_API_KEY}" \
  -F "portrait=@/path/to/portrait.jpg" \
  -F "audio=@/path/to/speech.wav" \
  -F "profile=@/path/to/face-profile.json;type=application/json" \
  -o render.mp4
```

프로필을 보내지 않으면 v7에서 선택한 smooth 기본값을 사용한다.

```text
cropScale=2.3
smoothingKernel=5
samplingTimesteps=50
seed=1024
```

선택 필드로 클립의 앞뒤를 원래 초상화 자세로 되돌릴 수 있다. 통화 서버는 답변과
대기 클립을 이어 붙일 때 얼굴이 튀지 않도록 이 값을 보낸다
(`docs/realtime-video-transitions.md`).

```text
fade_in_frames=2      # 시작 N프레임을 초상화 자세에서 출발 (0~250)
fade_out_frames=8     # 마지막 N프레임 동안 초상화 자세로 복귀 (0~250)
fade_type=s           # s=초상화 자세, d0=첫 생성 프레임
fade_keys=exp,pitch,yaw,roll,t
```

두 프레임 값이 모두 없거나 0이면 기존과 똑같이 렌더한다. 범위를 벗어나면 `422`를 반환한다.

GPU 렌더는 한 번에 한 요청만 처리한다. 이미 렌더 중이면 HTTP `429`를 반환하므로
호출 측에서 지수 백오프로 재시도해야 한다.

모델 로드와 렌더는 같은 전용 작업 스레드에서 실행한다. CUDA와 ONNX Runtime을
서로 다른 스레드에서 초기화하고 사용하면 GPU 컨텍스트가 중복 생성되어 RTX 4090의
메모리를 소진할 수 있으므로 이 실행 구조를 변경하지 않는다.

동시 통화가 필요하면 한 프로세스의 CUDA 스레드를 늘리지 않고 Ditto 서비스를 서로
다른 포트의 독립 프로세스로 실행한다. `tools/gpu/start-ditto-worker-pool.sh`는 기본적으로
8080과 8081에 두 워커를 순차 로드한다. 워커마다 모델과 CUDA 컨텍스트를 별도로
보유하므로 운영 워커 수를 늘리기 전에는 반드시 실제 렌더 중 최고 VRAM과 처리 시간을
측정한다.

2026-09-15 실제 GPU API 검증 결과:

```text
model load: 30.4초
first render: 16.9초
second render with the same model: 2.4초
output: 1080x1920, 25fps, 2.12초, H.264 + AAC
```

## 외부 연결 보안

얼굴과 음성은 생체정보이므로 공개 네트워크에서 평문 HTTP로 전송하지 않는다.
기본값은 `127.0.0.1` 바인딩이다. 개발 중에는 VS Code 포트 포워딩이나 SSH 터널을
사용한다. AWS 통화 서버와 연결할 때는 다음 중 하나를 먼저 구성한다.

- WireGuard 또는 Tailscale 같은 암호화된 사설 네트워크
- TLS 인증서를 설정한 HTTPS

TLS를 직접 사용할 때는 인증서와 개인 키를 함께 설정한다.

```env
DITTO_SERVICE_HOST=0.0.0.0
DITTO_SERVICE_SSL_CERTFILE=/path/to/fullchain.pem
DITTO_SERVICE_SSL_KEYFILE=/path/to/private-key.pem
```

호스트 방화벽에서도 AWS 통화 서버의 주소만 허용해야 한다.
