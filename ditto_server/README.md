# Ditto GPU Rendering Service

Ditto 모델을 GPU 메모리에 한 번 로드한 뒤 얼굴 이미지와 음성 요청을 순차 처리하는
내부 렌더링 서비스다. 회원가입 자동화가 생성한 `face-profile.json`의 설정을 받아
MP4를 반환한다.

현재 API는 통화 서버 연결 전 검증용 MP4 렌더 단계다. 프레임 스트리밍과 WebRTC
비디오 트랙 연결은 다음 단계다.

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
export DITTO_SERVICE_HOST=127.0.0.1
export DITTO_SERVICE_PORT=8080

/shareHost/C084003-ditto/conda-env/bin/python -m ditto_server.main
```

모델 로드에는 현재 RTX 4090 기준 약 25초가 걸린다. `/ready`가 HTTP 200을 반환한
뒤 렌더 요청을 보낸다.

```bash
curl http://127.0.0.1:8080/ready
```

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

GPU 렌더는 한 번에 한 요청만 처리한다. 이미 렌더 중이면 HTTP `429`를 반환하므로
호출 측에서 지수 백오프로 재시도해야 한다.

모델 로드와 렌더는 같은 전용 작업 스레드에서 실행한다. CUDA와 ONNX Runtime을
서로 다른 스레드에서 초기화하고 사용하면 GPU 컨텍스트가 중복 생성되어 RTX 4090의
메모리를 소진할 수 있으므로 이 실행 구조를 변경하지 않는다.

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
