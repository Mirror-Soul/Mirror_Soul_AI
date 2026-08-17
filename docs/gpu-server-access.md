# Mirror Soul GPU 서버 접속 가이드

이 문서는 홍익대학교 GPU 5번 서버의 Mirror Soul 팀 컨테이너에 다시 접속해
얼굴 모델 개발 환경을 이어서 사용하는 절차를 정리한다.

비밀번호는 이 저장소에 기록하지 않는다. 호스트 계정 비밀번호와 컨테이너
root 비밀번호는 서로 다른 비밀번호일 수 있다.

## 서버 정보

| 구분 | 값 |
| --- | --- |
| GPU 호스트 | `203.249.75.55` |
| 호스트 SSH 포트 | `20405` |
| 호스트 계정 | `C084003` |
| 팀 컨테이너 | `C084003` |
| 컨테이너 SSH 포트 | `40053` |
| API 포트 | `42053` -> 컨테이너 `8080` |
| DB 포트 | `44053` -> 컨테이너 `3306` |
| Jupyter 포트 | `46053` -> 컨테이너 `8888` |
| 작업 경로 | `/workspace/mirror-soul-face` |
| Conda 환경 | `mirrorsoul-face` |
| tmux 세션 | `liveportrait-setup` |

## 권장 접속 순서

### 1. Windows CMD에서 GPU 호스트 접속

실행 장소: Windows CMD 또는 PowerShell

정상 프롬프트 예시:

```text
E:\Mirror_Soul_AI>
```

명령:

```powershell
ssh C084003@203.249.75.55 -p 20405
```

호스트 계정 비밀번호를 입력한다. 입력 중에는 화면에 문자가 표시되지 않는 것이
정상이다. 접속되면 아래 프롬프트가 나타난다.

```text
C084003@HGW5:~$
```

### 2. 팀 컨테이너 상태 확인 및 시작

실행 장소: GPU 호스트

정상 프롬프트:

```text
C084003@HGW5:~$
```

명령:

```bash
sudo docker ps
```

목록에 `C084003`이 없으면 컨테이너를 시작한다.

```bash
sudo docker start C084003
sudo docker ps
```

`STATUS`가 `Up ...`이고 이름이 `C084003`이면 정상이다.

학교 서버는 허용된 sudo 명령을 정확히 검사할 수 있다. 옵션을 임의로 추가한
`sudo docker ps --filter ...` 대신 위의 명령을 그대로 사용한다.

### 3. 팀 컨테이너 진입

실행 장소: GPU 호스트

정상 프롬프트:

```text
C084003@HGW5:~$
```

명령을 정확히 입력한다.

```bash
sudo docker exec -it C084003 /bin/bash
```

`bash`만 쓰면 sudo 허용 규칙과 일치하지 않을 수 있으므로 `/bin/bash`를 사용한다.
진입 후 프롬프트는 다음과 같다.

```text
root@TeamC084003:/#
```

### 4. 얼굴 모델 개발 환경 활성화

실행 장소: 팀 컨테이너

정상 프롬프트 예시:

```text
root@TeamC084003:/#
```

명령:

```bash
cd /workspace/mirror-soul-face
source /opt/conda/etc/profile.d/conda.sh
conda activate mirrorsoul-face
```

성공하면 프롬프트 앞에 `(mirrorsoul-face)`가 표시된다.

```text
(mirrorsoul-face) root@TeamC084003:/workspace/mirror-soul-face#
```

LivePortrait 작업 폴더로 이동하려면 다음을 실행한다.

```bash
cd /workspace/mirror-soul-face/liveportrait
```

### 5. tmux 작업 이어서 열기

먼저 세션을 확인한다.

```bash
tmux ls
```

`liveportrait-setup` 세션이 있으면 이어서 연다.

```bash
tmux attach -t liveportrait-setup
```

세션이 없다는 메시지가 나오면 새로 만든다.

```bash
tmux new -s liveportrait-setup
```

tmux 화면 아래에 상태 표시줄이 보이면 정상이다. 네트워크가 끊겨도 tmux 안에서
실행 중인 설치나 작업은 계속 진행될 수 있다.

## 안전하게 접속 종료하기

### 1. tmux 작업 유지한 채 분리

명령어를 입력하는 것이 아니라 키보드로 다음 순서대로 누른다.

```text
Ctrl+B를 누르고 떼기 -> D 누르기
```

`[detached ...]`가 나오면 작업은 유지된 채 tmux 화면에서 나온 것이다.

### 2. 컨테이너와 호스트에서 나오기

컨테이너 프롬프트에서:

```bash
exit
```

`C084003@HGW5:~$`가 나오면 호스트로 나온 것이다. 다시 실행한다.

```bash
exit
```

Windows의 `E:\Mirror_Soul_AI>` 프롬프트로 돌아오면 접속이 완전히 종료된 것이다.

### 3. 컨테이너를 중지해야 할 때

GPU 예약 종료나 학교 운영 지침에 따라 중지가 필요할 때만 GPU 호스트에서 실행한다.

```bash
sudo docker stop C084003
```

컨테이너의 `stop`과 `start` 사이에는 설치 및 작업 파일이 유지된다. 다만 컨테이너가
삭제되거나 새로 생성되면 내부 `/workspace` 데이터가 사라질 수 있으므로 중요한 결과물과
환경 잠금 파일은 별도로 백업한다.

## 빠른 재접속 명령 모음

Windows에서:

```powershell
ssh C084003@203.249.75.55 -p 20405
```

GPU 호스트에서:

```bash
sudo docker start C084003
sudo docker exec -it C084003 /bin/bash
```

팀 컨테이너에서:

```bash
cd /workspace/mirror-soul-face
source /opt/conda/etc/profile.d/conda.sh
conda activate mirrorsoul-face
tmux attach -t liveportrait-setup
```

## 자주 발생한 문제

### `port 40053: Connection refused`

컨테이너가 중지되어 있거나 컨테이너의 SSH가 아직 준비되지 않은 경우다. 직접 root SSH를
반복하지 말고, 먼저 호스트 포트 `20405`로 접속한 후 아래를 실행한다.

```bash
sudo docker start C084003
sudo docker ps
sudo docker exec -it C084003 /bin/bash
```

### `root@203.249.75.55: Permission denied`

컨테이너 root 비밀번호와 호스트 `C084003` 비밀번호가 다르거나 root SSH 인증에 실패한
경우다. 권장 접속 경로인 호스트 접속 후 `docker exec`를 사용한다. 컨테이너 root
비밀번호를 변경해야 한다면 컨테이너 안에서 `passwd root`를 실행한다.

### `not allowed to execute docker exec`

명령이 학교 sudo 허용 규칙과 정확히 일치하지 않을 가능성이 크다. 다음 명령을 그대로 쓴다.

```bash
sudo docker exec -it C084003 /bin/bash
```

### 비밀번호 만료 또는 IP 차단

학교 장애 해결 양식에서 비밀번호 초기화 또는 현재 Public IP의 차단 해제를 요청한다.
비밀번호가 초기화되면 최초 호스트 로그인에서 새 비밀번호 변경을 요구할 수 있다.

### 현재 위치를 모르겠을 때

| 프롬프트 | 현재 위치 |
| --- | --- |
| `E:\Mirror_Soul_AI>` | 내 Windows PC |
| `C084003@HGW5:~$` | 학교 GPU 호스트 |
| `root@TeamC084003:/#` | 팀 Docker 컨테이너 |
| `(mirrorsoul-face) root@TeamC084003:...#` | 컨테이너 + 얼굴 모델 Conda 환경 |
| 화면 아래 tmux 상태 표시줄 | tmux 세션 내부 |

## 현재 환경 기록 파일

환경 복구를 위해 다음 파일을 생성해 두었다.

```text
/workspace/mirror-soul-face/liveportrait-requirements.lock.txt
/workspace/mirror-soul-face/mirrorsoul-face-environment.yml
```

이 파일들은 현재 Python 패키지와 Conda 환경 구성을 기록한 스냅샷이다.
