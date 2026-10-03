# 회원 RAG 저장·검색 구조 (schema v2)

AI 서버의 회원 RAG는 ChromaDB 컬렉션 하나(`RAG_DB_PATH`, `RAG_COLLECTION_NAME`)에 저장한다.
모든 쓰기는 `model_training/services.py`의 `upsert_rag_documents()`를 거치고, 문서 규격과 ID
규칙은 `model_training/rag_documents.py`에 모여 있다.

## 문서 종류

```text
회원 RAG
├─ profile_snapshot      회원 핵심 프로필 (profileKey당 1개)        ← 구현
├─ interview_memory      인터뷰 질문·답변 1건당 1개                  ← 구현
├─ conversation_memory   통화 중 기억 (예약, 아직 쓰기 경로 없음)
└─ preference_memory     선호 기억 (예약, 아직 쓰기 경로 없음)
```

문서는 거대한 단일 문서가 아니라 검색 가능한 단위로 나눈다. 클론 상태, 활성 음성 ID 같은
운영 데이터는 RAG에 저장하지 않는다.

## 공통 metadata

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `userId` | str | 회원 UUID. 모든 조회·삭제의 격리 기준 |
| `cloneId` | int | 전달된 경우에만 기록 |
| `sourceType` | str | 위 문서 종류 |
| `sourceId` | str | 원본 식별자 |
| `profileKey` | str | `aiProfileId`, 없으면 `default` (v1과 동일) |
| `schemaVersion` | int | `2` |
| `updatedAt` | str | ISO-8601 UTC |
| `importance` | float | 프로필 1.0, 인터뷰 0.8 |
| `confidence` | float | 사용자가 직접 입력한 값은 1.0 |
| `profileManaged` | bool | `/profiles` 동기화가 소유한 문서인지 |

그 밖에 `questionId`, `questionCategory`, `aiProfileId`, `audioUrl`, `mbti`, `keywords` 등은 값이 있을
때만 기록한다. ChromaDB가 지원하는 문자열·숫자·불리언만 사용하고 `None`·리스트는 저장하지 않는다.

## 문서 ID

```text
{userId}:{sourceType}:{sourceId}
```

- `profile_snapshot`: `sourceId = profileKey`
- `interview_memory`: `sourceId` 우선순위
  1. 백엔드가 보낸 `sourceId`
  2. `question-{questionId}`
  3. `question-text-{질문 문장 해시}` (구버전 요청에 questionId가 없을 때)
  4. `answer-text-{답변 해시}` (질문 문장도 없을 때)

인터뷰 순서와 무관하므로 순서가 바뀌어도 같은 ID가 나온다. 같은 원본을 다시 보내면 upsert된다.

## 저장 경로

| API | 역할 | 규칙 |
| --- | --- | --- |
| `POST /api/v1/training/profiles` | 온보딩 RAG 대표 경로 | snapshot + 인터뷰를 한 번에 upsert → 저장 성공 후 이전 문서 정리 → 완료 콜백 |
| `POST /api/v1/training/samples` | 개별 인터뷰(호환 유지) | 같은 문서 빌더·ID·upsert 사용 |

- 같은 인터뷰를 두 API로 보내도 ID가 같으므로 문서는 하나다.
- `/samples`가 이미 `/profiles`가 소유한 문서를 갱신하면 `profileManaged`, `profileKey`를 유지한다.
- `/profiles`가 `/samples` 문서와 같은 인터뷰를 보내면 그 문서는 프로필 동기화 소유가 된다.

### 재학습 시 정리 범위

`/profiles` 저장이 끝난 뒤, 같은 `userId`와 `profileKey`에서 다음 문서만 삭제한다.

- `profileManaged=true`인 `profile_snapshot`·`interview_memory` 중 이번 요청에 없는 문서
- 구버전 `member_profile_summary`, `member_profile_interview` (이번 요청으로 v2 문서가 대체함)

삭제하지 않는 문서: 다른 회원, 다른 profileKey, `/samples`로만 저장된 인터뷰(`profileManaged=false`),
구버전 `interview_answer`, `conversation_memory`·`preference_memory`.

완료 콜백(`/internal/clone-training/{cloneId}/personality/complete`)은 upsert와 정리가 모두 성공한
뒤에만 호출된다. 중간에 실패하면 API가 실패를 반환하고, 같은 요청을 재시도하면 같은 결과가 된다.

## 검색 규칙 (`search_user_memories`)

1. 프로필: `userId`와 프로필 sourceType으로 **정확 조회**해 한 번만 맨 앞에 넣는다.
   v2 `profile_snapshot`이 구버전 `member_profile_summary`보다 우선이고, 여러 개면 최근 `updatedAt`을 쓴다.
   벡터 유사도와 관계없이 포함되며 `topK` 자리를 차지하지 않는다.
2. 기억: 프로필을 제외한 문서만 `userId`로 필터링해 벡터 검색한다.
   - `RAG_MAX_DISTANCE`(기본 0.75)를 넘는 문서는 제외
   - 같은 질문(`questionId`) 또는 같은 텍스트는 하나로 합치고, v2 문서를 우선
   - 중복 제거 여유분을 위해 내부적으로 `topK`보다 많이 조회한 뒤 최대 `topK`개 반환
3. 프롬프트 구성(`model_calling/services.py`)은 프로필·인터뷰·기타 기억으로 묶고
   `RAG_CONTEXT_MAX_CHARS`(기본 3000자) 안에서 자른다.

구버전 sourceType(`member_profile_summary`, `member_profile_interview`, `interview_answer`)도
마이그레이션 기간 동안 그대로 검색·프롬프트에 반영된다.

## 저장소 위치와 통화 중 검색 경로

RAG 저장소(ChromaDB)는 **AWS AI API 서버 한 곳**에만 있다. 저장(`/api/v1/training/profiles`,
`/samples`)과 검색은 모두 이 저장소를 사용한다.

```text
통화 서버 (실시간 답변)
  → POST {RAG_SEARCH_BASE_URL}/internal/rag/search   (X-Rag-Internal-Key)
  → AI API 서버의 search_user_memories()  → AI API 서버의 ChromaDB
```

- 통화 서버는 자체 ChromaDB를 열지 않는다. 예전에는 통화 서버가 자기 디스크의 빈
  `rag_store/chroma`를 검색해 `RAG lookup count=0`이 나왔다.
- 검색 실패·시간 초과(`RAG_SEARCH_TIMEOUT_SECONDS`, 기본 3초)는 통화를 끊지 않는다.
  `RAG lookup skipped ... error_code=RAG_SEARCH_*`를 남기고 RAG 없이 답변한다.
- 통화 서버 `/health`의 `ragSearch`가 `remote`이면 정상, `local`이면 설정 누락이다.
  모니터에는 `RAG search : OK (AI server store)` / `WARNING (local store ...)`로 표시된다.

| 서버 | 설정 |
| --- | --- |
| AI API 서버 | `RAG_INTERNAL_API_KEY=<공유 비밀값>`, `RAG_SEARCH_BASE_URL`은 비움 |
| 통화 서버 | `RAG_SEARCH_BASE_URL=http://<AI 서버 사설 IP>:8000`, 같은 `RAG_INTERNAL_API_KEY` |

AI API 서버에 키가 없으면 내부 검색 API는 `503`으로 거절하고, 키가 다르면 `401`이다.
AWS 보안그룹에서 통화 서버 → AI 서버 8000 포트를 허용해야 하며, 이후 8000 포트를
백엔드 전용으로 제한할 때도 통화 서버는 허용 목록에 남겨야 한다.

| 거절·실패 코드 (통화 로그) | 의미 |
| --- | --- |
| `RAG_SEARCH_UNAVAILABLE` | AI 서버 연결 실패, 시간 초과, 5xx |
| `RAG_SEARCH_CONFIG_ERROR` | 키 미설정·불일치(401/403), AI 서버 키 미설정(503) |
| `RAG_SEARCH_REJECTED` | 요청 형식 오류(4xx) |

## 기존 데이터 마이그레이션

운영 ChromaDB를 초기화하거나 삭제하지 않는다. 구버전 문서는 그대로 읽히므로 마이그레이션은
필수가 아니며, 회원이 다시 `/profiles`를 거치면 해당 profileKey의 구버전 프로필 문서는 자동으로 v2로 대체된다.
한 번에 v2로 옮기려면 다음 스크립트를 사용한다. 기존 임베딩과 텍스트를 재사용하므로 OpenAI 호출이 없다.

```bash
# 1. 백업
cp -r ./rag_store/chroma ./rag_store/chroma.backup-$(date +%Y%m%d%H%M)

# 2. 계획만 확인 (기본값, 쓰기 없음)
python -m tools.migrate_rag_v2 --dry-run

# 3. v2 문서 생성 (구버전 문서는 유지, 재실행해도 결과 동일)
python -m tools.migrate_rag_v2 --apply

# 4. (선택) 검증 후 v2 사본이 생긴 구버전 문서 삭제
python -m tools.migrate_rag_v2 --apply --delete-legacy
```

- `--user-id <uuid>`로 한 회원만 처리할 수 있다.
- 출력은 건수만 표시하고 문서 내용·개인정보는 출력하지 않는다.
- 같은 v2 ID로 모이는 구버전 문서가 여럿이면 `/profiles` 문서를 우선 복사하고 나머지는
  `duplicateLegacyDocuments`로 보고만 하며 `--delete-legacy`로도 지우지 않는다.
- AI 서버가 동시에 쓰는 중에도 upsert라 안전하지만, 가능하면 트래픽이 적은 시간에 실행한다.
