# Model Calling

## Realtime call

The realtime call pipeline runs automatically when `main.py` starts.

```text
WebRTC microphone track
-> utterance detection
-> Whisper STT
-> MBTI + RAG + GPT response
-> ElevenLabs TTS
-> WebRTC output audio track
```

For a `VIDEO` call, the same synthesized reply is rendered by the persistent
Ditto GPU service before playback.

```text
ElevenLabs MP3
-> Ditto GPU render API
-> queued 25 fps video track
-> audio and video playback
-> idle portrait after the reply ends
```

`VOICE` calls keep the existing audio-only behavior. `VIDEO` calls add both an
audio and video sender before the answer SDP is created. The caller's offer must
therefore include a video transceiver capable of receiving the AI video track.

Required video-call environment variables:

```env
DITTO_CALL_SERVICE_URL=https://ditto.internal:8080
DITTO_CALL_SERVICE_URLS=
DITTO_CALL_SERVICE_API_KEY=
DITTO_CALL_S3_BUCKET=mirrorsoul-storage-64
DITTO_CALL_FACE_RESULT_PREFIX=face-results
DITTO_CALL_RETRY_ATTEMPTS=6
DITTO_CALL_RETRY_BASE_SECONDS=1.0
DITTO_CALL_AUDIO_ONLY_FALLBACK=false
REALTIME_IDLE_MOTION_ENABLED=true
REALTIME_IDLE_MOTION_SCALE=0.012
REALTIME_IDLE_MOTION_PERIOD_SECONDS=6.0
REALTIME_VIDEO_WIDTH=540
REALTIME_VIDEO_HEIGHT=960
REALTIME_VIDEO_FPS=25
```

The call server loads the newest
`face-results/{userUuid}/job-*/face-profile.json` and its portrait from S3 once
per call. Its IAM role needs `s3:ListBucket` on the configured prefix and
`s3:GetObject` for profile and portrait objects. A local portrait can be used
for an integration test before the backend profile tables are ready:

```env
DITTO_CALL_SERVICE_URL=http://127.0.0.1:8080
DITTO_CALL_SERVICE_API_KEY=
DITTO_CALL_LOCAL_PORTRAIT_PATH=/path/to/portrait.jpg
DITTO_CALL_LOCAL_PROFILE_PATH=/path/to/face-profile.json
```

Plain HTTP is accepted only for loopback by default. For a Tailscale,
WireGuard, or equivalent encrypted private tunnel, set
`DITTO_CALL_ALLOW_INSECURE_HTTP=true`; use HTTPS on other networks.

For concurrent GPU rendering, set `DITTO_CALL_SERVICE_URLS` to a
comma-separated list such as
`http://127.0.0.1:18080,http://127.0.0.1:18081`. Each URL has one independent
render slot, so different workers render concurrently while requests waiting
for the same pool remain FIFO. `DITTO_CALL_SERVICE_URL` remains compatible with
single-worker deployments. If both settings are present, duplicate URLs are
removed and the singular URL is used as the first worker.

`DITTO_CALL_PARALLEL_MAX_AUDIO_SECONDS` defaults to 8 seconds. A longer reply
reserves every worker slot and renders alone because long Ditto sequences can
consume most of a 24 GB GPU. Set it to `0` only after a GPU-specific benchmark
has shown that unrestricted parallel renders are safe.

The current implementation is turn-based: it waits for the complete TTS audio
and Ditto MP4, then sends decoded frames over WebRTC. It is suitable for the
first video-call integration but does not yet provide frame-by-frame streaming
while the answer is being generated.

Before a rendered reply is queued, the call service decodes the complete MP4
and verifies its video stream, frame count, duration, frame rate, dimensions,
and duration agreement with the TTS audio. Invalid output fails the VIDEO stage
with a stable `DITTO_VIDEO_*` error code, so a successful HTTP response cannot
silently become static portrait plus audio. Thresholds use the
`DITTO_CALL_VIDEO_*` settings documented in `.env.example`.

The first implementation is turn based. A user utterance is finalized after a
short silence, then the answer is generated and played.

Required environment variables:

```env
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
BACKEND_API_BASE_URL=https://api.mirrorsoul64.com
AI_INTERNAL_API_KEY=<shared-secret>
BACKEND_CALL_CONTEXT_TIMEOUT_SECONDS=5
```

On `CALL_INVITE`, the call server sends one authenticated request to
`GET /internal/ai/calls/{callId}/context`. It validates and caches the returned
clone, persona, voice, and media type for the lifetime of that call. Realtime
turn processing reads only this in-memory context and does not query MySQL.
`AI_INTERNAL_API_KEY` must match the backend setting and must never be logged.

The voice training worker below still writes training results to RDS, so its
deployment continues to require the `DB_*` variables documented in the root
`.env.example`.

## Voice training worker

Backend publishes voice clone requests to SQS after onboarding interview audio
or voice update audio is uploaded to S3. The AI worker downloads the files,
creates an ElevenLabs voice clone, and publishes status to a separate result
queue. Backend owns job state, active voice profile storage, and aggregate
similarity score updates.

Expected SQS message contract:

```json
{
  "jobType": "VOICE_TRAINING",
  "source": "ONBOARDING_INTERVIEW",
  "jobId": 1,
  "userUuid": "d6bfd311-3c88-40b5-992c-31f12b4f06fd",
  "bucket": "mirrorsoul-bucket",
  "audioObjectKeys": [
    "interviews/d6bfd311-3c88-40b5-992c-31f12b4f06fd/sample.wav"
  ],
  "requestedAt": "2026-07-14T00:00:00Z"
}
```

Additional environment variables:

```env
AWS_REGION=ap-northeast-2
AWS_SQS_VOICE_TRAINING_QUEUE_URL=
AWS_SQS_VOICE_TRAINING_RESULT_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/447343943160/mirrorsoul-voice-training-result-queue
VOICE_TRAINING_WAIT_SECONDS=20
VOICE_TRAINING_VISIBILITY_TIMEOUT=600
VOICE_TRAINING_MAX_ATTEMPTS=3
VOICE_TRAINING_IDEMPOTENCY_CACHE_SIZE=1000
VOICE_TRAINING_AUDIO_QUALITY_ENABLED=true
VOICE_TRAINING_FFMPEG_BINARY=ffmpeg
VOICE_TRAINING_MIN_ACCEPTED_SAMPLES=3
VOICE_TRAINING_MIN_BATCH_DURATION_SECONDS=8
```

Before sending audio to ElevenLabs, the worker converts every supported input
to 16 kHz mono PCM WAV and checks duration, loudness, silence ratio, and
clipping. Invalid individual samples are excluded. The job fails before voice
creation when fewer than three valid samples remain or their combined duration
is below eight seconds. Thresholds can be tuned with the
`VOICE_TRAINING_*` values documented in `.env.example`; disable the gate only
as a temporary rollback with `VOICE_TRAINING_AUDIO_QUALITY_ENABLED=false`.

Quality logs use sample numbers instead of S3 keys:

```text
[VOICE_TRAINING_QUALITY] sample: job_id=17 sample=1 status=ACCEPTED ...
[VOICE_TRAINING_QUALITY] sample: job_id=17 sample=4 status=REJECTED reasons=too_much_silence
[VOICE_TRAINING_QUALITY] batch: job_id=17 status=PASSED accepted=4 rejected=1 duration=22.41s
```

Run once for a manual smoke test:

```bash
python -m model_calling.voice_training.worker --once
```

Run continuously on the AI server:

```bash
python -m model_calling.voice_training.worker
```

Successful worker flow:

```text
SQS message
-> PROCESSING result event
-> S3 audio download
-> input normalization and quality gate
-> ElevenLabs /v1/voices/add
-> COMPLETED result event with voice ID and voice score
-> backend stores active profile, job status, and aggregate score
```

The worker requires non-empty `audioObjectKeys` in every request. It deletes a
request only after publishing `COMPLETED` or a final `FAILED`. A retryable
failure retains the request and publishes another final result after retry.
Completed results are cached by `jobId` in the running worker so ordinary SQS
redelivery does not create another ElevenLabs voice. The backend consumer must
also process duplicate or stale result events idempotently by `jobId`.

Result event contract (`PROCESSING` and `FAILED` omit `result`):

```json
{
  "eventType": "VOICE_TRAINING_STATUS",
  "jobId": 1,
  "userUuid": "d6bfd311-3c88-40b5-992c-31f12b4f06fd",
  "status": "COMPLETED",
  "result": {
    "elevenlabsVoiceId": "example-voice-id",
    "voiceScore": 62.0
  }
}
```

`voiceScore` is always present on completion, is clamped to `0..100`, and has
at most two decimal places. `introAudio` is optional and is currently omitted.
If it is added, upload it to S3 before publishing the result and include
`bucket`, `objectKey`, `contentType`, `sizeBytes`, and `durationMs`.

On failure, `error` has `code`, `message`, and `retryable`. Retryable failures
are attempted up to `VOICE_TRAINING_MAX_ATTEMPTS`; the last failure is published
with `retryable: false`.
Backend should validate job/member/clone identity, ignore stale or duplicate
events, activate the returned voice ID on `COMPLETED`, and recompute the total
score with the supplied `voiceScore`. The backend resolves the clone from
`userUuid`; voice result events do not contain `cloneId`. When speaker embedding
cannot be computed, `voiceScore` uses the existing accepted-sample readiness
proxy rather than a measured voice match.

## Clone similarity score

After a voice clone is created, the worker computes only the voice component.
The backend stores that component from the result event and recalculates the
user-facing clone completeness score in the same transaction. The worker does
not read or write RDS while processing voice training jobs.

After each voice clone is created, the worker generates and saves a fixed
reference sentence with the newly created ElevenLabs voice:

```text
안녕하세요! 처음뵙겠습니다.
```

When speaker embedding evaluation is enabled, the worker extracts speaker
embeddings from the original member recordings and this saved clone reference
audio, then converts their cosine similarity into the voice score. Before
evaluation, FFmpeg normalizes m4a, mp3, webm, and wav inputs to 16 kHz mono PCM
WAV so the speaker model does not depend on container codec support. If FFmpeg,
the optional model dependencies, or evaluation fails, the worker keeps the
voice clone result and falls back to the conservative readiness score.

Optional speaker similarity dependencies:

```bash
pip install -r requirements-voice-similarity.txt
```

If embedding evaluation is unavailable, the worker uses accepted audio sample
count as a readiness proxy and marks the result `sample_coverage_proxy`. The
backend stores `voiceScore` in `clones.voice_similarity_score` and combines it
with face, profile, and reliability components. The displayed aggregate is
capped at 95.0 and is not a biometric probability.

Optional tuning:

```env
CLONE_SIMILARITY_ENABLE_SPEAKER_EMBEDDING=false
CLONE_SIMILARITY_EXPECTED_VOICE_SAMPLES=5
CLONE_SIMILARITY_EXCELLENT_VOICE_SAMPLES=20
CLONE_SIMILARITY_SPEAKER_MODEL=speechbrain/spkrec-ecapa-voxceleb
CLONE_SIMILARITY_COSINE_LOW=0.20
CLONE_SIMILARITY_COSINE_HIGH=0.70
CLONE_SIMILARITY_MAX_ACTUAL_VOICE_SCORE=100
CLONE_SIMILARITY_REFERENCE_TEXT=안녕하세요! 처음뵙겠습니다.
CLONE_SIMILARITY_REFERENCE_AUDIO_DIR=model_calling/assets/clone_similarity
FFMPEG_BIN=ffmpeg
```

Overall score calculation and the backend migration contract are documented in
`docs/clone-similarity-score-contract.md`.

The final aggregate score is owned by the backend; see the score contract for
its weighting, penalties, and display range.

Optional realtime response and voice activity detection settings:

```env
RAG_MAX_DISTANCE=0.75
RAG_TOP_K=6
RAG_CONTEXT_MAX_CHARS=3000
REALTIME_HISTORY_MAX_TURNS=8
REALTIME_RAG_QUERY_HISTORY_TURNS=2
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.4
LLM_MAX_OUTPUT_TOKENS=200
LLM_RESPONSE_MAX_CHARS=240
LLM_RESPONSE_MAX_SENTENCES=3
LLM_PERSONALITY_SIGNAL_THRESHOLD=12
LLM_REASONING_EFFORT=none
STT_MODEL=whisper-1
STT_LANGUAGE=ko
REALTIME_STT_TIMEOUT_SECONDS=30
REALTIME_CONTEXT_TIMEOUT_SECONDS=15
REALTIME_RAG_TIMEOUT_SECONDS=10
REALTIME_LLM_TIMEOUT_SECONDS=30
REALTIME_TTS_TIMEOUT_SECONDS=45
REALTIME_VIDEO_TIMEOUT_SECONDS=180
REALTIME_UTTERANCE_MAX_QUEUE_SECONDS=20
DITTO_CALL_QUEUE_TIMEOUT_SECONDS=90
REALTIME_VAD_ENERGY_THRESHOLD=900
REALTIME_VAD_SILENCE_SECONDS=0.8
REALTIME_VAD_MIN_SPEECH_SECONDS=0.7
REALTIME_VAD_MAX_SPEECH_SECONDS=15
REALTIME_VAD_STARTUP_GRACE_SECONDS=1.5
```

`RAG_MAX_DISTANCE` filters semantically distant member memories before they
reach the LLM. Realtime calls keep only the latest
`REALTIME_HISTORY_MAX_TURNS` user/assistant pairs, scoped to one call and
discarded when that WebRTC session closes. The member profile summary is always
included before the filtered interview memories. `LLM_MODEL` and
`LLM_TEMPERATURE` make controlled model comparisons possible without code
changes. GPT-5/6 and o-series models automatically use
`LLM_REASONING_EFFORT` and the completion-token parameter expected by those
models.

Realtime RAG searches with the current transcript plus the latest
`REALTIME_RAG_QUERY_HISTORY_TURNS` user turns. Assistant replies are excluded
from the search query so a generated claim cannot become retrieval evidence.
The prompt separates verified profile fields, direct interview answers,
personality guidance, speech style, and weak MBTI fallback information. Big Five
values within `LLM_PERSONALITY_SIGNAL_THRESHOLD` of 50 are treated as neutral
and do not influence the persona. Retrieved memory text is deduplicated and
bounded by `RAG_CONTEXT_MAX_CHARS`.

Generated replies are normalized to `LLM_RESPONSE_MAX_SENTENCES` and
`LLM_RESPONSE_MAX_CHARS`, with duplicate sentences removed. An empty provider
response becomes a short conversational fallback. These limits reduce TTS and
video latency while keeping realtime answers concise.

Each realtime stage has an independent timeout so one external dependency does
not leave the call pipeline stuck indefinitely. RAG timeout is a soft failure
and the reply continues without retrieved memories. STT, member context, LLM,
TTS, and required video failures stop only the affected turn and emit a stable
`error_code`. Utterances older than the queue-age limit are skipped, and a full
queue replaces its oldest item with the caller's latest speech. Ending a call
cancels and awaits every receiver, processing, and video preparation task.

Successful conversation logs:

```text
[WEBRTC] track received: callId=81 kind=audio
[REALTIME] STT user=... callId=81 turn=1 elapsed_ms=...: ...
[REALTIME] RAG lookup complete: callId=81 turn=1 ... elapsed_ms=...
[REALTIME] LLM user=... callId=81 turn=1 elapsed_ms=...: ...
[REALTIME] TTS complete: callId=81 turn=1 ... elapsed_ms=...
[CALL_TRACE] turn completed: callId=81 turn=1 ... total_ms=...
[CALL_TRACE] call closed: callId=81 ... status=COMPLETED ...
```

Every realtime stage includes the call ID and utterance sequence. The turn
summary contains per-stage latency, while the call summary reports completed,
failed, skipped, and cancelled turn counts. This keeps concurrent calls
separable in `tools/realtime_call_monitor.py`.

If `data/{user_id}/persona.json` exists, the realtime pipeline uses its
personality, speech style, and ElevenLabs voice ID. Otherwise it loads the
member profile and MBTI from RDS.

For RDS-backed calls, the pipeline first looks for an active voice profile:

```sql
SELECT avp.elevenlabs_voice_id
FROM ai_voice_profiles avp
JOIN clones c ON c.id = avp.clone_id
JOIN users u ON u.id = c.user_id
WHERE u.uuid = ?
  AND avp.status = 'ACTIVE'
  AND avp.is_active = TRUE
ORDER BY avp.updated_at DESC
LIMIT 1;
```

An active `ai_voice_profiles` row matching both the invited member UUID and
`cloneId` is required. Calls fail explicitly when the member voice is missing,
the clone IDs differ, or RDS cannot be queried; the server never substitutes
another member's voice from a global environment variable.
