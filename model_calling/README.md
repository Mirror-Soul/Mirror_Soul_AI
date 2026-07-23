# Model Calling

## Realtime voice call

The realtime call pipeline runs automatically when `main.py` starts.

```text
WebRTC microphone track
-> utterance detection
-> Whisper STT
-> MBTI + RAG + GPT response
-> ElevenLabs TTS
-> WebRTC output audio track
```

The first implementation is turn based. A user utterance is finalized after a
short silence, then the answer is generated and played.

Required environment variables:

```env
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=

DB_HOST=
DB_PORT=3306
DB_NAME=
DB_USERNAME=
DB_PASSWORD=
```

## Voice training worker

Backend publishes voice clone requests to SQS after onboarding interview audio
or voice update audio is uploaded to S3. The AI worker consumes that message,
downloads the audio files, creates an ElevenLabs voice clone, and stores the
active voice profile in RDS.

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
VOICE_TRAINING_WAIT_SECONDS=20
VOICE_TRAINING_VISIBILITY_TIMEOUT=600
VOICE_TRAINING_DELETE_FAILED_MESSAGES=true
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
-> S3 audio download
-> ElevenLabs /v1/voices/add
-> ai_voice_profiles active row
-> clone similarity score update
-> voice_training_jobs COMPLETED
```

## Clone similarity score

After a voice clone is created, the worker calculates an internal clone
similarity score from the member's active voice clone, onboarding interview
coverage, and basic profile completeness. Frontend can show the total score as
the clone similarity. If the user taps the score, show the generated explanation
text instead of exposing the full formula.

After each voice clone is created, the worker generates and saves a fixed
reference sentence with the newly created ElevenLabs voice:

```text
안녕하세요! 처음뵙겠습니다.
```

When speaker embedding evaluation is enabled, the worker extracts speaker
embeddings from the original member recordings and this saved clone reference
audio, then converts their cosine similarity into the voice score. If the
optional model dependencies are not installed or evaluation fails, the worker
keeps the voice clone result and falls back to the conservative readiness score.

Optional speaker similarity dependencies:

```bash
pip install -r requirements-voice-similarity.txt
```

The total score is always written to `clones.sync_rate` so existing backend
responses can use it. If the optional detail table exists, the worker also
stores score history and explanation there:

```sql
CREATE TABLE ai_clone_similarity_scores (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    clone_id BIGINT NOT NULL,
    voice_profile_id BIGINT NULL,
    voice_training_job_id BIGINT NULL,
    voice_score DECIMAL(5,2) NULL,
    interview_score DECIMAL(5,2) NULL,
    profile_score DECIMAL(5,2) NULL,
    total_score DECIMAL(5,2) NOT NULL,
    explanation TEXT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'COMPLETED',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

Optional tuning:

```env
CLONE_SIMILARITY_ENABLE_SPEAKER_EMBEDDING=false
CLONE_SIMILARITY_EXPECTED_VOICE_SAMPLES=5
CLONE_SIMILARITY_EXCELLENT_VOICE_SAMPLES=20
CLONE_SIMILARITY_EXPECTED_INTERVIEWS=5
CLONE_SIMILARITY_EXCELLENT_INTERVIEWS=15
CLONE_SIMILARITY_VOICE_WEIGHT=0.60
CLONE_SIMILARITY_INTERVIEW_WEIGHT=0.25
CLONE_SIMILARITY_PROFILE_WEIGHT=0.15
CLONE_SIMILARITY_SPEAKER_MODEL=speechbrain/spkrec-ecapa-voxceleb
CLONE_SIMILARITY_COSINE_LOW=0.20
CLONE_SIMILARITY_COSINE_HIGH=0.70
CLONE_SIMILARITY_MAX_ACTUAL_VOICE_SCORE=95
CLONE_SIMILARITY_REFERENCE_TEXT=안녕하세요! 처음뵙겠습니다.
CLONE_SIMILARITY_REFERENCE_AUDIO_DIR=model_calling/assets/clone_similarity
```

The first complete onboarding clone is intentionally calibrated around the
low-to-mid 60s. More interviews and voice samples raise the score over time;
scores above 90 should feel exceptional rather than automatic.

Optional voice activity detection settings:

```env
REALTIME_VAD_ENERGY_THRESHOLD=900
REALTIME_VAD_SILENCE_SECONDS=0.8
REALTIME_VAD_MIN_SPEECH_SECONDS=0.7
REALTIME_VAD_MAX_SPEECH_SECONDS=15
REALTIME_VAD_STARTUP_GRACE_SECONDS=1.5
```

Successful conversation logs:

```text
[WEBRTC] track received: kind=audio
[REALTIME] STT user=...: ...
[REALTIME] LLM user=...: ...
[REALTIME] reply audio queued
```

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

If no active `ai_voice_profiles` row exists, or the table is not available yet,
the call continues with the default `ELEVENLABS_VOICE_ID`.
